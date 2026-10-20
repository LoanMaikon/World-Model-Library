from src.methods.leworldmodel.models import Encoder, ActionEncoder, Predictor, Projector, JEPA, SIGReg
from stable_worldmodel import World, PlanConfig
from src.methods.leworldmodel.planning import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from src.methods.leworldmodel.solver import GradientSolver
from src.datasets import get_dataset_config, get_environment_config, load_dataset, make_image_transform, ActionNormalizer
from src.states import create_state, save_state, load_state, restore_state
from torch.nn.parallel import DistributedDataParallel as DDP
from src.dataloaders import load_dataloader
from src.schedulers import OptimizerScheduler
from src.optimizers import load_optimizer, load_planning_optimizer
from src.utils import get_dtype, write_on_log, write_on_csv, save_json
from copy import deepcopy
from time import perf_counter
import numpy as np
import torch
import os

class LeWorldModel():
    def __init__(self,
                 opened_config,
                 output_folder,
                 device,
                 rank,
                 world_size,
                 resume,
                 evaluate_state=None,
                ):

        self.config = opened_config
        self.output_folder = output_folder
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.evaluate_state = evaluate_state
        self.is_evaluating = self.config["mode"] == "evaluate"

        self._load_config()
        self._load_state()

    def train(self):
        for epoch in range(self.start_epoch, self.optimization_num_epochs + 1):
            self.model.train()
            self.train_sampler.set_epoch(epoch)

            lr_warmup = self.optimization_lr_interval == "epoch" and epoch <= self.lr_scheduler.warmup_steps
            if lr_warmup:
                self.lr_scheduler.step()

            self._train_epoch(epoch)

            if self.optimization_lr_interval == "epoch" and not lr_warmup:
                self.lr_scheduler.step()
            if self.optimization_weight_decay_interval == "epoch":
                self.wd_scheduler.step()

            self._save_state(epoch)

    def _train_epoch(self, epoch):
        write_on_log(f"Training epoch {epoch}", self.output_folder)

        dtype = get_dtype(self.optimization_precision)
        loss_sums = torch.zeros(4, dtype=torch.float64, device=self.device)

        for iteration, batch in enumerate(self.train_dataloader, start=1):
            batch = {
                key: value.to(self.device, non_blocking=self.data_pin_memory) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            lr = self.optimizer.param_groups[0]["lr"]
            weight_decay = self.wd_scheduler.get_value()

            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = self.model(batch)
                pred_loss = self.criterion(output["pred_emb"], output["emb"][:, 1:])
                sigreg_loss = self.sigreg(output["emb"].transpose(0, 1))
                loss = pred_loss + self.optimization_sigreg_weight * sigreg_loss

            self.scaler.scale(loss).backward()

            if self.optimization_gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimization_gradient_clip_norm)

            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_updated = self.scaler.get_scale() >= scale

            if optimizer_updated:
                if self.optimization_lr_interval == "step":
                    self.lr_scheduler.step()
                if self.optimization_weight_decay_interval == "step":
                    self.wd_scheduler.step()

            batch_size = batch["pixels"].size(0)
            batch_losses = torch.stack([
                loss.detach(), pred_loss.detach(), sigreg_loss.detach(), loss.new_tensor(batch_size),
            ]).to(torch.float64)
            batch_losses[:3] *= batch_size

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(batch_losses)

            loss_sums += batch_losses
            write_on_csv(
                self.output_folder,
                epoch,
                iteration,
                (batch_losses[0] / batch_losses[3]).item(),
                lr,
                weight_decay,
            )

        if loss_sums[3].item() == 0:
            raise ValueError("The training dataloader has no batches. Check batch_size and drop_last.")

        losses = (loss_sums[:3] / loss_sums[3]).tolist()
        write_on_log(
            f"Epoch {epoch}: loss={losses[0]:.6f}, pred_loss={losses[1]:.6f}, sigreg_loss={losses[2]:.6f}",
            self.output_folder,
        )
        return {"loss": losses[0], "pred_loss": losses[1], "sigreg_loss": losses[2]}

    def test(self):
        solvers = []
        if self.planning_cem_solver_use:
            solvers.append("cem")
        if self.planning_gradient_solver_use:
            solvers.append("gradient")
        if not solvers:
            raise ValueError("Enable at least one solver for evaluation.")
        if self.evaluation_num_episodes < 1 or self.evaluation_goal_offset_steps < 1 or self.evaluation_max_steps < 1:
            raise ValueError("num_episodes, goal_offset_steps and max_steps must be positive.")
        if self.planning_horizon < 1 or not 1 <= self.planning_replanning_horizon <= self.planning_horizon:
            raise ValueError("replanning_horizon must be between 1 and horizon.")

        counts = np.maximum(np.asarray(self.dataset.lengths) - self.evaluation_goal_offset_steps, 0)
        cumulative = np.cumsum(counts)
        num_starting_points = int(cumulative[-1]) if len(cumulative) else 0
        if self.evaluation_num_episodes > num_starting_points:
            raise ValueError("The dataset has too few valid starting points for evaluation.")
        generator = np.random.default_rng(self.meta_seed)
        selected = np.sort(generator.choice(num_starting_points, size=self.evaluation_num_episodes, replace=False))
        episodes_idx = np.searchsorted(cumulative, selected, side="right")
        start_steps = selected - np.concatenate([[0], cumulative[:-1]])[episodes_idx]

        dist = torch.distributed
        rank = dist.get_rank() if dist.is_initialized() else self.rank
        world_size = dist.get_world_size() if dist.is_initialized() else self.world_size
        self.model.eval()
        self.model.requires_grad_(False)
        results = {}

        for solver_name in solvers:
            local_episodes = episodes_idx[rank::world_size].tolist()
            local_starts = start_steps[rank::world_size].tolist()
            local_result = self._evaluate(solver_name, local_episodes, local_starts) if local_episodes else {
                "episode_successes": [], "seeds": None, "elapsed_seconds": 0.0,
            }
            gathered = [local_result]
            if dist.is_initialized():
                gathered = [None] * world_size
                dist.all_gather_object(gathered, local_result)

            successes = np.zeros(self.evaluation_num_episodes, dtype=bool)
            seeds = [None] * self.evaluation_num_episodes
            for process, result in enumerate(gathered):
                successes[process::world_size] = result["episode_successes"]
                if result["seeds"] is not None:
                    seeds[process::world_size] = result["seeds"]

            results[solver_name] = {
                "success_rate": float(successes.mean() * 100),
                "episode_successes": successes.tolist(),
                "episodes_idx": episodes_idx.tolist(),
                "start_steps": start_steps.tolist(),
                "seeds": seeds,
                "elapsed_seconds": max(result["elapsed_seconds"] for result in gathered),
                "evaluate_state": self.evaluate_state,
            }
            if rank == 0:
                folder = os.path.join(self.output_folder, solver_name)
                os.makedirs(folder, exist_ok=True)
                save_json(results[solver_name], folder, "results")
                write_on_log(f"{solver_name}: success_rate={results[solver_name]['success_rate']:.2f}%", self.output_folder)
                print(f"{solver_name}: success_rate={results[solver_name]['success_rate']:.2f}%", flush=True)

        if rank == 0:
            save_json(results, self.output_folder, "results")
        return results

    def _evaluate(self, solver_name, episodes_idx, start_steps):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else self.rank
        seed = self.meta_seed + rank
        environment = get_environment_config(self.data_dataset)
        for call in environment["callables"]:
            for argument in call["args"].values():
                if argument.get("in_dataset", True):
                    column = argument["value"].removeprefix("goal_")
                    if column not in self.dataset.column_names:
                        raise ValueError(f"Evaluation of {self.data_dataset} requires dataset column '{column}'.")

        match solver_name:
            case "cem":
                if not 2 <= self.planning_cem_solver_num_elites <= self.planning_cem_solver_num_samples:
                    raise ValueError("CEM num_elites must be between 2 and num_samples.")
                solver = CEMSolver(
                    model=self.model,
                    batch_size=self.planning_cem_solver_batch_size,
                    num_samples=self.planning_cem_solver_num_samples,
                    n_steps=self.planning_cem_solver_num_iterations,
                    topk=self.planning_cem_solver_num_elites,
                    var_scale=self.planning_cem_solver_var_scale,
                    device=self.device,
                    seed=seed,
                )
            case "gradient":
                solver = GradientSolver(
                    model=self.model,
                    batch_size=self.planning_gradient_solver_batch_size,
                    num_samples=self.planning_gradient_solver_num_samples,
                    n_steps=self.planning_gradient_solver_num_iterations,
                    var_scale=self.planning_gradient_solver_var_scale,
                    device=self.device,
                    seed=seed,
                    optimizer_cls=load_planning_optimizer,
                    optimizer_kwargs={
                        "optimizer_type": self.planning_gradient_solver_optimizer_type,
                        "lr": self.planning_gradient_solver_optimizer_lr,
                        "action_noise": self.planning_gradient_solver_action_noise,
                        "generator": torch.Generator(device=self.device).manual_seed(seed),
                    },
                    action_noise=0.0,
                )
            case _:
                raise ValueError(f"Unsupported evaluation solver: {solver_name}")

        if solver.batch_size < 1 or solver.num_samples < 1 or solver.n_steps < 1 or solver.var_scale < 0:
            raise ValueError("Solver batch_size, num_samples and num_iterations must be positive, and var_scale nonnegative.")
        if solver_name == "gradient" and self.planning_gradient_solver_action_noise < 0:
            raise ValueError("action_noise must be nonnegative.")

        policy = WorldModelPolicy(
            solver=solver,
            config=PlanConfig(
                horizon=self.planning_horizon,
                receding_horizon=self.planning_replanning_horizon,
                history_len=self.meta_context_length,
                action_block=self.meta_action_block,
            ),
            process={"action": ActionNormalizer(self.action_mean, self.action_std)},
            transform={"pixels": self.image_transform.transform, "goal": self.image_transform.transform},
        )

        os.environ.setdefault("MUJOCO_GL", "egl")
        world = World(
            env_name=environment["env_name"],
            num_envs=len(episodes_idx),
            image_shape=(self.data_image_size, self.data_image_size),
            max_episode_steps=self.evaluation_max_steps,
            **environment["kwargs"],
        )
        started = perf_counter()
        try:
            for call in environment["callables"]:
                if not hasattr(world.envs.envs[0].unwrapped, call["method"]):
                    raise ValueError(f"The simulator does not implement '{call['method']}'.")
            if int(np.prod(world.envs.single_action_space.shape)) != self.action_dim:
                raise ValueError("The simulator action dimension differs from the saved model.")
            action_space = world.envs.single_action_space
            normalizer = policy.process["action"]
            self.model.set_action_bounds(
                np.tile(normalizer.transform(action_space.low).reshape(-1), self.meta_action_block),
                np.tile(normalizer.transform(action_space.high).reshape(-1), self.meta_action_block),
            )
            world.set_policy(policy)
            video = None
            if self.evaluation_save_video:
                video = os.path.join(self.output_folder, solver_name, "videos", f"rank_{rank}")
            write_on_log(f"Evaluating {solver_name} on {len(episodes_idx)} episodes", self.output_folder)
            metrics = world.evaluate(
                dataset=self.dataset,
                episodes_idx=episodes_idx,
                start_steps=start_steps,
                goal_offset=self.evaluation_goal_offset_steps,
                eval_budget=self.evaluation_max_steps,
                callables=environment["callables"],
                video=video,
            )
            return {
                "episode_successes": np.asarray(metrics["episode_successes"], dtype=bool).tolist(),
                "seeds": np.asarray(metrics["seeds"]).tolist() if metrics["seeds"] is not None else None,
                "elapsed_seconds": perf_counter() - started,
            }
        finally:
            world.close()

    def _load_model(self):
        encoder = Encoder(
            model_name=self.meta_model_name,
            image_size=self.data_image_size,
            patch_size=self.meta_patch_size,
        )

        action_encoder = ActionEncoder(
            input_dim=self.meta_action_block * self.action_dim,
            embed_dim=self.meta_embed_dim,
        )

        predictor = Predictor(
            embed_dim=self.meta_embed_dim,
            context_length=self.meta_context_length,
            depth=self.meta_predictor_depth,
            num_heads=self.meta_predictor_num_heads,
            dim_head=self.meta_predictor_dim_head,
            mlp_dim=self.meta_predictor_mlp_dim,
            dropout=self.meta_predictor_dropout,
            emb_dropout=self.meta_predictor_emb_dropout,
        )

        projector = Projector(
            input_dim=encoder.embed_dim,
            hidden_dim=self.meta_projector_hidden_dim,
            output_dim=self.meta_embed_dim,
            norm=self.meta_projector_norm,
        )

        pred_proj = Projector(
            input_dim=self.meta_embed_dim,
            hidden_dim=self.meta_projector_hidden_dim,
            output_dim=self.meta_embed_dim,
            norm=self.meta_projector_norm,
        )

        self.model = JEPA(
            encoder=encoder,
            action_encoder=action_encoder,
            predictor=predictor,
            projector=projector,
            pred_proj=pred_proj,
            context_length=self.meta_context_length,
        )

    def _load_criterion(self):
        match self.optimization_criterion:
            case "mse":
                self.criterion = torch.nn.MSELoss()

            case _:
                raise ValueError(f"Unsupported optimization criterion: {self.optimization_criterion}")

        self.sigreg = SIGReg(
            knots=self.optimization_sigreg_knots,
            num_proj=self.optimization_sigreg_num_proj,
        ).to(self.device)

    def _load_state(self):
        self.states_folder = os.path.join(self.output_folder, "states")
        self.state_path = os.path.join(self.states_folder, "state.pth")
        self.last_epoch = 0
        self.normalization_stats = {}
        self.pretraining_config = deepcopy(self.config)
        state = None

        if self.is_evaluating or self.resume:
            path = self.evaluate_state if self.is_evaluating else self.state_path
            if not self.is_evaluating and not os.path.isfile(path):
                legacy_path = os.path.join(self.output_folder, "models", "state.pth")
                if os.path.isfile(legacy_path):
                    path = legacy_path
            state = load_state(path)
            self.pretraining_config = state["config"]

            if self.pretraining_config["mode"] != "pretraining" or self.pretraining_config["method"] != self.config["method"]:
                raise ValueError("The saved state does not belong to this pretraining method.")

            if self.is_evaluating:
                self._load_config_pretraining(self.pretraining_config)
                self.data_dataset_config = state["extra"].get("dataset_config", self.data_dataset_config)
                self._load_config_evaluation()
                self.data_datasets_folder_path = os.path.join(self.data_datasets_folder_path, "")
            elif self.pretraining_config != self.config:
                raise ValueError("The saved state configuration differs from the pretraining configuration.")
            else:
                self.data_dataset_config = state["extra"].get("dataset_config", self.data_dataset_config)

            if not self.is_evaluating and len(state["rng_states"]) != self.world_size:
                raise ValueError("Resume training with the same number of processes used to save the state.")

            self.last_epoch = int(state["epoch"])
            self.action_dim = int(state["extra"]["action_dim"])
            self.normalization_stats = state["extra"]["normalization_stats"]
            self.action_mean = self.normalization_stats["action"]["mean"]
            self.action_std = self.normalization_stats["action"]["std"]

        self.start_epoch = self.last_epoch + 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else self.rank
        torch.manual_seed(self.meta_seed + rank)

        self._load_dataset()
        self._load_dataloader()
        self._load_model()
        if not self.is_evaluating and self.world_size > 1 and self.device.type == "cuda":
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

        if state is not None:
            restore_state(state, model=self.model)

        if self.is_evaluating:
            self.model.eval()
            self.model.requires_grad_(False)
            return

        self.model.train()
        self.model.requires_grad_(True)
        self._load_criterion()
        self._load_optimizer()
        self._load_schedulers()
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.optimization_precision == "float16",
        )

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None)

        self.train_sampler.set_epoch(self.start_epoch)
        if state is not None:
            restore_state(
                state,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                scaler=self.scaler,
                dataloader=self.train_dataloader,
                device=self.device,
                rank=rank,
                restore_rng=True,
            )

    def _load_optimizer(self):
        self.optimizer = load_optimizer(
            model=self.model,
            optimizer_type=self.optimization_optimizer,
            lr=self.optimization_lr_values[0],
            weight_decay=self.optimization_weight_decay_values[0],
            decay_bias=self.optimization_decay_bias,
            decay_norm=self.optimization_decay_norm,
        )
        self.optimizers = {"model": self.optimizer}

    def _load_schedulers(self):
        total_steps = {
            "epoch": self.optimization_num_epochs,
            "step": self.optimization_num_epochs * len(self.train_dataloader),
        }
        for interval in [self.optimization_lr_interval, self.optimization_weight_decay_interval]:
            if interval not in total_steps:
                raise ValueError(f"Unsupported scheduler interval: {interval}")

        lr_steps = total_steps[self.optimization_lr_interval]
        warmup_steps = 0
        if self.optimization_lr_type.startswith("warmup_") and self.optimization_lr_warmup_fraction > 0:
            warmup_steps = min(max(1, int(lr_steps * self.optimization_lr_warmup_fraction)), lr_steps - 1)

        self.lr_scheduler = OptimizerScheduler(
            optimizer=self.optimizer,
            parameter="lr",
            values=self.optimization_lr_values,
            total_steps=lr_steps,
            schedule_type=self.optimization_lr_type,
            warmup_steps=warmup_steps,
        )

        self.wd_scheduler = OptimizerScheduler(
            optimizer=self.optimizer,
            parameter="weight_decay",
            values=self.optimization_weight_decay_values,
            total_steps=total_steps[self.optimization_weight_decay_interval],
            schedule_type=self.optimization_weight_decay_type,
        )

        self.schedulers = {"lr": self.lr_scheduler, "weight_decay": self.wd_scheduler}

    def _save_state(self, epoch, state_name=None):
        state = create_state(
            model=self.model,
            config=self.pretraining_config,
            epoch=epoch,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=self.scaler,
            dataloader=self.train_dataloader,
            device=self.device,
            extra={
                "action_dim": self.action_dim,
                "normalization_stats": self.normalization_stats,
                "dataset_config": self.data_dataset_config,
            },
        )
        if epoch % self.meta_save_every == 0:
            state_name = f"state_{epoch}" if state_name is None else state_name
            save_state(state, self.states_folder, state_name=state_name, rank=self.rank)

        save_state(state, self.states_folder, rank=self.rank)
        self.last_epoch = int(epoch)
        self.start_epoch = self.last_epoch + 1

    def _load_dataset(self):
        self.dataset_path = os.path.join(
            self.data_datasets_folder_path,
            self.data_dataset_config["dataset_file"],
        )

        self.image_transform = make_image_transform(
            image_size=self.data_image_size,
            mean=self.data_normalize_mean,
            std=self.data_normalize_std,
        )

        if self.is_evaluating:
            self.dataset, self.normalization_stats = load_dataset(
                path=self.dataset_path,
                normalization_stats=self.normalization_stats,
            )

            if self.dataset.get_dim("action") != self.action_dim:
                raise ValueError("The evaluation dataset action dimension differs from the saved model.")
            return

        expected_action_dim = self.action_dim if self.normalization_stats else None
        self.dataset, self.normalization_stats = load_dataset(
            path=self.dataset_path,
            action_block=self.meta_action_block,
            num_steps=self.meta_context_length + 1, # meta_context_length + 1 for prediction
            keys_to_load=self.data_dataset_config["keys_to_load"],
            keys_to_cache=self.data_dataset_config["keys_to_cache"],
            keys_to_merge=self.data_dataset_config["keys_to_merge"],
            transform=self.image_transform,
            normalize_keys=self.data_dataset_config["normalize_keys"],
            normalization_stats=self.normalization_stats,
            zero_nan_keys=["action"],
        )

        action_dim = self.dataset.get_dim("action")
        if expected_action_dim is not None and action_dim != expected_action_dim:
            raise ValueError("The dataset action dimension differs from the saved model.")
        self.action_dim = action_dim
        self.action_mean = self.normalization_stats["action"]["mean"]
        self.action_std = self.normalization_stats["action"]["std"]

        self.train_dataset = self.dataset

    def _load_dataloader(self):
        if self.is_evaluating:
            return

        self.train_dataloader, self.train_sampler = load_dataloader(
            dataset=self.train_dataset,
            batch_size=self.data_batch_size,
            num_workers=self.data_num_workers,
            prefetch_factor=self.data_prefetch_factor,
            persistent_workers=self.data_persistent_workers,
            pin_memory=self.data_pin_memory,
            drop_last=self.data_drop_last,
            seed=self.meta_seed,
            rank=self.rank,
            world_size=self.world_size,
        )

    def _load_config(self):
        self._load_config_evaluation() if self.is_evaluating else self._load_config_pretraining()
        self.data_datasets_folder_path = os.path.join(self.data_datasets_folder_path, "")

    def _load_config_evaluation(self):
        dataset = str(self.config["data"]["dataset"])
        defaults = self.data_dataset_config if hasattr(self, "data_dataset_config") and self.data_dataset == dataset else None
        self.data_dataset_config = get_dataset_config(dataset, defaults=defaults)
        self.data_datasets_folder_path = str(self.config["data"]["datasets_folder_path"])
        self.data_dataset = dataset
        self.meta_seed = int(self.config["meta"]["seed"])
        self.planning_horizon = int(self.config["planning"]["horizon"])
        self.planning_replanning_horizon = int(self.config["planning"]["replanning_horizon"])
        self.planning_cem_solver_use = bool(self.config["planning"]["cem_solver"]["use"])
        self.planning_cem_solver_batch_size = int(self.config["planning"]["cem_solver"]["batch_size"])
        self.planning_cem_solver_num_samples = int(self.config["planning"]["cem_solver"]["num_samples"])
        self.planning_cem_solver_num_iterations = int(self.config["planning"]["cem_solver"]["num_iterations"])
        self.planning_cem_solver_num_elites = int(self.config["planning"]["cem_solver"]["num_elites"])
        self.planning_cem_solver_var_scale = float(self.config["planning"]["cem_solver"]["var_scale"])
        self.planning_gradient_solver_use = bool(self.config["planning"]["gradient_solver"]["use"])
        self.planning_gradient_solver_batch_size = int(self.config["planning"]["gradient_solver"]["batch_size"])
        self.planning_gradient_solver_num_samples = int(self.config["planning"]["gradient_solver"]["num_samples"])
        self.planning_gradient_solver_num_iterations = int(self.config["planning"]["gradient_solver"]["num_iterations"])
        self.planning_gradient_solver_var_scale = float(self.config["planning"]["gradient_solver"]["var_scale"])
        self.planning_gradient_solver_action_noise = float(self.config["planning"]["gradient_solver"]["action_noise"])
        self.planning_gradient_solver_optimizer_type = str(self.config["planning"]["gradient_solver"]["optimizer"]["type"])
        self.planning_gradient_solver_optimizer_lr = float(self.config["planning"]["gradient_solver"]["optimizer"]["lr"])
        self.evaluation_num_episodes = int(self.config["evaluation"]["num_episodes"])
        self.evaluation_goal_offset_steps = int(self.config["evaluation"]["goal_offset_steps"])
        self.evaluation_max_steps = int(self.config["evaluation"]["max_steps"])
        self.evaluation_save_video = bool(self.config["evaluation"]["save_video"])

    def _load_config_pretraining(self, config=None):
        config = self.config if config is None else config
        self.data_datasets_folder_path = str(config["data"]["datasets_folder_path"])
        self.data_dataset = str(config["data"]["dataset"])
        self.data_dataset_config = get_dataset_config(self.data_dataset)
        self.data_batch_size = int(config["data"]["batch_size"])
        self.data_image_size = int(config["data"]["image_size"])
        self.data_num_workers = int(config["data"]["num_workers"])
        self.data_prefetch_factor = int(config["data"]["prefetch_factor"])
        self.data_persistent_workers = bool(config["data"]["persistent_workers"])
        self.data_pin_memory = bool(config["data"]["pin_memory"])
        self.data_drop_last = bool(config["data"]["drop_last"])
        self.data_normalize_mean = list(map(float, config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, config["data"]["normalize"]["std"]))
        self.meta_seed = int(config["meta"]["seed"])
        self.meta_model_name = str(config["meta"]["model_name"])
        self.meta_patch_size = int(config["meta"]["patch_size"])
        self.meta_embed_dim = int(config["meta"]["embed_dim"])
        self.meta_action_block = int(config["meta"]["action_block"])
        self.meta_context_length = int(config["meta"]["context_length"])
        self.meta_predictor_depth = int(config["meta"]["predictor"]["depth"])
        self.meta_predictor_num_heads = int(config["meta"]["predictor"]["num_heads"])
        self.meta_predictor_dim_head = int(config["meta"]["predictor"]["dim_head"])
        self.meta_predictor_mlp_dim = int(config["meta"]["predictor"]["mlp_dim"])
        self.meta_predictor_dropout = float(config["meta"]["predictor"]["dropout"])
        self.meta_predictor_emb_dropout = float(config["meta"]["predictor"]["emb_dropout"])
        self.meta_projector_hidden_dim = int(config["meta"]["projector"]["hidden_dim"])
        self.meta_projector_norm = str(config["meta"]["projector"]["norm"])
        self.meta_save_every = int(config["meta"]["save_every"])
        if self.meta_save_every <= 0:
            raise ValueError("save_every must be positive.")
        self.optimization_num_epochs = int(config["optimization"]["num_epochs"])
        self.optimization_optimizer = str(config["optimization"]["optimizer"])
        self.optimization_weight_decay_values = list(map(float, config["optimization"]["weight_decay"]["values"]))
        self.optimization_weight_decay_type = str(config["optimization"]["weight_decay"]["type"])
        self.optimization_weight_decay_interval = str(config["optimization"]["weight_decay"]["interval"])
        self.optimization_decay_bias = bool(config["optimization"]["weight_decay"]["decay_bias"])
        self.optimization_decay_norm = bool(config["optimization"]["weight_decay"]["decay_norm"])
        self.optimization_lr_values = list(map(float, config["optimization"]["lr"]["values"]))
        self.optimization_lr_type = str(config["optimization"]["lr"]["type"])
        self.optimization_lr_interval = str(config["optimization"]["lr"]["interval"])
        self.optimization_lr_warmup_fraction = float(config["optimization"]["lr"]["warmup_fraction"])
        self.optimization_precision = str(config["optimization"]["precision"])
        gradient_clip_norm = config["optimization"]["gradient_clip_norm"]
        self.optimization_gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
        self.optimization_criterion = str(config["optimization"]["criterion"])
        self.optimization_sigreg_weight = float(config["optimization"]["sigreg"]["weight"])
        self.optimization_sigreg_knots = int(config["optimization"]["sigreg"]["knots"])
        self.optimization_sigreg_num_proj = int(config["optimization"]["sigreg"]["num_proj"])
