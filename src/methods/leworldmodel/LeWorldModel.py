from src.datasets import get_dataset_config, load_dataset, make_image_transform
from src.states import create_state, save_state, load_state, restore_state
from torch.nn.parallel import DistributedDataParallel as DDP
from src.dataloaders import load_dataloader
from src.schedulers import OptimizerScheduler
from src.optimizers import load_optimizer
from src.utils import write_on_log
from copy import deepcopy
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

            self._train_epoch(epoch)

            if self.optimization_lr_interval == "epoch":
                self.lr_scheduler.step()
            if self.optimization_weight_decay_interval == "epoch":
                self.wd_scheduler.step()

            self._save_state(epoch)

    def _train_epoch(self, epoch):
        write_on_log(f"Training epoch {epoch}", self.output_folder)

        # ...

    def test(self):
        pass # calls self._evaluate() with the chosen solver

    def _evaluate(self):
        pass

    def _load_model(self):
        pass

    def _load_criterion(self):
        match self.optimization_criterion:
            case "mse":
                self.criterion = torch.nn.MSELoss()

            case _:
                raise ValueError(f"Unsupported optimization criterion: {self.optimization_criterion}")

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
            "cuda",
            enabled=self.device.type == "cuda" and self.optimization_precision == "float16",
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
