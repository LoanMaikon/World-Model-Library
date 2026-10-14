from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
from unittest.mock import patch
import json
import os
import subprocess
import sys
import unittest

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from src.Runner import Runner
from src.datasets import ActionNormalizer
from src.methods.leworldmodel.models import ActionEncoder, Predictor, Projector, JEPA
from src.methods.leworldmodel.LeWorldModel import GradientSolver, WorldModelGradientSolver
from src.optimizers import load_planning_optimizer
from src.states import load_state

ROOT = Path(__file__).resolve().parents[1]


class EvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rollout_keeps_action_gradients_and_cached_observations(self):
        for context_length in [1, 2]:
            model = JEPA(
                nn.Sequential(nn.Flatten(), nn.Linear(12, 8)),
                ActionEncoder(4, 8),
                Predictor(8, context_length, depth=2, num_heads=2, dim_head=4, mlp_dim=16, dropout=0),
                Projector(8, 16, 8, norm="layer_norm"),
                Projector(8, 16, 8, norm="layer_norm"),
                context_length,
            ).eval().requires_grad_(False)
            for block in model.predictor.transformer.layers:
                nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)
            actions = torch.randn(2, 3, 4, 4, requires_grad=True)
            info = {name: torch.randn(2, 1, 1, 3, 2, 2).expand(-1, 3, -1, -1, -1, -1)
                    for name in ["pixels", "goal"]}
            cost = model.get_cost(info, actions)
            self.assertEqual(cost.shape, (2, 3))
            cost.sum().backward()
            self.assertTrue(torch.isfinite(actions.grad).all())
            self.assertGreater(actions.grad.abs().sum().item(), 0)
            self.assertTrue(all(param.grad is None for param in model.parameters()))
            with torch.no_grad():
                latent = info["emb"][:, None].expand(-1, 3, -1).reshape(6, 1, 8)
                action_embeddings = model.action_encoder(actions.detach().flatten(0, 1))
                for step in range(4):
                    prediction = model.predict(latent[:, -context_length:],
                                               action_embeddings[:, max(0, step + 1 - context_length):step + 1])[:, -1:]
                    latent = torch.cat([latent, prediction], dim=1)
                expected = (latent[:, -1].reshape(2, 3, 8) - info["goal_emb"][:, None]).square().sum(-1)
                torch.testing.assert_close(cost, expected)
            cached = info["emb"]
            model.get_cost(info, actions.detach())
            self.assertIs(info["emb"], cached)

    def test_action_normalization_roundtrip_and_planning_noise(self):
        normalizer = ActionNormalizer(torch.tensor([[3., -2.]]), torch.tensor([[2., 4.]]))
        actions = np.array([[5., 6.], [1., 2.]], dtype=np.float32)
        np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(actions)), actions)
        outputs = []
        for _ in range(2):
            parameter = nn.Parameter(torch.zeros(2, 3, 4))
            optimizer = load_planning_optimizer([parameter], "adamw", 0.1, action_noise=0.2,
                                                generator=torch.Generator().manual_seed(42))
            parameter.sum().backward()
            optimizer.step()
            outputs.append(parameter.detach().clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertGreater(outputs[0].std().item(), 0)

    def test_gradient_solver_moves_initial_actions_to_its_device(self):
        solver = GradientSolver(model=None, n_steps=1)
        solver.device = torch.device("meta")
        actions = torch.zeros(2, 3, 4)
        with patch.object(WorldModelGradientSolver, "init_action") as initialize:
            solver.init_action(2, actions)
        passed_actions = initialize.call_args.args[1]
        self.assertEqual(passed_actions.device.type, "meta")
        self.assertEqual(passed_actions.shape, actions.shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this machine")
    def test_planning_noise_on_cuda(self):
        parameter = nn.Parameter(torch.zeros(2, 3, 4, device="cuda"))
        optimizer = load_planning_optimizer([parameter], "adamw", 0.1, action_noise=0.2,
                                            generator=torch.Generator(device="cuda").manual_seed(42))
        parameter.sum().backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())

        from gymnasium.spaces import Box
        from stable_worldmodel import PlanConfig

        class CostModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1, device="cuda"), requires_grad=False)

            def get_cost(self, info, actions):
                return actions.square().sum(dim=(2, 3))

        solver = GradientSolver(
            model=CostModel(), n_steps=2, batch_size=1, num_samples=3, device="cuda",
            optimizer_cls=load_planning_optimizer,
            optimizer_kwargs={"optimizer_type": "adamw", "lr": 0.1, "action_noise": 0.2,
                              "generator": torch.Generator(device="cuda").manual_seed(42)},
        )
        solver.configure(action_space=Box(-1., 1., shape=(2, 2)), n_envs=2,
                         config=PlanConfig(horizon=3, receding_horizon=1, action_block=2))
        result = solver.solve({"pixels": torch.zeros(2, 1, 3, 2, 2)})
        self.assertTrue(torch.isfinite(result["actions"]).all())
        self.assertEqual(solver.init.device.type, "cuda")


class FullExecutionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.make_dataset()

    def make_dataset(self):
        from stable_worldmodel.envs.two_room import TwoRoomEnv

        directory = self.folder / "data" / "tworoom"
        directory.mkdir(parents=True)
        frames, actions, proprio = [], [], []
        env = TwoRoomEnv()
        try:
            for episode in range(2):
                position = [40., 40.] if episode == 0 else [175., 175.]
                env.reset(seed=episode, options={"state": position})
                for step in range(8):
                    frames.append(np.array(Image.fromarray(env.render()).resize((28, 28))))
                    proprio.append(env.agent_position.numpy().copy())
                    action = np.array([1. if episode == 0 else -1., 0.2 if step % 2 else -0.2], dtype=np.float32)
                    actions.append(action if step < 7 else np.full(2, np.nan, dtype=np.float32))
                    env.step(action)
        finally:
            env.close()
        with h5py.File(directory / "tworoom.h5", "w") as file:
            file["ep_len"] = [8, 8]
            file["ep_offset"] = [0, 8]
            file["ep_idx"] = np.repeat([0, 1], 8)
            file["step_idx"] = np.tile(np.arange(8), 2)
            file["pixels"] = frames
            file["action"] = actions
            file["proprio"] = proprio

    def configs(self, num_workers=0):
        pretraining = yaml.safe_load((ROOT / "configs/pretraining/pretraining_leworldmodel_tworoom.yaml").read_text())
        pretraining["data"].update(datasets_folder_path=str(self.folder / "data"), batch_size=2,
                                   image_size=28, num_workers=num_workers)
        pretraining["meta"].update(embed_dim=8, action_block=2)
        pretraining["meta"]["predictor"].update(depth=2, num_heads=2, dim_head=4, mlp_dim=16, dropout=0)
        pretraining["meta"]["projector"]["hidden_dim"] = 16
        pretraining["optimization"]["num_epochs"] = 2
        pretraining["optimization"]["sigreg"]["num_proj"] = 8
        pretraining["optimization"]["lr"].update(values=[0, 0.001, 0], warmup_fraction=0.25)

        evaluation = yaml.safe_load((ROOT / "configs/evaluate/evaluate_leworldmodel_tworoom.yaml").read_text())
        evaluation["data"]["datasets_folder_path"] = str(self.folder / "data")
        evaluation["planning"].update(horizon=3, replanning_horizon=1)
        for name in ["cem_solver", "gradient_solver"]:
            evaluation["planning"][name].update(batch_size=2, num_samples=6, num_iterations=2)
        evaluation["planning"]["cem_solver"]["num_elites"] = 2
        evaluation["planning"]["gradient_solver"]["action_noise"] = 0.05
        evaluation["evaluation"].update(num_episodes=3, goal_offset_steps=4, max_steps=6)
        return pretraining, evaluation

    def write_config(self, config, name):
        path = self.folder / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def execute(self, config_path, output_folder, state=None, distributed=False):
        command = [sys.executable]
        if distributed:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
        command += [str(ROOT / "main.py"), "--config", str(config_path), "--devices", "cpu",
                    "--output_folder", str(output_folder)]
        if state is not None:
            command += ["--evaluate_state", str(state)]
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pretraining_resume_and_both_evaluation_solvers_with_videos(self):
        pretraining, evaluation = self.configs(num_workers=1)
        train_config = self.write_config(pretraining, "pretraining")
        output = self.folder / "pretraining"
        self.execute(train_config, output)
        expected = load_state(output / "states" / "state.pth")
        self.assertEqual(expected["epoch"], 2)
        self.assertEqual(expected["schedulers"]["lr"]["_step"], 10)

        resumed_output = self.folder / "resumed"
        runner = Runner(str(train_config), str(resumed_output), 0, 1)
        method = runner.method
        method.train_sampler.set_epoch(1)
        method._train_epoch(1)
        method.wd_scheduler.step()
        method._save_state(1)
        if method.dataset.h5_file is not None:
            method.dataset.h5_file.close()
        del runner, method
        self.execute(train_config, resumed_output)
        resumed = load_state(resumed_output / "states" / "state.pth")
        for name, value in expected["model"].items():
            torch.testing.assert_close(resumed["model"][name], value, rtol=0, atol=0)

        eval_config = self.write_config(evaluation, "evaluation")
        eval_output = self.folder / "evaluation"
        self.execute(eval_config, eval_output, output / "states" / "state.pth")
        results = json.loads((eval_output / "results.json").read_text())
        self.assertEqual(set(results), {"cem", "gradient"})
        self.assertEqual(results["cem"]["episodes_idx"], results["gradient"]["episodes_idx"])
        self.assertEqual(results["cem"]["start_steps"], results["gradient"]["start_steps"])
        for name, result in results.items():
            self.assertEqual(len(result["episode_successes"]), 3)
            self.assertTrue(0 <= result["success_rate"] <= 100)
            videos = list((eval_output / name / "videos").rglob("*.mp4"))
            self.assertEqual(len(videos), 3)
            self.assertGreater(next(iio.imiter(videos[0])).size, 0)

        evaluation["planning"]["gradient_solver"]["use"] = False
        evaluation["evaluation"]["save_video"] = False
        single_config = self.write_config(evaluation, "cem_only")
        self.execute(single_config, self.folder / "cem_only", output / "states" / "state_1.pth")
        single = json.loads((self.folder / "cem_only" / "results.json").read_text())
        self.assertEqual(set(single), {"cem"})
        self.assertFalse(list((self.folder / "cem_only").rglob("*.mp4")))

    def test_distributed_pretraining_and_evaluation(self):
        pretraining, evaluation = self.configs()
        pretraining["optimization"]["num_epochs"] = 1
        output = self.folder / "distributed_pretraining"
        self.execute(self.write_config(pretraining, "distributed_pretraining"), output, distributed=True)
        state = load_state(output / "states" / "state.pth")
        self.assertEqual(len(state["rng_states"]), 2)
        self.assertFalse(any(name.startswith("module.") for name in state["model"]))
        evaluation["evaluation"].update(num_episodes=3, save_video=False)
        eval_output = self.folder / "distributed_evaluation"
        self.execute(self.write_config(evaluation, "distributed_evaluation"), eval_output,
                     output / "states" / "state.pth", distributed=True)
        results = json.loads((eval_output / "results.json").read_text())
        for result in results.values():
            self.assertEqual(len(result["episode_successes"]), 3)
            self.assertTrue(0 <= result["success_rate"] <= 100)


if __name__ == "__main__":
    unittest.main()
