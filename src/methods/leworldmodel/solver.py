from stable_worldmodel.solver import GradientSolver as WorldModelGradientSolver
import numpy as np
import torch

class GradientSolver(WorldModelGradientSolver):
    def init_action(self, n_envs, actions=None):
        if actions is not None:
            actions = actions.to(self.device)
        super().init_action(n_envs, actions)

    def solve(self, info_dict, init_action=None):
        result = super().solve(info_dict, init_action)

        actions = []
        batch_size = self.batch_size or len(self.init)
        with torch.no_grad():
            for start in range(0, len(self.init), batch_size):
                candidates = self.init[start:start + batch_size]
                expanded = {}
                for key, value in info_dict.items():
                    value = value[start:start + batch_size]
                    if torch.is_tensor(value):
                        dtype = self.dtype if value.is_floating_point() else None
                        value = value.to(device=self.device, dtype=dtype)
                        value = value[:, None].expand(-1, self.num_samples, *value.shape[1:])
                    elif isinstance(value, np.ndarray):
                        value = np.repeat(value[:, None], self.num_samples, axis=1)
                    expanded[key] = value
                costs = self.model.get_cost(expanded, candidates)
                rows = torch.arange(len(candidates), device=candidates.device)
                actions.append(candidates[rows, costs.argmin(dim=1)].cpu())
        result["actions"] = torch.cat(actions)
        return result
