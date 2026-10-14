from stable_worldmodel.data import load_dataset as load_worldmodel_dataset
from torchvision.transforms import v2
from copy import deepcopy
import torch
import os

DATASETS = {
    "tworoom": {
        "dataset_file": "tworoom/tworoom.h5",
        "keys_to_load": ["pixels", "action", "proprio"],
        "keys_to_merge": {},
    },
    "pusht": {
        "dataset_file": "pusht/pusht_expert_train.h5",
        "keys_to_load": ["pixels", "action", "proprio", "state"],
        "keys_to_merge": {},
    },
    "reacher": {
        "dataset_file": "reacher/reacher.h5",
        "keys_to_load": ["pixels", "action", "observation"],
        "keys_to_merge": {},
    },
    "cube": {
        "dataset_file": "ogbench/cube_single_expert.h5",
        "keys_to_load": ["pixels", "action", "observation"],
        "keys_to_merge": {"proprio": "proprio"},
    },
}

ENVIRONMENTS = {
    "tworoom": {
        "env_name": "swm/TwoRoom-v1",
        "kwargs": {},
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    },
    "pusht": {
        "env_name": "swm/PushT-v1",
        "kwargs": {},
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    },
    "reacher": {
        "env_name": "swm/ReacherDMControl-v0",
        "kwargs": {"task": "qpos_match"},
        "callables": [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
    },
    "cube": {
        "env_name": "swm/OGBCube-v0",
        "kwargs": {"env_type": "single", "ob_type": "states", "multiview": False,
                   "width": 224, "height": 224, "visualize_info": False, "terminate_at_goal": True},
        "callables": [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            }},
        ],
    },
}

def get_environment_config(name):
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unsupported evaluation environment: {name}")
    return deepcopy(ENVIRONMENTS[name])

class ActionNormalizer():
    def __init__(self, mean, std):
        self.mean = mean.detach().cpu().numpy()
        self.std = std.detach().cpu().numpy()

    def transform(self, actions):
        return (actions - self.mean) / self.std

    def inverse_transform(self, actions):
        return actions * self.std + self.mean

def get_dataset_config(name, defaults=None):
    if defaults is None:
        if name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        defaults = DATASETS[name]

    dataset_config = deepcopy(defaults)
    numeric_keys = [key for key in dataset_config["keys_to_load"] if not key.startswith("pixels")]
    for key in ["keys_to_cache", "normalize_keys"]:
        dataset_config.setdefault(key, numeric_keys.copy())

    return dataset_config

class ImageTransform():
    def __init__(self, image_size, mean, std):
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.Resize((image_size, image_size)),
        ])

    def __call__(self, sample):
        for column in sample:
            if column.startswith("pixels"):
                sample[column] = self.transform(sample[column])
        return sample

def make_image_transform(image_size, mean, std):
    return ImageTransform(image_size, mean, std)

def load_dataset(
    path,
    action_block=1,
    num_steps=1,
    keys_to_load=None,
    keys_to_cache=None,
    keys_to_merge=None,
    transform=None,
    normalize_keys=None,
    normalization_stats=None,
    zero_nan_keys=None,
):
    dataset = load_worldmodel_dataset(
        os.path.abspath(path),
        frameskip=action_block,
        num_steps=num_steps,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        keys_to_merge=keys_to_merge,
        transform=transform,
    )
    normalization_stats = {} if normalization_stats is None else normalization_stats

    for column in normalize_keys or []:
        data = torch.from_numpy(dataset.get_col_data(column)).float()
        if column not in normalization_stats:
            valid_data = data[~torch.isnan(data).reshape(len(data), -1).any(dim=1)]
            if len(valid_data) < 2:
                raise ValueError(f"Cannot normalize '{column}' with fewer than two valid samples.")

            mean = valid_data.mean(dim=0, keepdim=True)
            std = valid_data.std(dim=0, keepdim=True).clamp_min(torch.finfo(data.dtype).eps)
            normalization_stats[column] = {"mean": mean, "std": std}
        else:
            mean = normalization_stats[column]["mean"]
            std = normalization_stats[column]["std"]
            if mean.shape[1:] != data.shape[1:] or std.shape[1:] != data.shape[1:]:
                raise ValueError(f"The saved normalization dimensions differ for '{column}'.")

        normalized_data = (data - mean) / std
        if column in (zero_nan_keys or []):
            normalized_data = torch.nan_to_num(normalized_data)
        dataset._cache[column] = normalized_data.numpy()

    return dataset, normalization_stats
