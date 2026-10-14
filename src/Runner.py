import shutil
import torch
import yaml
import os

class Runner():
    def __init__(self,
                 config,
                 output_folder,
                 rank,
                 world_size,
                 evaluate_state=None,
                ):

        self.config_path = config
        self.output_folder = output_folder
        self.rank = rank
        self.world_size = world_size
        self.evaluate_state = evaluate_state

        self._load_config()
        self._load_device()
        self._load_method()
        self._prepare_output_folder()

        method_args = dict(
            opened_config=self.config,
            output_folder=self.output_folder,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            resume=self.resume,
        )

        if self.is_evaluating():
            method_args["evaluate_state"] = self.evaluate_state

        self.method = self.method_class(**method_args)

    def train(self):
        self.method.train()

    def test(self):
        self.method.test()

    def is_evaluating(self):
        return self.mode == "evaluate"

    def _prepare_output_folder(self):
        dist = torch.distributed

        if not dist.is_available() or not dist.is_initialized():
            self.resume = self._create_output_folder()
            return

        result = [False, None]
        if dist.get_rank() == 0:
            try:
                result[0] = self._create_output_folder()
            except (OSError, ValueError, yaml.YAMLError) as error:
                result[1] = str(error)

        dist.broadcast_object_list(result, src=0)
        if result[1] is not None:
            raise ValueError(result[1])

        self.resume = result[0]

    def _create_output_folder(self):
        output_config = os.path.join(self.output_folder, "config.yaml")

        if os.path.exists(self.output_folder):
            if not os.path.isdir(self.output_folder):
                raise ValueError(f"Output path '{self.output_folder}' is not a directory.")

            if not os.path.isfile(output_config):
                raise ValueError(f"Cannot resume: '{output_config}' does not exist.")

            with open(output_config, "r") as f:
                saved_config = yaml.safe_load(f)

            if saved_config != self.config:
                raise ValueError(
                    f"Cannot resume: configuration differs from '{output_config}'."
                )

            return True

        os.makedirs(self.output_folder, exist_ok=False)
        shutil.copy(self.config_path, output_config)

        return False

    def _load_device(self):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        cuda_available = torch.cuda.is_available()

        if visible_devices and not cuda_available:
            raise RuntimeError(
                f"CUDA was requested (CUDA_VISIBLE_DEVICES={visible_devices}), but PyTorch cannot access it. "
                "Check the selected GPU, the NVIDIA driver, and the PyTorch installation."
            )

        self.device = torch.device(f"cuda:{self.rank}" if cuda_available and visible_devices != "" else "cpu")
        print(f"Device: {self.device} (CUDA_VISIBLE_DEVICES={visible_devices})", flush=True)

    def _load_config(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config["mode"]

        if self.is_evaluating():
            if self.evaluate_state is None:
                raise ValueError("--evaluate_state should be passed when mode is 'evaluate'.")

            if not os.path.isfile(self.evaluate_state):
                raise ValueError(f"'{self.evaluate_state}' does not exist for evaluation.")

        elif self.evaluate_state is not None:
            raise ValueError("--evaluate_state should only be passed when mode is 'evaluate'.")

    def _load_method(self):
        method = self.config["method"]

        match method:
            case "leworldmodel":
                from src.methods.leworldmodel.LeWorldModel import LeWorldModel

                self.method_class = LeWorldModel

            case _:
                raise NotImplementedError(
                    f"No method registered for '{method}'."
                )
