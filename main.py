import argparse
import os

from src.utils import is_distributed

def main():
    args = get_args()
    try:
        local_rank, world_size = setup_distributed()
        from src.Runner import Runner

        runner = Runner(
            config=args.config,
            output_folder=args.output_folder,
            rank=local_rank,
            world_size=world_size,
            evaluate_state=args.evaluate_state,
        )

        runner.train() if not runner.is_evaluating() else runner.test()
    finally:
        cleanup_distributed()

def get_args():
    parser = argparse.ArgumentParser(description="World Model Library")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file in configs/.")
    parser.add_argument("--devices", type=str, required=True, nargs="+", help="Devices (e.g., cuda:0, cuda:1, or cpu).")
    parser.add_argument("--output_folder", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--evaluate_state", type=str, required=False, help="Path to the saved state for evaluation.")

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise ValueError(f"Config file '{args.config}' does not exist.")

    if not is_distributed():
        device_str = str(args.devices[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if device_str == "cpu" else device_str.replace("cuda:", "")

    args.output_folder = os.path.join(args.output_folder, "")

    return args

def setup_distributed():
    if not is_distributed():
        return 0, 1

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    return local_rank, world_size

def cleanup_distributed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
