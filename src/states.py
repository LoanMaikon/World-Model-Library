from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
import torch
import os

def create_state(
    model,
    config,
    epoch,
    optimizers=None,
    schedulers=None,
    scaler=None,
    dataloader=None,
    device="cpu",
    extra=None,
):
    device = torch.device(device)
    generator = dataloader.generator if dataloader is not None else None
    rng_state = {
        "torch": torch.get_rng_state(),
        "dataloader": generator.get_state() if generator is not None else None,
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    rng_states = [rng_state]
    if torch.distributed.is_initialized():
        rng_states = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rng_states, rng_state)

    model = model.module if isinstance(model, DDP) else model

    return {
        "config": deepcopy(config),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in (optimizers or {}).items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in (schedulers or {}).items()},
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": deepcopy(extra) if extra is not None else {},
        "rng_states": rng_states,
    }

def save_state(state, folder, state_name="state", rank=0):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else rank

    if rank != 0:
        return

    filename = state_name if state_name.endswith(".pth") else f"{state_name}.pth"
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)

def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)

    if "optimizer" in state:
        state["optimizers"] = {"model": state.pop("optimizer")}

    if "lr_scheduler" in state:
        state["schedulers"] = {
            "lr": state.pop("lr_scheduler"),
            "weight_decay": state.pop("wd_scheduler"),
        }

    if "action_dim" in state:
        state["extra"] = {
            "action_dim": state.pop("action_dim"),
            "normalization_stats": state.pop("normalization_stats"),
        }

    return state

def restore_state(
    state,
    model=None,
    optimizers=None,
    schedulers=None,
    scaler=None,
    dataloader=None,
    device="cpu",
    rank=0,
    restore_rng=False,
):
    if model is not None:
        model = model.module if isinstance(model, DDP) else model
        model.load_state_dict(state["model"])

    for name, optimizer in (optimizers or {}).items():
        optimizer.load_state_dict(state["optimizers"][name])

    for name, scheduler in (schedulers or {}).items():
        scheduler.load_state_dict(state["schedulers"][name])

    if scaler is not None:
        scaler.load_state_dict(state["scaler"])

    if restore_rng:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else rank
        if rank >= len(state["rng_states"]):
            raise ValueError("The saved state has no random state for this process.")

        rng_state = state["rng_states"][rank]
        torch.set_rng_state(rng_state["torch"])
        if rng_state["dataloader"] is not None:
            if dataloader is None or dataloader.generator is None:
                raise ValueError("A dataloader with a generator is required to restore its random state.")
            dataloader.generator.set_state(rng_state["dataloader"])

        device = torch.device(device)
        if device.type == "cuda" and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state(rng_state["cuda"], device)
