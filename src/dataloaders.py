from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch

def load_dataloader(
    dataset,
    batch_size,
    num_workers=0,
    prefetch_factor=2,
    persistent_workers=False,
    pin_memory=False,
    drop_last=False,
    shuffle=True,
    seed=0,
    rank=0,
    world_size=1,
):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed + rank),
    )

    return dataloader, sampler
