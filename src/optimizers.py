from torch import nn
import torch

_NORM_MODULES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.RMSNorm,
)

def get_optimizer_class(optimizer_type):
    match optimizer_type:
        case "adamw":
            return torch.optim.AdamW

        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def load_planning_optimizer(params, optimizer_type, lr, action_noise=0.0, generator=None):
    optimizer = get_optimizer_class(optimizer_type)(params, lr=lr)

    if action_noise > 0:
        def add_noise(optimizer, args, kwargs):
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        noise = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator)
                        param.add_(noise * action_noise)

        optimizer.register_step_post_hook(add_noise)

    return optimizer

def make_param_groups(
    model,
    weight_decay,
    decay_bias=True,
    decay_norm=True,
    lr=None,
    fix_lr=False,
):
    param_groups = []
    seen_parameters = set()

    for module_name, module in model.named_modules():
        is_norm_module = isinstance(module, _NORM_MODULES)

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen_parameters:
                continue
            seen_parameters.add(id(param))

            is_bias = param_name == "bias"
            is_norm = is_norm_module
            use_decay = True

            if is_bias and not decay_bias:
                use_decay = False

            if is_norm and not decay_norm:
                use_decay = False

            group = {
                "params": [param],
                "weight_decay": weight_decay if use_decay else 0.0,
                "weight_decay_exclude": not use_decay,
                "is_bias": is_bias,
                "is_norm": is_norm,
                "decay_bias": decay_bias,
                "decay_norm": decay_norm,
                "fix_lr": fix_lr,
            }

            if lr is not None:
                group["lr"] = lr
                group["initial_lr"] = lr

            param_groups.append(group)

    return param_groups

def load_optimizer(
    model,
    optimizer_type,
    lr,
    weight_decay=0.0,
    decay_bias=True,
    decay_norm=True,
    fix_lr=False,
    **kwargs,
):
    param_groups = make_param_groups(
        model=model,
        weight_decay=weight_decay,
        decay_bias=decay_bias,
        decay_norm=decay_norm,
        lr=lr,
        fix_lr=fix_lr,
    )

    return get_optimizer_class(optimizer_type)(param_groups, lr=lr, weight_decay=weight_decay, **kwargs)
