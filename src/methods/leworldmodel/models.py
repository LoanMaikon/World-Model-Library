# Code from https://github.com/lucas-maes/le-wm/tree/main with some adaptations

from transformers import ViTConfig, ViTModel
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch

class Encoder(nn.Module):
    def __init__(self,
                 model_name="vit_tiny",
                 image_size=224,
                 patch_size=14,
                ):

        super().__init__()

        match model_name:
            case "vit_tiny":
                embed_dim, depth, num_heads = 192, 12, 3
            case "vit_small":
                embed_dim, depth, num_heads = 384, 12, 6
            case "vit_base":
                embed_dim, depth, num_heads = 768, 12, 12
            case "vit_large":
                embed_dim, depth, num_heads = 1024, 24, 16
            case "vit_huge":
                embed_dim, depth, num_heads = 1280, 32, 16
            case _:
                raise ValueError(f"Unsupported encoder: {model_name}")

        self.embed_dim = embed_dim
        config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * 4,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.model = ViTModel(config, add_pooling_layer=False, use_mask_token=False)

    def forward(self, pixels):
        output = self.model(pixels.float(), interpolate_pos_encoding=True)
        return output.last_hidden_state[:, 0]

class ActionEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 smoothed_dim=10,
                 mlp_scale=4,
                ):

        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * embed_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * embed_dim, embed_dim),
        )

    def forward(self, actions):
        actions = self.patch_embed(actions.float().transpose(1, 2))
        return self.embed(actions.transpose(1, 2))

class Projector(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 norm="batch_norm",
                ):

        super().__init__()

        match norm:
            case "batch_norm":
                norm_layer = nn.BatchNorm1d(hidden_dim)
            case "layer_norm":
                norm_layer = nn.LayerNorm(hidden_dim)
            case "none" | "identity":
                norm_layer = nn.Identity()
            case _:
                raise ValueError(f"Unsupported projector normalization: {norm}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        inner_dim = num_heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if inner_dim != dim or num_heads != 1 else nn.Identity()

    def forward(self, x):
        batch_size, num_frames = x.shape[:2]
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = [value.reshape(batch_size, num_frames, self.num_heads, self.dim_head).transpose(1, 2) for value in qkv]
        dropout = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        x = x.transpose(1, 2).reshape(batch_size, num_frames, -1)
        return self.to_out(x)

class ConditionalBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, num_heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, actions):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(actions).chunk(6, dim=-1)
        x = x + gate_attn * self.attn(modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalBlock(embed_dim, num_heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, actions):
        for layer in self.layers:
            x = layer(x, actions)
        return self.norm(x)

class Predictor(nn.Module):
    def __init__(self,
                 embed_dim,
                 context_length,
                 depth=6,
                 num_heads=16,
                 dim_head=64,
                 mlp_dim=2048,
                 dropout=0.1,
                 emb_dropout=0.0,
                ):

        super().__init__()
        self.context_length = context_length
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(embed_dim, depth, num_heads, dim_head, mlp_dim, dropout)

    def forward(self, embeddings, actions):
        num_frames = embeddings.size(1)
        if num_frames > self.context_length:
            raise ValueError("Predictor input exceeds context_length.")
        embeddings = self.dropout(embeddings + self.pos_embedding[:, :num_frames])
        return self.transformer(embeddings, actions)

class JEPA(nn.Module):
    def __init__(self,
                 encoder,
                 action_encoder,
                 predictor,
                 projector,
                 pred_proj,
                 context_length=1,
                ):

        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.projector = projector
        self.pred_proj = pred_proj
        self.context_length = context_length
        self.register_buffer("action_low", None, persistent=False)
        self.register_buffer("action_high", None, persistent=False)

    def set_action_bounds(self, low, high):
        device = next(self.parameters()).device
        self.action_low = torch.as_tensor(low, device=device, dtype=torch.float32).flatten()
        self.action_high = torch.as_tensor(high, device=device, dtype=torch.float32).flatten()

    def encode(self, batch):
        batch_size, num_frames = batch["pixels"].shape[:2]
        pixels = batch["pixels"].flatten(0, 1)
        embeddings = self.projector(self.encoder(pixels))
        output = dict(batch)
        output["emb"] = embeddings.reshape(batch_size, num_frames, -1)
        if "action" in batch:
            output["act_emb"] = self.action_encoder(batch["action"])
        return output

    def predict(self, embeddings, actions):
        predictions = self.predictor(embeddings, actions)
        batch_size, num_frames = predictions.shape[:2]
        predictions = self.pred_proj(predictions.flatten(0, 1))
        return predictions.reshape(batch_size, num_frames, -1)

    def forward(self, batch):
        output = self.encode(batch)
        if "act_emb" in output:
            output["pred_emb"] = self.predict(
                output["emb"][:, :self.context_length],
                output["act_emb"][:, :self.context_length],
            )
        return output

    def rollout(self, embeddings, action_sequence, past_actions=None):
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        if embeddings.ndim == 2:
            embeddings = embeddings[:, None]
        history_length = embeddings.size(1)
        if history_length > 1:
            if past_actions is None or past_actions.size(1) != history_length - 1:
                raise ValueError("Each pair of observed frames requires its intervening action block.")
            past_actions = past_actions[:, None].expand(-1, num_samples, -1, -1)
            action_sequence = torch.cat([past_actions, action_sequence], dim=2)
        embeddings = embeddings[:, None].expand(-1, num_samples, -1, -1).flatten(0, 1)
        actions = self.action_encoder(action_sequence.flatten(0, 1))

        for step in range(horizon):
            end = history_length + step
            start = max(0, end - self.context_length)
            prediction = self.predict(
                embeddings[:, start:end],
                actions[:, start:end],
            )[:, -1:]
            embeddings = torch.cat([embeddings, prediction], dim=1)

        return embeddings.reshape(batch_size, num_samples, history_length + horizon, -1)

    def get_cost(self, info_dict, action_candidates):
        device = next(self.parameters()).device

        if "emb" not in info_dict or "goal_emb" not in info_dict:
            with torch.no_grad():
                pixels = info_dict["pixels"][:, 0, -self.context_length:].to(device)
                goal = info_dict["goal"][:, 0, -1:].to(device)
                embeddings = self.encode({"pixels": pixels})["emb"]
                info_dict["emb"] = embeddings[:, 0] if embeddings.size(1) == 1 else embeddings
                info_dict["goal_emb"] = self.encode({"pixels": goal})["emb"][:, -1]

        action_candidates = action_candidates.to(device)
        if self.action_low is not None:
            action_candidates = action_candidates.clamp(min=self.action_low, max=self.action_high)

        embeddings = info_dict["emb"]
        if embeddings.ndim == 2:
            prediction = self.rollout(embeddings, action_candidates)[:, :, -1]
        else:
            past_actions = info_dict["past_action"][:, 0].to(device)
            lengths = info_dict.get("history_length")
            lengths = (
                lengths[:, 0].to(device=device, dtype=torch.long) if lengths is not None
                else torch.full((embeddings.size(0),), embeddings.size(1), device=device, dtype=torch.long)
            )
            prediction = embeddings.new_empty(*action_candidates.shape[:2], embeddings.size(-1))
            # Ignore padding for environments that have just reset.
            for length in lengths.unique().tolist():
                selected = lengths == length
                prediction[selected] = self.rollout(
                    embeddings[selected, -length:],
                    action_candidates[selected],
                    past_actions[selected, -(length - 1):] if length > 1 else None,
                )[:, :, -1]
        return (prediction - info_dict["goal_emb"][:, None]).square().sum(dim=-1)

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.clone()
        dist.all_reduce(tensor)
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient)
        return gradient

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("SIGReg requires at least two knots and one projection.")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        weights = torch.full((knots,), 2 * 3 / (knots - 1), dtype=torch.float32)
        weights[[0, -1]] /= 2
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, embeddings):
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            embeddings = embeddings.float()
            projections = torch.randn(embeddings.size(-1), self.num_proj, device=embeddings.device)
            projections = F.normalize(projections, dim=0)
            if is_distributed:
                dist.broadcast(projections, src=0)

            x = (embeddings @ projections).unsqueeze(-1) * self.t
            moments = torch.stack([x.cos().sum(dim=-3), x.sin().sum(dim=-3)])
            batch_size = embeddings.new_tensor(embeddings.size(-2))
            if is_distributed:
                moments = AllReduce.apply(moments)
                dist.all_reduce(batch_size)

            cosine, sine = moments / batch_size
            error = (cosine - self.phi).square() + sine.square()
            return ((error @ self.weights) * batch_size).mean()
