from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features), device=self.device),
                std=1.0 / (in_features ** 0.5)
            )
        )
        self.bias = nn.Parameter(torch.zeros((out_features,), device=self.device)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.weight.device, non_blocking=True)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias=bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_std, cast_to, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(cast_to, torch.dtype):
            self.cast_to = cast_to
        else:
            self.cast_to = getattr(torch, cast_to)

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((num_embeddings, embedding_dim), device=self.device),
                std=init_std
            )
        )

    def forward(self, x):
        x = x.to(self.embedding_weight.device, non_blocking=True)
        emb = torch.index_select(self.embedding_weight, 0, x.view(-1))
        emb = emb.view(*x.shape, -1).to(self.cast_to)
        return emb


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(self.device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.device), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False, device=self.device)
        self.o_proj = CastedLinear(num_heads * head_dim, hidden_size, bias=False, device=self.device)
        self.causal = causal

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.qkv_proj.weight.device, non_blocking=True)
        bs, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(bs, seq_len, -1, self.head_dim)
        query, key, value = torch.split(qkv, [self.num_heads, self.num_heads, self.num_heads], dim=2)
        if cos_sin:
            query, key = apply_rotary_pos_emb(query, key, *cos_sin)
        query, key, value = map(lambda t: einops.rearrange(t, 'b s h d -> b h s d'), (query, key, value))
        attn_output = scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        return self.o_proj(einops.rearrange(attn_output, 'b h s d -> b s (h d)'))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False, device=self.device)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False, device=self.device)

    def forward(self, x):
        x = x.to(self.gate_up_proj.weight.device, non_blocking=True)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return (hidden_states * torch.rsqrt(variance + variance_epsilon)).to(input_dtype)


