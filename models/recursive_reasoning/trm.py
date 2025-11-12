
from typing import List, Dict, Optional
from dataclasses import dataclass
import math, torch, torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.common import trunc_normal_init_


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict

class TRMConfig(BaseModel):
    model_config = {"extra": "allow"}
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    forward_dtype: str
    halt_max_steps: int
    halt_exploration_prob: float
    mlp_t: bool
    puzzle_emb_len: int
    no_ACT_continue: bool
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0



def select_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16



class TRMBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
        if config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=config.seq_len + self.puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(config.hidden_size, config.hidden_size // config.num_heads, config.num_heads, config.num_heads, causal=False)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, x: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            x = x.transpose(1, 2)
            x = rms_norm(x + self.mlp_t(x), self.norm_eps).transpose(1, 2)
        else:
            x = rms_norm(x + self.self_attn(cos_sin, x), self.norm_eps)
        return rms_norm(x + self.mlp(x), self.norm_eps)



class TRMModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, inject: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        x = x + inject
        for layer in self.layers:
            x = layer(cos_sin, x)
        return x



class TRM_Inner(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        self.compute_dtype = select_compute_dtype()
        self.dtype = getattr(torch, config.forward_dtype) if isinstance(config.forward_dtype, str) else config.forward_dtype

        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, 1.0 / math.sqrt(config.hidden_size), self.dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size) if config.puzzle_emb_len == 0 else config.puzzle_emb_len
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim, config.batch_size, 0, self.compute_dtype)

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.seq_len + self.puzzle_emb_len, config.rope_theta)

        self.L_level = TRMModule([TRMBlock(config) for _ in range(config.L_layers)])

        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.dtype)), persistent=False)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.dtype)), persistent=False)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    
    def _embed(self, x, p_ids):
        x = x.to(self.embed_tokens.embedding_weight.device)
        emb = self.embed_tokens(x.int()) * math.sqrt(self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            p_ids = p_ids.to(self.puzzle_emb.local_weights.device) if hasattr(self.puzzle_emb, "local_weights") else p_ids.to(self.compute_dtype)
            p_emb = self.puzzle_emb(p_ids)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - p_emb.shape[-1]
            if pad_count > 0:
                p_emb = F.pad(p_emb, (0, pad_count))
            emb = torch.cat((p_emb.view(-1, self.puzzle_emb_len, self.config.hidden_size), emb), dim=-2)
        return emb


    def empty_carry(self, bs: int, device: Optional[torch.device] = None) -> TRMInnerCarry:
        device = device if device is not None else next(self.parameters()).device
        shape = (bs, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        return TRMInnerCarry(
            z_H=torch.zeros(shape, dtype=self.compute_dtype, device=device),
            z_L=torch.zeros(shape, dtype=self.compute_dtype, device=device),
        )

    def reset_carry(self, flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        device = carry.z_H.device
        flag = flag.to(device).view(-1, 1, 1)
        H_init = self.H_init.to(device).view(1, 1, -1).to(self.compute_dtype)
        L_init = self.L_init.to(device).view(1, 1, -1).to(self.compute_dtype)
        z_H = carry.z_H.to(device)
        z_L = carry.z_L.to(device)
        return TRMInnerCarry(
            z_H=torch.where(flag, H_init, z_H),
            z_L=torch.where(flag, L_init, z_L),
        )


    def forward(self, carry: TRMInnerCarry, batch: Dict):
        seq_info = {"cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None}
        embeds = self._embed(batch["inputs"], batch["puzzle_identifiers"]).to(self.compute_dtype, non_blocking=True)

        z_H = carry.z_H.to(self.compute_dtype, non_blocking=True)
        z_L = carry.z_L.to(self.compute_dtype, non_blocking=True)

        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + embeds, seq_info["cos_sin"])
                z_H = self.L_level(z_H, z_L, seq_info["cos_sin"])

        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + embeds, seq_info["cos_sin"])
        z_H = self.L_level(z_H, z_L, seq_info["cos_sin"])

        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())

        # logits and q_head: cast to param dtype for stable softmax / loss if needed
        logits = self.lm_head(z_H.to(self.dtype))[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0].to(self.dtype)).float()
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])



class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, cfg_dict):
        super().__init__()
        self.config = TRMConfig(**cfg_dict)
        self.inner = TRM_Inner(self.config)
        self.compute_dtype = self.inner.compute_dtype

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict) -> TRMCarry:
        bs = batch["inputs"].shape[0]
        device = batch["inputs"].device
        inner = self.inner.empty_carry(bs, device=device)
        steps = torch.zeros(bs, dtype=torch.int32, device=device)
        halted = torch.ones(bs, dtype=torch.bool, device=device)
        current_data = {k: v.to(device) for k, v in batch.items()}
        return TRMCarry(inner_carry=inner, steps=steps, halted=halted, current_data=current_data)

    def forward(self, carry: TRMCarry, batch: Dict):
        data = {k: torch.where(carry.halted.view((-1,) + (1,) * (v.ndim - 1)), v, carry.current_data[k]) for k, v in batch.items()}

        new_inner_carry, logits, (q_halt, q_cont) = self.inner(self.inner.reset_carry(carry.halted, carry.inner_carry), data)
        steps = torch.where(carry.halted, 0, carry.steps) + 1

        with torch.no_grad():
            halted = steps >= self.config.halt_max_steps
            if self.training:
                if self.config.no_ACT_continue:
                    halted |= (q_halt > 0)
                else:
                    halted |= (q_halt > q_cont)
                min_steps = (torch.rand_like(q_halt) < self.config.halt_exploration_prob) * torch.randint_like(steps, 2, self.config.halt_max_steps + 1)
                halted &= (steps >= min_steps)


        out_carry = TRMCarry(inner_carry=new_inner_carry, steps=steps, halted=halted, current_data=data)
        return out_carry, {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_cont}



def move_carry_to_cpu(carry: TRMCarry) -> TRMCarry:
    inner = TRMInnerCarry(
        z_H=carry.inner_carry.z_H.detach().to("cpu"),
        z_L=carry.inner_carry.z_L.detach().to("cpu"),
    )
    steps = carry.steps.to("cpu")
    halted = carry.halted.to("cpu")
    current_data = {k: v.to("cpu") for k, v in carry.current_data.items()}
    return TRMCarry(inner_carry=inner, steps=steps, halted=halted, current_data=current_data)



def move_carry_to_device(carry: TRMCarry, device: torch.device, compute_dtype: Optional[torch.dtype] = None) -> TRMCarry:
    compute_dtype = compute_dtype if compute_dtype is not None else select_compute_dtype()
    inner = TRMInnerCarry(
        z_H=carry.inner_carry.z_H.to(device=device, dtype=compute_dtype),
        z_L=carry.inner_carry.z_L.to(device=device, dtype=compute_dtype),
    )
    steps = carry.steps.to(device=device)
    halted = carry.halted.to(device=device)
    current_data = {k: v.to(device=device) for k, v in carry.current_data.items()}
    return TRMCarry(inner_carry=inner, steps=steps, halted=halted, current_data=current_data)


