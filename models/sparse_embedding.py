from typing import Any
import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from models.common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float,
        cast_to: torch.dtype,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.cast_to = cast_to
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), device=self.device), std=init_std)
        self.register_buffer("weights", weights)

        local_weights = torch.zeros(batch_size, embedding_dim, device=self.device, requires_grad=True)
        local_ids = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.register_buffer("local_weights", local_weights, persistent=False)
        self.register_buffer("local_ids", local_ids, persistent=False)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.weights.device, non_blocking=True)

        if not self.training:
            return self.weights[inputs].to(self.cast_to)

        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)



class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params: Any, world_size: int, lr: float = 1e-3, weight_decay: float = 1e-2):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            local_weights_grad, local_ids, weights = None, None, None
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p

            if local_weights_grad is not None:
                dev = weights.device
                local_weights_grad = local_weights_grad.to(dev, non_blocking=True)
                local_ids = local_ids.to(dev, non_blocking=True)
                _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, **group)



def _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, world_size, lr, weight_decay, **kwargs):
    N, D = local_weights_grad.shape
    dev = weights.device
    local_weights_grad = local_weights_grad.to(dev, non_blocking=True)
    local_ids = local_ids.to(dev, non_blocking=True)

    all_grads, all_ids = (local_weights_grad, local_ids)
    if world_size > 1:
        all_grads = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=dev)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=dev)
        dist.all_gather_into_tensor(all_grads, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)

    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_grads.dtype, device=dev)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_grads)

    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p


