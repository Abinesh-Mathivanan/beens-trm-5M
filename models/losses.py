from typing import Any, Tuple, Dict, Sequence, Optional
import torch, torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, eps=1e-30): 
    return torch.where(x < 0, 1 / (1 - x + eps), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    if valid_mask is None: valid_mask = (labels != ignore_index)
    labels = torch.where(valid_mask, labels, 0)
    pred_logprobs = torch.gather(logprobs, index=labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, pred_logprobs, 0)



class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, **kwargs):
        super().__init__()
        self.model, self.loss_fn = model, globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs): 
        return self.model.initial_carry(*args, **kwargs)
    
    def forward(self, return_keys: Sequence[str], **kwargs) -> Tuple[Any, torch.Tensor, Dict, Optional[Dict], torch.Tensor]:
        carry, outputs = self.model(**kwargs)
        labels = carry.current_data["labels"]
        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1).clamp_min(1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid = carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid.sum(),
                "accuracy": torch.where(valid, is_correct.sum(-1).float() / loss_counts, 0).sum(),
                "exact_accuracy": (valid & seq_is_correct).sum(),
                "q_halt_accuracy": (valid & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid, carry.steps, 0).sum(),
            }
        lm_loss = (self.loss_fn(outputs["logits"], labels, valid_mask=mask) / loss_counts.unsqueeze(-1)).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})
        total_loss = lm_loss + 0.5 * q_halt_loss
        if "target_q_continue" in outputs:
            q_cont_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_cont_loss.detach()
            total_loss += 0.5 * q_cont_loss
        return carry, total_loss, metrics, {k: v.detach() for k, v in outputs.items() if k in return_keys}, carry.halted.all()

