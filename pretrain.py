from typing import Optional, Any, Sequence, Dict
from dataclasses import dataclass
import os, math, torch, torch.optim as optim, torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import tqdm, pydantic
from omegaconf import OmegaConf

from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from models.recursive_reasoning.trm import move_carry_to_device as trm_move_carry_to_device
from models.recursive_reasoning.trm import move_carry_to_cpu as trm_move_carry_to_cpu


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class PretrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    arch: ArchConfig
    data_paths: list[str]
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    run_name: Optional[str] = None
    seed: int = 0
    eval_interval: Optional[int] = None
    ema: bool = False
    ema_rate: float = 0.999

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence
    optimizer_lrs: Sequence
    carry: Any
    step: int
    total_steps: int

def create_dataloader(config, rank, world_size, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ),
        split="train"
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    ), dataset.metadata

def create_model(config, metadata, rank, world_size):
    arch_cfg = config.arch.dict()
    loss_cfg = arch_cfg.pop('loss')
    arch_cfg.pop('name') # not a model param
    
    model_constructor_cfg = {
        **arch_cfg,
        "batch_size": config.global_batch_size // world_size,
        "vocab_size": metadata.vocab_size,
        "seq_len": metadata.seq_len,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
    }
    

    model = load_model_class(config.arch.name)(model_constructor_cfg)
    loss_head_cls = load_model_class(loss_cfg['name'])
    model = loss_head_cls(model, **loss_cfg)
    

    with torch.device("cuda"):
        model.cuda()
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
    
    main_optimizer = optim.AdamW(
        model.parameters(),
        lr=0,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
    
    if config.arch.get('puzzle_emb_ndim', 0) > 0: # Check if puzzle_emb_ndim exists
        emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),
            lr=0,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size
        )
        optimizers = [emb_optimizer, main_optimizer]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]
    else:
        optimizers = [main_optimizer]
        optimizer_lrs = [config.lr]
    
    return model, optimizers, optimizer_lrs


def cosine_lr(step, base_lr, warmup, total, min_ratio):
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    prog = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * (
        min_ratio + max(0., (1 - min_ratio) * 0.5 * (1. + math.cos(math.pi * prog)))
    )

def train_batch(state, batch, b_size, world_size, config):
    state.step += 1
    if state.step > state.total_steps:
        return None, True 
    
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    
    if state.carry is not None:
        state.carry = trm_move_carry_to_device(state.carry, "cuda")
    
    if state.carry is None:
        state.carry = state.model.initial_carry(batch)
    
    state.carry, loss, metrics, _, _ = state.model(
        carry=state.carry,
        batch=batch,
        return_keys=[]
    )
    
    (loss / b_size).backward()
    
    if world_size > 1:
        for p in state.model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
    
    lr = 0 # initialize lr
    for optim, base_lr in zip(state.optimizers, state.optimizer_lrs):
        lr = cosine_lr(
            state.step,
            base_lr,
            config.lr_warmup_steps,
            state.total_steps,
            config.lr_min_ratio
        )
        for pg in optim.param_groups:
            pg['lr'] = lr
        optim.step()
        optim.zero_grad(set_to_none=True)
    
    state.carry = trm_move_carry_to_cpu(state.carry)
    torch.cuda.empty_cache()
    
    metrics['lr'] = lr
    return {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}, False


def launch(config: PretrainConfig):
    RANK, WORLD_SIZE = 0, 1 # assuming single-process, single-GPU
    
    torch.manual_seed(config.seed + RANK)
    
    loader, metadata = create_dataloader(
        config, RANK, WORLD_SIZE,
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size
    )
    
    total_steps = int(
        config.epochs * metadata.total_groups * metadata.mean_puzzle_examples / config.global_batch_size
    )
    
    model, optimizers, lrs = create_model(config, metadata, RANK, WORLD_SIZE)
    state = TrainState(model, optimizers, lrs, None, 0, total_steps)
    
    print(f"--- Starting Training for {config.run_name} ---")
    print(f"Total training steps: {total_steps}")
    print(f"Number of parameters: {sum(p.numel() for p in state.model.parameters()):,}")
    
    progress_bar = tqdm.tqdm(total=total_steps, dynamic_ncols=True)
    ema_helper = EMAHelper(mu=config.ema_rate) if config.ema else None
    if ema_helper:
        ema_helper.register(state.model)
    
    state.model.train()
    done = False
    for _, batch, b_size in loader:
        if done:
            break
        metrics, done = train_batch(state, batch, b_size, WORLD_SIZE, config)
        if metrics is not None:
            progress_bar.update(1)
            count = metrics.get('count', 0)
            if count > 0:
                log_str = (
                    f"Step {state.step}/{total_steps} | "
                    f"Loss: {metrics.get('lm_loss', 0)/count:.4f} | "
                    f"Acc: {metrics.get('accuracy', 0)*100/count:.2f}% | "
                    f"LR: {metrics.get('lr'):.2e}"
                )
                progress_bar.set_description(log_str)
            
            if ema_helper:
                ema_helper.update(state.model)
    
    progress_bar.close()
    print("\n--- Training Finished ---")


def main():
    hydra_config = OmegaConf.load("configs/pretrain_sudoku.yaml")
    config = PretrainConfig(**hydra_config)

    torch._dynamo.config.suppress_errors = True # supress warnings
    
    try:
        launch(config)
    except Exception as e:
        print(f"⚠️ PyTorch compile failed, falling back to eager mode. Reason: {e}")
        torch._dynamo.disable()
        launch(config)

if __name__ == "__main__":
    main()