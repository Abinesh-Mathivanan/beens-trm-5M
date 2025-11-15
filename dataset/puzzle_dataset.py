from typing import List
import os, json, numpy as np, pydantic, torch
from torch.utils.data import IterableDataset
from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

def _sample_batch(rng, group_order, p_indices, g_indices, start_idx, b_size):
    batch, p_batch, size = [], [], 0
    while (start_idx < group_order.size) and (size < b_size):
        g_id = group_order[start_idx]
        p_id = rng.integers(g_indices[g_id], g_indices[g_id + 1])
        start_idx += 1
        p_start, p_size = p_indices[p_id], int(p_indices[p_id + 1] - p_indices[p_id])
        add_size = min(p_size, b_size - size)
        p_batch.append(np.full(add_size, p_id, dtype=np.int32))
        batch.append(p_start + np.random.choice(p_size, add_size, replace=False))
        size += add_size
    return start_idx, np.concatenate(batch), np.concatenate(p_batch)

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config, self.split = config, split
        with open(os.path.join(config.dataset_paths[0], split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        self.local_batch_size = config.global_batch_size // config.num_replicas
        self._data, self._iters = None, 0

    def _lazy_load(self):
        if self._data: return
        self._data = {}
        path = self.config.dataset_paths[0]
        modes = {"inputs": "r", "labels": "r", "puzzle_identifiers": None, "puzzle_indices": None, "group_indices": None}
        for s in self.metadata.sets:
            self._data[s] = {f: np.load(os.path.join(path, self.split, f"{s}__{f}.npy"), mmap_mode=m) for f, m in modes.items()}

    def _collate(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}
        batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad = self.local_batch_size - batch["puzzle_identifiers"].size
            pads = {"inputs": self.metadata.pad_id, "labels": IGNORE_LABEL_ID, "puzzle_identifiers": self.metadata.blank_identifier_id}
            batch = {k: np.pad(v, ((0, pad),) + ((0,0),)*(v.ndim-1), 'constant', constant_values=pads[k]) for k,v in batch.items()}
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def __iter__(self):
        self._lazy_load()
        for _, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            g_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _ in range(self.config.epochs_per_iter)])
            start_idx = 0
            while start_idx < g_order.size:
                start_idx, b_indices, p_indices_batch = _sample_batch(rng, g_order, dataset["puzzle_indices"], dataset["group_indices"], start_idx, self.config.global_batch_size)
                if p_indices_batch.size < self.config.global_batch_size: break
                rank_slice = slice(self.config.rank * self.local_batch_size, (self.config.rank + 1) * self.local_batch_size)
                batch = self._collate({
                    "inputs": dataset["inputs"][b_indices[rank_slice]],
                    "labels": dataset["labels"][b_indices[rank_slice]],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][p_indices_batch[rank_slice]]
                })
                yield "train", batch, self.config.global_batch_size