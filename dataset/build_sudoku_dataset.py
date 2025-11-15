import os, csv, json, numpy as np, argparse
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from dataset.common import PuzzleDatasetMetadata

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str
    subsample_size: int
    num_aug: int

def shuffle_sudoku(board, solution):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose = np.random.rand() < 0.5
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in np.random.permutation(3)])
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in np.random.permutation(3)])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])
    def transform(x):
        if transpose: x = x.T
        return digit_map[x.flatten()[mapping].reshape(9, 9)]
    return transform(board), transform(solution)

def convert_subset(name, config):
    inputs, labels = [], []
    with open(hf_hub_download(config.source_repo, f"{name}.csv", repo_type="dataset")) as f:
        r = csv.reader(f); next(r)
        for _, q, a, _ in r:
            inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
    
    if name == "train" and config.subsample_size < len(inputs):
        idx = np.random.choice(len(inputs), config.subsample_size, replace=False)
        inputs, labels = [inputs[i] for i in idx], [labels[i] for i in idx]
    
    res = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    res["puzzle_indices"].append(0); res["group_indices"].append(0)
    ex_id, p_id = 0, 0
    for inp, out in tqdm(zip(inputs, labels), total=len(inputs), desc=f"Processing {name}"):
        for i in range(1 + (config.num_aug if name == "train" else 0)):
            i_aug, o_aug = (inp, out) if i == 0 else shuffle_sudoku(inp, out)
            res["inputs"].append(i_aug); res["labels"].append(o_aug)
            ex_id += 1; p_id += 1
            res["puzzle_indices"].append(ex_id); res["puzzle_identifiers"].append(0)
        res["group_indices"].append(p_id)

    for k in ["inputs", "labels"]: res[k] = np.concatenate(res[k]).reshape(len(res[k]), -1) + 1
    for k in ["group_indices", "puzzle_indices", "puzzle_identifiers"]: res[k] = np.array(res[k], dtype=np.int32)

    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    md = PuzzleDatasetMetadata(seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0, blank_identifier_id=0, num_puzzle_identifiers=1, total_groups=len(res["group_indices"])-1, mean_puzzle_examples=1, total_puzzles=len(res["group_indices"])-1, sets=["all"])
    with open(os.path.join(save_dir, "dataset.json"), "w") as f: json.dump(md.dict(), f)
    for k, v in res.items(): np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsample-size", type=int, default=1000)
    parser.add_argument("--num-aug", type=int, default=1000)
    args = parser.parse_args()
    config = DataProcessConfig(**vars(args))
    convert_subset("train", config)
    convert_subset("test", config)

if __name__ == "__main__":
    main()