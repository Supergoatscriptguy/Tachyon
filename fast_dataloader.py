"""Fast DataLoader — memory-mapped .npy shards"""

import os, json, random
import numpy as np
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class ShardedDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, split: str = "train", train_ratio: float = 0.98):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.shard_paths = sorted(self.data_dir.glob("shard_*.npy"))
        if not self.shard_paths:
            raise ValueError(f"No shards in {data_dir}")

        split_idx = int(len(self.shard_paths) * train_ratio)
        self.shard_paths = self.shard_paths[:split_idx] if split == "train" else self.shard_paths[split_idx:]
        print(f"[{split}] {len(self.shard_paths)} shards")

        sample = np.load(self.shard_paths[0], mmap_mode="r")
        self.sequences_per_shard = sample.shape[0]
        self.total_sequences = len(self.shard_paths) * self.sequences_per_shard
        self.cached_shard_idx = -1
        self.cached_shard = None

    def _load_shard(self, idx):
        if idx != self.cached_shard_idx:
            self.cached_shard = np.load(self.shard_paths[idx])
            self.cached_shard_idx = idx
        return self.cached_shard

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        shard_idx, seq_idx = idx // self.sequences_per_shard, idx % self.sequences_per_shard
        tokens = self._load_shard(shard_idx)[seq_idx]
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        elif len(tokens) < self.seq_len:
            tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), constant_values=1)
        return {"input_ids": torch.from_numpy(tokens.astype(np.int64))}


class StreamingShardDataset(IterableDataset):
    def __init__(self, data_dir: str, seq_len: int = 2048, split: str = "train", train_ratio: float = 0.98, shuffle: bool = True, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.shard_paths = sorted(self.data_dir.glob("shard_*.npy"))
        if not self.shard_paths:
            raise ValueError(f"No shards in {data_dir}")

        split_idx = int(len(self.shard_paths) * train_ratio)
        self.shard_paths = self.shard_paths[:split_idx] if split == "train" else self.shard_paths[split_idx:]
        print(f"[{split}] {len(self.shard_paths)} shards (streaming)")

    def _get_worker_shards(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return list(self.shard_paths)
        per_worker = len(self.shard_paths) // info.num_workers
        start = info.id * per_worker
        return self.shard_paths[start:] if info.id == info.num_workers - 1 else self.shard_paths[start:start + per_worker]

    def __iter__(self):
        shards = self._get_worker_shards()
        if self.shuffle:
            info = torch.utils.data.get_worker_info()
            rng = random.Random(self.seed + (info.id if info else 0))
            shards = list(shards)
            rng.shuffle(shards)

        for shard_path in shards:
            data = np.load(shard_path)
            indices = list(range(data.shape[0]))
            if self.shuffle:
                random.shuffle(indices)
            for i in indices:
                tokens = data[i]
                if len(tokens) > self.seq_len:
                    tokens = tokens[:self.seq_len]
                elif len(tokens) < self.seq_len:
                    tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), constant_values=1)
                yield {"input_ids": torch.from_numpy(tokens.astype(np.int64))}


def create_dataloader(data_dir: str, batch_size: int = 64, seq_len: int = 2048, num_workers: int = 4, split: str = "train", streaming: bool = True):
    if streaming:
        dataset = StreamingShardDataset(data_dir, seq_len, split, shuffle=(split == "train"))
        shuffle = False
    else:
        dataset = ShardedDataset(data_dir, seq_len, split)
        shuffle = (split == "train")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, prefetch_factor=2 if num_workers > 0 else None,
                      drop_last=True, persistent_workers=num_workers > 0)


if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=100)
    args = p.parse_args()

    loader = create_dataloader(args.data_dir, args.batch_size, args.seq_len, args.num_workers)
    for i, batch in enumerate(loader):
        if i >= 3: break
        print(f"Batch {i}: {batch['input_ids'].shape}")

    start, tokens = time.time(), 0
    for i, batch in enumerate(loader):
        tokens += batch["input_ids"].numel()
        if i >= args.num_batches - 1: break
    print(f"{tokens/(time.time()-start)/1e6:.2f}M tok/s")
