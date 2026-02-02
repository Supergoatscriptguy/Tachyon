# Tachyon 2

A Mixture of Experts transformer I built for fast pretraining. Uses modern techniques like GQA, RoPE, Flash Attention, and SwiGLU.

## What's in it

**MoE (Mixture of Experts)** — Each layer has 8 routed experts + 1 shared expert. Only the top-2 experts are used per token, so you get more total params but the same compute cost as a smaller dense model.

**GQA (Grouped Query Attention)** — 4 KV heads shared across 16 Q heads. Saves memory during inference without hurting quality.

**RoPE** — Rotary position embeddings. The standard now.

**Flash Attention** — Uses flash-attn if installed, otherwise falls back to PyTorch's SDPA.

**SwiGLU** — The gated activation everyone uses now. `w2(silu(w1(x)) * w3(x))`

**Z-loss + Load Balancing** — Keeps the router from collapsing and makes sure all experts get used.

## Model sizes

| Size | Layers | Experts | Total | Active |
|------|--------|---------|-------|--------|
| tiny | 8 | 4 | 110M | 75M |
| small | 16 | 8 | 350M | 150M |
| medium | 20 | 8 | 1B | 400M |
| large | 24 | 8 | 2B | 700M |

The "Active" column is how many params actually run per forward pass. That's what determines your speed/memory.

## Setup

### 1. Install dependencies

```bash
pip install torch numpy tokenizers huggingface_hub
```

Optional but recommended for faster attention:
```bash
pip install flash-attn --no-build-isolation
```

### 2. Get the tokenizer

Download `pile_tokenizer.json` from my tokenizers repo: https://github.com/Supergoatscriptguy/Tokenizers

Put it in the same folder as the scripts.

### 3. Get training data

I have a preprocessed dataset on HuggingFace: [SuperGoatScriptGuy/PreprocessedMIXED](https://huggingface.co/datasets/SuperGoatScriptGuy/PreprocessedMIXED)

It's ~215GB total (2685 shards), but you don't need all of it. For testing, 50 shards (~4GB) is plenty.

```python
from huggingface_hub import snapshot_download

# Download just 50 shards for testing
allow_patterns = [f'shard_{i:05d}.npy' for i in range(50)]
snapshot_download(
    repo_id='SuperGoatScriptGuy/PreprocessedMIXED',
    repo_type='dataset',
    local_dir='./data',
    allow_patterns=allow_patterns,
)
```

Or download more if you want a real training run. Each shard is ~82MB with ~20M tokens.

The shards are numpy arrays with shape `(sequences, 2048)` containing token IDs.

## Training

Once you have the tokenizer and data:

```bash
python train.py --model_size tiny --data_dir ./data --batch_size 32 --grad_accum 4
```

On an H100 with the tiny model, I get ~95k tokens/sec with torch.compile enabled.

Useful flags:
- `--no_compile` — Skip torch.compile if it's giving you issues
- `--no_gradient_checkpointing` — Faster but uses more memory
- `--save_interval 1000` — Save checkpoints more often
- `--max_steps 10000` — For quick test runs

## Generation

After training, test your checkpoint:

```bash
python generate.py --checkpoint checkpoint_5000.pt
```

This drops you into an interactive prompt. Or pass `--prompt "your text"` for a single generation.

## Files

- `model.py` — The actual model
- `train.py` — Training loop
- `fast_dataloader.py` — Streaming dataloader for the .npy shards
- `generate.py` — Text generation
