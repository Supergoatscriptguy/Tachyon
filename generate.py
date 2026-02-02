"""Tachyon 2 — Text Generation"""

import argparse
import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import Tachyon, TachyonConfig, get_model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" in ckpt and ckpt["config"]:
        model = Tachyon(ckpt["config"])
    else:
        d_model = ckpt["model"]["embed.weight"].shape[1]
        size = {512: "tiny", 1024: "small", 1536: "medium"}.get(d_model, "large")
        model = get_model(size)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).to(torch.bfloat16).eval()
    print(f"Loaded step {ckpt.get('step', '?')}")
    return model


def load_tokenizer(path=None):
    path = Path(path) if path else Path(__file__).parent / "pile_tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=128, temperature=0.8, top_k=50, top_p=0.9):
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        if ids.shape[1] > model.config.max_seq_len:
            ids = ids[:, -model.config.max_seq_len:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(ids)["logits"][:, -1, :]

        if temperature > 0:
            logits = logits / temperature
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")

        next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if next_tok.item() == 0:
            break
        ids = torch.cat([ids, next_tok], dim=1)

    return tokenizer.decode(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)

    if args.prompt:
        print(generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p))
    else:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if prompt.strip():
                print(generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p) + "\n")


if __name__ == "__main__":
    main()
