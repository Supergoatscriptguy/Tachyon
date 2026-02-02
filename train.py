"""Tachyon 2 Training Script"""

import os, sys, argparse, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from model import Tachyon, TachyonConfig, get_model
from fast_dataloader import create_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="large", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--memory_test", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_lr(step, warmup_steps, max_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


def enable_gradient_checkpointing(model):
    for layer in model.layers:
        layer._orig_forward = layer.forward
        def make_ckpt_fwd(module):
            def ckpt_fwd(x, mask=None, kv_cache=None):
                return torch.utils.checkpoint.checkpoint(lambda *args: module._orig_forward(*args), x, mask, kv_cache, use_reentrant=False)
            return ckpt_fwd
        layer.forward = make_ckpt_fwd(layer)


def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(int(n))


def memory_test(args):
    print("=" * 60 + "\nMEMORY TEST\n" + "=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA not available")
        return False

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"Config: {args.model_size}, batch={args.batch_size}, seq={args.seq_len}")
    model = get_model(args.model_size).to(device).to(torch.bfloat16)
    total, active = model.count_parameters()
    print(f"Params: {fmt(total)} total, {fmt(active)} active")

    if not args.no_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if not args.no_compile:
        model = torch.compile(model)

    x = torch.randint(0, 50258, (args.batch_size, args.seq_len), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, labels=x)["loss"]
    loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1e9
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Peak: {peak:.2f}GB / {total_mem:.1f}GB ({100*peak/total_mem:.1f}%)")
    print("PASS" if peak < total_mem * 0.9 else "FAIL")
    return peak < total_mem


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    model = get_model(args.model_size).to(device).to(torch.bfloat16)
    total, active = model.count_parameters()
    print(f"Params: {fmt(total)} total, {fmt(active)} active")

    if not args.no_gradient_checkpointing:
        enable_gradient_checkpointing(model)
    if not args.no_compile:
        print("Compiling...")
        model = torch.compile(model)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if "norm" in name or "bias" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), fused=True)

    dataloader = create_dataloader(args.data_dir, args.batch_size, args.seq_len, args.num_workers)
    data_iter = iter(dataloader)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training from step {start_step}, effective batch={args.batch_size * args.grad_accum}")

    model.train()
    opt.zero_grad()
    tps = args.batch_size * args.seq_len * args.grad_accum
    run_loss = run_aux = 0.0
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            x = batch["input_ids"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, labels=x)
                (out["loss"] / args.grad_accum).backward()
            run_loss += out["loss"].item() / args.grad_accum
            run_aux += out["aux_loss"].item() / args.grad_accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad()

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            print(f"step {step+1:6d} | loss {run_loss/args.log_interval:.4f} | z_loss {run_aux/args.log_interval:.6f} | lr {lr:.2e} | tok/s {fmt(tps*args.log_interval/elapsed)} | mem {mem:.1f}GB")
            run_loss = run_aux = 0.0
            t0 = time.time()

        if (step + 1) % args.save_interval == 0:
            path = Path(args.save_dir) / f"checkpoint_{step+1}.pt"
            torch.save({"step": step+1, "model": model.state_dict(), "optimizer": opt.state_dict(), "config": getattr(model, "config", None)}, path)
            print(f"Saved {path}")

    print("Done!")


def main():
    args = parse_args()
    if args.no_compile: args.compile = False
    if args.no_gradient_checkpointing: args.gradient_checkpointing = False
    if args.memory_test:
        sys.exit(0 if memory_test(args) else 1)
    train(args)


if __name__ == "__main__":
    main()
