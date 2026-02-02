"""Tachyon 2 — MoE Transformer with GQA, RoPE, Flash Attention, SwiGLU"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class TachyonConfig:
    vocab_size: int = 50258
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 5632
    d_shared_ff: int = 2816
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    top_k_experts: int = 2
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    z_loss_coef: float = 0.001
    balance_loss_coef: float = 0.01

    @classmethod
    def tiny(cls):
        return cls(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, d_ff=1408,
                   d_shared_ff=704, n_routed_experts=4, n_shared_experts=1, top_k_experts=2)

    @classmethod
    def small(cls):
        return cls(d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, d_ff=2816,
                   d_shared_ff=1408, n_routed_experts=8, n_shared_experts=1, top_k_experts=2)

    @classmethod
    def medium(cls):
        return cls(d_model=1536, n_layers=20, n_heads=16, n_kv_heads=4, d_ff=4096,
                   d_shared_ff=2048, n_routed_experts=8, n_shared_experts=1, top_k_experts=2)

    @classmethod
    def large(cls):
        return cls(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=5632,
                   d_shared_ff=2816, n_routed_experts=8, n_shared_experts=1, top_k_experts=2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cached_seq_len = 0

    def _build_cache(self, seq_len, device, dtype):
        if self.inv_freq is None or self.inv_freq.device != device:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, seq_len, device, dtype):
        if self.cos_cached is None or seq_len > self._cached_seq_len or self.cos_cached.device != device:
            self._build_cache(seq_len, device, dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class KVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(self, k, v):
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = self.v_cache = None

    @property
    def seq_len(self):
        return 0 if self.k_cache is None else self.k_cache.size(2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TachyonConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.layer_idx = layer_idx
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x, mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        cache_len = kv_cache.seq_len if kv_cache else 0
        cos, sin = self.rotary(cache_len + seq_len, x.device, x.dtype)
        q = apply_rotary_emb(q, cos[cache_len:], sin[cache_len:])
        k = apply_rotary_emb(k, cos[cache_len:], sin[cache_len:])

        if kv_cache:
            k, v = kv_cache.update(k, v)

        if FLASH_ATTN_AVAILABLE and mask is None and kv_cache is None:
            q_fa, k_fa, v_fa = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=self.dropout_p if self.training else 0.0, causal=True)
            attn_out = attn_out.view(batch, seq_len, -1)
        else:
            k_exp = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
            v_exp = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0, is_causal=mask is None and kv_cache is None)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, config: TachyonConfig):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model
        self.z_loss_coef = config.z_loss_coef
        self.balance_loss_coef = config.balance_loss_coef

        self.gate = nn.Linear(config.d_model, config.n_routed_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(config.n_routed_experts, config.d_ff, config.d_model))
        self.w2 = nn.Parameter(torch.empty(config.n_routed_experts, config.d_model, config.d_ff))
        self.w3 = nn.Parameter(torch.empty(config.n_routed_experts, config.d_ff, config.d_model))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        nn.init.normal_(self.w3, std=0.02)
        self.shared_experts = nn.ModuleList([SwiGLU(config.d_model, config.d_shared_ff) for _ in range(config.n_shared_experts)])

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]

        logits = self.gate(x_flat)
        z_loss = self.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        router_probs = weights.mean(dim=0)
        expert_frac = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        expert_frac.scatter_add_(0, top_indices.view(-1), torch.ones_like(top_indices.view(-1), dtype=x.dtype))
        expert_frac = expert_frac / (n_tokens * self.top_k)
        balance_loss = self.balance_loss_coef * self.n_experts * (router_probs * expert_frac).sum()

        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        expert_mask = torch.zeros(n_tokens, self.n_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_add_(1, top_indices, top_weights)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            mask_i = expert_mask[:, i]
            if mask_i.sum() == 0:
                continue
            active = mask_i > 0
            x_active, w_active = x_flat[active], mask_i[active]
            h = F.silu(F.linear(x_active, self.w1[i])) * F.linear(x_active, self.w3[i])
            routed_out[active] += w_active.unsqueeze(-1) * F.linear(h, self.w2[i])

        output = (shared_out + routed_out).view(batch, seq_len, d_model)
        expert_counts = torch.bincount(top_indices.view(-1), minlength=self.n_experts).float()

        return output, z_loss + balance_loss, {"expert_counts": expert_counts}


class TransformerBlock(nn.Module):
    def __init__(self, config: TachyonConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model)
        self.moe = MoELayer(config)

    def forward(self, x, mask=None, kv_cache=None):
        x = x + self.attn(self.norm1(x), mask, kv_cache)
        moe_out, aux_loss, aux_info = self.moe(self.norm2(x))
        return x + moe_out, aux_loss, aux_info


class Tachyon(nn.Module):
    def __init__(self, config: TachyonConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_kv_caches(self):
        return [KVCache() for _ in range(len(self.layers))]

    def forward(self, input_ids, labels=None, mask=None, kv_caches=None):
        x = self.drop(self.embed(input_ids))
        total_aux_loss = 0.0
        all_expert_counts = []

        for i, layer in enumerate(self.layers):
            x, aux_loss, aux_info = layer(x, mask, kv_caches[i] if kv_caches else None)
            total_aux_loss += aux_loss
            all_expert_counts.append(aux_info["expert_counts"])

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            total_loss = torch.tensor(0.0, device=shift_logits.device, dtype=shift_logits.dtype)
            for i in range(0, shift_logits.shape[0], 4096):
                total_loss += F.cross_entropy(shift_logits[i:i+4096], shift_labels[i:i+4096], ignore_index=-100, reduction="sum")
            loss = total_loss / (shift_labels != -100).sum().clamp(min=1) + total_aux_loss

        return {"logits": logits, "loss": loss, "aux_loss": total_aux_loss,
                "expert_counts": torch.stack(all_expert_counts).sum(dim=0)}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        attn = router = shared = routed = 0
        for layer in self.layers:
            attn += sum(p.numel() for p in layer.attn.parameters()) + sum(p.numel() for p in layer.norm1.parameters()) + sum(p.numel() for p in layer.norm2.parameters())
            router += sum(p.numel() for p in layer.moe.gate.parameters())
            shared += sum(p.numel() for expert in layer.moe.shared_experts for p in expert.parameters())
            routed += layer.moe.w1.numel() + layer.moe.w2.numel() + layer.moe.w3.numel()
        active = int(embed + attn + router + shared + routed * self.config.top_k_experts / self.config.n_routed_experts + sum(p.numel() for p in self.norm.parameters()))
        return total, active


def get_model(size: str = "large"):
    configs = {"tiny": TachyonConfig.tiny, "small": TachyonConfig.small, "medium": TachyonConfig.medium, "large": TachyonConfig.large}
    return Tachyon(configs[size]())


if __name__ == "__main__":
    model = Tachyon(TachyonConfig.large())
    total, active = model.count_parameters()
    print(f"Tachyon Large: {total/1e9:.2f}B total, {active/1e6:.0f}M active")
    x = torch.randint(0, 50258, (2, 128))
    out = model(x, labels=x)
    print(f"Loss: {out['loss'].item():.4f}, Aux: {out['aux_loss'].item():.6f}")
