"""Self-contained Qwen3 dense model.

Loads HF Qwen3 checkpoints (0.6B/1.7B/4B/8B). Three attention paths:
  - SDPA causal (default, training without packing)
  - SDPA with explicit bool mask (batched inference over a static KV cache)
  - FlexAttention with an externally-built BlockMask (packing / multi-region masks)

Design constraints: everything shape-static per bucket so torch.compile
(fullgraph) and CUDA graphs work on the decode path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

_flex_attention_c = torch.compile(flex_attention, dynamic=False)


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True

    @staticmethod
    def from_hf(model_dir: str | Path) -> "Qwen3Config":
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        assert cfg["model_type"] == "qwen3", cfg["model_type"]
        return Qwen3Config(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg["tie_word_embeddings"],
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rope_cache(cfg: Qwen3Config, device, dtype=torch.float32):
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim)
    )
    t = torch.arange(cfg.max_position_embeddings, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, D/2]
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [B, T, D/2] or [T, D/2] (HF "rotate_half" convention)
    if cos.dim() == 2:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    else:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    xf1, xf2 = x1.float(), x2.float()
    out = torch.cat(
        [xf1 * cos - xf2 * sin, xf2 * cos + xf1 * sin], dim=-1
    )
    return out.to(x.dtype)


class KVCache(nn.Module):
    """Static KV cache with per-row write offsets (for spec-decode rollback)."""

    def __init__(self, batch: int, max_len: int, cfg: Qwen3Config, device, dtype):
        super().__init__()
        shape = (cfg.num_hidden_layers, batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.register_buffer("k", torch.zeros(shape, device=device, dtype=dtype))
        self.register_buffer("v", torch.zeros(shape, device=device, dtype=dtype))
        self.max_len = max_len

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor):
        # k,v: [B, Hkv, q, D]; pos: [B, q] absolute slot indices
        B, H, q, D = k.shape
        idx = pos[:, None, :, None].expand(B, H, q, D)
        self.k[layer].scatter_(2, idx, k)
        self.v[layer].scatter_(2, idx, v)
        return self.k[layer], self.v[layer]


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        H, Hkv, D, E = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
        self.q_proj = nn.Linear(E, H * D, bias=False)
        self.k_proj = nn.Linear(E, Hkv * D, bias=False)
        self.v_proj = nn.Linear(E, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, E, bias=False)
        self.q_norm = RMSNorm(D, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(D, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, *, layer_idx=None, kv_cache: KVCache | None = None,
                cache_pos=None, attn_mask=None, block_mask=None, is_causal=False):
        B, T, E = x.shape
        cfg = self.cfg
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = self.q_norm(self.q_proj(x).view(B, T, H, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, Hkv, D)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.write(layer_idx, k, v, cache_pos)

        if block_mask is not None:
            g = H // k.shape[1]
            y = _flex_attention_c(q, k, v, block_mask=block_mask, enable_gqa=(g > 1))
        else:
            g = H // k.shape[1]
            if g > 1:
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, **kw):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, **kw)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = _rope_cache(cfg, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Qwen3-style init (initializer_range=0.02); from_pretrained overwrites.
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=0.02)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T]
        *,
        positions: torch.Tensor | None = None,  # [B, T] or [T] rope positions
        kv_cache: KVCache | None = None,
        cache_pos: torch.Tensor | None = None,  # [B, T] cache slots to write
        attn_mask: torch.Tensor | None = None,   # bool [B, 1, T, S]
        block_mask=None,                          # flex BlockMask
        is_causal: bool | None = None,
        logit_positions: torch.Tensor | None = None,  # [B, P] gather before lm_head
        return_hidden: bool = False,
    ):
        B, T = input_ids.shape
        if positions is None:
            positions = torch.arange(T, device=input_ids.device)
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        if is_causal is None:
            is_causal = attn_mask is None and block_mask is None and kv_cache is None
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(
                x, cos, sin,
                layer_idx=i, kv_cache=kv_cache, cache_pos=cache_pos,
                attn_mask=attn_mask, block_mask=block_mask, is_causal=is_causal,
            )
        x = self.norm(x)
        if logit_positions is not None:
            x = x.gather(1, logit_positions[..., None].expand(-1, -1, x.shape[-1]))
        if return_hidden:
            return x
        return self.lm_head(x)

    @staticmethod
    def from_pretrained(model_dir: str | Path, device="cuda", dtype=torch.bfloat16) -> "Qwen3":
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        cfg = Qwen3Config.from_hf(model_dir)
        with torch.device("meta"):
            model = Qwen3(cfg)
        sd = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            sd.update(load_file(f))
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        if cfg.tie_word_embeddings:
            sd["lm_head.weight"] = sd["embed_tokens.weight"]
        model.load_state_dict(sd, strict=True, assign=True)
        if cfg.tie_word_embeddings:  # assign=True breaks tying; restore
            model.lm_head.weight = model.embed_tokens.weight
        model = model.to(device=device, dtype=dtype)
        cos, sin = _rope_cache(cfg, device=device)
        model.rope_cos, model.rope_sin = cos, sin  # keep rope in fp32
        return model
