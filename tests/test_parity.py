"""Logit parity: our Qwen3 vs HF transformers, and KV-cache path vs full forward."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.qwen3 import KVCache, Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    torch.manual_seed(0)
    ids = torch.randint(10, 150000, (2, 64), device="cuda")

    ours = Qwen3.from_pretrained(MODEL_DIR, dtype=torch.float32)
    with torch.no_grad():
        logits_ours = ours(ids).float()
    del ours
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa").cuda()
    with torch.no_grad():
        logits_hf = hf(ids).logits.float()

    diff = (logits_ours - logits_hf).abs()
    rel = diff.max() / logits_hf.abs().max()
    agree = (logits_ours.argmax(-1) == logits_hf.argmax(-1)).float().mean()
    print(f"max abs diff {diff.max().item():.4f}  rel {rel.item():.2e}  argmax agree {agree.item():.4f}")
    assert agree.item() == 1.0, "argmax disagreement vs HF (fp32)"
    del hf
    torch.cuda.empty_cache()

    ours = Qwen3.from_pretrained(MODEL_DIR, dtype=torch.float32)

    # KV-cache incremental path == full forward
    B, T = 2, 48
    ids = torch.randint(10, 150000, (B, T), device="cuda")
    cache = KVCache(B, 128, ours.cfg, "cuda", torch.float32)
    chunks = [ids[:, :20], ids[:, 20:21], ids[:, 21:40], ids[:, 40:]]
    outs, off = [], 0
    with torch.no_grad():
        full = ours(ids).float()
        for ch in chunks:
            q = ch.shape[1]
            pos = torch.arange(off, off + q, device="cuda").expand(B, q)
            # explicit mask: query i attends to slots <= its absolute position
            S = cache.max_len
            qpos = torch.arange(off, off + q, device="cuda")
            mask = (torch.arange(S, device="cuda")[None, :] <= qpos[:, None])[None, None]
            mask = mask.expand(B, 1, q, S)
            out = ours(ch, positions=pos, kv_cache=cache, cache_pos=pos, attn_mask=mask)
            outs.append(out.float())
            off += q
    inc = torch.cat(outs, dim=1)
    d = (inc - full).abs().max().item()
    agree2 = (inc.argmax(-1) == full.argmax(-1)).float().mean().item()
    print(f"kv-cache max diff {d:.6f}  argmax agree {agree2:.4f}")
    assert d < 1e-3 and agree2 == 1.0
    print("PARITY OK")


if __name__ == "__main__":
    main()
