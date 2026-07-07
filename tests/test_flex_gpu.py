"""GPU: FlexAttention BlockMask path == dense SDPA mask path on the multi-region
layout (bf16 tolerance), on the real 0.6B model. Guards against the known
flex batch-dependent-closure pitfalls before using flex in long runs."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.multiregion import MultiRegionPacker, dense_mask, make_mask_mod
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MASK_ID = 151935


def main():
    torch.manual_seed(0)
    m = Qwen3.from_pretrained(MODEL_DIR)  # bf16
    docs = []
    g = torch.Generator().manual_seed(7)
    for _ in range(12):
        L = int(torch.randint(80, 400, (1,), generator=g))
        docs.append({"ids": torch.randint(10, 150000, (L,), generator=g).tolist(),
                     "resp_start": int(torch.randint(5, 40, (1,), generator=g))})
    packer = MultiRegionPacker(MASK_ID, T=1024, k_max=8, region_every=48, seed=1)
    seqs, used = packer.pack(docs)
    batch = packer.build(seqs[:2])
    print(f"batch {batch.ids.shape}, {int((batch.pair_w>0).sum())} pairs")

    ids = batch.ids.cuda()
    pos = batch.pos.cuda()

    with torch.no_grad():
        dm = dense_mask(batch, "cuda")
        out_dense = m(ids, positions=pos, attn_mask=dm, is_causal=False).float()

    from torch.nn.attention.flex_attention import create_block_mask

    B, T = ids.shape
    bm = create_block_mask(make_mask_mod(batch, "cuda"), B, 1, T, T, device="cuda")
    with torch.no_grad():
        out_flex = m(ids, positions=pos, block_mask=bm).float()

    d = (out_dense - out_flex).abs()
    denom = out_dense.abs().max()
    agree = (out_dense.argmax(-1) == out_flex.argmax(-1)).float().mean().item()
    nan = torch.isnan(out_flex).any().item()
    print(f"max abs {d.max().item():.4f} rel {(d.max()/denom).item():.2e} argmax agree {agree:.4f} nan={nan}")
    assert not nan, "flex produced NaN"
    assert agree > 0.985, "flex/dense disagree beyond bf16 noise"
    print("FLEX GPU OK")


if __name__ == "__main__":
    main()
