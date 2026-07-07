"""Property tests for the multi-region fused layout (CPU, tiny random model).

1. Teacher purity: logits at real slots in the fused masked forward == logits of
   a plain causal forward over the real tokens alone (per doc). Masks invisible.
2. Mask context: each mask slot's logits equal a stage-1-style forward where the
   doc is truncated at its region start and its own group appended causally.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.multiregion import MultiRegionPacker, dense_mask
from masquerade.qwen3 import Qwen3, Qwen3Config

MASK_ID = 151935


def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    return Qwen3(cfg).float()


def main():
    m = tiny().eval()
    torch.manual_seed(1)
    docs = [
        {"ids": torch.randint(10, 1000, (57,)).tolist(), "resp_start": 9},
        {"ids": torch.randint(10, 1000, (43,)).tolist(), "resp_start": 5},
        {"ids": torch.randint(10, 1000, (61,)).tolist(), "resp_start": 12},
    ]
    packer = MultiRegionPacker(MASK_ID, T=256, k_max=4, region_every=10, seed=3)
    seqs, used = packer.pack(docs)
    assert used == 3
    batch = packer.build(seqs)
    B, T = batch.ids.shape
    n_regions = int((batch.region.max() + 1).item())
    n_pairs = int((batch.pair_w > 0).sum().item())
    print(f"packed {used} docs into {B} seq(s) of T={T}, {n_regions} regions, {n_pairs} pairs")
    assert n_regions >= 6, "want many regions for a meaningful test"

    amask = dense_mask(batch, device="cpu")
    with torch.no_grad():
        fused = m(batch.ids, positions=batch.pos, attn_mask=amask, is_causal=False)

    # --- 1. teacher purity per doc ---
    for b in range(B):
        for did in range(int(batch.doc[b].max().item()) + 1):
            sel = ((batch.doc[b] == did) & (batch.region[b] == -1)).nonzero().squeeze(-1)
            toks = batch.ids[b, sel]
            with torch.no_grad():
                plain = m(toks[None])
            d = (fused[b, sel] - plain[0]).abs().max().item()
            assert d < 2e-4, f"teacher polluted: doc {did} diff {d}"
    print("teacher purity OK")

    # --- 2. mask-slot equivalence to stage-1 layout ---
    checked = 0
    for b in range(B):
        pairs = [(int(batch.student_idx[b, i]), int(batch.teacher_idx[b, i]))
                 for i in range(batch.pair_w.shape[1]) if batch.pair_w[b, i] > 0]
        # group masks by region
        for rid in range(n_regions):
            sel = (batch.region[b] == rid).nonzero().squeeze(-1)
            if sel.numel() == 0:
                continue
            did = int(batch.doc[b, sel[0]].item())
            s = int(batch.pos[b, sel[0]].item())          # region start (orig pos)
            k = sel.numel()
            docsel = ((batch.doc[b] == did) & (batch.region[b] == -1)).nonzero().squeeze(-1)
            toks = batch.ids[b, docsel]
            ref_in = torch.cat([toks[:s], torch.full((k,), MASK_ID)])[None]
            with torch.no_grad():
                ref = m(ref_in)
            d = (fused[b, sel] - ref[0, s:]).abs().max().item()
            assert d < 2e-4, f"mask context mismatch region {rid}: {d}"
            checked += 1
    print(f"mask-slot equivalence OK ({checked} regions)")

    # --- 3. pair indices point at matching rope positions ---
    for b in range(B):
        for i in range(batch.pair_w.shape[1]):
            if batch.pair_w[b, i] > 0:
                si, ti = int(batch.student_idx[b, i]), int(batch.teacher_idx[b, i])
                assert batch.pos[b, si] == batch.pos[b, ti]
                assert batch.region[b, si] >= 0 and batch.region[b, ti] == -1
    print("pair mapping OK")


if __name__ == "__main__":
    main()
