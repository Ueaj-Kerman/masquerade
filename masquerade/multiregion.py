"""Multi-region fused mask training (stage 3): doc packing + inserted mask groups.

Flattened layout per packed sequence:
  [d0: x0..x_{s-1}, M_1..M_k, x_s .. ][d1: ...] ...
Masks for region r=(s,k) are inserted immediately before x_s and carry rope
positions s..s+k-1 (shadowing the real region tokens that follow them).

Attention (mask_mod):
  real  q -> real kv, same doc, flat kv_idx <= flat q_idx   (pure NTP stream)
  mask  q -> real kv, same doc, flat kv_idx <  flat q_idx   (prior context;
             reals of its own region sit AFTER the group so are excluded)
        -> mask kv, same doc, same region, kv_idx <= q_idx
Real tokens never see masks => real-slot logits equal a plain causal forward
(the live teacher). Distill pairs: student mask slot (rope pos p) <- teacher
real slot x_p (both predict x_{p+1}).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

REAL = -1


def sample_regions(rng, resp_start: int, doc_len: int, k_max: int, region_every: int):
    """Non-overlapping regions inside [resp_start, doc_len-1), >=1 gap between."""
    regions = []
    p = resp_start + rng.randint(0, max(region_every // 2, 1))
    while p + 2 < doc_len - 1:
        k = rng.randint(1, k_max)
        k = min(k, doc_len - 1 - p)
        if k < 1:
            break
        regions.append((p, k))
        p += k + 1 + rng.randint(1, region_every)
    return regions


@dataclass
class PackedBatch:
    ids: torch.Tensor           # [B,T] token ids (with mask_id at mask slots)
    pos: torch.Tensor           # [B,T] rope positions
    doc: torch.Tensor           # [B,T] doc id (-1 = padding)
    region: torch.Tensor        # [B,T] region id, REAL(-1) for real slots
    student_idx: torch.Tensor   # [B,P] flat idx of mask slots (0-padded)
    teacher_idx: torch.Tensor   # [B,P] flat idx of paired real slots
    pair_w: torch.Tensor        # [B,P] weights (0=pad), position-decayed
    hard_labels: torch.Tensor   # [B,P]
    ntp_labels: torch.Tensor    # [B,T] next real token at real slots, -100 else


class MultiRegionPacker:
    def __init__(self, mask_id: int, T: int = 2048, k_max: int = 8,
                 region_every: int = 48, decay_gamma: float | None = None,
                 max_pairs: int = 256, seed: int = 0):
        self.mask_id, self.T, self.k_max = mask_id, T, k_max
        self.region_every = region_every
        self.gamma = decay_gamma or float(k_max)
        self.max_pairs = max_pairs
        self.rng = random.Random(seed)

    def pack(self, docs: list[dict]) -> tuple[list[list], list[dict]]:
        """docs: {ids: [...], resp_start: int}. Greedy-pack flattened docs into T."""
        seqs, cur, cur_len, used = [], [], 0, 0
        for d in docs:
            regions = sample_regions(self.rng, d["resp_start"], len(d["ids"]),
                                     self.k_max, self.region_every)
            flat_len = len(d["ids"]) + sum(k for _, k in regions)
            if flat_len > self.T:
                continue
            if cur_len + flat_len > self.T:
                seqs.append(cur)
                cur, cur_len = [], 0
            cur.append((d, regions))
            cur_len += flat_len
            used += 1
        if cur:
            seqs.append(cur)
        return seqs, used

    def build(self, seqs: list[list]) -> PackedBatch:
        B, T, P = len(seqs), self.T, self.max_pairs
        ids = torch.zeros(B, T, dtype=torch.long)
        pos = torch.zeros(B, T, dtype=torch.long)
        doc = torch.full((B, T), -1, dtype=torch.long)
        region = torch.full((B, T), REAL, dtype=torch.long)
        s_idx = torch.zeros(B, P, dtype=torch.long)
        t_idx = torch.zeros(B, P, dtype=torch.long)
        pw = torch.zeros(B, P)
        hard = torch.zeros(B, P, dtype=torch.long)
        ntp = torch.full((B, T), -100, dtype=torch.long)

        for b, seq in enumerate(seqs):
            f = 0          # flat cursor
            did = 0
            rid = 0
            np_ = 0
            for d, regions in seq:
                toks = d["ids"]
                reg_i = 0
                real_flat = {}  # orig pos -> flat idx
                for p_orig in range(len(toks)):
                    if reg_i < len(regions) and p_orig == regions[reg_i][0]:
                        s, k = regions[reg_i]
                        for j in range(k):
                            ids[b, f] = self.mask_id
                            pos[b, f] = s + j
                            doc[b, f] = did
                            region[b, f] = rid
                            f += 1
                        reg_i += 1
                        rid += 1
                    ids[b, f] = toks[p_orig]
                    pos[b, f] = p_orig
                    doc[b, f] = did
                    real_flat[p_orig] = f
                    if p_orig + 1 < len(toks):
                        ntp[b, f] = toks[p_orig + 1]
                    f += 1
                # distill pairs
                for (s, k) in regions:
                    mask_start = real_flat[s] - k
                    for j in range(k):
                        if np_ >= P:
                            break
                        s_idx[b, np_] = mask_start + j
                        t_idx[b, np_] = real_flat[s + j]
                        pw[b, np_] = math.exp(-j / self.gamma)
                        hard[b, np_] = toks[s + j + 1]
                        np_ += 1
                did += 1
            assert f <= T
        return PackedBatch(ids, pos, doc, region, s_idx, t_idx, pw, hard, ntp)


def make_mask_mod(batch: PackedBatch, device="cuda"):
    """mask_mod closure over the batch metadata tensors (for flex or dense ref)."""
    doc = batch.doc.to(device)
    region = batch.region.to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = (doc[b, q_idx] == doc[b, kv_idx]) & (doc[b, q_idx] >= 0)
        q_real = region[b, q_idx] == REAL
        kv_real = region[b, kv_idx] == REAL
        causal = kv_idx <= q_idx
        rr = q_real & kv_real & causal
        mr = (~q_real) & kv_real & (kv_idx < q_idx)
        mm = (~q_real) & (~kv_real) & (region[b, q_idx] == region[b, kv_idx]) & causal
        return same_doc & (rr | mr | mm)

    return mask_mod


def dense_mask(batch: PackedBatch, device="cuda") -> torch.Tensor:
    """[B,1,T,T] bool reference mask (for SDPA and tests)."""
    B, T = batch.ids.shape
    mm = make_mask_mod(batch, device)
    q = torch.arange(T, device=device)
    out = torch.zeros(B, 1, T, T, dtype=torch.bool, device=device)
    for b in range(B):
        out[b, 0] = mm(b, 0, q[:, None], q[None, :])
    return out
