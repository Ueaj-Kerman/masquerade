"""Data pipeline for mask-region distillation.

Each sample: chat-templated prompt + response. One masked region per sample
(stage 1/2a): pick region [s, s+k) inside the response, truncate at s+k,
student sees [MASK]*k at the region, teacher sees the original tokens.

Layout produced by the collator (all shape-static per batch):
  student_ids [B,T]  teacher_ids [B,T]  lengths [B]
  mask_pos [B,K]     mask_w [B,K]   position-decay weights, 0 on invalid slots
  hard_labels [B,K]  next real token for each mask slot
  anchor_pos [B,A]   anchor_w [B,A] prefix positions for NTP-preservation KL
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class RegenDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 1024, max_samples: int | None = None):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = []
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))
                if max_samples and len(self.rows) >= max_samples:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt_txt = self.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = self.tok(prompt_txt, add_special_tokens=False)["input_ids"]
        resp_ids = self.tok(r["response"], add_special_tokens=False)["input_ids"]
        resp_ids = resp_ids + [self.tok.eos_token_id]
        ids = (prompt_ids + resp_ids)[: self.max_len]
        resp_start = min(len(prompt_ids), len(ids))
        return {"ids": ids, "resp_start": resp_start}


@dataclass
class MaskCollator:
    mask_id: int
    k_max: int = 8
    n_anchor: int = 16
    decay_gamma: float | None = None  # None -> k_max
    pad_to_multiple: int = 64
    rng: random.Random = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random(0)
        if self.decay_gamma is None:
            self.decay_gamma = float(self.k_max)

    def __call__(self, batch):
        K, A = self.k_max, self.n_anchor
        rows = []
        for ex in batch:
            ids, rs = ex["ids"], ex["resp_start"]
            L = len(ids)
            # need s >= max(rs, 1); s + k <= L - 1 (hard label for last mask)
            if L - rs < 3 or rs < 1:
                continue
            k = self.rng.randint(1, K)
            k = min(k, L - 1 - rs)
            s = self.rng.randint(max(rs, 1), L - 1 - k)
            rows.append((ids, s, k))
        assert rows, "empty batch after filtering"

        T = max(s + k for _, s, k in rows)
        T = math.ceil(T / self.pad_to_multiple) * self.pad_to_multiple
        B = len(rows)
        student = torch.zeros(B, T, dtype=torch.long)
        teacher = torch.zeros(B, T, dtype=torch.long)
        mask_pos = torch.zeros(B, K, dtype=torch.long)
        mask_w = torch.zeros(B, K)
        hard = torch.zeros(B, K, dtype=torch.long)
        anchor_pos = torch.zeros(B, A, dtype=torch.long)
        anchor_w = torch.zeros(B, A)
        lengths = torch.zeros(B, dtype=torch.long)

        for b, (ids, s, k) in enumerate(rows):
            trunc = torch.tensor(ids[: s + k], dtype=torch.long)
            teacher[b, : s + k] = trunc
            student[b, : s + k] = trunc
            student[b, s : s + k] = self.mask_id
            lengths[b] = s + k
            for j in range(k):
                mask_pos[b, j] = s + j
                mask_w[b, j] = math.exp(-j / self.decay_gamma)
                hard[b, j] = ids[s + j + 1]
            # anchors: position s-1 (predicts first region token) + random prefix
            cand = [s - 1] + self.rng.sample(range(s - 1), min(A - 1, s - 1))
            for a, p in enumerate(cand[:A]):
                anchor_pos[b, a] = p
                anchor_w[b, a] = 1.0

        # normalize weights to mean 1 over valid slots
        mask_w = mask_w * (mask_w.numel() / max(mask_w.sum().item(), 1e-6)) * (mask_w > 0).float().mean()
        return {
            "student_ids": student, "teacher_ids": teacher, "lengths": lengths,
            "mask_pos": mask_pos, "mask_w": mask_w, "hard_labels": hard,
            "anchor_pos": anchor_pos, "anchor_w": anchor_w,
        }
