"""Stage 5: pretraining from scratch — NTP vs NTP+mask-distill (fused, multi-region).

Model: Qwen3 arch at small scales, GPT-2 vocab (50304), tied embeddings.
Data: FineWeb-Edu memmap shards (scripts/tokenize_fineweb.py), doc-causal masking.
Optimizers: adamw (Qwen-style) | aurora (tilde-research, matrices) + AdamW rest.
Schedule: WSD (constant then linear cooldown over last 20%).

The mask objective inserts mask groups before sampled regions (multiregion.py
geometry, resp_start=1) and distills mask slots from the live real-slot logits
(stop-grad). NTP CE applies to all real slots in both arms.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/aurora-release/src"))

from masquerade.multiregion import MultiRegionPacker, dense_mask
from masquerade.qwen3 import Qwen3, Qwen3Config

EOT = 50256
MASK_ID = 50303

PRESETS = {
    "30m": dict(hidden_size=384, intermediate_size=1536, num_hidden_layers=8,
                num_attention_heads=6, num_key_value_heads=3, head_dim=64),
    "50m": dict(hidden_size=512, intermediate_size=2048, num_hidden_layers=10,
                num_attention_heads=8, num_key_value_heads=4, head_dim=64),
    "124m": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                 num_attention_heads=12, num_key_value_heads=6, head_dim=64),
}


def _load_bin(f):
    """llm.c format: 256 int32 header (magic 20240520) then uint16 tokens;
    also accepts raw uint16 files."""
    head = np.fromfile(f, dtype=np.int32, count=2)
    if len(head) >= 1 and head[0] == 20240520:
        n = int(np.fromfile(f, dtype=np.int32, count=3)[2])
        return np.memmap(f, dtype=np.uint16, mode="r", offset=1024)[:n]
    return np.memmap(f, dtype=np.uint16, mode="r")


class Shards:
    def __init__(self, path):
        files = sorted(Path(path).glob("*train*.bin")) or sorted(Path(path).glob("shard_*.bin"))
        assert files, f"no shards in {path}"
        self.arrs = [_load_bin(f) for f in files]
        self.sizes = [len(a) for a in self.arrs]
        self.total = sum(self.sizes)
        vf = sorted(Path(path).glob("*val*.bin"))
        self.val = _load_bin(vf[0]) if vf else None

    def sample(self, rng, T):
        while True:
            i = rng.integers(len(self.arrs))
            hi = self.sizes[i] - T - 1
            if hi <= 0:
                continue
            off = rng.integers(hi)
            return np.asarray(self.arrs[i][off:off + T + 1], dtype=np.int64)

    def val_windows(self, T, n):
        a = self.val if self.val is not None else self.arrs[-1]
        return [np.asarray(a[j * (T + 1): (j + 1) * (T + 1)], dtype=np.int64)
                for j in range(n)]


def window_to_docs(win: np.ndarray):
    """Split a raw window at EOT into docs (each doc includes its EOT prefix)."""
    idx = np.flatnonzero(win == EOT)
    bounds = [0] + [int(i) for i in idx if i > 0] + [len(win)]
    docs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 8:
            docs.append({"ids": win[a:b].tolist(), "resp_start": 1})
    return docs


def build_batch(shards, rng, packer, B, T, use_mask):
    if not use_mask:
        wins = [shards.sample(rng, T) for _ in range(B)]
        x = torch.from_numpy(np.stack([w[:-1] for w in wins]))
        y = torch.from_numpy(np.stack([w[1:] for w in wins]))
        doc = (x == EOT).cumsum(1)
        return {"plain": True, "ids": x, "labels": y, "docsep": doc}
    seqs = []
    while len(seqs) < B:
        win = shards.sample(rng, T)  # raw longer than needed; packer budgets masks
        docs = window_to_docs(win[: T - T // 8])
        s, _ = packer.pack(docs)
        seqs.extend(s)
    return {"plain": False, "batch": packer.build(seqs[:B])}


def plain_attn_mask(x, doc, device):
    B, T = x.shape
    d = doc.to(device)
    q = torch.arange(T, device=device)
    m = (q[None, None, :, None] >= q[None, None, None, :]) & \
        (d[:, None, :, None] == d[:, None, None, :])
    return m


def chunked_ce(hidden, lm_head, labels, ignore=-100, chunk=8192):
    Hf = hidden.flatten(0, 1)
    Lf = labels.flatten()
    tot, cnt = hidden.new_zeros(()), 0
    for i in range(0, Hf.shape[0], chunk):
        h, l = Hf[i:i + chunk], Lf[i:i + chunk]
        with torch.autocast("cuda", torch.bfloat16):
            lg = lm_head(h)
        v = l != ignore
        if v.any():
            tot = tot + F.cross_entropy(lg.float()[v], l[v], reduction="sum")
            cnt += int(v.sum())
    return tot / max(cnt, 1), cnt


class AuroraWrap:
    """aurora for 2D hidden matrices; AdamW for embeddings/norms/head."""

    def __init__(self, model, lr=0.02, wd=0.025, mu=0.95, adamw_lr=4e-3):
        from aurora import aurora
        self._aurora = aurora
        self.mats, self.rest = [], []
        for n, p in model.named_parameters():
            (self.mats if (p.ndim == 2 and "embed" not in n and "lm_head" not in n)
             else self.rest).append(p)
        self.mom = [torch.zeros_like(p) for p in self.mats]
        self.lr, self.wd, self.mu = lr, wd, mu
        self.adamw = torch.optim.AdamW(self.rest, lr=adamw_lr, betas=(0.9, 0.95),
                                       eps=1e-8, weight_decay=0.0)
        self.base_lr, self.base_adamw_lr = lr, adamw_lr

    def set_scale(self, s):
        self.lr = self.base_lr * s
        for g in self.adamw.param_groups:
            g["lr"] = self.base_adamw_lr * s

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.mats, self.mom):
            if p.grad is not None:
                self._aurora(p, p.grad, m, eta=self.lr, weight_decay=self.wd, mu=self.mu)
        self.adamw.step()

    def zero_grad(self, set_to_none=True):
        for p in self.mats:
            p.grad = None
        self.adamw.zero_grad(set_to_none=set_to_none)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="30m", choices=list(PRESETS))
    ap.add_argument("--objective", default="ntp", choices=["ntp", "ntp+mask"])
    ap.add_argument("--optimizer", default="aurora", choices=["adamw", "aurora"])
    ap.add_argument("--data", default="data/fineweb")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--adamw-lr", type=float, default=4e-3)
    ap.add_argument("--w-tv", type=float, default=0.2)
    ap.add_argument("--w-ce-pair", type=float, default=0.05)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--region-every", type=int, default=64)
    ap.add_argument("--attn", default="dense", choices=["dense", "flex"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=100000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    logf = (out / "log.jsonl").open("a")

    cfg = Qwen3Config(vocab_size=50304, rope_theta=10000.0, tie_word_embeddings=True,
                      max_position_embeddings=args.T * 2, **PRESETS[args.preset])
    model = Qwen3(cfg).cuda().float()
    nparam = sum(p.numel() for p in model.parameters()) - cfg.vocab_size * cfg.hidden_size
    print(f"non-emb params: {nparam/1e6:.1f}M")
    if args.compile:
        model = torch.compile(model, dynamic=False)

    shards = Shards(args.data)
    packer = MultiRegionPacker(MASK_ID, T=args.T, k_max=args.k_max,
                               region_every=args.region_every, max_pairs=512, seed=args.seed)
    use_mask = args.objective == "ntp+mask"

    lr = args.lr or (0.02 if args.optimizer == "aurora" else 3e-3)
    if args.optimizer == "aurora":
        opt = AuroraWrap(model, lr=lr, adamw_lr=args.adamw_lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.1)

    def lr_scale(step):
        warm = 200
        cool_start = int(args.steps * 0.8)
        if step < warm:
            return (step + 1) / warm
        if step > cool_start:
            return max(0.0, (args.steps - step) / (args.steps - cool_start))
        return 1.0

    vwins = shards.val_windows(args.T, 16)

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot, cnt = 0.0, 0
        for j in range(0, len(vwins), 8):
            w = np.stack(vwins[j:j + 8])
            x = torch.from_numpy(w[:, :-1]).cuda()
            y = torch.from_numpy(w[:, 1:]).cuda()
            doc = (x == EOT).cumsum(1)
            am = plain_attn_mask(x, doc, "cuda")
            with torch.autocast("cuda", torch.bfloat16):
                h = model(x, attn_mask=am, is_causal=False, return_hidden=True)
            l, c = chunked_ce(h, model.lm_head, y)
            tot += float(l) * c; cnt += c
        model.train()
        return tot / cnt

    tokens_seen, t0 = 0, time.time()
    for step in range(1, args.steps + 1):
        s = lr_scale(step)
        if args.optimizer == "aurora":
            opt.set_scale(s)
        else:
            for g in opt.param_groups:
                g["lr"] = lr * s

        b = build_batch(shards, rng, packer, args.batch_size, args.T, use_mask)
        if b["plain"]:
            x, y = b["ids"].cuda(), b["labels"].cuda()
            am = plain_attn_mask(x, b["docsep"], "cuda")
            with torch.autocast("cuda", torch.bfloat16):
                h = model(x, attn_mask=am, is_causal=False, return_hidden=True)
            ntp, n_tok = chunked_ce(h, model.lm_head, y)
            loss = ntp
            tv = torch.tensor(0.0)
        else:
            pb = b["batch"]
            am = dense_mask(pb, "cuda") if args.attn == "dense" else None
            kw = {"attn_mask": am, "is_causal": False} if am is not None else {}
            if args.attn == "flex":
                from masquerade.multiregion import make_mask_mod
                from torch.nn.attention.flex_attention import create_block_mask
                kw = {"block_mask": create_block_mask(make_mask_mod(pb, "cuda"),
                                                      pb.ids.shape[0], 1, args.T, args.T,
                                                      device="cuda")}
            with torch.autocast("cuda", torch.bfloat16):
                h = model(pb.ids.cuda(), positions=pb.pos.cuda(), return_hidden=True, **kw)
            ntp, n_tok = chunked_ce(h, model.lm_head, pb.ntp_labels.cuda())
            E = h.shape[-1]
            hs = h.gather(1, pb.student_idx.cuda()[..., None].expand(-1, -1, E))
            ht = h.gather(1, pb.teacher_idx.cuda()[..., None].expand(-1, -1, E)).detach()
            with torch.autocast("cuda", torch.bfloat16):
                sl = model.lm_head(hs).float()
                tl = model.lm_head(ht).float()
            pw = pb.pair_w.cuda(); den = pw.sum().clamp(min=1e-6)
            tv = (0.5 * (tl.softmax(-1) - sl.softmax(-1)).abs().sum(-1) * pw).sum() / den
            pce = (F.cross_entropy(sl.flatten(0, 1), pb.hard_labels.cuda().flatten(),
                                   reduction="none").view_as(pw) * pw).sum() / den
            loss = ntp + args.w_tv * tv + args.w_ce_pair * pce
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            model.parameters() if not hasattr(opt, "mats") else
            [p for p in model.parameters()], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        tokens_seen += int(n_tok)

        if step % args.log_every == 0:
            rec = {"step": step, "loss": round(float(loss), 4), "ntp": round(float(ntp), 4),
                   "tv": round(float(tv), 4), "gnorm": round(float(gn), 3),
                   "tok": tokens_seen, "tok_s": round(tokens_seen / (time.time() - t0)),
                   "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            v = {"step": step, "val_loss": round(val_loss(), 4), "tok": tokens_seen,
                 "min": round((time.time() - t0) / 60, 1)}
            print("VAL " + json.dumps(v), flush=True)
            logf.write(json.dumps(v) + "\n"); logf.flush()
        if step % args.save_every == 0 or step == args.steps:
            torch.save(model.state_dict(), out / f"ckpt_{step:06d}.pt")


if __name__ == "__main__":
    main()
