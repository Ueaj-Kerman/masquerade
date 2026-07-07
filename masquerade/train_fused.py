"""Fused multi-region trainer (stages 2a/2b/3/5).

One forward per step over doc-packed sequences with inserted mask groups:
  - real slots = live teacher stream (stop-grad targets), also NTP loss site
  - mask slots = student draft stream, distilled from paired real slots
--teacher live   : self-distillation, no reference model (stage 2a semantics)
--teacher frozen : extra frozen forward over the same layout for targets
--w-ntp > 0      : adds hard NTP CE at real slots (stage 5 combined objective)
--attn flex|dense: FlexAttention BlockMask or SDPA dense bool mask
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import RegenDataset
from .multiregion import MultiRegionPacker, dense_mask, make_mask_mod
from .qwen3 import Qwen3
from .train import MASK_ID


def batch_iterator(ds, packer, B, shuffle_seed=0, epochs=1000):
    import random as _r
    order = list(range(len(ds)))
    rng = _r.Random(shuffle_seed)
    for _ in range(epochs):
        rng.shuffle(order)
        buf = []
        for i in order:
            buf.append(ds[i])
            if len(buf) >= B * 8:  # pack in chunks; ~8 docs/seq typical
                seqs, _ = packer.pack(buf)
                for j in range(0, len(seqs) - B + 1, B):
                    yield packer.build(seqs[j:j + B])
                buf = []


def build_attn(batch, kind: str, device: str):
    if kind == "dense":
        return {"attn_mask": dense_mask(batch, device), "is_causal": False}
    from torch.nn.attention.flex_attention import create_block_mask

    B, T = batch.ids.shape
    mm = make_mask_mod(batch, device)
    bm = create_block_mask(mm, B, 1, T, T, device=device)
    return {"block_mask": bm}


def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if args.tiny:
        from .qwen3 import Qwen3Config
        cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        student = Qwen3(cfg).to(device=device, dtype=torch.float32)
    else:
        student = Qwen3.from_pretrained(args.model_dir, dtype=torch.float32, device=device)
    with torch.no_grad():
        student.embed_tokens.weight[MASK_ID] = \
            student.embed_tokens.weight[:args.vocab_used].mean(0)

    teacher = None
    if args.teacher == "frozen":
        teacher = copy.deepcopy(student)
        if device == "cuda":
            teacher = teacher.to(torch.bfloat16)
        teacher.requires_grad_(False).eval()

    markov = None
    if args.markov_rank > 0:
        # DSpark-style low-rank sequential head: bias(prev_token) added to mask
        # logits; W2 zero-init so training starts from the parallel drafter.
        V = student.cfg.vocab_size
        markov = torch.nn.ModuleDict({
            "w1": torch.nn.Embedding(V, args.markov_rank),
            "w2": torch.nn.Linear(args.markov_rank, V, bias=False),
        }).to(device=device, dtype=torch.float32)
        torch.nn.init.normal_(markov["w1"].weight, std=0.02)
        torch.nn.init.zeros_(markov["w2"].weight)

    if args.grad_ckpt and not args.tiny:
        from torch.utils.checkpoint import checkpoint
        for blk in student.layers:
            blk._orig_forward = blk.forward
            blk.forward = (lambda bk: lambda *a, **kw: checkpoint(
                bk._orig_forward, *a, use_reentrant=False, **kw))(blk)

    ds = RegenDataset(args.data, tok, max_len=args.T - 64, max_samples=args.max_samples)
    n_val = min(256, len(ds) // 20)
    val_docs = [ds[i] for i in range(n_val)]
    train_ds = torch.utils.data.Subset(ds, range(n_val, len(ds)))
    packer = MultiRegionPacker(MASK_ID, T=args.T, k_max=args.k_max,
                               region_every=args.region_every, max_pairs=args.max_pairs,
                               seed=args.seed)
    it = batch_iterator(train_ds, packer, args.batch_size, shuffle_seed=args.seed)
    val_packer = MultiRegionPacker(MASK_ID, T=args.T, k_max=args.k_max,
                                   region_every=args.region_every, max_pairs=args.max_pairs,
                                   seed=123)
    vseqs, _ = val_packer.pack(val_docs)
    val_batches = [val_packer.build(vseqs[j:j + args.batch_size])
                   for j in range(0, max(len(vseqs) - args.batch_size + 1, 1), args.batch_size)][:4]

    params = list(student.parameters())
    if markov is not None:
        params += list(markov.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-6, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / args.warmup)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, st / args.steps)))))

    log_f = (out_dir / "log.jsonl").open("a")

    def fwd(model, batch, attn, need_grad):
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx, torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            h = model(batch.ids.to(device), positions=batch.pos.to(device),
                      return_hidden=True, **attn)
        return h

    def losses(batch, h_student, h_teacher):
        E = h_student.shape[-1]
        si = batch.student_idx.to(device)[..., None].expand(-1, -1, E)
        ti = batch.teacher_idx.to(device)[..., None].expand(-1, -1, E)
        hs = h_student.gather(1, si)
        ht = (h_teacher if h_teacher is not None else h_student).gather(1, ti).detach()
        with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            sl = student.lm_head(hs).float()
            # teacher logits FULLY stop-grad: detached hidden is not enough —
            # lm_head is tied to embeddings, so grads would leak into the
            # teacher stream and let the model self-simplify.
            with torch.no_grad():
                tm = teacher if args.teacher == "frozen" else student
                tl = tm.lm_head(ht).float()
        if markov is not None:
            prev = batch.ids.to(device).gather(1, batch.teacher_idx.to(device))
            sl = sl + markov["w2"](markov["w1"](prev)).float()
        s_lp = F.log_softmax(sl, -1)
        t_lp = F.log_softmax(tl, -1)
        pt, ps = t_lp.exp(), s_lp.exp()
        pw = batch.pair_w.to(device)
        denom = pw.sum().clamp(min=1e-6)
        tv = (0.5 * (pt - ps).abs().sum(-1) * pw).sum() / denom
        kl = ((pt * (t_lp - s_lp)).sum(-1) * pw).sum() / denom
        ce = (F.cross_entropy(sl.flatten(0, 1), batch.hard_labels.to(device).flatten(),
                              reduction="none").view_as(pw) * pw).sum() / denom
        agree = ((sl.argmax(-1) == tl.argmax(-1)).float() * (pw > 0)).sum() / (pw > 0).float().sum().clamp(min=1)

        ntp = torch.tensor(0.0, device=device)
        if args.w_ntp > 0:
            lab = batch.ntp_labels.to(device)
            valid = (lab != -100).nonzero()
            if valid.shape[0] > 0:
                sel = valid[torch.randperm(valid.shape[0], device=device)[:args.n_ntp * lab.shape[0]]]
                hsel = h_student[sel[:, 0], sel[:, 1]]
                with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
                    nl = student.lm_head(hsel).float()
                ntp = F.cross_entropy(nl, lab[sel[:, 0], sel[:, 1]])
        return tv, kl, ce, ntp, agree

    step, t0 = 0, time.time()
    while step < args.steps:
        batch = next(it)
        attn = build_attn(batch, args.attn, device)
        h_s = fwd(student, batch, attn, need_grad=True)
        h_t = fwd(teacher, batch, attn, need_grad=False) if args.teacher == "frozen" else None
        tv, kl, ce, ntp, agree = losses(batch, h_s, h_t)
        loss = args.w_tv * tv + args.w_kl * kl + args.w_ce * ce + args.w_ntp * ntp
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "tv": round(tv.item(), 4),
                   "ce": round(ce.item(), 4), "ntp": round(ntp.item(), 4),
                   "agree": round(agree.item(), 4), "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                vals = []
                for vb in val_batches:
                    va = build_attn(vb, args.attn, device)
                    h = fwd(student, vb, va, need_grad=False)
                    ht = fwd(teacher, vb, va, need_grad=False) if args.teacher == "frozen" else None
                    vals.append(losses(vb, h, ht))
                v = {"step": step, "val_tv": round(sum(x[0].item() for x in vals) / len(vals), 4),
                     "val_agree": round(sum(x[4].item() for x in vals) / len(vals), 4),
                     "min": round((time.time() - t0) / 60, 1)}
            print("VAL " + json.dumps(v), flush=True)
            log_f.write(json.dumps(v) + "\n"); log_f.flush()
        if step % args.save_every == 0 or step == args.steps:
            sd = {k: v.to(torch.bfloat16) for k, v in student.state_dict().items()
                  if not k.startswith("rope_")}
            if markov is not None:
                sd.update({f"markov.{k}": v.to(torch.bfloat16)
                           for k, v in markov.state_dict().items()})
            torch.save(sd, out_dir / f"ckpt_{step:06d}.pt")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--data", default="data/regen_qwen3_0.6b.jsonl")
    ap.add_argument("--out-dir", default="results/fused")
    ap.add_argument("--teacher", choices=["live", "frozen"], default="live")
    ap.add_argument("--attn", choices=["flex", "dense"], default="flex")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--region-every", type=int, default=48)
    ap.add_argument("--max-pairs", type=int, default=256)
    ap.add_argument("--n-ntp", type=int, default=64, help="NTP positions per row")
    ap.add_argument("--markov-rank", type=int, default=0, help="sequential head rank (0=off)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--w-tv", type=float, default=0.9)
    ap.add_argument("--w-ce", type=float, default=0.1)
    ap.add_argument("--w-kl", type=float, default=0.0)
    ap.add_argument("--w-ntp", type=float, default=0.0)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--vocab-used", type=int, default=151669)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
