"""Mask-region self-distillation trainer (stages 1 and 2a).

Stage 1 (--teacher frozen): teacher = frozen copy of the original weights.
Stage 2a (--teacher live):   teacher = stop-grad forward of the live weights.
Optional --teacher ema:      teacher = EMA of the live weights (stability aid).

Loss (per DSpark recipe, adapted):
  L = w_tv * TV(p_t, p_s) + w_ce * CE(hard next token)   at mask slots, position-decayed
    + w_kl * KL(p_t || p_s)                              at mask slots (optional alt to TV)
    + w_anchor * KL(p_t || p_s)                          at prefix anchor slots (NTP preservation)
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
from torch.utils.data import DataLoader

from .data import MaskCollator, RegenDataset
from .qwen3 import Qwen3

MASK_ID = 151935  # last (untrained) row of the Qwen3 vocab


def gathered_logprobs(model, ids, positions, autocast=True):
    dev = ids.device.type
    with torch.autocast(dev, torch.bfloat16, enabled=autocast and dev == "cuda"):
        logits = model(ids, logit_positions=positions)
    return logits.float()


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
    # init mask embedding to mean of real embeddings (untrained row otherwise)
    with torch.no_grad():
        emb = student.embed_tokens.weight
        emb[MASK_ID] = emb[: args.vocab_used].mean(0)

    teacher = None
    if args.teacher == "frozen":
        if args.tiny:
            teacher = copy.deepcopy(student).to(torch.bfloat16 if device == "cuda" else torch.float32)
        else:
            teacher = Qwen3.from_pretrained(args.model_dir, dtype=torch.bfloat16, device=device)
        teacher.requires_grad_(False).eval()
    elif args.teacher == "ema":
        teacher = copy.deepcopy(student).to(torch.bfloat16)
        teacher.requires_grad_(False).eval()

    mask_vec = None
    if args.mask_emb_lr:
        emb_mod = student.embed_tokens
        mask_vec = torch.nn.Parameter(emb_mod.weight[MASK_ID].detach().clone().float())
        orig_emb_forward = emb_mod.forward

        def emb_forward(ids):
            x = orig_emb_forward(ids)
            return torch.where((ids == MASK_ID)[..., None], mask_vec.to(x.dtype), x)

        emb_mod.forward = emb_forward

    markov = None
    if args.markov_rank > 0:
        V = student.cfg.vocab_size
        markov = torch.nn.ModuleDict({
            "w1": torch.nn.Embedding(V, args.markov_rank),
            "w2": torch.nn.Linear(args.markov_rank, V, bias=False),
        }).to(device=device, dtype=torch.float32)
        torch.nn.init.normal_(markov["w1"].weight, std=1.0)
        torch.nn.init.zeros_(markov["w2"].weight)

    if args.grad_ckpt:
        from torch.utils.checkpoint import checkpoint

        for blk in student.layers:
            blk._orig_forward = blk.forward
            blk.forward = (lambda b: lambda *a, **k: checkpoint(
                b._orig_forward, *a, use_reentrant=False, **k))(blk)

    ds = RegenDataset(args.data, tok, max_len=args.max_len, max_samples=args.max_samples)
    n_val = min(512, len(ds) // 20)
    val_ds = torch.utils.data.Subset(ds, range(n_val))
    train_ds = torch.utils.data.Subset(ds, range(n_val, len(ds)))
    collate = MaskCollator(mask_id=MASK_ID, k_max=args.k_max, n_anchor=args.n_anchor)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=collate, num_workers=2, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        collate_fn=MaskCollator(mask_id=MASK_ID, k_max=args.k_max,
                                                n_anchor=args.n_anchor))

    # Qwen3 pretraining optimizer: AdamW b=(0.9,0.95) eps=1e-6 wd=0.1 clip=1.0
    groups = [{"params": list(student.parameters()), "lr": args.lr}]
    if markov is not None:
        groups.append({"params": list(markov.parameters()),
                       "lr": args.markov_lr or args.lr, "weight_decay": 0.0})
    if mask_vec is not None:
        groups.append({"params": [mask_vec], "lr": args.mask_emb_lr,
                       "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-6, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / args.warmup)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, st / args.steps)))))

    log_f = (out_dir / "log.jsonl").open("a")
    step, t0 = 0, time.time()
    data_iter = iter(dl)

    def teacher_forward(t_ids, pos):
        if args.teacher == "live":
            with torch.no_grad():
                return gathered_logprobs(student, t_ids, pos)
        with torch.no_grad():
            return gathered_logprobs(teacher, t_ids, pos)

    @torch.no_grad()
    def validate():
        stats = {"agree": [], "agree_pos": torch.zeros(args.k_max), "cnt_pos": torch.zeros(args.k_max), "tv": []}
        for vb in val_dl:
            vb = {k: v.to(device) for k, v in vb.items()}
            pos = torch.cat([vb["anchor_pos"], vb["mask_pos"]], 1)
            tl = teacher_forward(vb["teacher_ids"], pos)[:, args.n_anchor:]
            sl = gathered_logprobs(student, vb["student_ids"], pos)[:, args.n_anchor:]
            if markov is not None:
                prev = vb["teacher_ids"].gather(1, vb["mask_pos"])
                sl = sl + markov["w2"](markov["w1"](prev)).float()
            valid = vb["mask_w"] > 0
            ag = (tl.argmax(-1) == sl.argmax(-1)).float()
            stats["agree"].append(ag[valid].mean().item())
            for j in range(args.k_max):
                v = valid[:, j]
                stats["agree_pos"][j] += ag[:, j][v].sum().cpu()
                stats["cnt_pos"][j] += v.sum().cpu()
            pt, ps = tl.softmax(-1), sl.softmax(-1)
            tv = 0.5 * (pt - ps).abs().sum(-1)
            stats["tv"].append(tv[valid].mean().item())
        agree_pos = (stats["agree_pos"] / stats["cnt_pos"].clamp(min=1)).tolist()
        return {"val_agree": sum(stats["agree"]) / len(stats["agree"]),
                "val_tv": sum(stats["tv"]) / len(stats["tv"]),
                "val_agree_pos": [round(x, 4) for x in agree_pos]}

    while step < args.steps:
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            b = next(data_iter)
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        pos = torch.cat([b["anchor_pos"], b["mask_pos"]], 1)  # [B, A+K]

        t_logits = teacher_forward(b["teacher_ids"], pos)
        s_logits = gathered_logprobs(student, b["student_ids"], pos)
        A = args.n_anchor
        if markov is not None:
            prev = b["teacher_ids"].gather(1, b["mask_pos"])
            s_logits = torch.cat([s_logits[:, :A],
                                  s_logits[:, A:] + markov["w2"](markov["w1"](prev)).float()], 1)
        t_lp, s_lp = F.log_softmax(t_logits, -1), F.log_softmax(s_logits, -1)

        # draft losses at mask slots
        mw = b["mask_w"]
        pt, ps = t_lp[:, A:].exp(), s_lp[:, A:].exp()
        tv = 0.5 * (pt - ps).abs().sum(-1)
        kl_draft = (pt * (t_lp[:, A:] - s_lp[:, A:])).sum(-1)
        ce = F.cross_entropy(
            s_logits[:, A:].flatten(0, 1), b["hard_labels"].flatten(), reduction="none"
        ).view_as(mw)
        denom = mw.sum().clamp(min=1e-6)
        L_tv = (tv * mw).sum() / denom
        L_kl = (kl_draft * mw).sum() / denom
        L_ce = (ce * mw).sum() / denom

        # anchor KL (NTP preservation)
        aw = b["anchor_w"]
        kl_anchor = (t_lp[:, :A].exp() * (t_lp[:, :A] - s_lp[:, :A])).sum(-1)
        L_anchor = (kl_anchor * aw).sum() / aw.sum().clamp(min=1e-6)

        loss = args.w_tv * L_tv + args.w_ce * L_ce + args.w_kl * L_kl + args.w_anchor * L_anchor
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if args.teacher == "ema" and step % args.ema_every == 0:
            with torch.no_grad():
                for pt_, ps_ in zip(teacher.parameters(), student.parameters()):
                    pt_.lerp_(ps_.to(pt_.dtype), 1 - args.ema_beta)

        step += 1
        if step % args.log_every == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "tv": round(L_tv.item(), 4),
                   "kl": round(L_kl.item(), 4), "ce": round(L_ce.item(), 4),
                   "anchor": round(L_anchor.item(), 5), "gnorm": round(gnorm.item(), 3),
                   "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if step % args.eval_every == 0 or step == args.steps:
            v = validate()
            v["step"] = step
            v["min"] = round((time.time() - t0) / 60, 1)
            print("VAL " + json.dumps(v), flush=True)
            log_f.write(json.dumps(v) + "\n")
            log_f.flush()
        if step % args.save_every == 0 or step == args.steps:
            if mask_vec is not None:
                with torch.no_grad():
                    student.embed_tokens.weight[MASK_ID] = mask_vec
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
    ap.add_argument("--out-dir", default="results/stage1")
    ap.add_argument("--teacher", choices=["frozen", "live", "ema"], default="frozen")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-anchor", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--w-tv", type=float, default=0.9)
    ap.add_argument("--w-ce", type=float, default=0.1)
    ap.add_argument("--w-kl", type=float, default=0.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--ema-beta", type=float, default=0.999)
    ap.add_argument("--ema-every", type=int, default=1)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--markov-rank", type=int, default=0)
    ap.add_argument("--markov-lr", type=float, default=None)
    ap.add_argument("--mask-emb-lr", type=float, default=None)
    ap.add_argument("--vocab-used", type=int, default=151669)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="random tiny model, CPU smoke test")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
