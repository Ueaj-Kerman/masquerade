"""Generate all figures from experiment outputs into results/figs/.

Inputs (whichever exist):
  results/stage1*/acceptance_curve.jsonl  -> acceptance/gsm8k vs training step
  results/pareto_local.jsonl              -> tok/s vs batch (ar vs spec k)
  results/dspark_repro_*.json             -> modal vllm throughput curves
  results/pretrain/*/log.jsonl            -> val loss curves ntp vs ntp+mask
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results/figs"
FIGS.mkdir(parents=True, exist_ok=True)

# validated categorical palette (dataviz reference, light mode, fixed order)
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
       "#e87ba4", "#eb6834"]
GRAY = "#6b6a60"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CAT)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
C = {"ar": GRAY, "base": GRAY, "spec": CAT[0], "dspark": CAT[5],
     "dflash": CAT[4], "eagle3": CAT[1], "ntp": GRAY, "ntp+mask": CAT[0]}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def fig_acceptance_curves():
    for run in sorted(ROOT.glob("results/*/acceptance_curve.jsonl")):
        rows = load_jsonl(run)
        if not rows:
            continue
        rows.sort(key=lambda r: r["step"])
        name = run.parent.name
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sets = list(rows[0]["acceptance"].keys())
        for s in sets:
            steps = [r["step"] for r in rows]
            tau = [r["acceptance"][s]["committed_per_round"] for r in rows]
            axes[0].plot(steps, tau, marker="o", label=s)
        axes[0].set(xlabel="training step", ylabel="committed tokens / round (τ-like)",
                    title=f"{name}: acceptance vs training")
        axes[0].legend()
        last = rows[-1]
        for s in sets:
            axes[1].plot(range(1, len(last["acceptance"][s]["pos_cond_accept"]) + 1),
                         last["acceptance"][s]["pos_cond_accept"], marker="o", label=s)
        axes[1].set(xlabel="draft position", ylabel="conditional acceptance",
                    title=f"final ckpt (step {last['step']})", ylim=(0, 1))
        axes[1].legend()
        if "gsm8k_acc" in rows[0]:
            axes[2].plot([r["step"] for r in rows], [r["gsm8k_acc"] for r in rows],
                         marker="o", color="#0b6e99")
            axes[2].set(xlabel="training step", ylabel="GSM8K accuracy",
                        title="base capability")
        fig.tight_layout()
        fig.savefig(FIGS / f"acceptance_{name}.png", dpi=150)
        plt.close(fig)
        print("wrote", FIGS / f"acceptance_{name}.png")


def fig_pareto_local():
    p = ROOT / "results/pareto_tight.jsonl"
    if not p.exists():
        return
    rows = load_jsonl(p)
    tags = sorted({r.get("ckpt") or "base" for r in rows})
    for tag in tags:
        sel = [r for r in rows if (r.get("ckpt") or "base") == tag]
        fig, ax = plt.subplots(figsize=(7, 5))
        ar = sorted([r for r in sel if r["mode"] == "ar"], key=lambda r: r["B"])
        if ar:
            ax.plot([r["B"] for r in ar], [r["tok_s"] for r in ar], marker="s",
                    color=C["ar"], label="autoregressive")
        for k in sorted({r["k"] for r in sel if r["mode"] == "spec"}):
            sp = sorted([r for r in sel if r["mode"] == "spec" and r["k"] == k],
                        key=lambda r: r["B"])
            ax.plot([r["B"] for r in sp], [r["tok_s"] for r in sp], marker="o",
                    color=CAT[[4,8,12,2,6].index(k) % len(CAT)] if k in (2,4,6,8,12) else None,
                    label=f"masquerade k={k}", lw=2)
        ax.set(xlabel="batch size", ylabel="tokens/s (aggregate)", xscale="log",
               title=f"throughput vs batch — {Path(tag).stem if tag != 'base' else 'base model'}")
        ax.legend()
        fig.tight_layout()
        name = "base" if tag == "base" else Path(tag).stem
        fig.savefig(FIGS / f"pareto_{name}.png", dpi=150)
        plt.close(fig)
        print("wrote pareto", name)


def fig_dspark_repro():
    files = sorted(ROOT.glob("results/dspark_repro_*.json"))
    if not files:
        return
    data = {}
    for f in files:
        algo = f.stem.replace("dspark_repro_", "")
        data[algo] = json.loads(f.read_text())
    for pset in ("gsm8k", "chat"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for algo, rows in data.items():
            sel = sorted([r for r in rows if r["set"] == pset], key=lambda r: r["B"])
            if not sel:
                continue
            axes[0].plot([r["B"] for r in sel], [r["tok_s"] for r in sel], marker="o",
                         color=C.get(algo), label=algo)
            axes[1].plot([r["tok_s_per_seq"] for r in sel], [r["tok_s"] for r in sel],
                         marker="o", color=C.get(algo), label=algo)
        axes[0].set(xlabel="concurrency (batch)", ylabel="tokens/s", xscale="log",
                    title=f"Qwen3-4B on H100 (vLLM) — {pset}")
        axes[1].set(xlabel="tokens/s per request", ylabel="aggregate tokens/s",
                    title="throughput vs per-user speed (DSpark Fig.7 style)")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / f"dspark_repro_{pset}.png", dpi=150)
        plt.close(fig)
        print("wrote dspark repro", pset)


def fig_pretrain():
    runs = sorted(ROOT.glob("results/pretrain/*/log.jsonl"))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ci = 0
    for run in runs:
        rows = [r for r in load_jsonl(run) if "val_loss" in r]
        if not rows:
            continue
        name = run.parent.name
        if "mask" in name:
            color = CAT[ci % len(CAT)]
            ci += 1
            style = dict(color=color)
        else:
            style = dict(color=GRAY, ls="--")
        ax.plot([r["tok"] / 1e6 for r in rows], [r["val_loss"] for r in rows],
                marker="o", ms=3, label=name, alpha=0.9, **style)
    ax.set(xlabel="training tokens (M)", ylabel="val loss (fineweb)", xscale="log",
           title="pretraining: NTP (dashed gray) vs NTP+mask variants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "pretrain_valloss.png", dpi=150)
    plt.close(fig)
    print("wrote pretrain fig")


def fig_4b_vs_dspark():
    """Relative speedup vs own-AR baseline per batch: DSpark (vLLM/H100, temp 1.0)
    vs masquerade-4B (our engine/RTX 5090, greedy), GSM8K prompts."""
    import json as _j
    try:
        base = {r["B"]: r["tok_s"] for r in
                _j.load(open(ROOT / "results/dspark_repro_base.json")) if r["set"] == "gsm8k"}
        dsp = {r["B"]: r["tok_s"] for r in
               _j.load(open(ROOT / "results/dspark_repro_dspark.json")) if r["set"] == "gsm8k"}
    except FileNotFoundError:
        return
    p = ROOT / "results/pareto_4b_hot.jsonl"
    if not p.exists():
        return
    rows = [r for r in load_jsonl(p) if r.get("ckpt")]
    ar = {r["B"]: r["tok_s"] for r in rows if r["mode"] == "ar"}
    ours = {}
    for r in rows:
        if r["mode"] == "spec" and r["B"] in ar:
            ours.setdefault(r["k"], {})[r["B"]] = r["tok_s"] / ar[r["B"]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    Bs = sorted(set(base) & set(dsp))
    ax.plot(Bs, [dsp[b] / base[b] for b in Bs], marker="s", color=C["dspark"],
            label="DSpark (vLLM, H100, temp 1.0)", lw=2)
    for k in sorted(ours):
        d = ours[k]
        Bs2 = sorted(d)
        ax.plot(Bs2, [d[b] for b in Bs2], marker="o",
                color=CAT[0] if k == 8 else CAT[1],
                label=f"masquerade k={k} (our engine, RTX 5090, greedy)", lw=2)
    ax.axhline(1.0, color=GRAY, lw=1, ls=":")
    ax.set(xlabel="batch size (concurrency)", ylabel="speedup vs own AR baseline",
           xscale="log", title="Qwen3-4B GSM8K: lossless speedup vs batch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "speedup_4b_vs_dspark.png", dpi=150)
    plt.close(fig)
    print("wrote speedup_4b_vs_dspark")


def fig_train_agree(runs: dict | None = None):
    """val_agree vs step across training runs (stage1 local + modal logs)."""
    if runs is None:
        runs = {}
        p = ROOT / "results/stage1_frozen/log.jsonl"
        if p.exists():
            runs["stage1 frozen (single-region)"] = p
        for d in sorted(ROOT.glob("results/modal/*/log.jsonl")):
            runs[d.parent.name] = d
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, p in runs.items():
        rows = [r for r in load_jsonl(p) if "val_agree" in r]
        if not rows:
            continue
        ax.plot([r["step"] for r in rows], [r["val_agree"] for r in rows],
                marker="o", ms=3, label=name)
    ax.set(xlabel="training step", ylabel="val mask-slot argmax agreement",
           title="draft agreement vs training", ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "train_agree.png", dpi=150)
    plt.close(fig)
    print("wrote train_agree")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "acc"):
        fig_acceptance_curves()
    if which in ("all", "pareto"):
        fig_pareto_local()
    if which in ("all", "dspark"):
        fig_dspark_repro()
    if which in ("all", "pretrain"):
        fig_pretrain()
    if which in ("all", "train"):
        fig_train_agree()
    if which in ("all", "4b"):
        fig_4b_vs_dspark()
