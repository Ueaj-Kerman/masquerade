"""Evaluate acceptance + GSM8K across training checkpoints -> acceptance-vs-training-time curve."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.evals import bench_acceptance, gsm8k_accuracy
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def load_ckpt(model_dir, ckpt_path=None):
    m = Qwen3.from_pretrained(model_dir)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cuda", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("rope_") or k == "lm_head.weight" for k in missing), missing
        m.lm_head.weight = m.embed_tokens.weight
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gsm-n", type=int, default=128)
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--only-step", type=int, default=None)
    ap.add_argument("--no-gsm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    run = Path(args.run_dir)
    ckpts = sorted(run.glob("ckpt_*.pt"))
    if args.only_step is not None:
        ckpts = [c for c in ckpts if int(c.stem.split("_")[1]) == args.only_step]
    todo = ([(0, None)] if args.include_base else []) + \
           [(int(c.stem.split("_")[1]), c) for c in ckpts]

    out_path = Path(args.out or run / "acceptance_curve.jsonl")
    done_steps = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done_steps.add(json.loads(line)["step"])

    for step, ck in todo:
        if step in done_steps:
            print(f"skip step {step} (done)")
            continue
        m = load_ckpt(args.model_dir, ck)
        rec = {"step": step}
        rec["acceptance"] = bench_acceptance(m, tok, k=args.k, n_prompts=args.n_prompts,
                                             max_new=args.max_new)
        if not args.no_gsm:
            rec.update(gsm8k_accuracy(m, tok, n=args.gsm_n))
        print(json.dumps(rec, indent=None), flush=True)
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
