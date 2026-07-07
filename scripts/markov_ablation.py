"""Same weights, Markov head on vs off: isolates the sequential head's tau gain."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.evals import bench_acceptance, load_ckpt_into
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "results/final06b_lr1e-5.pt"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    m = Qwen3.from_pretrained(MODEL_DIR)
    markov = load_ckpt_into(m, ckpt)
    for name, mk in [("markov_on", markov), ("markov_off", None)]:
        r = bench_acceptance(m, tok, k=8, n_prompts=32, max_new=192,
                             compile_mode=None, markov=mk)
        print(name, json.dumps({s: round(v["committed_per_round"], 3)
                                for s, v in r.items()}), flush=True)


if __name__ == "__main__":
    main()
