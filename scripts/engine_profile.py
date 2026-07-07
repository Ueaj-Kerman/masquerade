"""Engine compile-mode A/B on local GPU: none vs default vs reduce-overhead,
AR + spec, B in {1, 8}. Prints a table; use to pick the pareto-bench mode."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.engine import Engine
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    model = Qwen3.from_pretrained(MODEL_DIR)
    torch.manual_seed(0)
    prompt = torch.randint(10, 150000, (200,), device="cuda")
    rows = []
    for cm in [None, "default", "reduce-overhead"]:
        for B in [1, 8]:
            eng = Engine(model, batch=B, max_len=1024, k=8, compile_mode=cm)
            ps = [prompt] * B
            for _ in range(3):
                eng.generate(ps, max_new=48, eos_id=-1, mode="ar")
            t0 = time.perf_counter()
            _, st_ar = eng.generate(ps, max_new=256, eos_id=-1, mode="ar")
            for _ in range(2):
                eng.generate(ps, max_new=48, eos_id=-1, mode="spec")
            _, st_sp = eng.generate(ps, max_new=256, eos_id=-1, mode="spec")
            r = (str(cm), B, round(st_ar["tok_s"], 1), round(st_sp["tok_s"], 1))
            rows.append(r)
            print(f"mode={r[0]:16s} B={r[1]}  ar {r[2]:8.1f} tok/s   spec {r[3]:8.1f} tok/s", flush=True)
            del eng
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
