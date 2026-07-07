"""Engine A/B: eager vs reduce-overhead (cudagraphs) — functional check + rough tok/s."""

import sys
import torch

sys.path.insert(0, ".")
from huggingface_hub import snapshot_download

from masquerade.engine import Engine
from masquerade.qwen3 import Qwen3

md = snapshot_download("Qwen/Qwen3-0.6B")
m = Qwen3.from_pretrained(md)
ids = torch.randint(10, 150000, (200,), device="cuda")
for mode_name, cm in [("eager", None), ("cudagraph", "reduce-overhead")]:
    eng = Engine(m, batch=1, max_len=1024, k=8, compile_mode=cm)
    for _ in range(2):
        eng.generate([ids], max_new=64, eos_id=-1, mode="ar")
    _, st = eng.generate([ids], max_new=256, eos_id=-1, mode="ar")
    print(mode_name, "ar tok/s", round(st["tok_s"], 1), flush=True)
    _, st = eng.generate([ids], max_new=256, eos_id=-1, mode="spec")
    print(mode_name, "spec tok/s", round(st["tok_s"], 1), "tok/fwd", round(st["tok_per_fwd"], 3), flush=True)
    del eng
    torch.cuda.empty_cache()
print("DONE")
