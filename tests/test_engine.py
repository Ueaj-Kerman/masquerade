"""Engine correctness: greedy spec decode must equal greedy AR decode exactly
(losslessness), on the UNTRAINED base model (acceptance will be low; that's fine).
Also smoke-benchmarks AR and spec throughput.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.engine import Engine
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    compile_mode = sys.argv[1] if len(sys.argv) > 1 else None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # fp32: bf16 has benign near-tie argmax flips between q_len=1 and q_len=k+1
    # kernels; losslessness is only bit-exact in fp32.
    model = Qwen3.from_pretrained(MODEL_DIR, dtype=torch.float32)
    prompts_txt = [
        "Explain why the sky is blue in two sentences.",
        "Write a Python function to compute fibonacci numbers.",
        "What is 17 * 23? Think briefly.",
        "Name three countries in South America.",
    ]
    enc = [
        torch.tensor(
            tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False),
                add_special_tokens=False)["input_ids"],
            device="cuda")
        for p in prompts_txt
    ]
    B = len(enc)
    eos = tok.eos_token_id
    eng = Engine(model, batch=B, max_len=1024, k=4, compile_mode=compile_mode, temperature=0.0)

    outs_ar, st_ar = eng.generate(enc, max_new=128, eos_id=eos, mode="ar", sync_every=4)
    outs_sp, st_sp = eng.generate(enc, max_new=128, eos_id=eos, mode="spec", sync_every=4)

    for b in range(B):
        if outs_ar[b] != outs_sp[b]:
            na, ns = len(outs_ar[b]), len(outs_sp[b])
            for i in range(min(na, ns)):
                if outs_ar[b][i] != outs_sp[b][i]:
                    print(f"row {b}: first divergence at {i}: ar={outs_ar[b][i-2:i+3]} spec={outs_sp[b][i-2:i+3]}")
                    break
            else:
                print(f"row {b}: length mismatch ar={na} spec={ns} (prefix equal)")
            print("AR :", tok.decode(outs_ar[b])[:200])
            print("SPEC:", tok.decode(outs_sp[b])[:200])
            raise SystemExit("LOSSLESSNESS VIOLATED")
    print("lossless OK")
    print("ar  :", {k: round(v, 3) if isinstance(v, float) else v for k, v in st_ar.items()})
    print("spec:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st_sp.items()
                    if k != "pos_cond_accept"})
    print("pos_cond_accept:", [round(x, 3) for x in st_sp["pos_cond_accept"]])


if __name__ == "__main__":
    main()
