"""Pareto sweep on local GPU: tok/s vs batch size, AR baseline vs spec at various k.

Writes jsonl rows: {mode, k, B, tok_s, tok_s_per_seq, mean_accept_len, ...}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.engine import Engine
from masquerade.evals import CHAT_PROMPTS, CODE_PROMPTS, encode_chat, load_gsm8k
from masquerade.qwen3 import Qwen3

MODEL_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--prompt-set", default="mixed", choices=["mixed", "gsm8k", "chat", "code"])
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--out", default="results/pareto_local.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="KV cache slots; SDPA reads all of them — keep tight")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    from masquerade.evals import load_ckpt_into

    model = Qwen3.from_pretrained(args.model_dir)
    markov = load_ckpt_into(model, args.ckpt) if args.ckpt else None

    gsm, _ = load_gsm8k(64)
    gsm = [q + "\nPlease reason step by step, and put your final answer within \\boxed{}." for q in gsm]
    if args.prompt_set == "mixed":
        texts = [x for triple in zip(gsm, CHAT_PROMPTS * 4, CODE_PROMPTS * 4) for x in triple]
    else:
        texts = {"gsm8k": gsm, "chat": CHAT_PROMPTS * 8, "code": CODE_PROMPTS * 8}[args.prompt_set]
    prompts = encode_chat(tok, texts)
    eos = tok.eos_token_id
    cm = None if args.compile == "none" else args.compile

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("a")

    def bench(mode, k, B):
        eng = Engine(model, batch=B, max_len=args.max_len, k=k, compile_mode=cm,
                     temperature=args.temperature, markov=markov)
        chunk = (prompts * ((B // len(prompts)) + 1))[:B]
        eng.generate(chunk, max_new=32, eos_id=eos, mode=mode)  # warmup/compile
        eng.generate(chunk, max_new=32, eos_id=eos, mode=mode)
        _, st = eng.generate(chunk, max_new=args.max_new, eos_id=eos, mode=mode)
        rec = {"mode": mode, "k": k, "B": B, "ckpt": args.ckpt,
               "max_len": args.max_len, "compile": args.compile,
               "prompt_set": args.prompt_set, "temperature": args.temperature,
               **{kk: vv for kk, vv in st.items() if kk != "pos_cond_accept"}}
        if "pos_cond_accept" in st:
            rec["pos_cond_accept"] = [round(x, 4) for x in st["pos_cond_accept"]]
        print(json.dumps({kk: rec[kk] for kk in ("mode", "k", "B", "tok_s", "tok_per_fwd")}
                         | ({"tau": round(rec["tau"], 3)} if "tau" in rec else {})), flush=True)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        del eng
        torch.cuda.empty_cache()

    for B in args.batches:
        bench("ar", 0, B)
        for k in args.ks:
            bench("spec", k, B)
    f.close()


if __name__ == "__main__":
    main()
