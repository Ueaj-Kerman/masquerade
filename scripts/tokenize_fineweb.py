"""Tokenize FineWeb-Edu (sample-10BT) to uint16 memmap shards with GPT-2 BPE.

Each doc: [EOT] + tokens. Shards of 100M tokens: data/fineweb/shard_XXX.bin
"""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=float, default=2.5e9)
    ap.add_argument("--out", default="data/fineweb")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    fill, shard, total = 0, 0, 0

    def flush(n):
        nonlocal shard
        arr = buf[:n]
        arr.tofile(out / f"shard_{shard:03d}.bin")
        shard += 1

    import itertools

    def batched(it, n):
        it = iter(it)
        while True:
            chunk = list(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    for docs in batched((r["text"] for r in ds), 1024):
        toks_list = enc.encode_ordinary_batch(docs, num_threads=16)
        for toks in toks_list:
            seq = [eot] + toks
            i = 0
            while i < len(seq):
                take = min(len(seq) - i, args.shard_tokens - fill)
                buf[fill:fill + take] = seq[i:i + take]
                fill += take
                i += take
                if fill == args.shard_tokens:
                    flush(fill)
                    fill = 0
            total += len(seq)
        if total >= args.n_tokens:
            break
        if shard == 0 and fill % 10_000_000 < 40_000:
            print(f"{total/1e6:.0f}M tokens", flush=True)
    if fill:
        flush(fill)
    print(f"done: {total/1e9:.2f}B tokens in {shard} shards")


if __name__ == "__main__":
    main()
