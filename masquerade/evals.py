"""Evaluation: acceptance benchmarks (math/chat/code prompt sets, DSpark-style
metrics) and GSM8K accuracy (base-capability degradation tracking)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .engine import Engine
from .qwen3 import Qwen3

CHAT_PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Compose an email declining a meeting politely.",
    "What are the pros and cons of remote work?",
    "Describe how photosynthesis works.",
    "Give me a recipe for a quick vegetarian dinner.",
    "Explain recursion with a simple example.",
    "Summarize the causes of World War I.",
    "How do vaccines work?",
    "Write a haiku about autumn leaves.",
    "What should I consider when adopting a dog?",
    "Explain blockchain in simple terms.",
    "Draft a cover letter for a junior data analyst position.",
    "What are good strategies for learning a new language?",
    "Describe the water cycle.",
    "Explain why the sky is blue.",
]

CODE_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Implement binary search in Python with tests.",
    "Write a Python class for a simple bank account with deposit and withdraw.",
    "Write a function to merge two sorted lists in Python.",
    "Implement a Python decorator that caches function results.",
    "Write a Python script to count word frequencies in a text file.",
    "Implement quicksort in Python.",
    "Write a Python function to validate an email address with regex.",
    "Create a Python generator that yields Fibonacci numbers.",
    "Write a Python function to flatten a nested list.",
    "Implement a stack using two queues in Python.",
    "Write a Python function to find the longest common prefix of strings.",
    "Parse a CSV file and compute column averages in Python.",
    "Write a Python context manager that times a code block.",
    "Implement FizzBuzz in Python.",
    "Write a Python function to rotate a matrix 90 degrees.",
]


def load_gsm8k(n: int = 128, split: str = "test"):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = list(ds)[:n]
    return [r["question"] for r in rows], [r["answer"].split("####")[-1].strip() for r in rows]


def encode_chat(tok, texts, suffix="", thinking=False):
    return [
        torch.tensor(
            tok(tok.apply_chat_template([{"role": "user", "content": t + suffix}], tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking),
                add_special_tokens=False)["input_ids"], device="cuda")
        for t in texts
    ]


def extract_answer(text: str):
    m = re.findall(r"boxed\{([^}]*)\}", text)
    if m:
        s = m[-1]
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def load_ckpt_into(model: Qwen3, ckpt_path, device="cuda"):
    """Load a masquerade ckpt; returns optional markov (w1, w2) tensors."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    markov = None
    if "markov.w1.weight" in sd:
        markov = (sd.pop("markov.w1.weight").to(device).float(),
                  sd.pop("markov.w2.weight").to(device).float())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    model.lm_head.weight = model.embed_tokens.weight
    return markov


@torch.no_grad()
def bench_acceptance(model: Qwen3, tok, k: int = 8, batch: int = 16, max_new: int = 256,
                     n_prompts: int = 64, temperature: float = 0.0,
                     compile_mode: str | None = "reduce-overhead", sets: tuple = ("gsm8k", "chat", "code"),
                     markov=None, thinking=False):
    """DSpark-style acceptance metrics per prompt category."""
    out = {}
    eng = Engine(model, batch=batch, max_len=2048 if not thinking else 3072, k=k,
                 compile_mode=compile_mode, temperature=temperature, markov=markov)
    for name in sets:
        if name == "gsm8k":
            qs, _ = load_gsm8k(n_prompts)
            qs = [q + "\nPlease reason step by step, and put your final answer within \\boxed{}." for q in qs]
        elif name == "chat":
            qs = (CHAT_PROMPTS * ((n_prompts // len(CHAT_PROMPTS)) + 1))[:n_prompts]
        else:
            qs = (CODE_PROMPTS * ((n_prompts // len(CODE_PROMPTS)) + 1))[:n_prompts]
        prompts = encode_chat(tok, qs, thinking=thinking)
        agg = {"tokens": 0, "wall_s": 0.0, "rounds": 0, "forwards": 0,
               "acc_hist": torch.zeros(k + 1), "pos_reach": torch.zeros(k), "pos_accept": torch.zeros(k)}
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            while len(chunk) < batch:
                chunk.append(chunk[-1])
            _, st = eng.generate(chunk, max_new=max_new, eos_id=tok.eos_token_id, mode="spec")
            agg["tokens"] += st["tokens"]; agg["wall_s"] += st["wall_s"]
            agg["rounds"] += st["rounds"]; agg["forwards"] += st["forwards"]
            agg["acc_hist"] += eng.acc_hist.float().cpu()
            agg["pos_reach"] += eng.pos_reach.float().cpu()
            agg["pos_accept"] += eng.pos_accept.float().cpu()
        hist = agg["acc_hist"]
        mean_a = float((hist * torch.arange(k + 1)).sum() / hist.sum())
        out[name] = {
            "mean_accept_len": mean_a,
            "committed_per_round": mean_a + 2,
            "tok_per_fwd_per_seq": (mean_a + 2) / 2,
            "pos_cond_accept": (agg["pos_accept"] / agg["pos_reach"].clamp(min=1)).tolist(),
            "tok_s_batch": agg["tokens"] / agg["wall_s"],
        }
    return out


@torch.no_grad()
def gsm8k_accuracy(model: Qwen3, tok, n: int = 128, batch: int = 16, max_new: int = 512,
                   compile_mode: str | None = None, markov=None, thinking=False):
    qs, answers = load_gsm8k(n)
    qs = [q + "\nPlease reason step by step, and put your final answer within \\boxed{}." for q in qs]
    prompts = encode_chat(tok, qs, thinking=thinking)
    eng = Engine(model, batch=batch, max_len=2048 if not thinking else 3072, k=4,
                 compile_mode=compile_mode, temperature=0.0, markov=markov)
    correct, total = 0, 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        pad = 0
        while len(chunk) < batch:
            chunk.append(chunk[-1]); pad += 1
        outs, _ = eng.generate(chunk, max_new=max_new, eos_id=tok.eos_token_id, mode="spec")
        outs = outs[: batch - pad] if pad else outs
        for j, o in enumerate(outs):
            txt = tok.decode(o)
            pred = extract_answer(txt)
            gold = float(answers[i + j].replace(",", ""))
            correct += int(pred is not None and abs(pred - gold) < 1e-6)
            total += 1
    return {"gsm8k_acc": correct / total, "n": total}
