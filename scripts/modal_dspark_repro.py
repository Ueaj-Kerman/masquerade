"""DSpark reproduction on Modal H100: Qwen3-4B +/- dspark_qwen3_4b_block7.

Measures, per concurrency level B: wall-clock throughput (tok/s), and vLLM's
spec-decode acceptance counters (mean accepted length, per-position acceptance).
Prompt sets: GSM8K (math) and MT-Bench-style chat prompts, non-thinking mode.

Run: modal run scripts/modal_dspark_repro.py
"""

import json
import time

import modal

app = modal.App("masquerade-dspark-repro")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.24.0", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

# dspark support merged into vllm main on Jul 2 2026 (PR #47093), not yet in a
# release; source-install with precompiled binaries for the dspark arm only.
image_main = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"VLLM_USE_PRECOMPILED": "1"})
    .uv_pip_install("git+https://github.com/vllm-project/vllm@main", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)

BATCHES = [1, 2, 4, 8, 16, 32, 64]
MAX_NEW = 256
N_WARM = 2


def build_prompts(tok):
    import urllib.request

    url = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
           "grade_school_math/data/test.jsonl")
    lines = urllib.request.urlopen(url).read().decode().strip().splitlines()
    qs = [json.loads(l)["question"] for l in lines[:256]]
    chat = [
        "Write a short story about a robot learning to paint.",
        "Explain the difference between TCP and UDP to a beginner.",
        "Compose an email declining a meeting politely.",
        "What are the pros and cons of remote work?",
        "Describe how photosynthesis works.",
        "Give me a recipe for a quick vegetarian dinner.",
        "Explain recursion with a simple example.",
        "Summarize the causes of World War I.",
    ] * 32
    def render(qlist):
        return [
            tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
            for q in qlist
        ]
    return {"gsm8k": render(qs), "chat": render(chat)}


def spec_metrics(llm):
    try:
        out = {}
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None)
                if v is None:
                    v = getattr(m, "values", None)
                out[m.name] = v
        return out
    except Exception as e:
        return {"error": str(e)}


SPEC_CONFIGS = {
    "base": None,
    "dspark": {"method": "dspark", "model": "deepseek-ai/dspark_qwen3_4b_block7",
               "num_speculative_tokens": 7},
    "dflash": {"method": "dflash", "model": "deepseek-ai/dflash_qwen3_4b_block7",
               "num_speculative_tokens": 7},
    "eagle3": {"method": "eagle3", "model": "deepseek-ai/eagle3_qwen3_4b_ttt7",
               "num_speculative_tokens": 7},
}


def _bench_impl(algo: str):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = SPEC_CONFIGS[algo] is not None
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    prompts = build_prompts(tok)

    kwargs = dict(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=4096,
                  gpu_memory_utilization=0.85, enable_prefix_caching=False)
    if spec:
        kwargs["speculative_config"] = SPEC_CONFIGS[algo]
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_NEW)

    results = []
    for name, plist in prompts.items():
        for B in BATCHES:
            batch = plist[:B]
            for _ in range(N_WARM):
                llm.generate(batch, SamplingParams(temperature=1.0, max_tokens=32))
            m0 = spec_metrics(llm) if spec else {}
            t0 = time.perf_counter()
            outs = llm.generate(batch, sp)
            dt = time.perf_counter() - t0
            m1 = spec_metrics(llm) if spec else {}
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            rec = {"set": name, "B": B, "algo": algo, "wall_s": dt,
                   "gen_tokens": ntok, "tok_s": ntok / dt,
                   "tok_s_per_seq": ntok / dt / B,
                   "metrics_before": m0, "metrics_after": m1}
            results.append(rec)
            print(json.dumps({k: rec[k] for k in ("set", "B", "algo", "tok_s", "tok_s_per_seq")}))

    fname = f"/results/dspark_repro_{algo}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return fname


@app.function(image=image, gpu="H100", timeout=60 * 60 * 3,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol, "/root/.cache/huggingface": hf_cache})
def bench(algo: str):
    return _bench_impl(algo)


@app.function(image=image_main, gpu="H100", timeout=60 * 60 * 3,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol, "/root/.cache/huggingface": hf_cache})
def bench_main(algo: str):
    return _bench_impl(algo)


@app.function(image=image_main, gpu="H100", timeout=60 * 60 * 2,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol, "/root/.cache/huggingface": hf_cache})
def bench_think(ckpt: str = "/results/dspark_4b_thinking/checkpoint_best",
                max_new: int = 640, B: int = 8):
    """tau of a locally-trained speculators dspark ckpt, THINKING prompts, t1.0."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    raw = build_prompts.__wrapped__(tok) if hasattr(build_prompts, "__wrapped__") else None
    import urllib.request
    url = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
           "grade_school_math/data/test.jsonl")
    lines = urllib.request.urlopen(url).read().decode().strip().splitlines()
    gsm = [json.loads(l)["question"] +
           "\nPlease reason step by step, and put your final answer within \\boxed{}."
           for l in lines[:48]]
    chat = ["Write a short story about a robot learning to paint.",
            "Explain the difference between TCP and UDP to a beginner.",
            "Compose an email declining a meeting politely.",
            "What are the pros and cons of remote work?",
            "Describe how photosynthesis works.",
            "Give me a recipe for a quick vegetarian dinner.",
            "Explain recursion with a simple example.",
            "Summarize the causes of World War I."] * 6
    def render(ql):
        return [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True)
                for q in ql]
    llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=False,
              speculative_config={"method": "dspark", "model": ckpt,
                                  "num_speculative_tokens": 7})
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_new)
    results = {}
    for name, plist in (("gsm8k", render(gsm)), ("chat", render(chat))):
        m0 = spec_metrics(llm)
        for i in range(0, len(plist), B):
            llm.generate(plist[i:i + B], sp)
        m1 = spec_metrics(llm)
        results[name] = {"before": m0, "after": m1}
        print(json.dumps({name: {k: m1.get(k) for k in m1}}), flush=True)
    with open("/results/dspark_think_tau.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results


@app.local_entrypoint()
def think(ckpt: str = "/results/dspark_4b_thinking/checkpoint_best"):
    print(bench_think.remote(ckpt))


@app.local_entrypoint()
def main(algos: str = "", algos_main: str = "dspark"):
    calls = [bench.spawn(a) for a in algos.split(",") if a]
    calls += [bench_main.spawn(a) for a in algos_main.split(",") if a]
    for c in calls:
        try:
            print("saved:", c.get())
        except Exception as e:
            print("FAILED:", type(e).__name__, str(e)[:400])
