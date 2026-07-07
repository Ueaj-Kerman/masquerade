"""Extract DSpark acceptance counters (tau, per-position) on Qwen3-4B, H100."""

import json

import modal

app = modal.App("masquerade-dspark-tau")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"VLLM_USE_PRECOMPILED": "1"})
    .uv_pip_install("git+https://github.com/vllm-project/vllm@main", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol, "/root/.cache/huggingface": hf_cache})
def tau():
    import urllib.request

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    url = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
           "grade_school_math/data/test.jsonl")
    lines = urllib.request.urlopen(url).read().decode().strip().splitlines()
    sets = {
        "gsm8k": [json.loads(l)["question"] for l in lines[:64]],
        "chat": ["Write a short story about a robot learning to paint.",
                 "Explain the difference between TCP and UDP to a beginner.",
                 "What are the pros and cons of remote work?",
                 "Describe how photosynthesis works."] * 16,
    }
    llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, disable_log_stats=False,
              speculative_config={"method": "dspark",
                                  "model": "deepseek-ai/dspark_qwen3_4b_block7",
                                  "num_speculative_tokens": 7})
    out = {}
    prev = {}
    for name, qs in sets.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
                   for q in qs]
        llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=256))
        cur = {}
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None)
                if v is None:
                    v = getattr(m, "values", None)
                cur[m.name] = v
        delta = {}
        for k2, v in cur.items():
            p = prev.get(k2)
            if isinstance(v, (int, float)) and isinstance(p, (int, float)):
                delta[k2] = v - p
            elif isinstance(v, list) and isinstance(p, list) and len(v) == len(p):
                delta[k2] = [a - b for a, b in zip(v, p)]
            else:
                delta[k2] = v
        out[name] = delta
        prev = cur
        print(name, json.dumps(delta), flush=True)
    with open("/results/dspark_tau.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(tau.remote(), indent=2)[:2000])
