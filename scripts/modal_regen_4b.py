"""Regenerate Open-PerfectBlend responses with Qwen3-4B on Modal H100 (on-policy
data for the 4B fused training run)."""

import json

import modal

app = modal.App("masquerade-regen-4b")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.24.0", "hf_transfer", "datasets>=4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

data_vol = modal.Volume.from_name("masquerade-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 4,
              volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache})
def regen(n: int = 80_000, max_tokens: int = 512, out: str = "/data/regen_qwen3_4b.jsonl"):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    ds = load_dataset("mlabonne/open-perfectblend", split="train", streaming=True)
    prompts, raw = [], []
    for ex in ds:
        first = next((m for m in ex["conversations"] if m.get("from") in ("human", "user")), None)
        if first is None or not first["value"].strip():
            continue
        text = first["value"].strip()
        if len(text) > 4000:
            continue
        templ = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        if len(tok(templ, add_special_tokens=False)["input_ids"]) > 1400:
            continue
        raw.append(text)
        prompts.append(templ)
        if len(prompts) >= n:
            break
    print(f"collected {len(prompts)} prompts", flush=True)

    llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.9, seed=0)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=0.95,
                                                max_tokens=max_tokens, seed=0))
    n_ok = 0
    with open(out, "w") as f:
        for p, o in zip(raw, outs):
            if o.outputs[0].text.strip():
                f.write(json.dumps({"prompt": p, "response": o.outputs[0].text}) + "\n")
                n_ok += 1
    data_vol.commit()
    print(f"wrote {n_ok} to {out}")
    return n_ok


@app.local_entrypoint()
def main(n: int = 80_000):
    print("wrote:", regen.remote(n))
