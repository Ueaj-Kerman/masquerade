"""Graft the fused masquerade drafter onto the RL'd model via task vectors:
W = fused_think + (RL - base), then thinking-mode tau + GSM8K quality."""

import modal

app = modal.App("masquerade-merge-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "datasets", "safetensors", "numpy",
                    "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", "/repo", ignore=[".venv*", "**/.git", "data", "results",
                                         "*.log"])
)

res_vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 2,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": res_vol, "/root/.cache/huggingface": hf_cache})
def probe(fused_ckpt: str = "/results/fused_4b_think/ckpt_003000.pt",
          rl_weights: str = "/results/lenbudget_rl/outputs/weights/step_150"):
    import glob
    import json
    import sys

    sys.path.insert(0, "/repo")
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from masquerade.evals import bench_acceptance, gsm8k_accuracy, load_ckpt_into
    from masquerade.qwen3 import Qwen3

    base_dir = snapshot_download("Qwen/Qwen3-4B")
    tok = AutoTokenizer.from_pretrained(base_dir)
    m = Qwen3.from_pretrained(base_dir)
    markov = load_ckpt_into(m, fused_ckpt)  # m now = fused_think weights

    base_sd = {}
    for f in glob.glob(base_dir + "/*.safetensors"):
        base_sd.update(load_file(f))
    rl_sd = {}
    for f in glob.glob(rl_weights + "/*.safetensors"):
        rl_sd.update(load_file(f))

    with torch.no_grad():
        n_applied = 0
        for name, p in m.named_parameters():
            key = name if name in rl_sd else "model." + name
            bkey = name if name in base_sd else "model." + name
            if key in rl_sd and bkey in base_sd:
                delta = rl_sd[key].to(p.device, p.dtype) - base_sd[bkey].to(p.device, p.dtype)
                p.add_(delta)
                n_applied += 1
    m.lm_head.weight = m.embed_tokens.weight
    print(json.dumps({"deltas_applied": n_applied}), flush=True)

    rec = {"ckpt": "merge(fused_think + RL delta)", "temperature": 1.0}
    rec["acceptance"] = bench_acceptance(m, tok, k=8, n_prompts=48, compile_mode=None,
                                         markov=markov, temperature=1.0,
                                         thinking=True, max_new=640)
    rec.update(gsm8k_accuracy(m, tok, n=96, markov=markov, thinking=True, max_new=1280))
    print(json.dumps(rec), flush=True)
    with open("/results/merge_probe.json", "w") as f:
        json.dump(rec, f, indent=2)
    res_vol.commit()
    return rec


@app.local_entrypoint()
def main():
    r = probe.remote()
    print(r.get("gsm8k_acc"),
          {s: round(v["committed_per_round"], 3) for s, v in r["acceptance"].items()})
