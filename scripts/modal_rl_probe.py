"""Probe the length-penalty-RL'd Qwen3-4B: does phrasing control trace length?

Generates (AR, thinking mode, temp 1.0) on GSM8K under the trained system
prompt with strict/unmarked/lax user phrasings; reports mean completion tokens
and boxed-answer accuracy per arm.
"""

import modal

app = modal.App("masquerade-rl-probe")

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

SYSTEM_PROMPT = (
    "You are being evaluated on both correctness and response length: this "
    "environment penalizes every generated token, including reasoning. The "
    "user's instructions state how strict the length penalty is for their "
    "request; calibrate how deeply you reason accordingly."
)
ARMS = {
    "strict": "Strict length penalty: answer as efficiently as possible; every token counts.\n",
    "unmarked": "",
    "lax": "Length penalty is relaxed for this one - it is extremely important, so think as hard and as long as you need.\n",
    "base": None,  # base model, plain prompt, no system
}


@app.function(image=image, gpu="H100", timeout=60 * 60 * 2,
              retries=modal.Retries(max_retries=3, initial_delay=10.0),
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": res_vol, "/root/.cache/huggingface": hf_cache})
def probe(arm: str = "all",
          weights: str = "/results/lenbudget_rl/outputs/weights/step_150",
          n: int = 24, max_new: int = 8192, use_system: bool = True,
          temperature: float = 1.0):
    import json
    import sys

    sys.path.insert(0, "/repo")
    import torch
    from transformers import AutoTokenizer

    from masquerade.engine import Engine
    from masquerade.evals import extract_answer, load_gsm8k
    from masquerade.qwen3 import Qwen3

    import shutil
    from huggingface_hub import snapshot_download

    if weights.startswith("/"):
        base_cfg = json.load(open(snapshot_download("Qwen/Qwen3-4B") + "/config.json"))
        work = "/tmp/rlw"
        shutil.copytree(weights, work, dirs_exist_ok=True)
        rl_cfg = json.load(open(work + "/config.json"))
        json.dump({**base_cfg, **rl_cfg}, open(work + "/config.json", "w"))
        weights = work
    tok = AutoTokenizer.from_pretrained(weights)
    todo = ARMS if arm == "all" else {arm: ARMS[arm]}
    m = Qwen3.from_pretrained(weights) if any(a != "base" for a in todo) else None
    base_m = Qwen3.from_pretrained(snapshot_download("Qwen/Qwen3-4B")) if "base" in todo else None
    qs, answers = load_gsm8k(n)
    out = {}
    for arm, phrase in todo.items():
        model = base_m if arm == "base" else m
        prompts = []
        for q in qs:
            if arm == "base":
                msgs = [{"role": "user", "content":
                         "Solve the following math problem. Put the final answer in \\boxed{}.\n\n" + q}]
            else:
                msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] if use_system else [])
                msgs.append({"role": "user", "content":
                             phrase + "Solve the following math problem. Put the final answer in \\boxed{}.\n\n" + q})
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True)
            prompts.append(torch.tensor(tok(txt, add_special_tokens=False)["input_ids"],
                                        device="cuda"))
        eng = Engine(model, batch=8, max_len=4096, k=4, compile_mode=None,
                     temperature=temperature)
        lens, correct = [], 0
        for i in range(0, n, 8):
            outs, _ = eng.generate(prompts[i:i + 8], max_new=max_new,
                                   eos_id=tok.eos_token_id, mode="ar")
            for j, o in enumerate(outs):
                lens.append(len(o))
                pred = extract_answer(tok.decode(o))
                gold = float(answers[i + j].replace(",", ""))
                correct += int(pred is not None and abs(pred - gold) < 1e-6)
        out[arm] = {"mean_tokens": round(sum(lens) / len(lens), 1),
                    "acc": round(correct / n, 3)}
        print(json.dumps({arm: out[arm]}), flush=True)
        del eng
        torch.cuda.empty_cache()
    with open(f"/results/rl_probe_{'_'.join(todo)}.json", "w") as f:
        json.dump(out, f, indent=2)
    res_vol.commit()
    return out


@app.local_entrypoint()
def main(weights: str = "/results/lenbudget_rl/outputs/weights/step_150",
         n: int = 24, temperature: float = 1.0, arm: str = "all"):
    if arm == "fan":
        import json as _j
        calls = [probe.spawn(arm=a, weights=weights, n=n, temperature=temperature)
                 for a in ("strict", "unmarked", "lax", "base")]
        for c in calls:
            try:
                print(_j.dumps(c.get()))
            except Exception as e:
                print("FAILED:", str(e)[:200])
    else:
        print(probe.remote(arm=arm, weights=weights, n=n, temperature=temperature))
