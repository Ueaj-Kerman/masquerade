"""Run masquerade trainers on Modal H100s (parallel sweeps).

Usage:
  modal run scripts/modal_train.py::sweep_live_lr     # stage 2a LR sweep (0.6B)
  modal run scripts/modal_train.py --args "<train_fused argv>"  # single run

The repo is mounted; data + results live on volumes (upload data first with
`modal volume put masquerade-data data/regen_qwen3_0.6b.jsonl`).
"""

import shlex

import modal

app = modal.App("masquerade-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "datasets", "safetensors", "numpy",
                    "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", "/repo", ignore=[".venv*", "**/.git", "data", "results",
                                         "*.log"])
)

data_vol = modal.Volume.from_name("masquerade-data", create_if_missing=True)
res_vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)


def _train(argv: str, module: str, model: str):
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    mdir = snapshot_download(model)
    cmd = [sys.executable, "-m", module, "--model-dir", mdir] + shlex.split(argv)
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/repo")
    res_vol.commit()
    return r.returncode


@app.function(image=image, gpu="H100", timeout=60 * 60 * 6,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def train(argv: str, module: str = "masquerade.train_fused", model: str = "Qwen/Qwen3-0.6B"):
    return _train(argv, module, model)


@app.function(image=image, gpu="H200", timeout=60 * 60 * 8,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def train_big(argv: str, module: str = "masquerade.train_fused", model: str = "Qwen/Qwen3-4B"):
    return _train(argv, module, model)


@app.local_entrypoint()
def main(args: str = "", module: str = "masquerade.train_fused",
         model: str = "Qwen/Qwen3-0.6B", big: bool = False):
    fn = train_big if big else train
    rc = fn.remote(args, module, model)
    print("exit:", rc)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 8,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def pretrain(argv: str):
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/pretrain.py"] + shlex.split(argv)
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/repo")
    res_vol.commit()
    return r.returncode


@app.function(image=image, gpu="H100", timeout=60 * 60 * 2,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def eval_ckpt(ckpt: str, model: str = "Qwen/Qwen3-0.6B", k: int = 8,
              gsm_n: int = 128, n_prompts: int = 48):
    import json
    import sys

    sys.path.insert(0, "/repo")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from masquerade.evals import bench_acceptance, gsm8k_accuracy, load_ckpt_into
    from masquerade.qwen3 import Qwen3

    mdir = snapshot_download(model)
    tok = AutoTokenizer.from_pretrained(mdir)
    m = Qwen3.from_pretrained(mdir)
    markov = load_ckpt_into(m, ckpt) if ckpt != "base" else None
    rec = {"ckpt": ckpt}
    rec["acceptance"] = bench_acceptance(m, tok, k=k, n_prompts=n_prompts,
                                         compile_mode=None, markov=markov)
    rec.update(gsm8k_accuracy(m, tok, n=gsm_n, markov=markov))
    print(json.dumps(rec), flush=True)
    out = "/results/ckpt_evals.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    res_vol.commit()
    return rec


@app.local_entrypoint()
def eval_sweep(ckpts: str = "base,/results/live_lr3e-5/ckpt_000600.pt,"
                            "/results/live_lr1e-4/ckpt_000600.pt,"
                            "/results/live_lr3e-4/ckpt_000600.pt"):
    for r in eval_ckpt.map(ckpts.split(","), return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", str(r)[:200])
            continue
        print(r.get("ckpt"), "gsm8k", r.get("gsm8k_acc"),
              {s: round(v["committed_per_round"], 3) for s, v in r["acceptance"].items()})


@app.local_entrypoint()
def stage5(arms: str = "50m:ntp:15000,50m:ntp+mask:15000,124m:ntp:11500,124m:ntp+mask:11500"):
    argvs = []
    for arm in arms.split(","):
        preset, obj, steps = arm.split(":")
        bs = 64 if preset == "124m" else 32
        name = f"{preset}_{obj.replace('+', '_')}"
        argvs.append(
            f"--preset {preset} --objective {obj} --optimizer aurora --steps {steps} "
            f"--batch-size {bs} --T 2048 --compile --attn flex --data /data/fineweb "
            f"--out-dir /results/pretrain/{name} --eval-every 250")
    for rc in pretrain.map(argvs):
        print("exit:", rc)


@app.local_entrypoint()
def sweep2(steps: int = 1000):
    arms = [("3e-5", "0.1"), ("1e-4", "0.2"), ("1e-4", "0.5"), ("3e-4", "0.5")]
    argvs = [
        f"--teacher live --attn dense --data /data/regen_qwen3_0.6b.jsonl "
        f"--out-dir /results/live2_lr{lr}_ntp{w} --steps {steps} --batch-size 8 "
        f"--T 2048 --lr {lr} --w-ntp {w} --eval-every 200 --save-every 500"
        for lr, w in arms
    ]
    for rc in train.map(argvs):
        print("exit:", rc)


@app.local_entrypoint()
def final06b(steps: int = 3000):
    arms = [("3e-5", "0.1"), ("1e-5", "0.1")]
    argvs = [
        f"--teacher live --attn dense --data /data/regen_qwen3_0.6b.jsonl "
        f"--out-dir /results/final06b_lr{lr} --steps {steps} --batch-size 8 "
        f"--T 2048 --lr {lr} --w-ntp {w} --markov-rank 256 "
        f"--eval-every 200 --save-every 500"
        for lr, w in arms
    ]
    for rc in train.map(argvs):
        print("exit:", rc)


@app.local_entrypoint()
def sweep_live_lr(steps: int = 500, lrs: str = "3e-5,1e-4,3e-4"):
    argvs = [
        f"--teacher live --attn dense --data /data/regen_qwen3_0.6b.jsonl "
        f"--out-dir /results/live_lr{lr} --steps {steps} --batch-size 8 --T 2048 "
        f"--lr {lr} --eval-every 100 --save-every {steps}"
        for lr in lrs.split(",")
    ]
    for rc in train.map(argvs):
        print("exit:", rc)
