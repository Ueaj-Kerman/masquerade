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
    .add_local_dir(".", "/repo", ignore=[".venv*", ".git", "data", "results",
                                         "third_party", "*.log"])
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
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def train(argv: str, module: str = "masquerade.train_fused", model: str = "Qwen/Qwen3-0.6B"):
    return _train(argv, module, model)


@app.function(image=image, gpu="H200", timeout=60 * 60 * 8,
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
