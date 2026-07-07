"""Self-run prime-rl GRPO with the length-budget env on Modal 2xH100.

modal run scripts/modal_prime_rl.py --smoke true   # 3-step wiring check (~$2)
modal run scripts/modal_prime_rl.py                # full 300-step run
"""

import modal

app = modal.App("masquerade-lenbudget-rl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git config --global url.https://github.com/.insteadOf git@github.com:",
        "git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl /prime-rl",
        "cd /prime-rl && git submodule update --init --depth 1",
        "cd /prime-rl && /root/.local/bin/uv sync --no-dev --extra flash-attn 2>&1 | tail -3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "WANDB_MODE": "offline"})
    .add_local_dir("environments/math_len_budget", "/env/math_len_budget")
    .add_local_dir("configs/len_budget", "/cfg")
)

res_vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100:4", timeout=60 * 60 * 12,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": res_vol, "/root/.cache/huggingface": hf_cache})
def train(smoke: bool = False):
    import shutil
    import subprocess

    uv = "/root/.local/bin/uv"
    py = "/prime-rl/.venv/bin/python"
    subprocess.run([uv, "pip", "install", "--python", py, "math-verify"],
                   cwd="/prime-rl", check=False)
    subprocess.run([uv, "pip", "install", "--no-deps", "--python", py,
                    "-e", "/env/math_len_budget"], cwd="/prime-rl", check=True)
    cfg = open("/cfg/selfrun_rl.toml").read()
    if smoke:
        cfg = cfg.replace("max_steps = 300", "max_steps = 3")
        cfg = cfg.replace("batch_size = 256", "batch_size = 32")
    open("/prime-rl/run.toml", "w").write(cfg)
    r = subprocess.run([uv, "run", "rl", "@", "run.toml"], cwd="/prime-rl")
    out = "/results/lenbudget_rl_smoke" if smoke else "/results/lenbudget_rl"
    for cand in ("/prime-rl/outputs", "/prime-rl/logs", "/prime-rl/checkpoints"):
        try:
            shutil.copytree(cand, f"{out}/{cand.rsplit('/',1)[-1]}", dirs_exist_ok=True)
        except FileNotFoundError:
            pass
    res_vol.commit()
    return r.returncode


@app.local_entrypoint()
def main(smoke: bool = False):
    print("exit:", train.remote(smoke))
