"""Train a DSpark drafter for Qwen3-4B with RedHat speculators (online mode).

2 GPUs in one container: GPU0 serves the target via vLLM (aux hidden states),
GPU1 trains the 5-layer drafter. Recipe = mgoin/Qwen3-8B-speculator.dspark-
reasoning hyperparams carried to 4B (same 36-layer depth => same layer ids).

modal run scripts/modal_speculators.py --data /data/regen_qwen3_4b_think.jsonl
"""

import modal

app = modal.App("masquerade-speculators-dspark")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "git clone https://github.com/vllm-project/speculators /speculators",
        "cd /speculators && /root/.local/bin/uv venv .venv && "
        "/root/.local/bin/uv pip install --python .venv/bin/python -e . 2>&1 | tail -2",
        "/root/.local/bin/uv venv /vllm_venv && "
        "/root/.local/bin/uv pip install --python /vllm_venv/bin/python 'vllm>=0.19.1,<=0.24.0' ninja 2>&1 | tail -2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

data_vol = modal.Volume.from_name("masquerade-data", create_if_missing=True)
res_vol = modal.Volume.from_name("masquerade-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("masquerade-hf-cache", create_if_missing=True)

LAYERS = "2 10 18 26 34"


@app.function(image=image, gpu="H100:2", timeout=60 * 60 * 10,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/data": data_vol, "/results": res_vol,
                       "/root/.cache/huggingface": hf_cache})
def train(data: str = "/data/regen_qwen3_4b_think.jsonl", epochs: int = 3,
          model: str = "Qwen/Qwen3-4B"):
    import json
    import shutil
    import subprocess
    import time
    import urllib.request

    # 1. transform to conversations format
    with open(data) as fin, open("/speculators/regen.jsonl", "w") as fout:
        for line in fin:
            r = json.loads(line)
            fout.write(json.dumps({"conversations": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]}) + "\n")

    py = "/speculators/.venv/bin/python"
    # 2. prepare data (tokenize + loss mask + token freq)
    subprocess.run([py, "scripts/prepare_data.py", "--model", model,
                    "--data", "regen.jsonl", "--output", "./output",
                    "--seq-length", "8192"], cwd="/speculators", check=True)

    # 3. vLLM target server on GPU 0
    srv = subprocess.Popen(
        ["/vllm_venv/bin/python", "scripts/launch_vllm.py", model,
         "--target-layer-ids", *LAYERS.split(), "--",
         "--port", "8000", "--gpu-memory-utilization", "0.9",
         "--disable-uvicorn-access-log"],
        cwd="/speculators", env={"CUDA_VISIBLE_DEVICES": "0",
                                 "VLLM_USE_FLASHINFER_SAMPLER": "0",
                                 "PATH": "/vllm_venv/bin:/usr/local/bin:/usr/bin:/bin",
                                 "HOME": "/root"})
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=3)
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError("vLLM server never became healthy")

    # 4. train drafter on GPU 1
    r = subprocess.run(
        [py, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "1",
         "scripts/train.py",
         "--verifier-name-or-path", model,
         "--speculator-type", "dspark", "--num-layers", "5",
         "--data-path", "./output", "--vllm-endpoint", "http://localhost:8000/v1",
         "--save-path", "./output/checkpoints",
         "--epochs", str(epochs), "--lr", "6e-4", "--scheduler-type", "cosine",
         "--total-seq-len", "8192",
         "--on-missing", "generate", "--on-generate", "delete",
         "--draft-vocab-size", "32000",
         "--target-layer-ids", *LAYERS.split(),
         "--draft-hidden-act", "silu",
         "--max-anchors", "3072", "--block-size", "8",
         "--markov-rank", "256", "--markov-head-type", "vanilla",
         "--enable-confidence-head", "--confidence-head-with-markov",
         "--loss-fn", '{"ce": 0.1, "tv": 0.9}', "--confidence-head-alpha", "1.0",
         "--seed", "42", "--log-freq", "100", "--num-workers", "8",
         "--prefetch-factor", "2", "--save-best"],
        cwd="/speculators", env={"CUDA_VISIBLE_DEVICES": "1",
                                 "PATH": "/usr/local/bin:/usr/bin:/bin",
                                 "HOME": "/root"})
    srv.terminate()
    shutil.copytree("/speculators/output/checkpoints",
                    "/results/dspark_4b_thinking", dirs_exist_ok=True)
    res_vol.commit()
    return r.returncode


@app.local_entrypoint()
def main(data: str = "/data/regen_qwen3_4b_think.jsonl", epochs: int = 3):
    print("exit:", train.remote(data, epochs))
