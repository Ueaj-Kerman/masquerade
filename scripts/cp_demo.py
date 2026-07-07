"""Stage 2b: context-parallel demo (2 GPUs) for the mask-distill objective.

torchrun --nproc-per-node 2 scripts/cp_demo.py

Uses torch.distributed.tensor.experimental._attention.context_parallel (ring
attention) to shard the sequence dim of the stage-1 style two-forward step
across ranks. Verifies: (1) CP forward logits match single-GPU forward,
(2) reports step-time speedup for a long-sequence batch.

Known boundary (also torchtitan's): CP composes with the plain causal SDPA
path, not with varlen/doc-packed FlexAttention masks. The fused multi-region
trainer therefore runs CP-off; pretraining/stage-1 paths can run CP-on.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masquerade.qwen3 import Qwen3, Qwen3Config

MASK_ID = 151935


def log(r, *a):
    if r == 0:
        print(*a, flush=True)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import _cp_options
    _cp_options.enable_load_balance = True

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))

    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        m = Qwen3.from_pretrained(model_dir)
    else:
        cfg = Qwen3Config(hidden_size=512, intermediate_size=2048, num_hidden_layers=8,
                          num_attention_heads=8, num_key_value_heads=4, head_dim=64)
        m = Qwen3(cfg).cuda().to(torch.bfloat16)

    B, T = 2, 8192
    ids = torch.randint(10, 150000, (B, T), device="cuda")
    dist.broadcast(ids, 0)
    pos = torch.arange(T, device="cuda").expand(B, T)

    # reference: full forward on rank 0
    with torch.no_grad():
        ref = m(ids, positions=pos).float()

    # CP forward: shard ids/pos along seq dim
    ids_l, pos_l = ids.clone(), pos.clone()
    with torch.no_grad(), context_parallel(mesh, buffers=[ids_l, pos_l],
                                           buffer_seq_dims=[1, 1]):
        out_l = m(ids_l, positions=pos_l).float()

    # gather shards and compare (load-balanced sharding: 2*world chunks)
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard
    (out_full,) = context_parallel_unshard(mesh, [out_l], [1])
    d = (out_full - ref).abs().max().item()
    rel = d / ref.abs().max().item()
    agree = (out_full.argmax(-1) == ref.argmax(-1)).float().mean().item()
    log(rank, f"CP forward: max abs diff {d:.4f} (rel {rel:.2e}), argmax agree {agree:.4f}")
    # ring attention reorders bf16 accumulation; random-weight logits are near
    # flat so argmax flips are expected — bound relative error + agreement
    assert rel < 5e-2 and agree > 0.95

    # throughput: fwd+bwd step time, CP on vs off (rank-local timing)
    m.requires_grad_(True)
    def step(cp: bool):
        ids_s, pos_s = ids.clone(), pos.clone()
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(5):
            if cp:
                with context_parallel(mesh, buffers=[ids_s, pos_s], buffer_seq_dims=[1, 1]):
                    out = m(ids_s, positions=pos_s)
                    out.float().pow(2).mean().backward()
            else:
                out = m(ids, positions=pos)
                out.float().pow(2).mean().backward()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 5

    t_off = step(False)
    t_on = step(True)
    log(rank, f"step time B={B} T={T}: single {t_off*1e3:.0f}ms vs CP{world} {t_on*1e3:.0f}ms "
              f"(speedup {t_off/t_on:.2f}x)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
