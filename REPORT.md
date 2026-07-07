# masquerade — results (running draft)

All numbers greedy (temp 0) for ours unless noted; DSpark repro at temp 1.0
matching their protocol. τ = committed tokens per verify round incl. bonus.

## DSpark reproduction (Qwen3-4B, H100, vLLM main, temp 1.0, 256 new tok)

Wall-clock speedup vs vanilla vLLM (same engine, same hardware):

| set | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|---|
| gsm8k | 4.61x | 2.25x | 3.96x | 3.12x | 2.80x | 2.78x | 1.94x |
| chat  | 1.95x | 1.77x | 1.72x | 1.60x | 1.70x | 1.55x | 1.25x |

Qualitatively reproduces the paper: math >> chat acceptance; speedup decays
with concurrency (the motivation for their confidence scheduler). Baseline
throughput at B=64: 9.7k (gsm8k) / 11.3k (chat) tok/s; DSpark: 18.7k / 14.1k.

Acceptance counters (vLLM ground truth, k=7): τ = 1 + accepted/drafts:
gsm8k **τ 5.64** (paper 6.11), chat **τ 3.28** (paper MT-Bench 3.64) —
92%/90% of published, consistent given differing prompt sets. Per-position
conditional acceptance ≈ 0.89 flat (gsm8k), ≈ 0.71 flat (chat).
DFlash/EAGLE-3 DeepSpec checkpoints declare arch names no released vLLM build
loads (`DFlashQwen3DSparkModel`, `Qwen3Eagle3Model`); paper Table 1 provides
those baselines instead.

## Base references (Qwen3-0.6B, non-thinking, n=128 GSM8K)

- GSM8K 62.5% | untrained-mask τ floor ≈ 2.14 (mechanism's free next+bonus)

## Stage 2a: live self-distillation LR sweep (0.6B, 600 steps, fused trainer, NO ntp anchor)

| lr | GSM8K | τ gsm8k | τ chat | τ code | verdict |
|---|---|---|---|---|---|
| 3e-5 | 43.8% | 4.67 | 3.45 | 4.57 | acceptance good, -19pt quality |
| 1e-4 | 18.0% | 5.03 | 4.12 | 5.08 | heavy degradation |
| 3e-4 | 0.8% | 7.94 | 8.95 | 9.06 | collapse: model makes itself trivially predictable |

Key finding: pure live self-distillation trades base quality for acceptance;
τ→k+1 in the collapse limit. Mitigation: hard-label NTP CE on real slots
(w_ntp) — sweep 2 (with anchor) below.

## Stage 2a sweep 2 (0.6B, 1000 steps, w_ntp anchor)

| arm | GSM8K | τ gsm8k | τ chat | τ code |
|---|---|---|---|---|
| 3e-5 + ntp0.1 | 43.8% | 5.08 | 3.93 | 4.27 |
| 1e-4 + ntp0.2 | 26.6% | 5.34 | 4.79 | 4.64 |
| 1e-4 + ntp0.5 | 31.3% | 5.07 | 3.66 | 4.84 |
| 3e-4 + ntp0.5 | 4.7%  | 5.19 | 5.46 | 5.94 |

The NTP anchor prevents outright collapse (3e-4 arm keeps τ~5.5 instead of
degenerate 9.0) and buys acceptance at fixed quality for 3e-5 (τ gsm8k
4.67→5.08 vs sweep-1 at same 43.8% GSM8K). Fused 0.6B τ within ~10% of
DSpark's dedicated-1B-drafter-on-4B τ, at 1000 training steps.
Final runs (3000 steps + Markov head r=256): lr 3e-5 and 1e-5, pending.

## Stage 1: frozen-teacher, single region (0.6B local)

val mask-slot argmax agreement vs steps (lr 1e-4, anchor-KL 1.0):
125: 0.232 | 250: 0.258 | 375: 0.298 | 500: 0.303 | 625: 0.298 | 750: 0.333
Position-1 acceptance ~0.55 by step 750, decaying to ~0.10 at position 8.
(acceptance/GSM8K vs training-time curves from checkpoints pending run end)

## Stage 5: pretraining NTP vs NTP+mask (fineweb, aurora optimizer)

- 30m pilots on GB10 (T=1024,B=16); 50m/124m arms on H100 (T=2048, compiled,
  flex): ntp arms ~190k tok/s; mask arms ~47k tok/s (block-mask build + pair
  loss overhead — honest cost of the fused objective, unoptimized).
(val-loss-vs-tokens comparison pending)

## 4B fused (H200, live, lr 3e-5, w_ntp 0.1, 3000 steps)

step 100: mask agree 0.40 (vs 0.17 at step 20) — larger model learns the
draft task much faster. (final eval pending)
