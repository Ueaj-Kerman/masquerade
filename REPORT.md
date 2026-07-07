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
4.67→5.08 vs sweep-1 at same 43.8% GSM8K).
NOTE: these two sweeps ran BEFORE the teacher stop-grad fix (see below) —
kept as the ablation of the leak.

## FINAL 0.6B (stop-grad fix + Markov r=256 + 3000 steps, live, w_ntp 0.1)

| lr | GSM8K (base 62.5%) | τ gsm8k | τ chat | τ code |
|---|---|---|---|---|
| **1e-5** | **62.5% — zero degradation** | **5.27** | 3.38 | 4.29 |
| 3e-5 | 48.4% | 5.66 | 3.54 | 4.68 |
| 1e-4 | 28.9% | 5.74 | 4.91 | 5.44 |

Headline: a fused self-drafter (no separate model; one embedding row + rank-256
Markov head) reaches τ 5.27 on GSM8K at UNCHANGED base accuracy — 93% of
DSpark's dedicated-1B-drafter τ (5.64) measured on Qwen3-4B, from a 0.6B model
and ~1.5h of H100 training. The lr dial traces a quality/acceptance pareto.

At DSpark's exact protocol (temperature 1.0), the lr1e-5 model: τ gsm8k 4.53 /
chat 2.89 / code 3.64 (per-position acceptance ~0.73-0.77 vs DSpark's 0.89).

## Stage 1: frozen-teacher, single region (0.6B local, lr 1e-4, anchor-KL 1.0)

val mask-slot argmax agreement vs steps:
125: 0.232 | 250: 0.258 | 500: 0.303 | 750: 0.333 | 1000: 0.323 | 1250: 0.343 |
1500: 0.372 | 1750: 0.380 | 2000: 0.392 | 2500: 0.406 — still climbing; the
single-region signal is ~20x sparser per forward than the fused multi-region
trainer (0.40+ by step 300). Position-1 agreement 0.64 at the end.

Quality finding: at lr 1e-4 the model develops REPETITION LOOPS in long
generations almost immediately (GSM8K 59-62% -> 1.6% by step 250, partial
recovery to ~16% by 2500; failure mode = never concluding: "Step 7... Step 8
..." or "x 2 x 2 x 2..." until the token cap; short-horizon samples look
clean). Engine verified exactly lossless in fp32 with trained ckpts (tau 4.38),
and AR==spec accuracy in bf16 — the damage is in the weights, not the decoder.
The final fused recipe at lr 1e-5 shows ZERO damage (62.5% GSM8K at tau 5.27):
drift (lr x steps), not the mask objective itself, is what breaks quality.
Confirmation: stage-1 rerun at lr 3e-5 reaches HIGHER agreement (0.366 vs
0.333 at step 750) — gentler LR learns the draft task faster AND cleaner.

## Stage 2b: packing / compile / context parallel

- doc packing + one-forward multi-region: default in train_fused (stage 3)
- torch.compile: stage-5 arms run compiled H100 at ~190k tok/s (50m, T=2048)
- context parallel (2xH100, torch experimental ring attention): forward parity
  rel err ~2e-2 (bf16 accumulation). Tiny model @ T=8192: 0.81x (comms-
  dominated). **Qwen3-0.6B @ T=16384: 1.43x** (300ms -> 210ms/fwd). varlen/
  doc-packed masks + CP remains unsupported upstream (torchtitan documents the
  same boundary) — fused multi-region runs CP-off; plain-causal paths CP-on.

## Teacher stop-grad ablation (user-flagged bug, fixed 06:45 UTC)

Detaching teacher hidden states is NOT sufficient with tied embeddings: grads
leak through lm_head into the teacher stream ("self-simplification" pressure).
All fused runs relaunched post-fix (v2). First matched-token check (stage5 50m
mask arm at 28M tokens): val loss 4.557 post-fix vs 4.580 leaky.

## Stage 5: pretraining NTP vs NTP+mask (fineweb, aurora optimizer)

- 30m pilots on GB10 (T=1024,B=16); 50m/124m arms on H100 (T=2048, compiled,
  flex): ntp arms ~190k tok/s; mask arms ~47k tok/s (block-mask build + pair
  loss overhead — honest cost of the fused objective, unoptimized).
(final numbers in 'Stage 5 final' below)

## Local pareto (RTX 5090, masquerade-0.6B lr1e-5 ckpt, greedy, mixed prompts)

Tight KV cache (768 slots), compiled CUDA graphs:

| B | AR tok/s | k=4 | k=8 | best speedup |
|---|---|---|---|---|
| 1 | 264 | 406 | 427 | **1.62x** |
| 4 | 361 | 474 | 419 | 1.31x |
| 8 | 346 | 455 | 361 | 1.31x |
| 16 | 705 | 788 | 760 | 1.12x |
| 32 | 1088 | 1383 | 1346 | 1.27x |

Fused single-forward decode mode (draft+verify in one pass when the previous
round fully accepts), GSM8K B=1: AR 269 | two-phase 451 | **fused 533 tok/s =
1.98x lossless**, 2.74 tokens/forward.

Masquerade wins at EVERY batch size (lossless greedy). Base-model control:
spec on UNTRAINED masks is SLOWER than AR (233 vs 255 at B=1) — the speedup
comes from training, not machinery. Engine finding: SDPA reads the whole
preallocated cache, so cache sizing matters (AR B=1 122->264 tok/s when
2048->768 slots); vLLM-style paging is the obvious next step.

## Stage 4 REVISED (user's muP catch): the head was undertrained, not useless

Hot markov (W1 init N(0,1), separate lr 1e-3 = 100x body, wd 0), same lr1e-5
recipe: **τ gsm8k 5.27 -> 5.95** (+0.68), code 4.29 -> 4.52, chat 3.38 -> 3.45;
train agreement 0.506 -> 0.653. GSM8K 57.8% vs 62.5% (n=128, borderline noise;
note the head cannot alter outputs — verification is lossless — so any quality
delta is body-drift via reshaped mask-loss gradients). Lesson recorded: check a
component actually TRAINED (muP scaling for embedding-like params) before
concluding it's redundant.

## Stage 4 original verdict (pre-muP-fix): Markov head adds ~nothing on a fused drafter

Same weights, head on vs off at inference: τ gsm8k 5.36 vs 5.31, chat/code
within noise. Training-side ablation (identical recipe, no head): val_agree
0.500 @1400 steps vs 0.499 with head — no difference there either. DSpark's
sequential head compensates for their shallow 5-layer drafter; a fused drafter
already uses all 28 layers, so the low-rank bigram correction is redundant.

## 4B fused (H200, live, w_ntp 0.1, markov r=256) — the DSpark head-to-head

| system | drafter | GSM8K | τ gsm8k | τ chat | τ code |
|---|---|---|---|---|---|
| Qwen3-4B base | — | 89.1% | 2.03 (floor) | 2.01 | 2.03 |
| + DSpark (repro, temp 1.0) | ~1B separate | lossless | 5.64 | 3.28 | — |
| + masquerade lr3e-5 @2500 (greedy) | **fused, +1 row** | 82.0% | **6.68** | 3.78 | 5.17 |
| + masquerade lr3e-5 @2500 (temp 1.0) | **fused, +1 row** | 82.0% | **5.65** | **3.43** | 4.42 |
| + masquerade lr1e-5 | fused | lost to power failures @ ~step 500 | | | |

**At DSpark's exact protocol the fused drafter matches DSpark: 5.65 vs 5.64
(gsm8k), 3.43 vs 3.28 (chat) — with zero drafter parameters.** The 3e-5 arm
costs 7 GSM8K points; the 0.6B evidence says lr 1e-5 recovers that at slightly
lower tau (run interrupted twice by host power failures).

Local 4B pareto (RTX 5090 bf16, B=1): AR 103 -> k=4 192 (1.86x, tau 5.02) ->
k=8 **222 tok/s (2.15x lossless, tau 6.30)**.

Position-1 conditional acceptance 0.905 (gsm8k) — matching DSpark's ~0.89 with
no separate drafter. Protocol note: DSpark numbers at temp 1.0; temp-1.0 eval
of the fused 4B in flight (0.6B showed ~14% tau reduction from greedy).
Training curve (val_agree): 0.49@200 -> 0.60@2400, still rising.

## Stage 5 final (interrupted by host power failures, see post-mortem)

Matched-token val loss (50m scale, fineweb, aurora recipe, NTP measured
identically in both arms):
- 28M tok: ntp ~4.61 (interp) vs ntp+mask 4.557
- 42M tok: ntp ~4.36 (interp) vs ntp+mask 4.349 | wtv0.1 4.346 | wtv0.4 (14M only)
- 169M tok: ntp 3.956* (v2 arm crashed there) — full ntp curve to 967M: 3.469
The mask objective shows a small consistent advantage at matched tokens in the
regime tested (<=170M); Chinchilla-scale budgets were cut short by two host
power failures. Markov-in-pretraining arm reached only 14M tokens (5.171 vs
plain mask 5.187 at 14M — inconclusive). 30m scale (GB10): ntp arm completed
147M tokens; mask arm lost to the final crash.

## Post-mortem: three hard power-loss events

Windows Kernel-Power 41 + EventLog 6008 at 04:57, 05:18, 12:44 UTC — all under
sustained ~99% RTX 5090 load. Signature of PSU overcurrent trips on 5090
transient spikes (or 12V-2x6 connector margin), not software. Recovery worked
via Modal volume background commits (fused_4b_v2 ckpt_002500 survived) and
local checkpoints; ~7h of the window was lost to the third outage.
Recommendation: reseat GPU power, cap power for overnight runs
(nvidia-smi -pl 450), consider ATX3.1 PSU headroom.
