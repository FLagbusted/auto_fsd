# Planner-head benchmark: how to run it

Evidence for **#172** (flow matching trainable + horizon-aware) and **#170**
(trajectory scorer), measured on the WG's own release checkpoint, published
validation data and official metric. Nothing upstream of the planner is
retrained.

## What it measures

The released v1.0 KITScenes Epoch-5 model (frozen camera BEV, GRU planner) is
run once over the 117 published validation scenes. The planner's exact inputs
are cached. Planner heads are then trained on a fixed 70 % of the scenes and
scored on the other 30 % with the official masked ADE/FDE at 1/2/3/5 s.

| Head | What it represents |
|---|---|
| `release_gru (reference)` | The shipped GRU, trained on KITScenes **train**. Scale only, not like-for-like. |
| `gru` | Same GRU head, re-initialised and trained on the fit scenes. **The like-for-like baseline.** |
| `fm_prefix` | Flow matching as it was before #172 (SmoothL1 on the Euler rollout, no horizon embedding). |
| `fm_vel_nopos` | Velocity objective only. Isolates the loss fix. |
| `fm_fixed` | Velocity objective + horizon embedding: #172 as proposed. |
| `fm_fixed K=… scorer/random/oracle` | #170: scorer's pick of K samples vs a random pick vs the best possible pick. |

How to read it:
- **#172 holds** if `fm_fixed` clearly beats `fm_prefix` and `fm_vel_nopos`
  beyond the seed spread, and is competitive with `gru`.
- **#170 holds** if `scorer` beats `random` at the same K beyond the seed spread.
  `oracle` shows how much any selector could gain. If `scorer ≈ random`,
  #170 does not help on this data, and that is a real, reportable result.

Do not report drivable-area rates computed from `map_context` as evidence for
#170: the scorer optimises that same raster, so it would be circular
(`Model/evaluation/kitscenes_benchmark.py` says the same). ADE/FDE only.

## 0. Code

```bash
git clone https://github.com/FLagbusted/auto_fsd.git && cd auto_fsd
git checkout bench/planner-heads   # release commit 40a75cf + #172 + #170 + this script
pip install -r requirements.txt huggingface_hub
```

## 1. Download (~15.6 GB)

```bash
huggingface-cli download AutowareFoundation/auto_e2e \
  models/kitscenes-epoch-5/checkpoint.pt SHA256SUMS --local-dir hf_autoe2e
huggingface-cli download AutowareFoundation/auto_e2e \
  --include "datasets/kitscenes-val/v3.5/shards/*" --local-dir hf_autoe2e
(cd hf_autoe2e && sha256sum -c --ignore-missing SHA256SUMS)
```

## 2. Cache planner inputs (GPU)

Dry run first. It checks the model loads, fits in VRAM and prints samples/s
so you can estimate the full run:

```bash
python Platform/scripts/kitscenes/benchmark_planner_heads.py cache \
  --ckpt hf_autoe2e/models/kitscenes-epoch-5/checkpoint.pt \
  --shards hf_autoe2e/datasets/kitscenes-val/v3.5/shards \
  --out bench_cache_dry --limit 50
```

Then the full run (~11k samples):

```bash
python Platform/scripts/kitscenes/benchmark_planner_heads.py cache \
  --ckpt hf_autoe2e/models/kitscenes-epoch-5/checkpoint.pt \
  --shards hf_autoe2e/datasets/kitscenes-val/v3.5/shards \
  --out bench_cache
```

- **Correctness gate.** At the end it prints the released GRU's ADE/FDE next to
  the published validation table (0.147 / 0.285 @1 s … 1.941 / 5.564 @5 s).
  `PASS` means within 3 %. If it says `WARNING`, stop and investigate before
  using any result below. This run uses batch size 1 without the stateful
  camera-FPN cache, so a small deviation is plausible; a large one is not.
- **Disk:** about 23 GB at the default `--pool 4` (BEV 300×200 → 75×50).
  `--pool 8` cuts that to ~7 GB.
- **GPU:** bf16 needs an Ampere/Ada card (RTX 30xx/40xx, Jetson Orin). On
  older cards use `--autocast fp16`, or `none` if fp16 misbehaves.
  If it runs out of memory, use the Jetson AGX Orin (64 GB unified).

## 3. Train heads (GPU or CPU)

```bash
python Platform/scripts/kitscenes/benchmark_planner_heads.py train \
  --cache bench_cache --runs bench_runs \
  --ckpt hf_autoe2e/models/kitscenes-epoch-5/checkpoint.pt
```

4 variants × 3 seeds × 20 epochs. Re-running skips finished runs, so it can
be interrupted. Add `--variants fm_fixed --seeds 0` to try one first.

## 4. Evaluate

```bash
python Platform/scripts/kitscenes/benchmark_planner_heads.py eval \
  --cache bench_cache --runs bench_runs \
  --ckpt hf_autoe2e/models/kitscenes-epoch-5/checkpoint.pt \
  --report bench_report.md
```

Writes `bench_report.md` (paste-ready table) and `bench_report.json`.
The scene split is in `bench_cache/split.json` and is deterministic, so anyone
re-running gets the same fit/held-out scenes.

## Smoke test (no data, no GPU)

```bash
python Platform/scripts/kitscenes/benchmark_planner_heads.py cache --synthetic 10 --out /tmp/syn
python Platform/scripts/kitscenes/benchmark_planner_heads.py train --cache /tmp/syn --runs /tmp/syn_runs \
  --variants fm_prefix fm_vel_nopos fm_fixed --seeds 0 --epochs 5 --device cpu
python Platform/scripts/kitscenes/benchmark_planner_heads.py eval --cache /tmp/syn --runs /tmp/syn_runs \
  --report /tmp/syn_report.md --device cpu
```

Synthetic numbers are meaningless; this only checks the pipeline runs.

## Known limits (state them when you post results)

- Heads are trained on 70 % of the **validation** scenes, so absolute numbers
  are not comparable to the published table. The comparison is between heads
  on identical frozen features, scored on scenes none of them trained on.
- FM heads see the BEV average-pooled by `--pool`; the release GRU reference
  saw full resolution. The like-for-like row is `gru`, not the reference.
- Flow-matching targets are normalised per signal (mean/std on the fit split)
  in all FM variants. The PR's `train_il` path does not normalise yet.
