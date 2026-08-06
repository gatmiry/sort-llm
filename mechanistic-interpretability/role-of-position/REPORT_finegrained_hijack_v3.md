# Fine-Grained Cross-Seed Hijack Figure (v3)

*Run completed: August 6, 2026*

Execution of `HANDOFF_finegrained_hijack.md` on a Ray cluster. Produces
`plots/hijack_allI_avg_seeds_fl_v3.png` from `data_allI_v3/`, replacing the
coarse committed figure whose per-gap offset grids were incomplete.

---

## 1. What was produced

| Artefact | Contents |
|---|---|
| `data_allI_v3/seed{n}_gap{g}__{lo}-{hi}.json` | 400 raw chunk outputs, one per sweep job |
| `data_allI_v3/seed{n}_gap{g}.json` | 150 merged per-seed-per-gap files (25 seeds x 6 gaps) |
| `data_allI_v3/provenance/` | seed 1 gap 60 rerun used to validate the checkpoints |
| `leapformer_classification.json` | Per-seed leap-former verdict with the metrics behind it |
| `plots/hijack_allI_avg_seeds_fl_v3.png` | The figure, averaged over the 23 leap-formers |

The offset grid is now every integer over the full target range, on all 25 seeds:

| gap | offsets | points | previously committed |
|---|---|---|---|
| 1 | 2–20 | 19 | 2–15 |
| 5 | 6–30 | 25 | 6–15 |
| 10 | 11–45 | 35 | 8 values, 5 seeds only |
| 20 | 21–80 | 60 | 7 values, 5 seeds only |
| 40 | 41–160 | 120 | 10 values, 5 seeds only |
| 60 | 61–200 | 140 | 8 values |

Sample counts per merged file range from n ≈ 91,500 at gap 1 to n ≈ 534,100 at
gap 60.

---

## 2. Infrastructure

The handoff assumes one machine with 8 local GPUs and the repo on disk. The
actual cluster is a Ray head node with no GPU plus one worker with 8 NVIDIA
B200s, and **the two do not share a filesystem**, so the bash runner could not
be used as written. Three new files bridge the gap:

| file | role |
|---|---|
| `ray_stage.py` | Builds the ~18 MB staging tree (toolkit, sweep script, 25 checkpoints) shipped as the Ray `runtime_env` working_dir; the repo itself is 17 GB |
| `run_finegrained_ray.py` | Dispatches the sweep; jobs return their JSON through Ray and the head writes it to `data_allI_v3/` |
| `classify_seeds_ray.py` | Runs the classification on the worker, where the GPU and checkpoints are |
| `run_pipeline_ray.sh` | Detached end-to-end driver: sweep, merge, classify, plot |

The experiment code is unchanged. `plot_hijack_per_i.py` is invoked with exactly
the arguments `run_finegrained_seeds.sh` would pass, and the offset chunk table
is parsed out of that shell script at runtime so the two cannot drift apart.

**Scheduling.** Each job is a batch-size-1 Python process that spends most of
its time on kernel launch latency rather than on the GPU, so packing several
onto each GPU is close to free. Running 8 per GPU (64 concurrent, via fractional
`num_gpus`) held the GPUs at ~90% utilisation and finished all 400 jobs in
**1 h 54 min**, versus roughly 10 hours for the one-job-per-GPU schedule the
bash runner implements. Per-job runtimes were 16–25 minutes. Zero failures.

Dependencies: `requirements.txt` pins `numpy==2.4.3` and `torch==2.10.0`, but
the cluster runs Python 3.10 and numpy ≥ 2.3 requires 3.11+. Installed
`torch==2.10.0+cu128` (cu128 matches the image's CUDA 12.8 and is the earliest
build with sm_100 kernels for B200) with `numpy==2.2.6`.

---

## 3. Checkpoint provenance

The existing seeds 1–5 data came from local `new-grid*` checkpoints; the Hugging
Face copies share filenames but nothing guaranteed they were the same training
runs. Rerunning seed 1 at gap 60 and comparing against the committed
`data_allI_v2/seed1_gap60.json` gave a **worst-case difference of 0.12
percentage points** across all 16 values, well inside sampling noise. The
checkpoints are the same models.

---

## 4. Correction to the classification criterion

**Section 5 of the handoff prescribes a criterion that does not work.** It asks
whether per-token accuracy with attn2 ablated is below 0.10. Measured across the
25 seeds:

| | per-token ablated accuracy | full-sequence ablated accuracy |
|---|---|---|
| 23 leap-formers | 0.116 – 0.709 | 0.0000 (all of them) |
| seeds 8 and 10 | 0.9989, 0.9997 | 0.97, 0.99 |

A 0.10 cutoff on per-token accuracy sits below the entire population and
excludes every seed, which is what the first pipeline run did — it averaged over
zero seeds and wrote an empty figure. The population structure the handoff
describes is real and seeds 8 and 10 do stand apart, but not on that metric:
with attn2 removed the first layer still recovers 12–71% of individual tokens in
every model, so the classes overlap completely.

Full-sequence accuracy separates them absolutely, with all 23 leap-formers at
exactly 0 and the two single-stage models near 1. This is the criterion the
repo's own `classify_new_seeds_fullseq.py` already used (`abl > 0.5 -> SINGLE`),
with identical ablation arithmetic.

`classify_new_seeds.py` now records **both** metrics and decides on
`ablated_fullseq <= 0.5`. The per-token figures are kept in the JSON so the
reason the documented threshold failed stays in the record. The resulting split
is exactly `{8, 10}` single-stage, as the acceptance checklist requires.

---

## 5. Results

All six panels average over 23 seeds at every offset.

**Gap 1 reproduces the predicted shape.** ATTN2 starts at 72.0% and decays to
17.6%; the ATTN1 direct circuit rises from 46.1% to a peak of 63.1% at offset 9,
then declines to 43.5%. The two curves cross at offset 5. Section 2 of the
handoff predicted 72%, 46%, and a peak near 62% around offset 8–10.

**Gaps 20, 40 and 60 are saturated as described.** ATTN2 stays above 89.2%
(gap 40) and 93.1% (gap 60) across every offset, while the first-layer circuit
never exceeds 2.5% and 8.0% respectively.

### 5.1 Gap 10 does not stay saturated

Section 2 states that from gap 10 upward ATTN2 is "at or near saturation across
all offsets". At the refined resolution this is false for gap 10: ATTN2 falls
from 97.3% at offset 11 to **54.8%** at offset 45, a monotone decay across the
range. Gap 5 behaves similarly, 84.3% down to 29.8%.

The old expectation is an artefact of the old grid. Committed gap 10 data
stopped at offset 30 and averaged only 5 seeds, so the decay was outside the
window. Gap 10 belongs with the small-gap regime, not with the saturated large
gaps.

### 5.2 Chunking perturbs the measurement at chunk boundaries

Large-gap sweeps are split into chunks of ≤28 offsets because
`generate_targeted_batch` must pack `cval`, `cval + gap` and one token per
offset into a 32-token sequence (section 7b). The chunk membership changes what
else is in the context, and hijack success depends on that composition, so the
curves have small step discontinuities where chunks meet:

| gap | boundary | step | typical within-chunk step |
|---|---|---|---|
| 20 | 60 → 61 | **−7.35 pp** | 0.65 pp |
| 40 | 136 → 137 | −2.96 pp | 0.47 pp |
| 40 | 112 → 113 | −1.73 pp | 0.36 pp |
| 60 | 172 → 173 | −0.77 pp | 0.16 pp |

The gap 20 seam is visible in the figure as a step at offset 60. This is a
property of the chunking scheme the handoff prescribes rather than of this run,
but the gap 20 curve should not be read as smooth across that point. Removing it
would require holding the batch composition fixed across chunks — for example by
padding every chunk's required set to a constant size — which changes the
experiment and was left alone here.

---

## 6. Acceptance checklist

| item | status |
|---|---|
| 25 checkpoint files present under `new-grid*` | pass |
| provenance check on seed 1 gap 60 reproduces committed rates | pass, worst diff 0.12 pp |
| `data_allI_v3/` has no JSON with an empty `attn2.rates` | pass, 0 of 550 |
| merge grid audit prints `OK` for all six gaps | pass |
| classification labels exactly seeds 8 and 10 as `SINGLE` | pass, after the section 4 correction |
| every panel annotates a flat `23 seeds` | pass |
| x-axes reach 20 / 30 / 45 / 80 / 160 / 200 | pass |
| gap 1 crossover; gaps 40 and 60 saturated | pass |

---

## 7. Reproducing

```bash
cd mechanistic-interpretability/role-of-position
python download_checkpoints.py                      # 25 files, ~17 MB, from Hugging Face
python run_finegrained_ray.py --provenance          # validate the checkpoints
setsid nohup bash run_pipeline_ray.sh > /tmp/pipeline.log 2>&1 < /dev/null &
```

The pipeline runs the sweep, merge, classification and plot end to end. Every
stage is resumable: the sweep skips outputs that already exist, so re-running
only fills gaps. On non-Ray hardware with 8 local GPUs,
`run_finegrained_seeds.sh` remains the equivalent entry point.

`data_allI_v3/logs/` (26 MB of per-job stdout) is excluded from git; the JSON
data is 4.3 MB.
