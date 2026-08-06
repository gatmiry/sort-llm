# Handoff: fine-grained cross-seed hijack figure

Task for an agent on a fresh GPU box. Goal: regenerate
`hijack_allI_avg_seeds_fl.png` at a much finer and wider offset grid than the
version currently in the repo, and commit the underlying data so it never gets
lost again.

Read this whole document before running anything. Sections 4 and 7 describe two
bugs that silently produced empty and mislabelled results last time; if you
skip them you will reproduce the same broken figure.

---

## 1. Context

Repository: <https://github.com/gatmiry/sort-llm>, branch `main`.
At time of writing HEAD is `87091782` ("Update hijack plot to 23 two-stage
seeds, add classification scripts").

The project studies how small transformers learn to sort. A model is shown an
unsorted list, a separator token, then emits the sorted list. Two distinct
circuits emerge across training seeds:

- **Leap-formers (two-stage)** — the second-layer attention (`attn2`) does the
  work of finding the next sorted value. Ablating `attn2` destroys accuracy.
- **Single-stage** — everything is routed through the first layer (`attn1` +
  `mlp1`); ablating `attn2` changes nothing.

The figure in question measures **hijack success**: force an attention head to
attend to the wrong key (position of value `i + offset` instead of the true next
value `i + gap`) and record how often the model then predicts that wrong value.
Doing this separately for the first-layer circuit and for `attn2` shows which
component actually controls the prediction, as a function of how far away the
hijack target is.

Models used: `k32_N512`, meaning block size 32 (lists of 32 numbers) and vocab
512. Two layers, one head, 64-dim, pre-norm. Always the 100k-iteration
checkpoint. Seeds 1 through 25.

---

## 2. The figure to produce

Six panels in a 2x3 grid, one per gap, x-axis = offset, y-axis = hijack success
rate in percent. Three curves per panel (this is `--mode firstlayer`):

| curve | colour | meaning |
|---|---|---|
| ATTN1 direct circuit hijack | red | force `attn1` to the wrong key, recompute `mlp1`, replace both the `attn1` direct path and `mlp1` in the residual |
| ATTN2 hijack | blue | force `attn2` to the wrong key, `mlp1` untouched |
| ATTN1 direct + ATTN2 individually succeed | dashed purple | fraction where each hijack independently succeeds |

Each panel also draws faint per-seed traces and a mean +/- 1 std band, and
annotates the seed count in the lower right.

**Target offset ranges** (this is the refinement — the committed figure is much
coarser and stops much earlier):

| gap | target offsets | committed version has |
|---|---|---|
| 1 | 2–20, every integer | 2–15 |
| 5 | 6–30, every integer | 6–15 |
| 10 | 11–45, every integer | 11,13,15,17,19,21,25,30 (seeds 1–5 only) |
| 20 | 21–80, every integer | 21,25,30,35,40,45,50 (seeds 1–5 only) |
| 40 | 41–160, every integer | 41,45,50,55,60,70,80,100,120,150 (seeds 1–5 only) |
| 60 | 61–200, every integer | 61,65,70,80,90,100,120,150 |

Every range starts at `gap + 1`. This is required, not cosmetic — see section 7.

**Expected shape of the result**, useful as a sanity check. At gap 1 the two
curves cross: ATTN2 starts high (~72%) and decays with offset, ATTN1 rises from
~46% to a peak near 62% around offset 8–10, then declines. At gap 5 ATTN2
dominates throughout. From gap 10 upward ATTN2 is at or near saturation across
all offsets while ATTN1 and the joint curve sit near zero. If your gap 40 or 60
panel shows ATTN1 anywhere above a few percent, something is wrong.

---

## 3. Current state of the repository

**Scripts in `mechanistic-interpretability/role-of-position/`:**

| file | role |
|---|---|
| `plot_hijack_per_i.py` | the experiment. Runs hijacks, writes per-offset rates to JSON |
| `plot_hijack_avg_seeds.py` | the combiner. Reads per-seed JSON, plots the cross-seed mean |
| `classify_new_seeds.py` | labels each seed leap-former vs single-stage |
| `run_finegrained_seeds.sh` | **new** — corrected chunked sweep, use this |
| `merge_hijack_chunks.py` | **new** — merges chunk JSONs, audits the result |
| `run_all_seeds.sh`, `run_new_seeds.sh` | the old runners. **Do not use.** `run_new_seeds.sh` contains the bug in section 4 |

**Data directories:**

- `data_allI/`, `data_allI_v1_backup/` — 20 files each, original 5-seed runs, `mlp1` mode.
- `data_allI_v2/` — 150 files (25 seeds x 6 gaps), the `firstlayer` mode data
  behind the committed figure. Only 110 of the 150 contain measurements.
- `data_allI_v3/` — you will create this.

**Checkpoints are not in the repo.** `.gitignore` excludes `*.pt`, so the
`new-grid*/k32_N512/seed*/` directories contain only `logs/train.log`.

---

## 4. Bug 1: gaps 20 and 40 produced no data at all

In `run_new_seeds.sh`, gaps 10, 20 and 40 were launched with no offset flag:

```bash
if [ "$GAP" -le 5 ]; then
  OFFARGS="--fine-offsets"
elif [ "$GAP" -eq 60 ]; then
  OFFARGS="--offsets 61,65,70,80,90,100,120,150"
else
  OFFARGS=""          # <-- gaps 10, 20, 40
fi
```

With neither `--offsets` nor `--fine-offsets`, `plot_hijack_per_i.py` sets
`USE_RANDOM = False` and builds batches with `generate_gap_batch()`, which opens:

```python
max_start = vocab_n - GAP * block_size
if max_start <= 0:
    return None
```

For `k32_N512` that is `block_size = 32`, `vocab_n = 512`:

- gap 10 → 512 − 320 = 192, fine
- gap 20 → 512 − 640 = −128, **every batch returns `None`**
- gap 40 → 512 − 1280, **every batch returns `None`**

So all twenty seeds 6–25 wrote `"n_total": 0` with empty rate arrays for gaps 20
and 40. Gap 60 escaped only because it was given explicit `--offsets`, which
switches to `generate_random_batch()`.

**Fix:** always pass explicit `--offsets`. `run_finegrained_seeds.sh` does.

---

## 5. Bug 2: gap 10 data exists but on an incompatible grid

Gap 10 survived batch generation but, with no `--offsets`, fell back to the
default grid `[m * GAP for m in range(1, 8)]` = 10, 20, …, 70. Offset 10 is then
discarded (section 7), leaving `[20, 30, 40, 50, 60, 70]` for seeds 6–25 while
seeds 1–5 had been swept at `[11, 13, 15, 17, 19, 21, 25, 30]`.

`plot_hijack_avg_seeds.py` takes its x-axis from the first seed and looks every
other seed up **by exact offset value**, filling misses with `NaN`. Only offset
30 is common to both grids, so the gap 10 panel is a 5-seed average everywhere
except one point.

---

## 6. Bug 3: the seed count on the figure was wrong

The annotation was `len(seed_data)`, i.e. how many JSON files loaded, not how
many seeds contributed a number. Combined with bugs 1 and 2, the committed
figure claims "23 seeds" on all six panels while the reality is:

| gap | seeds actually averaged |
|---|---|
| 1 | 23 |
| 5 | 23 |
| 10 | 5, except 23 at offset 30 |
| 20 | 5 |
| 40 | 5 |
| 60 | 23 |

`plot_hijack_avg_seeds.py` has been fixed to count non-`NaN` contributors and
print a range like `5-23 seeds` when it varies. **If your regenerated figure
shows anything other than a flat 23 on every panel, the sweep is incomplete.**

---

## 7. Two hard constraints on the offset grid

**(a) An offset equal to the gap is always discarded.** In `collect_data()`:

```python
wval = cval + off
if wval < 0 or wval == tval or wval >= vocab_n or wval not in val_to_pos:
    continue
```

`tval` is `cval + GAP`, so `off == GAP` is skipped as a no-op. Start every range
at `gap + 1`.

**(b) At most ~28 offsets per run when gap >= 20.** For gap >= 20 with
`--offsets`, the script uses `generate_targeted_batch()`, which must pack
`cval`, `cval + gap`, and one token for every offset into a single `block_size =
32` sequence. Ask for more and it silently truncates its required set, so the
tail offsets are never sampled. Gaps 1, 5 and 10 use random batches and are not
affected.

This is why the large-gap sweeps are chunked:

| gap | chunks |
|---|---|
| 20 | 21-40, 41-60, 61-80 |
| 40 | 41-64, 65-88, 89-112, 113-136, 137-160 |
| 60 | 61-88, 89-116, 117-144, 145-172, 173-200 |

Chunks are merged afterwards by `merge_hijack_chunks.py`.

---

## 8. Seed exclusion

Seeds **8 and 10** are single-stage models and are excluded from the average.
Verified independently from the hijack data — at every gap they sit completely
outside the rest of the population:

| gap | seeds 8 & 10 | other 23 seeds |
|---|---|---|
| 1 | attn2 ~0%, first-layer ~100% | attn2 0–89%, first-layer 19–96% |
| 5 | attn2 0%, first-layer 100% | attn2 5–96%, first-layer 3–61% |
| 10 | attn2 0%, first-layer 100% | attn2 37–97%, first-layer 0.4–8.7% |
| 60 | attn2 ~0.1%, first-layer ~100% | attn2 99–100%, first-layer 0–2.2% |

The sweep deliberately **runs all 25 seeds** — the data is cheap and keeps the
classification re-derivable. Exclusion happens at plot time only.

Do not hardcode the pair. Run `classify_new_seeds.py --save-json` and pass the
result to the plotter with `--classification`; the criterion is per-token
accuracy with `attn2` ablated below 10%. Commit that JSON — the original
classification run left no record, which is why this had to be reverse-engineered.

---

## 9. Procedure

### Step 0 — environment

Requires a CUDA machine; `sortgpt_toolkit/model.py` picks the device and the
sweep assumes 8 GPUs by default (`NUM_GPUS` to change). Dependencies are pinned
at the repo root:

```bash
pip install -r requirements.txt        # torch, numpy, matplotlib
```

### Step 1 — code

```bash
git clone https://github.com/gatmiry/sort-llm.git
cd sort-llm
```

Confirm the four files below are present and are the corrected versions. If
`run_finegrained_seeds.sh` or `merge_hijack_chunks.py` are missing, the fixes
described in sections 4–7 have not been pushed and you must obtain them before
proceeding — **do not fall back to `run_new_seeds.sh`**, it is the buggy runner.

```bash
ls mechanistic-interpretability/role-of-position/{run_finegrained_seeds.sh,merge_hijack_chunks.py}
grep -c exclude-seeds mechanistic-interpretability/role-of-position/plot_hijack_avg_seeds.py   # expect 1
grep -c save-json     mechanistic-interpretability/role-of-position/classify_new_seeds.py      # expect 1
```

### Step 2 — checkpoints

All 25 are public at
<https://huggingface.co/gatmiry/sortgpt-checkpoints> under
`checkpoints/k32_N512/seed{n}/std0p01_iseed{n}__ckpt100000.pt`, about 0.7 MB
each, ~18 MB total. The repo has 20 checkpoints per seed (5k–100k in 5k steps);
**only the 100k one is used**.

The scripts expect four different local trees depending on seed number:

| seeds | path |
|---|---|
| 1 | `new-grid/k32_N512/checkpoints/` |
| 2–5 | `new-grid-multiple/k32_N512/seed{n}/checkpoints/` |
| 6–15 | `new-grid-multiple-2/k32_N512/seed{n}/checkpoints/` |
| 16–25 | `new-grid-multiple-3/k32_N512/seed{n}/checkpoints/` |

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

def dest(s):
    if s == 1:    return 'new-grid/k32_N512/checkpoints'
    if s <= 5:    return f'new-grid-multiple/k32_N512/seed{s}/checkpoints'
    if s <= 15:   return f'new-grid-multiple-2/k32_N512/seed{s}/checkpoints'
    return f'new-grid-multiple-3/k32_N512/seed{s}/checkpoints'

for s in range(1, 26):
    fn = f'std0p01_iseed{s}__ckpt100000.pt'
    src = hf_hub_download('gatmiry/sortgpt-checkpoints',
                          f'checkpoints/k32_N512/seed{s}/{fn}')
    os.makedirs(dest(s), exist_ok=True)
    shutil.copy(src, os.path.join(dest(s), fn))
    print('ok', dest(s), fn)
EOF
```

Verify 25 files landed:

```bash
find new-grid* -name 'std0p01_iseed*__ckpt100000.pt' | wc -l   # expect 25
```

### Step 3 — sweep

```bash
bash mechanistic-interpretability/role-of-position/run_finegrained_seeds.sh
```

500 jobs (25 seeds x 20 chunk-jobs) across 8 GPUs, versus 120 for the old coarse
run, which finished in about 13 minutes — though a third of those jobs were the
broken gap 20/40 ones that exited immediately without doing work. The new sweep
has far more (i, offset) cells to fill and higher batch caps, so plan for hours
rather than minutes. The script:

- skips outputs that already exist, so it is **resumable** — just re-run it
- warns and skips a seed whose checkpoint is missing rather than dying
- writes per-job logs to `data_allI_v3/logs/`

Tunable via environment: `NUM_GPUS`, `MB_RANDOM` (default 60000),
`MB_TARGETED` (default 40000), `SEED_LIST`, `DATADIR`.

While it runs, spot-check that files are non-empty:

```bash
python - <<'EOF'
import json, glob
bad = [p for p in glob.glob('mechanistic-interpretability/role-of-position/data_allI_v3/*.json')
       if not json.load(open(p)).get('attn2', {}).get('rates')]
print(len(bad), 'empty files'); print(*bad[:10], sep='\n')
EOF
```

Any empty file means the offsets for that job were rejected — investigate before
continuing, do not just re-run.

### Step 4 — merge

```bash
python mechanistic-interpretability/role-of-position/merge_hijack_chunks.py \
  --datadir mechanistic-interpretability/role-of-position/data_allI_v3
```

This writes `seed{n}_gap{g}.json` per seed and gap, then prints a grid audit.
**Every gap must report `OK`.** A `MISMATCH` means seeds were swept on different
offset grids and the figure will silently under-average, exactly as in bug 2.

### Step 5 — classify

```bash
cd mechanistic-interpretability/role-of-position
python classify_new_seeds.py --seeds 1-25 --save-json leapformer_classification.json
```

Expect seeds 8 and 10 to come out `SINGLE` and the other 23 `LEAP`. If the set
differs from `{8, 10}`, stop and report it — that would be a real finding, not a
config error, and it changes the paper's numbers.

### Step 6 — plot

```bash
python plot_hijack_avg_seeds.py --mode firstlayer \
  --datadir data_allI_v3 \
  --classification leapformer_classification.json \
  --out-suffix _v3
```

Produces `plots/hijack_allI_avg_seeds_fl_v3.png`.

Optionally also produce the `mlp1` variant with `--mode mlp1`, which swaps the
red curve for "MLP1 hijack" (first-layer direct path left intact).

---

## 10. Acceptance checklist

- [ ] 25 checkpoint files present under `new-grid*`
- [ ] `data_allI_v3/` has no JSON with an empty `attn2.rates`
- [ ] merge grid audit prints `OK` for all six gaps
- [ ] classification labels exactly seeds 8 and 10 as `SINGLE`
- [ ] every panel of the new figure annotates a flat `23 seeds`, no range
- [ ] x-axes reach 20 / 30 / 45 / 80 / 160 / 200 for gaps 1 / 5 / 10 / 20 / 40 / 60
- [ ] gap 1 shows the ATTN1-vs-ATTN2 crossover; gaps 40 and 60 show ATTN2 saturated and ATTN1 near zero

---

## 11. What to commit

The previous round committed only the PNG, and the extended-offset rerun was
lost entirely — it exists nowhere in git history, on any branch, in any dangling
object, or on Hugging Face. Avoid repeating that.

Commit all of:

- `data_allI_v3/*.json` (roughly a few hundred KB, safe to commit; the chunk
  files and `logs/` can be dropped once merged)
- `leapformer_classification.json`
- `plots/hijack_allI_avg_seeds_fl_v3.png`
- `run_finegrained_seeds.sh`, `merge_hijack_chunks.py` and the edits to
  `plot_hijack_avg_seeds.py` / `classify_new_seeds.py` if not already in

Then push to `origin/main`.

---

## 12. Downstream: the paper

The paper is a **separate** Overleaf git repository, not part of this repo:

- remote `https://git.overleaf.com/69c9a928b8ca815361b30519`
- main file `neurips_2026.tex`
- `newpics/` holds PNG figures, `figures/` holds PDFs

Section LaTeX lives in this repo as `paper-addon.tex` inside each analysis
directory, and those files reference figures by `newpics/...` path.

Note that `hijack_allI_avg_seeds_fl.png` is **not currently referenced by any
`.tex` file**. The 15 figures presently cited from `newpics/` are
`argmax_bias_analysis`, `argmax_saturation`, `attn2_accuracy_normal_vs_mlp1only`,
`attn_error_rates`, `attn_spread_comparison`, `l1_vs_l2_qk_smoothness`,
`no_attn2_acc_by_gap`, `probl1distance_all_leapformers`, `qk_heatmap_asymmetry`,
`qk_heatmap_split_comparison`, `qk_score_slices_band`,
`qk_slices_split_{2,3,4}tokens` and `residual_stream_cleanup`.

So wiring the hijack figure into the paper is a separate, still-open task. Do
not assume it has a home in the document yet.
