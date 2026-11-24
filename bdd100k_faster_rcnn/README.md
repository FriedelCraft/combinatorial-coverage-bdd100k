# BDD100K Faster R-CNN (ResNet-50 + FPN)

## Research baseline (Phase 1)

The existing 10-epoch and day/balanced/night subset results are historical
references. Their metrics are preserved under `outputs/`, but they are not
budget-matched evidence for the new target-aware selection study. See
`RESEARCH_BASELINE.md` for the experimental contract and recorded results.

For new comparisons, use the same `--max-steps`, `--lr-step-steps`, batch size,
seed, model, augmentation, and validation split for every selection method.
Each run writes `run_config.json`, including the manifest SHA-256 digest.
For example (choose a step budget after a GPU smoke run):

```powershell
python train.py --project-root .. --train-manifest outputs/subsets/balanced.csv `
  --output-dir outputs/experiments/fixed_balanced_seed42 `
  --max-steps 10000 --lr-step-steps 3000 --batch-size 2 --seed 42
```

The learning-rate schedule is in optimizer steps in this mode. The last
partial pass through the subset stops exactly at the requested step budget.
Epoch-based training remains available for reproducing the older runs.

Build the per-image metadata table used by the research selector:

```powershell
.\.venv\Scripts\python.exe build_master_table.py --project-root .. --split train
.\.venv\Scripts\python.exe build_master_table.py --project-root .. --split val
```

Freeze the research query, development, and final evaluation partitions:

```powershell
.\.venv\Scripts\python.exe make_research_splits.py
```

For research runs, train with `--val-manifest research_splits/development.csv`.
Evaluate checkpoints with `--val-manifest research_splits/final_evaluation.csv` in `evaluate.py`
or `evaluate_combinations.py`. The two query manifests contain only image
names and are for building target representations; never use their object
annotations for selection. Never tune checkpoints or method parameters on
the final evaluation set.

Build equal-size prototype selection baselines and run the matched benchmark:

```powershell
.\.venv\Scripts\python.exe select_subsets.py --budget 7500 --target night
.\run_research_benchmark.ps1
```

The benchmark is a long GPU job. For a workflow check, use
`.\run_research_benchmark.ps1 -Steps 1000 -EvalEverySteps 500 -LrStepSteps 400`.
Short runs are not publication-quality comparisons.

To construct the visual target signal, extract ResNet-50 features from the
training pool and the image-only query manifest, then rerun selection:

```powershell
.\.venv\Scripts\python.exe extract_features.py --project-root .. --split train `
  --output-prefix outputs/features/train_resnet50
.\.venv\Scripts\python.exe extract_features.py --project-root .. --split val `
  --manifest research_splits/query_night.csv --output-prefix outputs/features/query_night
.\.venv\Scripts\python.exe select_subsets.py --budget 7500 --target night `
  --train-features outputs/features/train_resnet50 `
  --query-features outputs/features/query_night
```

This is a deliberately small, readable baseline for training Faster R-CNN on
all 70,000 BDD100K training images. It uses TorchVision's pretrained
Faster R-CNN model, whose backbone is ResNet-50 plus a Feature Pyramid Network
(FPN), and replaces the final classifier with the 10 BDD100K object classes.

## 1. Project layout

Put this code folder beside your extracted dataset folders:

```text
SOP/
├── bdd100k_faster_rcnn/
├── bdd100k_images_100k/
│   └── 100k/
│       ├── train/                 # 70,000 JPG files
│       ├── val/                   # 10,000 JPG files
│       └── test/
└── bdd100k_labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

The loader also understands the layout you started coding earlier:

```text
bdd100k_labels/100k/train/<image-name>.json
bdd100k_labels/100k/val/<image-name>.json
```

If your JSON files are elsewhere, pass their exact paths with
`--train-labels` and `--val-labels`.

## 2. Create the environment

Open PowerShell in `bdd100k_faster_rcnn`. A project-local Python 3.11
environment is now available at `.venv`; the old Conda `sop` environment is
empty. Use `.\.venv\Scripts\python.exe` for commands. To recreate the
environment, install the CUDA build of PyTorch using the current official
PyTorch selector, then install the remaining project packages:

```powershell
pip install -r requirements.txt
```

Confirm that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

A CPU is enough for checking the dataset and understanding the code, but a full
70,000-image Faster R-CNN run is realistically a GPU workload.

## 3. Check the data first

From inside `bdd100k_faster_rcnn`, use `..` as the project root because the
dataset folders sit one level above the code folder:

```powershell
python check_dataset.py --project-root .. --count 3
```

You should see image tensor shapes, box tensor shapes, and class names. Fix any
path error here before training.

## 4. Run a tiny smoke test

Do not begin with a multi-day run. First prove that data loading, forward pass,
backpropagation, validation, and checkpoint saving all work:

```powershell
python train.py --project-root .. --epochs 1 --max-train-images 20 --max-val-images 10
```

Expected outputs:

```text
outputs/last_checkpoint.pth
outputs/best_checkpoint.pth
outputs/history.csv
```

## 5. Train on the entire dataset

Remove the two `--max-...` options to use all 70,000 training images and all
10,000 validation images:

```powershell
python train.py --project-root .. --epochs 10 --batch-size 2
```

The important defaults are intentionally kept in `train.py` so you can find
and change them easily:

- Optimizer: SGD
- Learning rate: `0.005`
- Momentum: `0.9`
- Weight decay: `0.0005`
- LR schedule: multiply the LR by `0.1` every 3 epochs
- Input resize inside Faster R-CNN: minimum side `640`, maximum side `1024`
- Augmentation: 50% chance of a horizontal flip
- Mixed precision: BF16 on a compatible NVIDIA GPU; use `--no-amp` otherwise
- DataLoader workers: `0`, which is the safest starting point on Windows

If CUDA runs out of memory, use batch size 1:

```powershell
python train.py --project-root .. --epochs 10 --batch-size 1
```

If you have enough RAM and loading is slow, try `--workers 2`. With the large
single-file JSON label format, Windows may use much more RAM for extra workers,
so start at zero.

## 6. Resume an interrupted run

`last_checkpoint.pth` stores the model, optimizer, scheduler, and completed
epoch. To continue toward a total of 10 epochs:

```powershell
python train.py --project-root .. --epochs 10 --resume outputs/last_checkpoint.pth
```

## 7. Evaluate precision, recall, and F1

This baseline uses class-aware matching at IoU 0.5. A prediction counts as a
true positive only when it has the correct class, its confidence is at least
0.5, and it overlaps an unmatched ground-truth box by at least 0.5 IoU.

```powershell
python evaluate.py --project-root .. --checkpoint outputs/best_checkpoint.pth
```

It prints per-class values and an overall micro-average. This is intentionally
separate from the later covered-versus-uncovered research evaluation.

## 8. Draw predictions on one image

```powershell
python predict.py `
  --image ..\bdd100k_images_100k\100k\val\b1c66a42-6f7d68ca.jpg `
  --checkpoint outputs\best_checkpoint.pth `
  --output outputs\prediction.jpg
```

Replace the sample filename with a real validation image on your computer.

## Research results

The completed five-seed, fixed-budget study is written up in
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md). Exact final metrics and paired
intervals are in [`FINAL_EVALUATION_RESULTS.md`](FINAL_EVALUATION_RESULTS.md),
and the frozen analysis plan is in
[`FINAL_EVALUATION_PROTOCOL.md`](FINAL_EVALUATION_PROTOCOL.md). The main figure
is [`figures/final_paired_comparison.png`](figures/final_paired_comparison.png),
with a PDF copy for presentations. Recompute the report table with
`python summarize_final_evaluation.py` and the figure with
`python plot_final_evaluation.py` in an environment with Matplotlib.

### Second target: rain

The exploratory rain study is defined in
[`RAIN_EXPERIMENT_PROTOCOL.md`](RAIN_EXPERIMENT_PROTOCOL.md). It uses the same
7,500-image and 7,500-update budget as the night study. Five adaptive rain
subsets and five generic-coverage controls with exactly the same rainy-image
count are recorded under `research_subsets/rain_seed*/`. The existing generic
coverage runs serve as the paired baseline because their selector does not
depend on the target condition. All ten rain-targeted training and final
evaluation runs are complete. Their per-seed scores and paired intervals are
in [`RAIN_EXPERIMENT_RESULTS.md`](RAIN_EXPERIMENT_RESULTS.md); the interpretation
is in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md). Recompute the rain summary
with `python summarize_rain_experiment.py`.
The post-final class and subset audit is in
[`RAIN_CLASS_ERROR_ANALYSIS.md`](RAIN_CLASS_ERROR_ANALYSIS.md); regenerate it
with `python analyze_rain_classes.py`.
The inference-only seed-42 checkpoint and rainy time-of-day diagnostic is in
[`RAIN_SEED42_CHECKPOINT_DIAGNOSTIC.md`](RAIN_SEED42_CHECKPOINT_DIAGNOSTIC.md).
Its scores come from `evaluate_rain_time_diagnostic.py`; regenerate the write-up
with `python summarize_rain_time_diagnostic.py`.

Rain results on the existing final partition are exploratory: rainy baseline
scores from this partition were already seen during the earlier night study.

## What each file does

| File | Purpose |
|---|---|
| `src/config.py` | The 10 classes and their numeric IDs |
| `src/bdd_dataset.py` | Reads images, filters valid `box2d` labels, and creates targets |
| `src/model.py` | Loads Faster R-CNN ResNet-50/FPN and replaces its classifier |
| `src/utils.py` | Device, batching, seed, and checkpoint helpers |
| `train.py` | Training loop, validation loss, LR scheduling, and checkpoints |
| `evaluate.py` | Per-class and overall precision/recall/F1 |
| `predict.py` | Runs one image and draws predicted boxes |
| `check_dataset.py` | Quickly verifies the dataset before training |

## Two details that prevent common bugs

1. Class ID `0` is background, so the ten real classes use IDs 1 through 10.
2. Detection images cannot be stacked normally because each image may contain
   a different number of boxes. `collate_fn` therefore returns lists.

The model expects images as floating-point tensors in the range `[0, 1]` and
boxes in `[x1, y1, x2, y2]` format. The dataset class produces exactly that.
