# Phase 1: baseline freeze

## Existing pipeline

- Detector: TorchVision Faster R-CNN with ResNet-50-FPN and COCO detection weights;
  replace the predictor with ten BDD100K classes plus background.
- Images: BDD100K train and validation splits. Bounding boxes are clipped to
  image bounds; invalid and non-detection labels are excluded.
- Training: SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005,
  batch size 2, horizontal flip with probability 0.5, input min/max 640/1024.
- Research runtime: project-local Python 3.11 `.venv`, PyTorch 2.11.0+cu128,
  TorchVision 0.26.0+cu128, RTX 5050 Laptop GPU. BF16 autocast is used for
  CUDA training because FP16 overflow skipped all optimizer updates in the
  first smoke run.
- Legacy schedule: StepLR every three epochs, gamma 0.1. This schedule must
  not be used for cross-subset comparisons because subset sizes change the
  number of optimizer updates per epoch.
- Evaluation: keep the existing full validation and matched day/night strata
  scripts. Report mAP and the per-class/condition breakdown alongside overall
  results. A single seed is descriptive; later experiments need repeated seeds.

## Recorded historical results

From `outputs/experiment_comparison/comparison_summary.txt`:

| Training data | Overall validation mAP | Matched day mAP, four classes | Matched night mAP, four classes |
|---|---:|---:|---:|
| Full 70k | 0.2427 | 0.3172 | 0.2788 |
| Day-heavy | 0.1880 | 0.2711 | 0.2208 |
| Balanced | 0.1629 | 0.2564 | 0.2274 |
| Night-heavy | 0.1610 | 0.2425 | 0.2346 |

These results were generated before a common optimizer-step budget was
established. They are context, not a controlled comparison with the new method.
The five matched strata and one run per condition limit statistical claims.

## Contract for new selection experiments

1. Freeze train and validation membership. Selection uses training data only;
   the validation labels and metrics must not guide per-image selection.
2. Save each subset as a CSV manifest with `image_name`; record its size,
   construction seed, and checksum. Use the same subset size for all methods.
3. Use `--max-steps` and `--lr-step-steps` with identical values across methods.
   Keep batch size, optimizer, initialization, preprocessing, and augmentation
   identical. Store each run in a separate output directory.
4. Evaluate each checkpoint on the same full validation split and the same
   fixed target and matched-strata definitions. Report target gains and
   general-domain retention with uncertainty across seeds.
5. Do a small smoke run before selecting the full step budget based on actual
   GPU time. The full fixed-step baseline has not yet been run.

## GPU preflight, 2026-09-28

- CUDA available and RTX 5050 recognized.
- Dataset unit tests: 2 passed; two real training images loaded correctly.
- COCO-pretrained two-step fixed-budget run: completed and saved checkpoints.
- COCO-pretrained batch-size-2, four-step run: completed without out-of-memory.
- These tiny runs establish functionality, not a defensible full-run time
  estimate. Their validation subsets are too small to interpret as accuracy.

## Metadata audit, 2026-09-28

`build_master_table.py` produced `outputs/metadata/master_train.csv` and
`master_val.csv` with one row per image: weather, scene, time of day, image
dimensions, ten detection-class counts, and relative box-size counts. Summary
JSON files accompany each table.

- Train: 70,000 images, 0 missing labels, 1,288,010 valid detection boxes.
- Validation: 10,000 images, 0 missing labels, 185,526 valid detection boxes.
- Train time of day: 36,800 daytime; 28,028 night; 5,033 dawn/dusk;
  139 undefined.
- The box count agrees with the earlier approximately 1.288 million figure.

The next selection-method design must keep the final validation labels out of
image selection. If target-domain examples are needed to build features,
reserve a separate query set and an untouched evaluation set.

## Frozen research partitions, 2026-09-28

`make_research_splits.py` deterministically partitions the 10,000 validation
images. It writes image-name-only CSV manifests and SHA-256 checksums under
`research_splits/`:

| Role | Images | Use |
|---|---:|---|
| Night query | 400 | Pixels only, to define the night target |
| Rain query | 100 | Pixels only, to define a second target |
| Development | 1,000 | Validation loss and checkpoint/method development |
| Final evaluation | 8,500 | Report final full and domain-specific metrics |

The four groups are disjoint and exhaustive. The final set retains 3,126 night,
4,650 daytime, and 532 rainy images. All five previously used weather/scene
strata retain at least 138 night and 174 daytime images where applicable.
The earlier 10,000-image validation results remain historical and are not
directly comparable to metrics from this 8,500-image final set.

## Selection benchmark prototype, 2026-09-28

`select_subsets.py` created five deterministic 7,500-image training manifests
under `research_subsets/`: random, target-only night, manual 80% night,
generic metadata coverage, and target-aware metadata coverage. Each manifest
has a SHA-256 digest in `night_selection_summary_7500.json`.

The coverage methods use image metadata, class presence, and box-size presence
as tokens. For a selected set S, the prototype objective is a positive
weighted sum of square-root token counts plus a nonnegative modular target
relevance term. Because square root has diminishing increments, the objective
is monotone submodular. Lazy greedy therefore has the standard 1−1/e
cardinality-constraint approximation guarantee for this *specified objective*;
that guarantee does not imply a detector mAP guarantee.

The current target-aware method knows that the target is night and increases
weights for night-related tokens. It has not yet used query-image visual
embeddings, so it is a metadata-only prototype, not the final proposed method.
Its 7,500-image subset has 6,050 night, 706 daytime, 738 dawn/dusk, and 6
undefined-time images. The manual baseline has 6,000 night and 1,500 daytime.

The visual target-aware variant uses L2-normalized ImageNet-pretrained
ResNet-50 vectors from all 70,000 training images and the 400 night-query
images. It computes each training image's cosine similarity to the normalized
query mean, converts that similarity to a training-pool percentile, and adds
it as a nonnegative modular reward. This leaves the weighted square-root
coverage objective monotone submodular. Query labels and final-evaluation
images are never read by feature extraction. The visual subset contains
4,238 night, 1,621 daytime, 1,609 dawn/dusk, and 32 undefined-time images.

`run_research_benchmark.ps1` applies a shared optimizer-step budget and fixed
validation intervals. A short matched pilot is used to verify the workflow;
its accuracy is not evidence of a winning method. The full benchmark design
is 7,500 optimizer updates, evaluations every 3,750 updates, and LR decays
every 3,000 updates, with batch size 2 and seed 42. This is a design choice,
not yet a completed full benchmark.

The matched 1,000-update development pilot is summarized in `PILOT_RESULTS.md`.
Visual target-aware coverage improved night mAP over generic coverage, while
generic coverage retained higher overall mAP. The final evaluation partition
has not been scored for this pilot.

## Repository and data boundary

This folder is currently not a Git repository. The source archive and output
summaries remain in the workspace. Keep BDD100K images/labels, checkpoints,
feature caches, and large generated artifacts out of a future repository.
