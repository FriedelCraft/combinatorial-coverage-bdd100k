# Frozen final evaluation protocol

Frozen after the five-seed development comparison and before final-set scoring.

## Primary comparison

- Methods: generic coverage versus adaptive target-aware class-time coverage.
- Training subsets: 7,500 images selected separately at seeds 42–46.
- Detector and compute: COCO-pretrained Faster R-CNN ResNet-50-FPN, BF16,
  batch size 2, 7,500 optimizer updates, validation at 3,750 and 7,500,
  learning-rate decay at 3,000 and 6,000, same configuration for both methods.
- Checkpoint: lowest development loss among the two scheduled checks.
- Evaluation: the frozen 8,500-image `research_splits/final_evaluation.csv`.
- **Primary endpoint:** mean across five paired seeds of adaptive minus generic
  night-domain COCO mAP@0.50:0.95.
- **Retention endpoint:** mean across the same seeds of overall COCO mAP
  difference. Daytime and rainy mAP are reported as secondary trade-offs.
- Report each seed and mean paired difference. Do not select a favorable seed.
  With only five runs, treat intervals and tests as exploratory.

## Additional context

Random, target-only, and manual 80% night methods have one full-budget run at
seed 42. If scored on the final set, label those results single-seed context,
not a repeated statistical comparison. The older 10-epoch/full-70k results
used a different training budget and validation set and remain historical.

No method weights, subset manifests, model checkpoints, or outcome definitions
will be changed after final-set metrics are inspected. The final evaluation
images were excluded from query and development membership and were never
used in subset selection or checkpoint selection.
