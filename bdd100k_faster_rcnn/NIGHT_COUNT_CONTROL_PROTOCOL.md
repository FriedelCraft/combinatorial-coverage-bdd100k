# Post-final night-count control

This ablation was designed **after** the five-seed primary final results were
inspected. It is exploratory and does not alter the frozen primary comparison
in `FINAL_EVALUATION_PROTOCOL.md`.

## Question

Does the adaptive method outperform generic metadata coverage because it
selects more night images, or does the development-gap weighting add something
beyond that night count?

## Control

For each seed 42–46, select 7,500 training images with exactly 3,851 night
images, matching that seed's adaptive manifest. Optimize the same weighted
square-root metadata coverage objective as generic coverage, subject to that
night quota. Use no per-class development AP gaps in the control selector.
Manifests and checksums are stored under
`research_subsets/night_count_control_seed*/`.

Train the same COCO-pretrained Faster R-CNN, 7,500 optimizer updates, batch 2,
learning-rate and checkpoint schedule used in the primary comparison. Score
the same 8,500 final images. The exploratory comparison is the paired
adaptive-minus-night-count-control night COCO mAP@0.50:0.95 across seeds 42–46.
Overall, day, and rain mAP are descriptive secondary outcomes.

The development results were viewed before this final control scoring:
night-count control minus generic coverage averaged +0.0131 night mAP on the
1,000-image development set. Adaptive minus generic coverage averaged +0.0237.
These observations motivated scoring the control but are not used to change
its manifests, trained checkpoints, or outcome definitions.
