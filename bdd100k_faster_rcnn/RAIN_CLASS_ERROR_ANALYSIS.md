# Rain experiment: class-level error analysis

This is a descriptive post-final analysis of the same previously viewed
532-image rainy final slice. It uses saved per-class AP and metadata;
image-level predictions were not saved, so it cannot identify individual
false positives or missed objects. No model was retrained or rerun.

## Rainy final-set AP by class

| Class | Final GT boxes | Generic AP | Count-control AP | Adaptive AP | Adaptive − control | Seed 42 adaptive − control |
|---|---:|---:|---:|---:|---:|---:|
| person | 623 | 0.2266 | 0.2315 | 0.2317 | +0.0002 | -0.0143 |
| car | 5014 | 0.4021 | 0.4063 | 0.4072 | +0.0008 | -0.0098 |
| rider | 27 | 0.2174 | 0.2278 | 0.1999 | -0.0279 | -0.0635 |
| truck | 262 | 0.3196 | 0.3258 | 0.3313 | +0.0054 | -0.0260 |
| bus | 95 | 0.3342 | 0.3235 | 0.3274 | +0.0039 | -0.0113 |
| train | 0 | — | — | — | — | — |
| motor | 22 | 0.1573 | 0.1692 | 0.1662 | -0.0030 | -0.0312 |
| bike | 39 | 0.1774 | 0.1858 | 0.1939 | +0.0080 | -0.0208 |
| traffic light | 1574 | 0.1610 | 0.1634 | 0.1644 | +0.0010 | -0.0134 |
| traffic sign | 1858 | 0.2591 | 0.2632 | 0.2634 | +0.0002 | -0.0157 |

AP is available for 9 classes with rainy final ground truth.
A class AP difference is descriptive; it does not establish why the model changed.
The box counts come from the filtered BDD100K metadata.

## Rainy training-box representation

Both rain-targeted methods select the same number of rainy images per seed.
The table compares boxes *within those rainy images*, not all selected boxes.

| Class | Development gap used | Mean count-control boxes | Mean adaptive boxes | Mean difference | Seed 42 difference | Final rainy images with class |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.0112 | 3871 | 5117 | +1246 | +1316 | 167 |
| car | 0.0171 | 26251 | 26883 | +632 | +579 | 525 |
| rider | 0.0000 | 196 | 196 | +0 | +0 | 24 |
| truck | 0.0122 | 1588 | 1882 | +294 | +296 | 160 |
| bus | 0.0382 | 789 | 908 | +119 | +119 | 78 |
| train | 0.0000 | 11 | 11 | +0 | +0 | 0 |
| motor | 0.0000 | 118 | 125 | +7 | +7 | 19 |
| bike | 0.0582 | 378 | 386 | +8 | +8 | 22 |
| traffic light | 0.0095 | 8776 | 10537 | +1761 | +1779 | 335 |
| traffic sign | 0.0184 | 9416 | 9974 | +558 | +589 | 426 |

## Per-seed context

| Seed | Adaptive rainy mAP − control | Adaptive rainy mAP − generic | Rainy images: generic / control / adaptive | Adaptive/control image overlap | Lowest development-loss step: adaptive / control |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.0229 | -0.0184 | 1455 / 2851 / 2851 | 6838/7500 | 3750 / 7500 |
| 43 | +0.0022 | -0.0010 | 1455 / 2851 / 2851 | 6838/7500 | 7500 / 7500 |
| 44 | +0.0111 | -0.0032 | 1455 / 2851 / 2851 | 6838/7500 | 7500 / 3750 |
| 45 | +0.0024 | +0.0154 | 1455 / 2851 / 2851 | 6838/7500 | 7500 / 7500 |
| 46 | +0.0010 | +0.0242 | 1455 / 2851 / 2851 | 6838/7500 | 7500 / 7500 |

The checkpoint step is selected by whole-development-set loss, not rainy AP.
Different checkpoints and selected image sets are both possible explanations
for a seed-level gap; this analysis cannot separate their effects.

## Interpretation

In seed 42, adaptive AP fell for 9/9 evaluable classes,
so the -0.0229 rainy mAP gap is broad rather than a single-class failure.
The largest class drop was rider (-0.0635 AP).
Across five seeds, rider was the largest negative class difference
(-0.0279 AP; 5/5 pairs negative), but it
has only 27 rainy final boxes across 24 images.
Its development AP gap g_c was zero, so it received no AP-gap uplift.
Its rainy training-box count was unchanged between methods; that
box-count measure does not explain the rider loss.

The adaptive and count-control manifests overlap by 6,838
of 7,500 images in each seed. Adaptive subsets also overlap seed 42 by
7,410–7,417 images across the other seeds.
Thus the five seeds mostly vary training randomness and a small
selection tie-break, not five substantially different data samples.
Despite having more rainy training boxes for most classes, the
adaptive subset did worse broadly in seed 42; raw box counts alone
do not explain the failure.

The seed-42 adaptive subset contains 1,513
rainy daytime and 1,011 rainy night
images; the count control contains 1,371
and 1,146, respectively.
This time-of-day shift and the seed-42 checkpoint difference are
plausible factors, but existing aggregate metrics cannot test either
mechanism. The available evidence supports a descriptive finding of
high run variability, not a causal explanation of seed 42.

A subsequent inference-only checkpoint and rainy time-of-day
comparison is reported in `RAIN_SEED42_CHECKPOINT_DIAGNOSTIC.md`.
