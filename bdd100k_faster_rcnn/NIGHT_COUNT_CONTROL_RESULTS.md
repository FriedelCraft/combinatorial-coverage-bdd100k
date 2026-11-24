# Exploratory night-count control results

This control was designed **after** the primary final evaluation was inspected.
It is a post-final ablation, not an independent confirmation of the method.
The protocol is in `NIGHT_COUNT_CONTROL_PROTOCOL.md`; all five training and
final evaluation runs completed.

Each control subset contains 7,500 training images, including exactly 3,851
night images, matching the adaptive subset for that seed. The control uses
generic metadata coverage weights and the same detector, training updates,
and checkpoint rule as the primary comparison.

## Held-out night COCO mAP@0.50:0.95

| Seed | Generic coverage | Night-count control | Adaptive | Control − generic | Adaptive − control |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.2050 | 0.2083 | 0.2196 | +0.0033 | +0.0113 |
| 43 | 0.2047 | 0.2110 | 0.2143 | +0.0063 | +0.0034 |
| 44 | 0.2046 | 0.2114 | 0.2139 | +0.0068 | +0.0025 |
| 45 | 0.1946 | 0.2163 | 0.2155 | +0.0217 | -0.0008 |
| 46 | 0.1914 | 0.2004 | 0.2154 | +0.0089 | +0.0151 |
| **Mean** | **0.2001** | **0.2095** | **0.2158** | **+0.0094** | **+0.0063** |

The 95% paired *t* interval for control minus generic is approximately
[+0.0005, +0.0183]. For adaptive minus control it is approximately
[-0.0019, +0.0145]. Both use only five seeds and the same fixed held-out
images; they do not include image-sampling uncertainty. The incremental
adaptive advantage over the matched-night-count control is positive in four
of five pairs but **not established clearly** by this experiment.

## Other domains

| Paired mean difference | Overall | Day | Rain |
|---|---:|---:|---:|
| Control − generic | -0.0011 | -0.0050 | +0.0008 |
| Adaptive − control | +0.0044 | +0.0040 | +0.0053 |

The control suggests that increasing the number of night training images
explains a substantial part of the primary night gain. The remaining
adaptive-minus-control difference is compatible with either a small added
benefit or run-to-run variation. Because this test was designed after the
primary results were seen, it should be treated as exploratory evidence.
