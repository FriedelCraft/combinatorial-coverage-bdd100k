# Seed 42 rain/time and checkpoint diagnostic

This is an exploratory **post-final** diagnostic on the already viewed
532-image rainy final slice. The original result used the checkpoint
with lower whole-development-set loss. This comparison does not change
that frozen checkpoint rule or any result in `RAIN_EXPERIMENT_RESULTS.md`.

| Rainy final slice | Images | Adaptive step 3,750 | Adaptive step 7,500 | Count control step 7,500 | Adaptive checkpoint gain | Adaptive step 7,500 − control |
|---|---:|---:|---:|---:|---:|---:|
| All rainy | 532 | 0.2368 | 0.2560 | 0.2597 | +0.0193 | -0.0036 |
| Rainy daytime | 292 | 0.2544 | 0.2773 | 0.2752 | +0.0230 | +0.0021 |
| Rainy night | 198 | 0.2430 | 0.2641 | 0.2847 | +0.0211 | -0.0206 |
| Rainy dawn/dusk | 42 | 0.2853 | 0.2886 | 0.2884 | +0.0033 | +0.0002 |

## What the comparison shows

The saved step-7,500 adaptive checkpoint raises seed-42 rainy mAP by +0.0193
compared with its selected step-3,750 checkpoint.
This closes 84% of the original 0.0229 gap to the
count control; 0.0036 remains.
The selected step-3,750 checkpoint had lower whole-development
loss (1.1121 versus 1.1243), which is why the frozen rule chose it.
Selecting step 7,500 now because of these final-set scores would
invalidate the original evaluation protocol.

At step 7,500, adaptive slightly exceeds the control on rainy
daytime images but remains lower on rainy night images. The
adaptive seed-42 training subset has 1,011 rainy night images
versus 1,146 in the count control, and 1,513 rainy daytime
images versus 1,371. This composition shift is compatible with
the nighttime gap, but these results cannot show that it caused it.
The rainy dawn/dusk slice has only 42 images.

Rider AP on all rainy images rises from 0.1873
to 0.2198 between adaptive checkpoints,
but the count control remains at 0.2508.
The class has only 27 ground-truth boxes in this rainy slice.

## Per-class rainy time-of-day AP

These are separate COCO AP calculations for each slice. Overall
rainy mAP is not the image-count-weighted average of the slices.

| Class | Day: adaptive 3,750 | Day: adaptive 7,500 | Day: control | Night: adaptive 3,750 | Night: adaptive 7,500 | Night: control |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.2509 | 0.2621 | 0.2693 | 0.1465 | 0.1519 | 0.1444 |
| car | 0.4257 | 0.4334 | 0.4326 | 0.3445 | 0.3627 | 0.3609 |
| rider | 0.2340 | 0.3110 | 0.3671 | 0.1459 | 0.1474 | 0.1921 |
| truck | 0.3219 | 0.3448 | 0.3406 | 0.3043 | 0.3387 | 0.3485 |
| bus | 0.2612 | 0.2767 | 0.2720 | 0.4454 | 0.4641 | 0.4738 |
| train | — | — | — | — | — | — |
| motor | 0.2356 | 0.2419 | 0.1953 | 0.1522 | 0.2166 | 0.2500 |
| bike | 0.1556 | 0.1832 | 0.1578 | 0.2506 | 0.2779 | 0.3704 |
| traffic light | 0.1741 | 0.1970 | 0.1951 | 0.1200 | 0.1251 | 0.1278 |
| traffic sign | 0.2304 | 0.2458 | 0.2470 | 0.2778 | 0.2928 | 0.2943 |

The first and third `All rainy` scores reproduce the stored
final-evaluation mAP within 0.00001. Evaluation code:
`evaluate_rain_time_diagnostic.py`; metric CSV:
`outputs/analysis/rain_seed42_checkpoint_diagnostic/rain_time_checkpoint_metrics.csv`.
No training was performed.
