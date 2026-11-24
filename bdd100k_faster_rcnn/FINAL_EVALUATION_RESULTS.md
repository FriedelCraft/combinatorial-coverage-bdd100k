# Frozen final evaluation

Five paired seeds (42–46), 7,500 optimizer updates and 7,500 selected training images per run.
Metrics are COCO mAP@0.50:0.95 on the same 8,500 held-out images.
The primary endpoint is the mean paired night difference (adaptive minus generic).

| Method | Seed | Overall | Day | Night | Rain |
|---|---:|---:|---:|---:|---:|
| generic_coverage | 42 | 0.2244 | 0.2340 | 0.2050 | 0.2551 |
| generic_coverage | 43 | 0.2241 | 0.2337 | 0.2047 | 0.2569 |
| generic_coverage | 44 | 0.2250 | 0.2347 | 0.2046 | 0.2598 |
| generic_coverage | 45 | 0.2127 | 0.2224 | 0.1946 | 0.2396 |
| generic_coverage | 46 | 0.2130 | 0.2221 | 0.1914 | 0.2411 |
| **generic_coverage mean** | — | 0.2198 | 0.2294 | 0.2001 | 0.2505 |
| adaptive_target_aware | 42 | 0.2234 | 0.2276 | 0.2196 | 0.2565 |
| adaptive_target_aware | 43 | 0.2225 | 0.2287 | 0.2143 | 0.2572 |
| adaptive_target_aware | 44 | 0.2231 | 0.2290 | 0.2139 | 0.2540 |
| adaptive_target_aware | 45 | 0.2234 | 0.2282 | 0.2155 | 0.2532 |
| adaptive_target_aware | 46 | 0.2230 | 0.2282 | 0.2154 | 0.2622 |
| **adaptive_target_aware mean** | — | 0.2231 | 0.2283 | 0.2158 | 0.2566 |

## Paired differences

| Seed | Overall | Day | Night | Rain |
|---:|---:|---:|---:|---:|
| 42 | -0.0010 | -0.0064 | +0.0147 | +0.0014 |
| 43 | -0.0017 | -0.0050 | +0.0096 | +0.0002 |
| 44 | -0.0019 | -0.0057 | +0.0093 | -0.0058 |
| 45 | +0.0107 | +0.0058 | +0.0209 | +0.0136 |
| 46 | +0.0100 | +0.0062 | +0.0240 | +0.0211 |
| **Mean** | **+0.0032** | **-0.0010** | **+0.0157** | **+0.0061** |

## Exploratory uncertainty

Two-sided 95% paired t intervals across five seeds (df=4). These describe run-to-run variation;
five seeds are too few for strong distributional claims. Images within a seed are shared,
so this interval does not estimate uncertainty from sampling new images or datasets.

| Domain | Mean difference | 95% paired t interval | Positive pairs |
|---|---:|---:|---:|
| night | +0.0157 | [+0.0075, +0.0239] | 5/5 |
| all | +0.0032 | [-0.0049, +0.0113] | 2/5 |
| daytime | -0.0010 | [-0.0090, +0.0070] | 2/5 |
| rainy | +0.0061 | [-0.0075, +0.0197] | 4/5 |

Rain is a secondary slice (532 held-out images) and was not the selection target.
The protocol and endpoint were frozen before final scoring; see `FINAL_EVALUATION_PROTOCOL.md`.
