# Repeated development benchmark

7,500 optimizer updates per run; seeds 42, 43, 44, 45, 46.
All metrics use the same 1,000-image development set. The final set is untouched.

| Method | Seed | Overall | Day | Night | Rain |
|---|---:|---:|---:|---:|---:|
| generic_coverage | 42 | 0.2446 | 0.2529 | 0.2295 | 0.2612 |
| generic_coverage | 43 | 0.2411 | 0.2494 | 0.2272 | 0.2418 |
| generic_coverage | 44 | 0.2422 | 0.2529 | 0.2287 | 0.2520 |
| generic_coverage | 45 | 0.2263 | 0.2366 | 0.2118 | 0.2349 |
| generic_coverage | 46 | 0.2225 | 0.2360 | 0.1933 | 0.2172 |
| **generic_coverage mean** | — | **0.2353** | **0.2456** | **0.2181** | **0.2414** |
| adaptive_target_aware | 42 | 0.2406 | 0.2437 | 0.2445 | 0.2404 |
| adaptive_target_aware | 43 | 0.2420 | 0.2477 | 0.2402 | 0.2288 |
| adaptive_target_aware | 44 | 0.2413 | 0.2481 | 0.2404 | 0.2161 |
| adaptive_target_aware | 45 | 0.2428 | 0.2473 | 0.2442 | 0.2263 |
| adaptive_target_aware | 46 | 0.2396 | 0.2440 | 0.2396 | 0.2435 |
| **adaptive_target_aware mean** | — | **0.2413** | **0.2462** | **0.2418** | **0.2310** |

Paired difference (adaptive_target_aware minus generic_coverage):

| Seed | Overall | Day | Night | Rain |
|---:|---:|---:|---:|---:|
| 42 | -0.0039 | -0.0092 | +0.0150 | -0.0209 |
| 43 | +0.0009 | -0.0017 | +0.0130 | -0.0130 |
| 44 | -0.0009 | -0.0048 | +0.0117 | -0.0359 |
| 45 | +0.0165 | +0.0107 | +0.0324 | -0.0087 |
| 46 | +0.0171 | +0.0080 | +0.0463 | +0.0263 |
| **Mean difference** | **+0.0059** | **+0.0006** | **+0.0237** | **-0.0104** |

5 seeds support descriptive comparisons; uncertainty is limited. The rainy
development domain contains 78 images; uncertainty is not estimated.
Final evaluation must wait until the method and analysis plan are frozen.
