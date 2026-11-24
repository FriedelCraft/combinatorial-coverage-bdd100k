# Repeated development benchmark

7,500 optimizer updates per run; seeds 42, 43, 44.
All metrics use the same 1,000-image development set. The final set is untouched.

| Method | Seed | Overall | Day | Night | Rain |
|---|---:|---:|---:|---:|---:|
| generic_coverage | 42 | 0.2446 | 0.2529 | 0.2295 | 0.2612 |
| generic_coverage | 43 | 0.2411 | 0.2494 | 0.2272 | 0.2418 |
| generic_coverage | 44 | 0.2422 | 0.2529 | 0.2287 | 0.2520 |
| **generic_coverage mean** | — | **0.2426** | **0.2517** | **0.2285** | **0.2517** |
| visual_cluster_aware | 42 | 0.2338 | 0.2406 | 0.2324 | 0.2350 |
| visual_cluster_aware | 43 | 0.2352 | 0.2428 | 0.2242 | 0.2298 |
| visual_cluster_aware | 44 | 0.2331 | 0.2382 | 0.2267 | 0.2303 |
| **visual_cluster_aware mean** | — | **0.2340** | **0.2405** | **0.2278** | **0.2317** |

Paired difference (visual_cluster_aware minus generic_coverage):

| Seed | Overall | Day | Night | Rain |
|---:|---:|---:|---:|---:|
| 42 | -0.0107 | -0.0123 | +0.0029 | -0.0262 |
| 43 | -0.0059 | -0.0066 | -0.0029 | -0.0121 |
| 44 | -0.0091 | -0.0147 | -0.0020 | -0.0217 |
| **Mean difference** | **-0.0086** | **-0.0112** | **-0.0007** | **-0.0200** |

Three seeds support descriptive comparisons only. The rainy
development domain contains 78 images; uncertainty is not estimated.
Final evaluation must wait until the method and analysis plan are frozen.
