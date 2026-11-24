# Repeated development benchmark

7,500 optimizer updates per run; seeds 42, 43, 44.
All metrics use the same 1,000-image development set. The final set is untouched.

| Method | Seed | Overall | Day | Night | Rain |
|---|---:|---:|---:|---:|---:|
| generic_coverage | 42 | 0.2446 | 0.2529 | 0.2295 | 0.2612 |
| generic_coverage | 43 | 0.2411 | 0.2494 | 0.2272 | 0.2418 |
| generic_coverage | 44 | 0.2422 | 0.2529 | 0.2287 | 0.2520 |
| **generic_coverage mean** | — | **0.2426** | **0.2517** | **0.2285** | **0.2517** |
| visual_target_aware | 42 | 0.2252 | 0.2255 | 0.2310 | 0.2244 |
| visual_target_aware | 43 | 0.2217 | 0.2253 | 0.2150 | 0.2275 |
| visual_target_aware | 44 | 0.2265 | 0.2302 | 0.2272 | 0.2192 |
| **visual_target_aware mean** | — | **0.2245** | **0.2270** | **0.2244** | **0.2237** |

Paired difference (visual_target_aware minus generic_coverage):

| Seed | Overall | Day | Night | Rain |
|---:|---:|---:|---:|---:|
| 42 | -0.0193 | -0.0274 | +0.0015 | -0.0368 |
| 43 | -0.0194 | -0.0241 | -0.0122 | -0.0143 |
| 44 | -0.0157 | -0.0227 | -0.0015 | -0.0328 |
| **Mean difference** | **-0.0182** | **-0.0248** | **-0.0041** | **-0.0280** |

Three seeds support descriptive comparisons only. The rainy
development domain contains 78 images; uncertainty is not estimated.
Final evaluation must wait until the method and analysis plan are frozen.
