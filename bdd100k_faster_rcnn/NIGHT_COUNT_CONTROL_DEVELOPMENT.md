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
| night_count_control | 42 | 0.2412 | 0.2459 | 0.2370 | 0.2366 |
| night_count_control | 43 | 0.2367 | 0.2427 | 0.2311 | 0.2386 |
| night_count_control | 44 | 0.2381 | 0.2424 | 0.2367 | 0.2337 |
| night_count_control | 45 | 0.2393 | 0.2437 | 0.2354 | 0.2319 |
| night_count_control | 46 | 0.2254 | 0.2343 | 0.2159 | 0.2286 |
| **night_count_control mean** | — | **0.2361** | **0.2418** | **0.2312** | **0.2339** |

Paired difference (night_count_control minus generic_coverage):

| Seed | Overall | Day | Night | Rain |
|---:|---:|---:|---:|---:|
| 42 | -0.0033 | -0.0070 | +0.0075 | -0.0247 |
| 43 | -0.0044 | -0.0067 | +0.0039 | -0.0033 |
| 44 | -0.0041 | -0.0105 | +0.0080 | -0.0183 |
| 45 | +0.0130 | +0.0070 | +0.0235 | -0.0030 |
| 46 | +0.0029 | -0.0017 | +0.0226 | +0.0113 |
| **Mean difference** | **+0.0008** | **-0.0038** | **+0.0131** | **-0.0076** |

5 seeds support descriptive comparisons; uncertainty is limited. The rainy
development domain contains 78 images; uncertainty is not estimated.
Final evaluation must wait until the method and analysis plan are frozen.
