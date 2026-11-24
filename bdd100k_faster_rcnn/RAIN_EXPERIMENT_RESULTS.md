# Rain target experiment: exploratory results

This second-target experiment reuses a final partition whose rainy baseline
scores were already viewed during the night study. It is exploratory.
The protocol is in `RAIN_EXPERIMENT_PROTOCOL.md`.

| Method | Seed | Rain mAP | Overall mAP | Day mAP | Night mAP |
|---|---:|---:|---:|---:|---:|
| generic_coverage | 42 | 0.2551 | 0.2244 | 0.2340 | 0.2050 |
| generic_coverage | 43 | 0.2569 | 0.2241 | 0.2337 | 0.2047 |
| generic_coverage | 44 | 0.2598 | 0.2250 | 0.2347 | 0.2046 |
| generic_coverage | 45 | 0.2396 | 0.2127 | 0.2224 | 0.1946 |
| generic_coverage | 46 | 0.2411 | 0.2130 | 0.2221 | 0.1914 |
| **generic_coverage mean** | — | **0.2505** | **0.2198** | **0.2294** | **0.2001** |
| rain_count_control | 42 | 0.2597 | 0.2224 | 0.2304 | 0.2067 |
| rain_count_control | 43 | 0.2537 | 0.2220 | 0.2313 | 0.2004 |
| rain_count_control | 44 | 0.2455 | 0.2087 | 0.2165 | 0.1924 |
| rain_count_control | 45 | 0.2526 | 0.2208 | 0.2289 | 0.2030 |
| rain_count_control | 46 | 0.2644 | 0.2227 | 0.2317 | 0.2026 |
| **rain_count_control mean** | — | **0.2552** | **0.2193** | **0.2278** | **0.2010** |
| adaptive_target_aware | 42 | 0.2368 | 0.2063 | 0.2152 | 0.1845 |
| adaptive_target_aware | 43 | 0.2559 | 0.2214 | 0.2302 | 0.2028 |
| adaptive_target_aware | 44 | 0.2567 | 0.2216 | 0.2306 | 0.2027 |
| adaptive_target_aware | 45 | 0.2550 | 0.2222 | 0.2295 | 0.2074 |
| adaptive_target_aware | 46 | 0.2654 | 0.2234 | 0.2330 | 0.2013 |
| **adaptive_target_aware mean** | — | **0.2539** | **0.2190** | **0.2277** | **0.1997** |

## adaptive_target_aware minus generic_coverage

| Domain | Mean paired difference | Exploratory 95% paired t interval | Positive pairs |
|---|---:|---:|---:|
| rainy | +0.0034 | [-0.0173, +0.0241] | 2/5 |
| all | -0.0009 | [-0.0153, +0.0136] | 2/5 |
| daytime | -0.0017 | [-0.0160, +0.0127] | 2/5 |
| night | -0.0003 | [-0.0166, +0.0159] | 2/5 |

Rainy paired differences by seed: 42: -0.0184, 43: -0.0010, 44: -0.0032, 45: +0.0154, 46: +0.0242.

## rain_count_control minus generic_coverage

| Domain | Mean paired difference | Exploratory 95% paired t interval | Positive pairs |
|---|---:|---:|---:|
| rainy | +0.0047 | [-0.0133, +0.0226] | 3/5 |
| all | -0.0005 | [-0.0135, +0.0124] | 2/5 |
| daytime | -0.0016 | [-0.0151, +0.0118] | 2/5 |
| night | +0.0010 | [-0.0108, +0.0128] | 3/5 |

Rainy paired differences by seed: 42: +0.0045, 43: -0.0032, 44: -0.0143, 45: +0.0130, 46: +0.0233.

## adaptive_target_aware minus rain_count_control

| Domain | Mean paired difference | Exploratory 95% paired t interval | Positive pairs |
|---|---:|---:|---:|
| rainy | -0.0012 | [-0.0171, +0.0146] | 4/5 |
| all | -0.0003 | [-0.0132, +0.0125] | 3/5 |
| daytime | -0.0000 | [-0.0130, +0.0129] | 3/5 |
| night | -0.0013 | [-0.0167, +0.0141] | 3/5 |

Rainy paired differences by seed: 42: -0.0229, 43: +0.0022, 44: +0.0111, 45: +0.0024, 46: +0.0010.

The intervals measure training-seed variation on a fixed evaluation set.
They do not account for a new sample of rainy driving scenes. The rainy
development slice contains 78 images and the final rainy slice 532 images.
