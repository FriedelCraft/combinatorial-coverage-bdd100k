# Matched 1,000-update development pilot

Run on 2026-09-28 with seed 42, batch size 2, COCO-pretrained Faster R-CNN,
BF16, the same optimizer schedule, and identical 1,000-image development set.
Each training manifest contains 7,500 images. Validation and checkpoint
selection occurred at 500 and 1,000 optimizer updates. LR decayed at updates
400 and 800. The final 8,500-image evaluation set was not scored.

| Method | Overall mAP | Day mAP | Night mAP | Rain mAP |
|---|---:|---:|---:|---:|
| Random | 0.1441 | 0.1531 | 0.1206 | 0.1364 |
| Target-only night | 0.1090 | 0.1109 | 0.1090 | 0.1007 |
| Manual 80% night | 0.1275 | 0.1332 | 0.1177 | 0.1143 |
| Generic coverage | **0.1806** | **0.1944** | **0.1502** | **0.1678** |
| Target-aware coverage | 0.1352 | 0.1433 | 0.1226 | 0.1319 |
| Visual target-aware coverage | 0.1664 | 0.1715 | **0.1658** | 0.1638 |

Generic coverage leads overall, daytime, and rainy mAP in this short pilot.
Visual target-aware coverage leads night mAP by 0.0156 over generic coverage,
while trailing it by 0.0142 overall. The metadata-only target-aware method is
only 0.0020 above random on night and below generic coverage by 0.0276.
Target-only and manual night-heavy selection also underperform generic coverage
at this early update count.

These numbers are **development diagnostics**, not research conclusions:
one seed, only 1,000 updates, and only about 2,000 training-image draws per
run. Most selected images were unseen by their detector. The rainy development
slice has only 78 images. No final-set claim or statistical significance is
supported. The visual method uses ResNet-50 features from the 400 night-query
images and all training images, without reading query bounding-box labels.
Its 7,500-image subset contains 4,238 night images. A longer matched
development run is needed before considering the final evaluation set.
