# Second-target rain experiment: prospective analysis record

Recorded before building rain-adaptive subsets or training any rain-targeted detector.
The existing final set's rainy baseline scores have already been inspected in
the night study. This is therefore an **exploratory second-target test on a
previously viewed final set**, not a fresh confirmatory test.

## Target and data

- Training pool: 70,000 BDD100K train images, including 5,083 marked rainy.
- Development: the existing disjoint 1,000-image development partition, with
  78 rainy images. These labels may guide weighting and checkpoint selection.
- Scoring: the existing 8,500-image final partition, with 532 rainy images.
  The rain query partition is excluded from training and scoring. Query pixels
  are not used by the adaptive metadata selector.

## Fixed method

Use the **same square-root token coverage selector** as the night study, with
target membership changed from `timeofday=night` to `weather=rainy`.
For rainy images containing class c, add `target_class=c`. Set its weight to
`f_target,c^(-1/4) * (1 + 12 g_c)`, where `g_c` is the positive part of the
mean *nonrain minus rainy* per-class development AP gap across completed
generic-coverage seeds 42, 43, 44. A class absent from either development
domain has gap zero. Keep the coefficient **12 unchanged**; it is the heuristic
value used in the night study, not a rain-tuned optimum.

Reuse the existing five generic-coverage 7,500-image manifests and completed
generic detector runs as the baseline: the generic selector is target-agnostic,
and those runs used the same training pool and fixed update budget. Their
prior rainy final scores are known, which is a limitation of this test.

For each seed 42-46, select a 7,500-image rain-adaptive subset. Also select a
7,500-image **rain-count control** by maximizing generic metadata coverage
subject to the exact rainy-image count of that seed's adaptive subset. The
control uses no class AP gaps. All methods use the same detector, batch size 2,
7,500 optimizer updates, learning-rate decays at 3,000 and 6,000, and the
lowest development-loss checkpoint from updates 3,750 and 7,500.

## Outcomes

- Main descriptive comparison: five paired differences in rainy COCO
  mAP@0.50:0.95, rain-adaptive minus generic coverage.
- Mechanism comparison: rain-adaptive minus rain-count control on rainy mAP.
- Report overall, daytime, and nighttime mAP changes as secondary outcomes.
- Report every seed, mean paired differences, and exploratory 95% paired t
  intervals. Do not choose a favorable seed or tune weights after scoring.

Interpretation is bounded by the 78-image rainy development slice, 532-image
rainy final slice, one detector/budget, and prior viewing of the final set.
The class-gap weighting may be noisy; matching rainy count probes whether a
gain is driven mainly by rainy-image representation.
