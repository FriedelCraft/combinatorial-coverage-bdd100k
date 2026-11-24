# BDD100K post-midsem speaker notes

Slides 1–10 are the main talk. Slides 11–13 are for questions.

## 1. Target-aware image selection for object detection

I will briefly recap the night result that was already in my midsem report. The new work asks whether the same data selection rule helps in rain, and uses two diagnostics to explain what happened.

## 2. The nighttime result motivated the rain study

The midsem report already included the Faster R-CNN baseline, manual day/night mixtures, combinatorial selector, final night results and the night-count ablation. Night-adaptive improved by 1.57 points on a 0–100 mAP scale. The adaptive-minus-control interval was [−0.0019, +0.0145], so the extra benefit from class weighting was uncertain. This motivated testing the same rule for rain.


## 3. Training-image selection as a set problem

Think of a bipartite incidence relation between images and properties. The square root rewards the first occurrence more than the tenth. Use the appendix if asked for the proof or what the greedy guarantee applies to.

## 4. Rain adaptation and the matched control

The bar in the formula averages three generic development models, seeds 42 to 44. A class missing AP in either slice gets a zero gap if no paired AP is available. The factor 12 was carried over from development of the night method. It is a heuristic, with no claim of optimality. For a gap of 0.05 the token multiplier is 1.60. The quota control asks whether extra rainy images are enough to match the adaptive selector.

## 5. Evaluation protocol

The development data supported two decisions: estimate class AP gaps and pick the checkpoint by overall development loss. All three methods were compared on the same final images at each seed. COCO mAP averages class AP across IoU thresholds from 0.50 to 0.95. The five repeats measure variation across training runs on the same scoring images.

## 6. The rain study found no clear adaptive gain

Start with the last row. Both methods selected exactly 2,851 rainy images, and the adaptive mean was 0.12 points lower on the 0–100 scale. All differences use unrounded run values. Report this as inconclusive evidence of an added rain benefit, with five seeds and a fixed set of images.

## 7. Four seed pairs favored adaptive coverage

Pair by seed because both methods are evaluated with the same seed index on the same evaluation images. Show all five points and keep seed 42 in the mean. The subset overlap is about 91%. The adaptive selections also overlap heavily across seeds, so these are repeated training runs on closely related selections.

## 8. Seed 42 fell across all nine evaluable classes

All nine classes with rainy ground truth scored lower for the selected adaptive checkpoint. There were no train-class boxes on the rainy final slice. Rider had 27 boxes in 24 images and no positive development AP gap. Highlight the limited sample when interpreting the result. Saved class AP and metadata give aggregate patterns rather than a count of missed objects per image.

## 9. The later checkpoint recovered most of the gap

I scored the saved early and late adaptive checkpoints and the late count-control checkpoint. The early checkpoint remains the reported result because its development loss was 1.1121 versus 1.1243 at the later step. The later score is a post hoc diagnostic. The rain–night mix is a possible factor in the remaining gap, but image identities and training dynamics also differ.

## 10. What the night and rain experiments establish

My contribution after midsem was the transfer test to rain, the rain-count comparison, the class audit and a checkpoint diagnostic by time of day. State that the fresh holdout is future work. Two endpoints worth declaring are rainy mAP as primary and rainy nighttime mAP as secondary.

## 11. Why the coverage objective is submodular

Rationalize √(n+1)−√n to get the displayed expression. Nonnegative weights also give monotonicity, and the empty set has score zero. The bound applies to the surrogate F. Detector mAP is evaluated empirically.

## 12. Per-seed rain scores and uncertainty

Compute the paired difference within each seed using full-precision AP, then average and calculate the t interval with four degrees of freedom. The interval does not include uncertainty from sampling new evaluation scenes.

## 13. References and experiment records

Use this slide if asked about references or reproducibility. Training manifests, checkpoint histories, final metric files and the diagnostic scripts are available in the project folder.
