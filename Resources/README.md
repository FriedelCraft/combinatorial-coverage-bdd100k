Yo! Let's pull those exact specs straight from the text of the "Coverage for Identifying Critical Metadata in Machine Learning Operating Envelopes" paper. I'll break it down exactly how they did it, box by box, no guessing involved.

### Metadata Preprocessing

* **What exactly did the paper do here?** The researchers extracted metadata from the RarePlanes dataset, cross-referenced location and time data to derive new human-meaningful features (like finding the "Biome" using latitude/longitude), added a randomly assigned "Control" variable, and grouped continuous data into discrete bins.


* **What metadata?** They selected six features: Hour of Day, Off Nadir Max (camera angle), Average Pan Resolution, Season, Biome, and a Control variable.


* **Categorical or continuous?**
* Categorical: Season, Biome, and Control.


* Continuous: Hour of Day, Off Nadir Max, and Average Pan Resolution.




* **If continuous, how is it discretized?** The continuous factors were discretized into bins that span equal size ranges.


* **How many values/levels?** There are three 3-valued factors, one 4-valued factor, one 5-valued factor, and one 12-valued factor.



### Combinatorial Analysis

* **What exactly did the paper do here?** They built an experimental framework called "Systematic Inclusion & Exclusion". They trained 31 different ML models, systematically withholding one specific metadata interaction from each model's training data, and then evaluated them on "covered" and "uncovered" test sets to see which missing factors caused the model to fail.


* **What is t?** For this pilot experiment, the interaction strength was limited to t = 1.


* **What are the factors?** Hour of Day, Off Nadir Max, Average Pan Resolution, Season, Biome, and Control.


* **What are the levels?**
* **Hour of Day:** [10.26, 11.27), [11.27, 12.28), [12.28, 13.29), [13.29, 14.30), [14.3, 15.32].


* **Off Nadir Max:** [3, 12], (12, 21], (21, 30].


* **Average Pan Resolution:** (0.31, 0.3419495), [0.3419495, 0.3729495], (0.3729495, 0.41].


* **Season:** Spring, Summer, Autumn, Winter.


* **Biome:** Temperate Grasslands, Savannas & Shrublands; Temperate Broadleaf & Mixed Forests; Temperate Conifer Forests; Boreal Forests/Taiga; Flooded Grasslands & Savannas; Deserts & Xeric Shrublands; Mediterranean Forests, Woodlands & Scrub; Mangroves; Tropical & Subtropical Moist Broadleaf Forests; Tropical & Subtropical Coniferous Forests; Tropical & Subtropical Grasslands, Savannas & Shrublands; Tundra.


* **Control:** 1, 2, 3.




* **Are constraints used?** They didn't artificially inject constraints, but they noted that building the test sets was constrained by the natural dependencies across interactions within the preexisting dataset (e.g., a specific combination of time, angle, and biome might not exist in the data to be sampled).


* **Is CC used?** The paper evaluates training pools for the *concept* of coverage (making sure non-withheld interactions are covered), but it does not formally calculate or report a Combinatorial Coverage (CC) percentage metric.


* **Is SDCC used?** No, Set Difference Combinatorial Coverage (SDCC) is not mentioned or explicitly used in this specific paper.



Let me know if you need to rip the methodology out of any of the other papers!


| Category | Question | The Paper's Blueprint |
| :--- | :--- | :--- |
| **Dataset** | What dataset are we using? | RarePlanes, an open-source dataset of satellite imagery[cite: 4]. |
| **ML task** | Classification? Detection? Regression? | Object detection, specifically drawing bounding boxes around aircraft and classifying them[cite: 4]. |
| **Model** | What baseline model? | YOLOv8 nano, which was trained entirely from scratch without using any pre-trained weights[cite: 4]. |
| **Metadata** | Which variables describe operating conditions? | Hour of Day, Off Nadir Max (camera angle), Average Pan Resolution, Season, Biome, and a randomized Control variable[cite: 4]. |
| **Levels** | What values can each metadata variable take? | Hour of Day (5 binned values), Off Nadir Max (3 binned values), Average Pan Resolution (3 binned values), Season (4 categories), Biome (12 categories), and Control (3 categories)[cite: 4]. |
| **Strength** | 2-way? 3-way? Higher? | 1-way (t = 1) interaction strength, which they chose due to time constraints for this pilot study[cite: 4]. |
| **Coverage** | CC? SDCC? Both? | Neither formally. They used an experimental design called "Systematic Inclusion & Exclusion" to intentionally withhold interactions during data splitting, but they didn't calculate strict CC or SDCC percentage metrics[cite: 4]. |
| **Performance** | Accuracy? F1? mAP? RMSE? | Precision, Recall, and F1 score, all calculated equally at the individual object level[cite: 4]. |
| **Analysis** | How will coverage be related to performance? | They used a statistical method called weighted logistic regression to test if the performance drop between the "covered" and "uncovered" test sets was statistically significant for each withheld factor[cite: 4]. |



# SOP Research Roadmap
## BDD100K + Critical Metadata + ML Operating Envelope
### Research Topic
**Coverage for Identifying Critical Metadata in Machine Learning
Operating Envelopes**
### Planned Application
- **Domain:** Autonomous Driving
- **Dataset:** BDD100K
- **Primary Task:** Object Detection
- **Baseline Model:** Faster R-CNN
- **Possible Extension:** Semantic/instance segmentation after
the detection pipeline is working
---
# 1. Research Objective
The immediate objective is to investigate whether information
about the operating conditions of an ML system, represented
through metadata, can help identify conditions associated with
degraded model performance.
The initial experimental approach will adapt the **Systematic
Inclusion &amp; Exclusion** methodology from the reference paper to
BDD100K.
The basic idea is:
1. Identify metadata describing driving conditions.
2. Select a metadata factor and one of its levels.
3. Remove that condition from the model&#39;s training data.
4. Train the model without exposure to that condition.
5. Evaluate the model on:
- covered conditions
- the withheld/uncovered condition
6. Compare the performance.
7. Repeat for multiple metadata conditions.
8. Identify conditions whose absence from training is associated
with significant performance degradation.
---

# 2. Overall Research Pipeline
```text
BDD100K
↓
Dataset &amp; Annotation Exploration
↓
Identify Available Metadata
↓
Select Relevant Metadata Factors
↓
Analyze Metadata Distributions
↓
Design Systematic Exclusion Experiments
↓
Create Modified Training Sets
↓
Train Faster R-CNN
↓
Evaluate Covered vs Uncovered Conditions
↓
Measure Precision / Recall / F1
↓
Compare Performance
↓
Statistical Analysis
↓
Identify Critical Metadata
↓
Define Potential Operating-Envelope Regions
↓
Identify Research Gap / Extension
```
---
# 3. Week 1 — Dataset &amp; Literature Understanding
## Goal
Understand the reference methodology and determine exactly what
BDD100K can provide for the experiment.
### Tasks

* Read the 2021 combinatorial coverage paper.
* Read the 2024 critical metadata paper carefully.
* Understand:
* combinatorial testing
* t-way interactions
* CC
* SDCC
* metadata
* operating envelopes
* systematic inclusion &amp; exclusion
* Trace the most relevant follow-up work.
* Obtain and inspect BDD100K annotations.
* Understand the BDD100K data format.
* Identify all metadata available in the actual dataset.
* Count samples for each metadata value.
* Check missing values and heavily underrepresented categories.
### Important
Do not assume that BDD100K has the same metadata as the
RarePlanes experiment.
Use only metadata that actually exists in the BDD100K version
being used.
### Week 1 Deliverable
Create:
## `dataset_metadata_report.md`
Containing:
```text
Dataset:
Number of images:
Object classes:
Available metadata:
Metadata types:
Metadata levels:
Samples per level:
Missing values:
Potential metadata factors:
```

Also create:
## `experimental_design_v1.md`
Containing:
* selected metadata
* reason for selecting each factor
* proposed train/test split
* first exclusion experiment
* proposed evaluation metrics
---
# 4. Metadata Selection
Initially target approximately **3–5 metadata factors**.
Each selected factor should satisfy:
1. It represents a meaningful driving/operating condition.
2. It has enough samples.
3. It is relevant to autonomous-driving model performance.
4. It can be represented as discrete levels.
Example structure:
| Factor | Type | Possible Levels | Samples |
| ----------- | ----------- | --------------- | ------: |
| Weather | Categorical | ... | ... |
| Time of Day | Categorical | ... | ... |
| Scene | Categorical | ... | ... |
Do not finalize the factors before inspecting BDD100K.
---
# 5. Week 2 — Dataset &amp; Baseline Setup
## Goal
Build the first complete experimental pipeline.
### Tasks

### A. Dataset Preparation
* Create training/test splits.
* Clean metadata.
* Handle missing values.
* Select usable metadata factors.
* Verify class distributions.
### B. Model Setup
Use:
**Faster R-CNN**
Do not focus on architectural improvements initially. The model
is a baseline for studying metadata effects.
### C. Implement Metadata-Based Dataset Selection
Create functions that can:
```text
Input:
Dataset + metadata condition
Output:
Subset containing that condition
```
and:
```text
Input:
Dataset + metadata condition
Output:
Dataset excluding that condition
```
### D. Design the first exclusion experiment
Example:
```text
Metadata factor:

Weather
Withheld level:
Rainy
```
Training set:
```text
Original training set
-
Rainy images
```
Then train Faster R-CNN on the modified training set.
### Week 2 Deliverable
A working pipeline that can perform:
```text
Select metadata condition
↓
Exclude it from training
↓
Train Faster R-CNN
↓
Evaluate on covered data
↓
Evaluate on withheld data
↓
Report performance
```
---
# 6. Week 3 — Systematic Experiments
## Goal
Scale the experiment after the first experiment works.
For each selected metadata level:
```text

Select metadata level
↓
Remove from training
↓
Train model
↓
Evaluate covered set
↓
Evaluate uncovered set
↓
Store results
```
Example:
```text
Experiment 1:
Weather = Rain
Experiment 2:
Weather = Fog
Experiment 3:
Time = Night
Experiment 4:
Scene = Highway
```
The exact levels will depend on the BDD100K metadata
distribution.
---
# 7. Evaluation Metrics
For every experiment, record:
* Precision
* Recall
* F1 score
Example:

| Experiment | Withheld Condition | Covered F1 | Uncovered F1 |
ΔF1 |
| ---------- | ------------------ | ---------: | -----------: | -
-: |
| 1 | Rain | | |
|
| 2 | Night | | |
|
Also keep:
* Number of training images
* Number of covered test images
* Number of uncovered test images
* Number of objects in each set
This is important because a performance difference can be
misleading if the uncovered set is extremely small.
---
# 8. Week 4 — Analysis
## Goal
Turn the experiments into research observations.
Investigate:
### A. Which metadata conditions matter?
Example:
```text
Rain:
Large F1 decrease
Night:
Large F1 decrease
Highway:
Small F1 decrease
```

Conditions causing larger degradation become candidates for
**critical metadata**.
### B. Which conditions appear less important?
If withholding a condition produces little change, investigate
whether it has little effect on this model/task.
### C. Does coverage relate to performance?
Compare:
```text
Representation / Coverage
vs
Model Performance
```
### D. Are there surprising cases?
For example:
```text
Well represented condition
+
Poor performance
```
or:
```text
Poorly represented condition
+
Only small performance drop
```
These cases are especially interesting because they may suggest
that coverage alone does not fully explain model behavior.
---
# 9. Statistical Analysis

The reference paper uses **weighted logistic regression** to
examine whether performance differences between covered and
uncovered test sets are statistically significant.
Do not begin by implementing the statistical model.
First:
```text
Get data
↓
Run experiments
↓
Collect performance results
↓
Understand the result structure
↓
Implement statistical analysis
```
Study the exact statistical setup from the paper before
reproducing it.
---
# 10. Connection to CC and SDCC
The 2021 paper provides the foundation through **Combinatorial
Coverage (CC)** and **Set Difference Combinatorial Coverage
(SDCC)**.
However, the 2024 paper&#39;s pilot experiment does **not simply
reproduce the CC/SDCC calculation**.
Its main experimental mechanism is:
**Systematic Inclusion &amp; Exclusion**
Therefore the initial SOP baseline should follow the 2024
methodology rather than immediately inventing a new CC/SDCC-based
experiment.
Later, depending on the results, possible extensions include
studying whether explicit combinatorial coverage measures can
better predict performance degradation.

---
# 11. Possible Research Extensions
Do not commit to these yet. Let the baseline experiments
determine which direction is justified.
Potential directions:
### Extension A — Coverage vs Performance
Investigate whether metadata coverage quantitatively predicts
model performance.
### Extension B — Frequency-Aware Coverage
Investigate whether the number of samples representing a
condition matters in addition to simple presence/absence.
### Extension C — Higher-Order Interactions
The pilot reference experiment uses:
```text
t = 1
```
A later extension could investigate:
```text
t = 2
t = 3
```
For example:
```text
Rain + Night
```
instead of only:
```text
Rain

Night
```
### Extension D — Robotics
After validating the approach on BDD100K, investigate whether a
similar methodology can be applied to a robotic/autonomous system
with controllable environmental conditions.
### Extension E — Multiple ML Tasks
Use object detection first. If the baseline is successful, test
whether the observations also hold for segmentation.
---
# 12. Recommended Experimental Progression
Do not attempt the full project immediately.
Use:
```text
Experiment 0
Dataset exploration
↓
Experiment 1
One metadata factor
↓
Experiment 2
Several levels of that factor
↓
Experiment 3
Second metadata factor
↓
Experiment 4+
Multiple factors
↓
Statistical analysis
↓
Higher-order / frequency analysis
↓
Research extension
```

---
# 13. First Concrete Milestone
Your first milestone is:
&gt; **Complete one end-to-end metadata exclusion experiment on
BDD100K using Faster R-CNN.**
That means you can demonstrate:
```text
BDD100K
↓
Metadata condition selected
↓
Condition excluded from training
↓
Faster R-CNN trained
↓
Covered test evaluation
↓
Uncovered test evaluation
↓
Precision / Recall / F1 comparison
```
Do not move to complicated research extensions until this works.
---
# 14. Project Structure
Start with:
```text
sop/
│
├── papers/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/

│ ├── 01_dataset_exploration.ipynb
│ └── 02_metadata_analysis.ipynb
│
├── src/
│ ├── dataset.py
│ ├── metadata.py
│ ├── splitting.py
│ ├── model.py
│ ├── evaluation.py
│ └── statistics.py
│
├── experiments/
│
├── results/
│
├── notes/
│ ├── dataset_metadata_report.md
│ └── experimental_design_v1.md
│
└── README.md
```
---
# 15. What You Should Have After One Month
By the end of Month 1, aim to have:
* A clear understanding of the reference methodology.
* A finalized BDD100K metadata subset.
* A working Faster R-CNN baseline.
* At least one complete metadata exclusion experiment.
* Multiple baseline experiments.
* Covered vs uncovered performance results.
* Initial statistical analysis.
* A list of observed limitations.
* 2–3 evidence-based research questions for Month 2 onward.
---
# Guiding Principle
Do not decide the final research contribution now.
Use:

```text
Literature
↓
Baseline reproduction/adaptation
↓
Experiments
↓
Observations
↓
Unexpected behavior
↓
Research gap
↓
New method / extension
```
The most valuable thing you can discover during this process may
be something that **doesn&#39;t behave the way the reference
methodology suggests**.
That is where the interesting research usually starts.