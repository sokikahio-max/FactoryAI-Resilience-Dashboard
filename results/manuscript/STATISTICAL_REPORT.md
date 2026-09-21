# Step 3 — Manuscript-Ready Statistical Analysis

## Analysis basis

This package analyzes the frozen v7 experiment with **350 condition–method runs** (10 repeated seeds × 7 data conditions × 5 methods). The primary predictive metric is **PR-AUC**, with precision, recall, F1, balanced accuracy, MCC, ROC-AUC, false-negative rate (FNR), Brier score, coverage, and selective accuracy treated as secondary outcomes. Means, standard deviations, and t-based 95% confidence intervals are computed across repeated seeds. Pairwise inferential comparisons use the paired Wilcoxon signed-rank test between the full framework and each comparator on the same seeds, with Holm correction applied across the four comparisons within each condition. Paired rank-biserial correlation is reported as effect size.

The **severe-missingness/full-framework** cell has no predictive metric because the controller escalated **10/10 runs before model training**; it is therefore reported as N/A rather than scored as a prediction failure.

## Dataset integrity

| Check                                |   Observed |   Expected | Pass   |
|:-------------------------------------|-----------:|-----------:|:-------|
| Rows                                 |        350 |        350 | True   |
| Unique seeds                         |         10 |         10 | True   |
| Unique methods                       |          5 |          5 | True   |
| Unique conditions                    |          7 |          7 | True   |
| Duplicate seed-method-condition rows |          0 |          0 | True   |


## Table 5. Clean-condition predictive performance

Values are mean ± SD [95% CI] across 10 seeds.

| Method               | PR-AUC                       | ROC-AUC                      | Recall                       | F1                           | Balanced Accuracy            | MCC                          | FNR                          |
|:---------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| Manual GBM           | 0.795 ± 0.038 [0.768, 0.822] | 0.974 ± 0.011 [0.966, 0.981] | 0.772 ± 0.046 [0.739, 0.805] | 0.697 ± 0.053 [0.659, 0.735] | 0.878 ± 0.021 [0.863, 0.893] | 0.691 ± 0.051 [0.655, 0.727] | 0.228 ± 0.046 [0.195, 0.261] |
| Standard H2O AutoML  | 0.785 ± 0.039 [0.757, 0.813] | 0.970 ± 0.011 [0.962, 0.978] | 0.803 ± 0.034 [0.779, 0.827] | 0.690 ± 0.077 [0.635, 0.745] | 0.892 ± 0.017 [0.880, 0.904] | 0.687 ± 0.073 [0.635, 0.739] | 0.197 ± 0.034 [0.173, 0.221] |
| Rule-based retry     | 0.784 ± 0.042 [0.754, 0.814] | 0.972 ± 0.009 [0.966, 0.978] | 0.762 ± 0.057 [0.721, 0.803] | 0.688 ± 0.068 [0.639, 0.736] | 0.872 ± 0.024 [0.855, 0.890] | 0.684 ± 0.062 [0.640, 0.728] | 0.238 ± 0.057 [0.197, 0.279] |
| LLM-guided, no retry | 0.767 ± 0.047 [0.733, 0.800] | 0.973 ± 0.010 [0.966, 0.980] | 0.766 ± 0.039 [0.738, 0.794] | 0.681 ± 0.052 [0.644, 0.718] | 0.874 ± 0.018 [0.862, 0.887] | 0.676 ± 0.050 [0.640, 0.711] | 0.234 ± 0.039 [0.206, 0.262] |
| Full framework       | 0.767 ± 0.044 [0.735, 0.798] | 0.972 ± 0.011 [0.965, 0.980] | 0.766 ± 0.045 [0.734, 0.798] | 0.678 ± 0.053 [0.640, 0.716] | 0.874 ± 0.021 [0.859, 0.889] | 0.672 ± 0.051 [0.635, 0.709] | 0.234 ± 0.045 [0.202, 0.266] |


## Table 6. PR-AUC under clean and degraded conditions

Values are mean ± SD [95% CI].

| Condition             | Manual GBM                   | Standard H2O AutoML          | Rule-based retry             | LLM-guided, no retry         | Full framework               |
|:----------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| Clean                 | 0.795 ± 0.038 [0.768, 0.822] | 0.785 ± 0.039 [0.757, 0.813] | 0.784 ± 0.042 [0.754, 0.814] | 0.767 ± 0.047 [0.733, 0.800] | 0.767 ± 0.044 [0.735, 0.798] |
| Sensor drift (mild)   | 0.666 ± 0.045 [0.634, 0.698] | 0.674 ± 0.049 [0.639, 0.710] | 0.679 ± 0.050 [0.644, 0.715] | 0.681 ± 0.049 [0.646, 0.716] | 0.688 ± 0.040 [0.659, 0.717] |
| Sensor drift (severe) | 0.454 ± 0.047 [0.421, 0.487] | 0.484 ± 0.052 [0.447, 0.521] | 0.474 ± 0.062 [0.429, 0.518] | 0.530 ± 0.052 [0.493, 0.567] | 0.517 ± 0.071 [0.466, 0.568] |
| Noise (mild)          | 0.712 ± 0.038 [0.685, 0.739] | 0.684 ± 0.042 [0.654, 0.713] | 0.693 ± 0.047 [0.660, 0.727] | 0.681 ± 0.048 [0.647, 0.716] | 0.679 ± 0.048 [0.645, 0.714] |
| Noise (severe)        | 0.578 ± 0.051 [0.542, 0.614] | 0.578 ± 0.064 [0.533, 0.624] | 0.576 ± 0.064 [0.530, 0.622] | 0.577 ± 0.060 [0.534, 0.620] | 0.579 ± 0.065 [0.532, 0.625] |
| Missingness (mild)    | 0.727 ± 0.039 [0.699, 0.755] | 0.735 ± 0.035 [0.710, 0.760] | 0.722 ± 0.059 [0.679, 0.764] | 0.706 ± 0.040 [0.677, 0.734] | 0.706 ± 0.044 [0.674, 0.737] |
| Missingness (severe)  | 0.491 ± 0.067 [0.443, 0.538] | 0.495 ± 0.074 [0.442, 0.547] | 0.491 ± 0.068 [0.442, 0.540] | 0.484 ± 0.070 [0.434, 0.534] | N/A                          |


## Primary paired inferential results

Only Holm-adjusted statistically significant PR-AUC comparisons are shown below.

| Condition             | Comparator       |   Mean_Difference_Full_minus_Comparator |   Diff_CI95_Low |   Diff_CI95_High |   p_raw |   p_Holm |   Rank_Biserial | Direction         |
|:----------------------|:-----------------|----------------------------------------:|----------------:|-----------------:|--------:|---------:|----------------:|:------------------|
| Clean                 | Manual GBM       |                                 -0.0288 |         -0.0460 |          -0.0116 |  0.0039 |   0.0156 |         -0.9636 | Full < comparator |
| Sensor drift (severe) | Manual GBM       |                                  0.0628 |          0.0324 |           0.0932 |  0.0039 |   0.0156 |          0.9636 | Full > comparator |
| Sensor drift (severe) | Rule-based retry |                                  0.0432 |          0.0179 |           0.0685 |  0.0078 |   0.0234 |          0.9556 | Full > comparator |
| Noise (mild)          | Manual GBM       |                                 -0.0324 |         -0.0536 |          -0.0112 |  0.0020 |   0.0078 |         -1.0000 | Full < comparator |

Interpretation: the full framework was significantly **lower than Manual GBM on clean data**, significantly **higher than Manual GBM and Rule-based retry under severe sensor drift**, and significantly **lower than Manual GBM under mild measurement noise**. No PR-AUC difference versus Standard H2O AutoML survived Holm correction. For FNR, **no pairwise comparison survived Holm correction** at p < 0.05.

## Table 8. Failure-aware controller outcomes

| Condition             |   accept |   abstain |   human_escalation |   accept_pct |   abstain_pct |   human_escalation_pct |
|:----------------------|---------:|----------:|-------------------:|-------------:|--------------:|-----------------------:|
| Clean                 |       10 |         0 |                  0 |          100 |             0 |                      0 |
| Sensor drift (mild)   |        3 |         7 |                  0 |           30 |            70 |                      0 |
| Sensor drift (severe) |        0 |         1 |                  9 |            0 |            10 |                     90 |
| Noise (mild)          |        3 |         7 |                  0 |           30 |            70 |                      0 |
| Noise (severe)        |        0 |        10 |                  0 |            0 |           100 |                      0 |
| Missingness (mild)    |        6 |         4 |                  0 |           60 |            40 |                      0 |
| Missingness (severe)  |        0 |         0 |                 10 |            0 |             0 |                    100 |

The controller accepted all clean runs (10/10), abstained on all severe-noise runs (10/10), and escalated all severe-missingness runs (10/10). Under severe sensor drift, 9/10 runs escalated and 1/10 abstained.

## Recovery analysis

| Condition             |   Recovery_Attempts |   Substantive_Execution_Changes |   Recovery_Models_Selected |   Mean_Validation_PR_AUC_Gain_All_Attempts |   Mean_Test_PR_AUC_Gain_Selected |   Selected_Improved_Test |   Selected_Worsened_Test |
|:----------------------|--------------------:|--------------------------------:|---------------------------:|-------------------------------------------:|---------------------------------:|-------------------------:|-------------------------:|
| Clean                 |                   0 |                               0 |                          0 |                                   nan      |                         nan      |                        0 |                        0 |
| Sensor drift (mild)   |                   8 |                               8 |                          3 |                                     0.0023 |                           0.0106 |                        2 |                        1 |
| Sensor drift (severe) |                  10 |                              10 |                          2 |                                     0.0070 |                          -0.0828 |                        0 |                        2 |
| Noise (mild)          |                   7 |                               7 |                          1 |                                     0.0002 |                          -0.0067 |                        0 |                        1 |
| Noise (severe)        |                   9 |                               9 |                          2 |                                     0.0063 |                           0.0251 |                        2 |                        0 |
| Missingness (mild)    |                   4 |                               4 |                          2 |                                     0.0084 |                           0.0061 |                        1 |                        1 |
| Missingness (severe)  |                   0 |                               0 |                          0 |                                   nan      |                         nan      |                        0 |                        0 |

Across the full framework there were **38 recovery attempts**, and **38/38** produced a substantive executable configuration change. Only **10/38 (26.3%)** recovery models were selected by the validation gate. Among those 10 selected recoveries, **5 improved** untouched-test PR-AUC and **5 worsened** it. This demonstrates that validation improvement did not guarantee test improvement and supports the use of rejection, abstention, and escalation in addition to reconfiguration.

## Quantitative resilience / performance retention

| Condition             | Method         |   n |   Mean_Performance_Retention |       SD |   CI95_Low |   CI95_High |
|:----------------------|:---------------|----:|-----------------------------:|---------:|-----------:|------------:|
| Sensor drift (mild)   | Full framework |  10 |                       0.8937 |   0.0499 |     0.8580 |      0.9294 |
| Sensor drift (severe) | Full framework |  10 |                       0.6968 |   0.0660 |     0.6496 |      0.7440 |
| Noise (mild)          | Full framework |  10 |                       0.8875 |   0.0444 |     0.8557 |      0.9192 |
| Noise (severe)        | Full framework |  10 |                       0.7476 |   0.0655 |     0.7008 |      0.7945 |
| Missingness (mild)    | Full framework |  10 |                       0.9197 |   0.0397 |     0.8913 |      0.9481 |
| Missingness (severe)  | Full framework |   0 |                     nan      | nan      |   nan      |    nan      |

The clearest predictive retention advantage appeared under sensor drift: full-framework mean retention was 0.894 under mild drift and 0.697 under severe drift. The framework was not uniformly superior under noise or mild missingness, which should be stated explicitly.

## Table 9. Direct LLM component evaluation

|   Calls |   Schema_Valid_Rate |   Task_Accuracy |   Metric_Appropriateness |   Action_Consistency |   Algorithm_Recommendation_Consistency |   Metric_Consistency |   Mean_Latency_s |   Median_Latency_s |   Min_Latency_s |   Max_Latency_s |   Input_Tokens |   Output_Tokens |   Estimated_Cost_USD |   Failures |
|--------:|--------------------:|----------------:|-------------------------:|---------------------:|---------------------------------------:|---------------------:|-----------------:|-------------------:|----------------:|----------------:|---------------:|----------------:|---------------------:|-----------:|
| 45.0000 |              1.0000 |          1.0000 |                   1.0000 |               1.0000 |                                 1.0000 |               1.0000 |           3.2595 |             2.8831 |          2.3111 |          7.9622 |     36415.0000 |       8349.0000 |               0.0173 |     0.0000 |

Across the 45 repeated LLM-evaluation calls, schema validity, task identification, metric appropriateness, action consistency, algorithm-recommendation consistency, and metric consistency were all 100%, with zero recorded failures.

## Table 10. Runtime, token usage, and API cost

| Method               |   n |   Mean_Runtime_s |   SD_Runtime_s |   Runtime_CI95_Low |   Runtime_CI95_High |   Total_Runtime_min |   Planner_Calls |   Input_Tokens |   Output_Tokens |   Estimated_LLM_Cost_USD |   Mean_Planner_Latency_s |
|:---------------------|----:|-----------------:|---------------:|-------------------:|--------------------:|--------------------:|----------------:|---------------:|----------------:|-------------------------:|-------------------------:|
| Manual GBM           |  70 |           7.5222 |         1.2558 |             7.2228 |              7.8216 |              8.7759 |               0 |              0 |               0 |                   0.0000 |                 nan      |
| Standard H2O AutoML  |  70 |          29.6433 |         3.3197 |            28.8517 |             30.4348 |             34.5838 |               0 |              0 |               0 |                   0.0000 |                 nan      |
| Rule-based retry     |  70 |          27.9346 |         6.9339 |            26.2813 |             29.5879 |             32.5904 |               0 |              0 |               0 |                   0.0000 |                 nan      |
| LLM-guided, no retry |  70 |          38.6339 |         3.6272 |            37.7690 |             39.4988 |             45.0729 |              70 |          67283 |           13552 |                   0.0297 |                   3.4589 |
| Full framework       |  70 |          32.1117 |        15.5256 |            28.4098 |             35.8137 |             37.4637 |             109 |         116712 |           27147 |                   0.0559 |                   5.8574 |

Across the integrated 350-run experiment, the planner was called **179** times, using **183,995 input tokens** and **40,699 output tokens**, for an estimated API cost of **$0.0856**. The sum of recorded end-to-end runtimes was approximately **2.64 hours**.

## Manuscript interpretation that the data support

The results do **not** support a claim that the proposed framework universally outperforms conventional AutoML in predictive accuracy. Instead, they support a more precise contribution: the framework maintains competitive predictive performance while adding failure-aware orchestration, validation-governed reconfiguration, calibrated abstention, and human escalation. Its strongest predictive advantage was observed under controlled sensor drift. Severe missingness was treated as a data-quality failure requiring escalation rather than as a model-selection problem.

A defensible resilience statement is therefore: **resilience is operationalized as a governed combination of performance retention, candidate reconfiguration, validation-based rejection, abstention, and human escalation—not as guaranteed recovery after every degradation event.**

## Limitations to carry into the Discussion

- The human-centric evaluation is architectural/technical; the experiment does not establish improved operator trust, usability, workload, or decision quality.
- The AI4I dataset is simulated and predictive rather than causal; LLM outputs should be described as diagnostic suggestions, not validated root causes.
- H2O XGBoost was unavailable on the Windows platform, so the manual baseline used H2O GBM.
- Recovery models that improved validation performance did not always improve untouched-test performance, reinforcing the need for cautious deployment claims.
- Predictive comparisons are not applicable to the full framework under severe missingness because it intentionally escalated before training.

## Generated figures

- `figures/Fig_PR_AUC_by_condition.png`
- `figures/Fig_FNR_by_condition.png`
- `figures/Fig_Performance_Retention.png`
- `figures/Fig_Controller_States.png`
- `figures/Fig_Recovery_Validation_vs_Test.png`
- `figures/Fig_Runtime_by_Method.png`

