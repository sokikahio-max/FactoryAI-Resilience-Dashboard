# Revised Experimental Protocol

## Scientific question

The experiments test whether structured LLM guidance and a failure-aware controller add measurable value beyond a manual model, standard AutoML, and deterministic retry rules under controlled industrial data degradation.

## Target and predictors

The main task is binary prediction of `Machine failure` (0 = no failure, 1 = failure). Identifiers and failure-mode outcome columns are excluded from predictors. The legitimate AI4I predictors are expected to be `Type`, air temperature, process temperature, rotational speed, torque and tool wear.

## Data separation

Each configured seed creates a stratified 60/20/20 train/validation/test split. Training fits models. Validation selects probability thresholds, model configurations and retry actions. The test partition is evaluated only after the strategy is locked.

## Perturbations

The target labels are never shuffled. Validation and test feature data are independently perturbed using reproducible seeds:

1. `sensor_drift`: gradual sensor offset increasing across the evaluation sequence.
2. `measurement_noise`: zero-mean Gaussian noise scaled to each feature's observed standard deviation.
3. `missingness`: reproducible random masking of selected sensor channels.

Each has mild and severe levels in the default configuration.

## Ablations

- B0: fixed manually configured H2O GBM.
- B1: standard H2O AutoML, no LLM and no retry.
- B2: AutoML + deterministic validation-triggered retry, no LLM.
- B3: structured LLM-guided AutoML, no retry.
- P: structured LLM guidance + data-quality assessment + ACCEPT/RECONFIGURE/ABSTAIN/ESCALATE controller + one constrained retry.

Automated methods use a maximum model-search budget of 10 candidate models. Recovery methods allocate five models to the initial search and five to the single recovery search when needed.

## Metrics

Primary predictive metric: PR-AUC. Secondary metrics: precision, recall, F1, balanced accuracy, MCC, ROC-AUC, false-negative rate, Brier score and confusion counts. The classification probability threshold is selected using validation data to minimize a configurable asymmetric cost function where missed failures can be weighted more heavily than false alarms.

## Resilience metrics

For test PR-AUC:

- Performance retention = degraded-before / clean.
- Recovery gain = recovered - degraded-before.
- Normalized recovery ratio = (recovered - degraded-before) / (clean - degraded-before).

The experiment also records recovery/modeling time, abstention state and human-escalation state.

## Statistics

The code reports mean, standard deviation and 95% t confidence intervals across repeated seeds. Paired comparisons between the proposed framework and each baseline use the Wilcoxon signed-rank test on matched seeds, Holm adjustment for multiple comparisons and a paired rank-biserial effect-size estimate.

## LLM evaluation

The LLM receives no raw sensor rows. It returns a Pydantic-constrained decision structure. The direct LLM evaluation measures schema-valid output rate, task identification, action/model recommendation consistency across repeated calls and paraphrased prompts, latency, token use, estimated API cost and failure cases.
