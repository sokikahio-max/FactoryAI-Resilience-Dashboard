# v1.1.0-r2 — Reviewer-2 reproducibility release

This release corresponds to the second-revision analysis for the manuscript:

**A Failure-Aware and Human-Centric LLM–AutoML Framework for Resilient Decision Support in Industrial Applications in Industry 5.0**

## What is added in R2

- Dependence-corrected resampled PR-AUC comparisons for overlapping repeated splits.
- Joint recovery-selection gate requiring non-worsening validation PR-AUC, asymmetric cost, and failure recall.
- Correct handling of zero clean-reference classification cost.
- Retrospective non-LLM governed-controller audit and controller-threshold sensitivity analysis.
- Diagnostic versus effective operational coverage analysis.
- Targeted risk–coverage/selective-error/selective-failure-recall rerun at target coverages 0.70, 0.80, 0.90, 0.95, and 1.00.
- Fixed-initial-model recovery-isolation experiment comparing deterministic and archived LLM recovery from the same starting configuration and budget.
- Repeated 45-call structured-planner verification with provider-returned model identifier and response IDs recorded.

## Planner verification snapshot

- 45 live calls across 9 pre-specified cases, 5 repetitions each.
- Requested model: `gpt-5.6-luna`.
- Provider-returned model: `gpt-5.6-luna` for 45/45 calls.
- Explicit temperature argument sent: no.
- Schema-valid, task-correct, metric-appropriate: 45/45.
- Provider/parser failures: 0.
- Mean latency: 2.73 s; median: 2.66 s; range: 1.98–4.73 s.
- Input/output tokens: 41,725 / 8,346.
- Estimated total API cost under configured accounting rates: USD 0.01836.

## Version relationship

The original 350-run experiment remains the archived v7 experimental base. R2 does not silently replace it. The R2 release adds explicitly identified post-hoc analyses and targeted reruns required by the second-round reviewer, while retaining the original run-level archive for traceability.
