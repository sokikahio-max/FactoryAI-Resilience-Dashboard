# Reviewer 2 R2 targeted protocol

This patch addresses the second-revision requests without changing the original v7 experimental archive.

## Post-hoc analyses from archived v7 run-level outputs

Run:

```powershell
python scripts/reviewer2_posthoc_analysis.py
```

The analysis reconstructs a non-LLM governed controller, audits acceptance/withholding against the untouched-test envelope, separates diagnostic instance coverage from effective system-level operational coverage, evaluates threshold/cost sensitivity, checks zero-reference-cost cases, and audits retained recoveries against validation cost and failure recall.

## Targeted rerun requiring the original H2O environment

The archived v7 CSV does not contain per-instance probabilities, so risk--coverage curves and selective failure recall cannot be reconstructed without refitting the archived executable configurations. Likewise, a deterministic recovery candidate starting from the exact same initial LLM-planned model was not part of v7.

Use the same Windows/H2O environment reported in the manuscript and run:

```powershell
python scripts/download_ai4i.py --output data/ai4i2020.csv
python scripts/reviewer2_targeted_rerun.py --data data/ai4i2020.csv --backend h2o
```

The script deliberately makes **no LLM API calls**. It reuses the exact initial and retry execution configurations stored in `results/manuscript/final_10seed_runs_v7.csv`. For each validation-triggered P-framework recovery it fits, from the same initial configuration and model-count budget:

1. no recovery (shared initial model),
2. deterministic recovery, and
3. archived LLM-guided recovery configuration.

Both recovery candidates are governed by the R2 joint validation gate: PR-AUC must not decrease, asymmetric cost must not increase, and failure recall must not decrease.

It also evaluates target coverages 0.70, 0.80, 0.90, 0.95, and 1.00 using a validation-calibrated decision-boundary abstention heuristic and reports selective error, selective failure recall/FNR, selective classification cost, diagnostic coverage, and effective operational coverage after the system-level accept/abstain/escalate gate.

## Direct planner metadata rerun

The original v7 planner telemetry stored the requested model name but not the provider-returned `response.model`. The R2 patch records requested model, provider-returned model, response ID, and whether a temperature parameter was sent. To refresh the 45-call direct planner evaluation:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-5.6-luna"
python scripts/evaluate_llm.py --config configs/experiment.yaml --cases configs/llm_eval_cases.yaml --repetitions 5 --output-dir results/llm_eval_r2
```

Do not commit `.env` or API credentials.
