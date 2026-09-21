# FactoryAI Resilience — Failure-Aware LLM–AutoML Research Pipeline

Code and reproducibility materials for the manuscript:

**A Failure-Aware and Human-Centric LLM–AutoML Framework for Resilient Decision Support in Industrial Applications in Industry 5.0**

This repository implements a failure-aware industrial decision-support pipeline that combines structured LLM planning with H2O AutoML, validation-governed reconfiguration, uncertainty-aware abstention, and human escalation.

## What the framework evaluates

The manuscript experiment compares five configurations on identical repeated data splits:

1. **B0 — Manual H2O GBM**
2. **B1 — Standard H2O AutoML**
3. **B2 — H2O AutoML + deterministic retry (no LLM)**
4. **B3 — Structured LLM-guided H2O AutoML (no retry)**
5. **P — Full failure-aware framework**

The evaluation uses the AI4I 2020 Predictive Maintenance Dataset with ten fixed stratified train/validation/test splits and seven conditions: clean data plus mild/severe sensor drift, measurement noise, and missingness. The final manuscript experiment contains **350 condition-method runs**.

## Key methodological safeguards

- **Leakage-free evaluation:** training/validation/test partitions are separated; the final test partition is not used for threshold selection, retries, recovery selection, or controller actions.
- **Failure-label leakage prevention:** `TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `Failure Type`, `UDI`, and `Product ID` are excluded when predicting `Machine failure`.
- **No label shuffling:** degradation is simulated through sensor drift, measurement noise, and missing sensor information.
- **Structured LLM planning:** the LLM returns a schema-constrained decision object; a deterministic policy maps only allowlisted fields into executable H2O settings.
- **Failure-aware control:** the controller can **ACCEPT**, **RECONFIGURE**, **ABSTAIN**, or request **HUMAN ESCALATION**.
- **Recovery guard:** seed-only/no-op retraining is not counted as LLM-guided recovery.
- **Imbalance-aware evaluation:** PR-AUC is the primary metric; precision, recall, F1, balanced accuracy, MCC, ROC-AUC, FNR, Brier score, confusion counts, coverage, and selective accuracy are also recorded.

## Repository structure

```text
FactoryAI-Resilience-Dashboard/
├── configs/
│   ├── experiment.yaml
│   └── llm_eval_cases.yaml
├── data/
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   └── THREAT_MODEL.md
├── notebooks/
│   └── Revision_Experiment_Quickstart.ipynb
├── results/
│   └── manuscript/
│       ├── final_10seed_runs_v7.csv
│       ├── llm_eval_raw_v5.csv
│       ├── llm_eval_summary_v5.csv
│       ├── STATISTICAL_REPORT.md
│       └── figures/
├── scripts/
│   ├── download_ai4i.py
│   ├── run_experiments.py
│   ├── analyze_results.py
│   ├── evaluate_llm.py
│   ├── generate_figures.py
│   └── capture_environment.py
├── src/factoryai_resilience/
├── tests/test_core.py
├── streamlit_app.py
├── requirements.txt
└── pyproject.toml
```

## 1. Environment

Recommended manuscript environment: **Python 3.12**, **64-bit Java 17**, and **H2O 3.46.0.12**.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 2. Obtain the AI4I 2020 dataset

The dataset is not duplicated in this repository. Download the public UCI dataset reproducibly with:

```bash
python scripts/download_ai4i.py --output data/ai4i2020.csv
```

## 3. Core tests

```bash
python tests/test_core.py
```

## 4. H2O installation check without LLM calls

```bash
python scripts/run_experiments.py \
  --data data/ai4i2020.csv \
  --backend h2o \
  --planner mock \
  --seeds 1 \
  --output results/h2o_check.csv
```

## 5. Final manuscript experiment

Set the API key through an environment variable. Never commit API keys.

PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-5.6-luna"
python scripts/run_experiments.py --data data/ai4i2020.csv --backend h2o --planner openai --seeds 10 --output results/final_10seed_runs_v7.csv
```

If an interrupted v7 run already contains valid checkpoints, resume with:

```powershell
python scripts/run_experiments.py --data data/ai4i2020.csv --backend h2o --planner openai --seeds 10 --resume --output results/final_10seed_runs_v7.csv
```

The ten fixed experiment seeds are:

```text
11, 22, 33, 44, 55, 66, 77, 88, 99, 110
```

## 6. Direct LLM evaluation

```bash
python scripts/evaluate_llm.py --planner openai --repetitions 5 --output-dir results/llm_eval
```

The direct evaluation measures schema validity, task identification, task-appropriate metric selection, repeated-call consistency, latency, token use, estimated cost, and failures.

## 7. Analysis and figures

```bash
python scripts/analyze_results.py --results results/final_10seed_runs_v7.csv --output-dir results/analysis
python scripts/generate_figures.py --analysis-dir results/analysis --output-dir results/figures
```

The archived manuscript outputs used for the reported analysis are in `results/manuscript/`.

## Important implementation notes

- The live OpenAI API call does **not** pass an explicit temperature parameter. The `temperature` value retained in the configuration object is a legacy, unused field and did not control the executed manuscript calls.
- The Windows H2O-3 runtime used for the manuscript did not expose H2O XGBoost. The manual baseline therefore uses native H2O GBM, and the AutoML allowlist is restricted to GBM, DRF, GLM, and Stacked Ensemble.
- The LLM receives metadata and locally computed diagnostic summaries rather than raw industrial rows.
- The project does not claim production MES/SCADA deployment, continuous production monitoring, causal root-cause identification, or experimentally validated operator trust/workload benefit.

## Reproducibility materials

The repository includes the final 350-run result table, the 45-call LLM evaluation outputs, manuscript analysis summary, and generated result figures under `results/manuscript/`.

## Citation

Please cite the associated manuscript when using this code. A `CITATION.cff` file is included for repository citation metadata.
