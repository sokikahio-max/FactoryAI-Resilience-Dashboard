[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17152407.svg)](https://doi.org/10.5281/zenodo.17152407)


# Factory AI Resilience Dashboard (LLM + AutoML) — Code & Data

This repository contains the latest notebook and datasets supporting the manuscript *Beyond Automation: An Evaluative and Resilient AI Systems for Sustainable I5.0 Manufacturing in the Second Half of AI — from Problem Definition to Human-Centric Decision Support*.

## Contents
- `notebooks/Latest_FactoryAI_Resilience_Dashboard_With_Save_Button.ipynb` — primary analysis notebook (Streamlit-enabled dashboard with save button).
- `data/` — input datasets for the two case studies.
- `requirements.txt` — Python dependencies.
- `CITATION.cff` — citation metadata.
- `.zenodo.json` — Zenodo metadata (used to prefill deposit info).
- `LICENSE` — MIT license (you can change this).
- `.gitignore` — standard Python/Jupyter ignores.

## Data Summary
- **case_study_1_high_accuracy.csv**: 5000 rows; columns: UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF
- **case_study_2_low_accuracy.csv**: 5000 rows; columns: UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF

## Quickstart

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch Jupyter and open the notebook
python -m ipykernel install --user --name factoryai-env
jupyter notebook notebooks/Latest_FactoryAI_Resilience_Dashboard_With_Save_Button.ipynb
```

## Publish & Mint a DOI via Zenodo

1. **Create a GitHub repository** (public recommended).
2. **Add and push** this folder:
   ```bash
   git init
   git remote add origin <your-repo-url>.git
   git add .
   git commit -m "Initial commit: latest dashboard, data, and manuscript context"
   git branch -M main
   git push -u origin main
   ```
3. **Connect Zenodo to GitHub** (one-time): https://zenodo.org/account/settings/github/ → authorize → toggle **ON** your repo.
4. **Create a GitHub release** (e.g., tag `v0.1.0`). Publishing the release triggers Zenodo to archive and mint a DOI.
5. **Update metadata**: Add Zenodo DOI badge to `README.md` and paste the DOI into `CITATION.cff` (`doi: "10.5281/zenodo.XXXXXXX"`).

## Citation
Please cite the software using the Zenodo DOI and the `CITATION.cff` file once the DOI is minted.
