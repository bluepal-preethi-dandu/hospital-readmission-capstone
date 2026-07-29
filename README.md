# Hospital Readmission Prediction

This capstone project predicts whether a patient with diabetes will be readmitted within 30 days of discharge. It develops an end-to-end machine-learning workflow that starts with exploratory analysis and patient-safe preprocessing, then evaluates models, explains predictions with SHAP, audits subgroup performance, maps risk to care-transition interventions, and documents a production deployment architecture.

The project uses the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), which contains 101,766 inpatient encounters from 130 U.S. hospitals between 1999 and 2008. Predicting readmission risk matters because preventable readmissions are disruptive for patients and can trigger financial penalties under the Hospital Readmissions Reduction Program. The goal is not to automate care decisions: it is to give clinicians a transparent decision-support signal that can guide post-discharge follow-up.

## Project highlights

- Patient-grouped 70/15/15 train, validation, and test split with zero patient overlap.
- Expired encounters (`discharge_disposition_id` 11, 19, 20, and 21) removed before modeling to avoid a non-clinical data artifact.
- SMOTE applied inside cross-validation folds through an `imblearn` pipeline, preventing synthetic-neighbor leakage.
- SHAP explanations, subgroup fairness analysis, and a threshold-calibration mitigation experiment.
- A Streamlit decision-support prototype that returns probability, risk tier, and a recommended intervention.

## Repository structure

```text
capstone/
├── notebooks/       # EDA, preprocessing, SHAP, fairness, and intervention notebooks
├── scripts/         # Reusable training and threshold-analysis scripts
├── app/             # Streamlit demo application and its deployment requirements
├── results/         # Saved model artifacts, metrics, figures, and CSV outputs
├── docs/            # Deployment architecture proposal
├── report/          # Technical report and presentation deliverables
├── requirements.txt # Pinned project dependencies
└── README.md
```

## Setup

Clone the repository and create an isolated Python environment.

```bash
git clone https://github.com/bluepal-preethi-dandu/hospital-readmission-capstone.git
cd hospital-readmission-capstone

python -m venv .venv
```

Activate the environment.

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the pinned dependencies.

```bash
pip install -r requirements.txt
```

The raw UCI data file is intentionally excluded from version control. Place `diabetic_data.csv` in `diabetes+130-us+hospitals+for+years+1999-2008/` before rerunning the data-preparation workflow.

## Execution guide

Run the workflow in the following order:

1. `notebooks/01_data_exploration.ipynb` — data quality audit, class balance, and exploratory analysis.
2. `notebooks/02_preprocessing.ipynb` — filtering, feature engineering, and patient-grouped splits; writes processed data.
3. `python scripts/train_pipeline.py` — trains and evaluates Logistic Regression, Random Forest, Decision Tree, and XGBoost; writes model artifacts and metrics.
4. `python scripts/threshold_analysis.py` — compares models at a matched recall target and selects operating thresholds.
5. `notebooks/04_interpretability.ipynb` — SHAP summary and patient-level waterfall explanations.
6. `notebooks/05_fairness_analysis.ipynb` — subgroup metrics, ethics discussion, and threshold-calibration mitigation experiment.
7. `notebooks/06_intervention_logic.ipynb` — data-driven risk tiers and intervention design.

Run the local demo from the repository root:

```bash
streamlit run app/demo_app.py
```

## Live demo

**Streamlit Community Cloud URL:** `https://<your-streamlit-subdomain>.streamlit.app`

Replace the placeholder above with the deployed app URL after the Streamlit Community Cloud deployment is live.

## Final model summary

The selected final model is **XGBoost**. It achieved the best validation ROC-AUC among the evaluated models: **0.6839**. Its default 0.50 decision threshold produced very low recall, so the deployment cutoff was set to **0.12** to prioritize catching true readmissions.

At the selected validation operating point, XGBoost achieved:

| Metric | Value |
| --- | ---: |
| Threshold | 0.12 |
| Precision | 0.1777 |
| Recall | 0.6369 |
| ROC-AUC | 0.6839 |

The resulting probability is translated into four operational tiers: Low, Moderate, High, and Very High risk. The Streamlit prototype presents the tier and a corresponding transition-of-care intervention, while keeping clinician judgment and override authority central to the workflow.

## Important note

This project is an educational capstone and a decision-support prototype. It is not a validated clinical tool and must not be used as an automated treatment, discharge, or eligibility decision.
