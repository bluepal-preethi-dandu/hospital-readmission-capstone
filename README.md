# Hospital Readmission Prediction Project

## Project Overview
This project focuses on building an AI/ML-based healthcare system to predict whether a patient will be readmitted to the hospital within 30 days of discharge. The work is based on the capstone assignment for an end-to-end machine learning solution in the healthcare domain.

## Problem Statement
Hospital readmissions create major financial and clinical burdens. The objective is to design a predictive system that:
- estimates 30-day readmission risk,
- identifies the most important clinical predictors,
- supports personalized intervention recommendations,
- evaluates fairness and ethical concerns,
- proposes a production-ready deployment architecture.

## Dataset
The project uses the UCI Diabetes 130-US Hospitals dataset, which contains approximately 100,000 patient encounters from 130 U.S. hospitals over the period 1999–2008.

### Files available in this folder
- capstone_assignment-Hospital readmission.docx: original assignment document
- diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv: main dataset
- diabetes+130-us+hospitals+for+years+1999-2008/IDS_mapping.csv: dataset code mapping file

## Expected Project Workflow
1. Data loading and inspection
2. Exploratory data analysis (EDA)
3. Data preprocessing and feature engineering
4. Handling class imbalance
5. Model training and comparison
   - Logistic Regression
   - Random Forest or Gradient Boosting
   - A third model of choice
6. Evaluation using recall, F1-score, ROC-AUC, and confusion matrix
7. Feature importance analysis using SHAP or equivalent methods
8. Fairness and bias evaluation
9. Intervention design and deployment architecture proposal

## Suggested Technical Stack
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- shap
- xgboost (optional)

## Suggested Project Structure
```text
capstone/
├── README.md
├── capstone_assignment-Hospital readmission.docx
├── diabetes+130-us+hospitals+for+years+1999-2008/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── notebooks/            # Jupyter notebooks
├── src/                  # Python modules
├── results/              # figures, metrics, model files
└── requirements.txt
```

## Deliverables to Prepare
- Jupyter notebook or Python scripts for modeling
- README with setup and execution instructions
- requirements.txt with dependencies
- results folder with charts and metrics
- technical report in IEEE/ACM style

## Setup Instructions
1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the dataset and begin preprocessing and modeling.

## Notes
This folder already contains the assignment brief and the dataset needed for the project. The next step is to turn the assignment into a working implementation notebook or Python pipeline.
