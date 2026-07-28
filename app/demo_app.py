"""Streamlit decision-support prototype for 30-day readmission risk.

The input contract is intentionally derived from the serialized preprocessing
pipeline.  This keeps the demo aligned with the features used at training
time instead of relying on a hand-maintained schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR_PATH = PROJECT_ROOT / "results" / "preprocessing_pipeline.joblib"
MODEL_PATH = PROJECT_ROOT / "results" / "xgboost.joblib"
REFERENCE_PATH = Path(__file__).resolve().parent / "training_reference.json"
DEPLOYMENT_THRESHOLD = 0.12
EXPIRED_DISPOSITIONS = {11, 19, 20, 21}
AGE_LEVELS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
MED_CHANGE_COLUMNS = [
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def assign_risk_tier(probability: float) -> str:
    if probability < 0.12:
        return "Low Risk"
    if probability < 0.20:
        return "Moderate Risk"
    if probability < 0.35:
        return "High Risk"
    return "Very High Risk"


def get_intervention(tier: str) -> str:
    interventions = {
        "Low Risk": "Standard discharge instructions and routine primary-care follow-up.",
        "Moderate Risk": "Post-discharge follow-up call within 72 hours and medication-reconciliation review.",
        "High Risk": "Project BOOST-style structured discharge: teach-back education, appointment scheduled before discharge, and a follow-up call within 48 hours.",
        "Very High Risk": "Care Transitions Intervention: transition coach, home visit within 72 hours, medication reconciliation, red-flag symptom education, and 30-day check-in calls.",
    }
    return interventions[tier]


@st.cache_resource
def load_artifacts() -> tuple[Any, Any]:
    """Load fitted preprocessing and model artifacts once per Streamlit process."""
    return joblib.load(PREPROCESSOR_PATH), joblib.load(MODEL_PATH)


@st.cache_data
def load_training_reference() -> dict[str, Any]:
    """Load aggregate training references without requiring the raw CSV at runtime."""
    with REFERENCE_PATH.open(encoding="utf-8") as reference_file:
        return json.load(reference_file)


def pipeline_schema(preprocessor: Any) -> tuple[list[str], dict[str, list[Any]]]:
    """Return the persisted raw schema and learned categorical levels."""
    # The saved artifact is a sklearn Pipeline containing a ColumnTransformer
    # named ``preprocessor``.  Accept a bare ColumnTransformer too so this
    # helper remains aligned with either serialization layout.
    column_transformer = preprocessor.named_steps.get("preprocessor", preprocessor) if hasattr(preprocessor, "named_steps") else preprocessor
    names = list(preprocessor.feature_names_in_)
    categorical_levels: dict[str, list[Any]] = {}
    for _, transformer, columns in column_transformer.transformers_:
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            encoder = transformer.named_steps["onehot"]
            categorical_levels.update({column: list(levels) for column, levels in zip(columns, encoder.categories_)})
    return names, categorical_levels


def clean_select_options(values: pd.Series, fallback: list[str] | None = None) -> list[str]:
    result = sorted({str(value) for value in values.dropna().tolist()})
    return result or (fallback or ["Unknown"])


def build_feature_row(inputs: dict[str, Any], dominant: dict[str, Any], diag_freq: dict[str, dict[str, float]], schema: list[str]) -> pd.DataFrame:
    """Apply the exact notebook-02 feature engineering required by the model."""
    row: dict[str, Any] = {
        "age_ordinal": AGE_LEVELS.index(inputs["age"]),
        "time_in_hospital": inputs["time_in_hospital"],
        "num_lab_procedures": inputs["num_lab_procedures"],
        "num_procedures": inputs["num_procedures"],
        "num_medications": inputs["num_medications"],
        "number_diagnoses": inputs["number_diagnoses"],
        "number_outpatient": inputs["number_outpatient"],
        "number_emergency": inputs["number_emergency"],
        "number_inpatient": inputs["number_inpatient"],
        "gender": inputs["gender"], "race": inputs["race"],
        "admission_type_id": inputs["admission_type_id"],
        "discharge_disposition_id": inputs["discharge_disposition_id"],
        "admission_source_id": inputs["admission_source_id"],
        "payer_code": inputs["payer_code"], "medical_specialty": inputs["medical_specialty"],
        "change": inputs["change"], "diabetesMed": inputs["diabetesMed"],
        "insulin": inputs["insulin"], "metformin": inputs["metformin"],
    }
    row["total_visits_prior"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    row["num_med_changes"] = sum(inputs[column] != dominant[column] for column in dominant)
    row["had_glu_serum_test"] = int(inputs["max_glu_serum"] != "Not recorded")
    row["had_A1C_test"] = int(inputs["A1Cresult"] != "Not recorded")
    for column in ["diag_1", "diag_2", "diag_3"]:
        row[f"{column}_freq"] = diag_freq[column].get(str(inputs[column]), 0.0)
    return pd.DataFrame([{column: row[column] for column in schema}])


LOW_RISK_EXAMPLE = {
    "age": "[50-60)", "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 0,
    "num_medications": 6, "number_diagnoses": 4, "number_outpatient": 0,
    "number_emergency": 0, "number_inpatient": 0,
}
HIGH_RISK_EXAMPLE = {
    "age": "[70-80)", "time_in_hospital": 10, "num_lab_procedures": 75, "num_procedures": 3,
    "num_medications": 20, "number_diagnoses": 9, "number_outpatient": 2,
    "number_emergency": 2, "number_inpatient": 4,
}


def apply_example(example: dict[str, Any], reference: dict[str, Any], dominant: dict[str, Any]) -> None:
    """Populate session-state keys; fields outside the profile use safe learned defaults."""
    defaults = {
        "gender": "Female", "race": "Caucasian", "admission_type_id": 1,
        "discharge_disposition_id": 1, "admission_source_id": 7, "payer_code": "MC",
        "medical_specialty": "InternalMedicine", "change": "No", "diabetesMed": "Yes",
        "insulin": "No", "metformin": "No", "max_glu_serum": "Not recorded", "A1Cresult": "Not recorded",
        "diag_1": reference["diagnosis_defaults"]["diag_1"],
        "diag_2": reference["diagnosis_defaults"]["diag_2"],
        "diag_3": reference["diagnosis_defaults"]["diag_3"],
    }
    defaults.update(example)
    for key, value in defaults.items():
        st.session_state[f"input_{key}"] = value
    for column, value in dominant.items():
        st.session_state[f"input_{column}"] = str(value)


def selectbox(label: str, key: str, options: list[Any], default: Any | None = None) -> Any:
    options = list(options)
    if key not in st.session_state and default in options:
        st.session_state[key] = default
    return st.selectbox(label, options, key=key)


def main() -> None:
    st.set_page_config(page_title="Readmission Risk Demo", page_icon="🏥", layout="wide")
    st.title("30-Day Hospital Readmission Risk")
    st.warning("This is a decision-support prototype for a capstone project, not a validated clinical tool. It does not replace clinical judgment.")

    preprocessor, model_pipeline = load_artifacts()
    reference = load_training_reference()
    dominant = reference["medication_dominant_values"]
    diag_freq = reference["diagnosis_frequency"]
    numeric_defaults = reference["numeric_defaults"]
    schema, learned_levels = pipeline_schema(preprocessor)

    with st.sidebar:
        st.subheader("Example patients")
        if st.button("Load low-risk example", use_container_width=True):
            apply_example(LOW_RISK_EXAMPLE, reference, dominant)
        if st.button("Load high-risk example", use_container_width=True):
            apply_example(HIGH_RISK_EXAMPLE, reference, dominant)
        st.caption("Examples are illustrative profiles; they are not real patients.")

    diagnosis_options = reference["diagnosis_options"]
    med_options = reference["medication_options"]

    with st.form("risk_form"):
        with st.expander("Demographics", expanded=True):
            left, right = st.columns(2)
            with left:
                age = selectbox("Age band", "input_age", AGE_LEVELS, "[60-70)")
                gender = selectbox("Gender", "input_gender", learned_levels["gender"], "Female")
            with right:
                race = selectbox("Race", "input_race", learned_levels["race"], "Caucasian")

        with st.expander("Admission details", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                admission_type = selectbox("Admission type ID", "input_admission_type_id", learned_levels["admission_type_id"], 1)
            with c2:
                discharge = selectbox("Discharge disposition ID", "input_discharge_disposition_id", learned_levels["discharge_disposition_id"], 1)
            with c3:
                admission_source = selectbox("Admission source ID", "input_admission_source_id", learned_levels["admission_source_id"], 7)
            payer_code = selectbox("Payer code", "input_payer_code", learned_levels["payer_code"], "MC")
            specialty = selectbox("Medical specialty", "input_medical_specialty", learned_levels["medical_specialty"], "InternalMedicine")

        with st.expander("Lab results, procedures, and prior utilization", expanded=True):
            numeric_inputs: dict[str, int] = {}
            columns = st.columns(3)
            for index, column in enumerate(numeric_defaults):
                minimum, maximum = reference["numeric_ranges"][column]
                numeric_inputs[column] = columns[index % 3].number_input(
                    column.replace("_", " ").title(), min_value=minimum, max_value=maximum,
                    value=int(numeric_defaults[column]), step=1, key=f"input_{column}",
                )
            max_glu = selectbox("Maximum glucose serum test", "input_max_glu_serum", ["Not recorded", "Norm", ">200", ">300"], "Not recorded")
            a1c = selectbox("A1C test", "input_A1Cresult", ["Not recorded", "Norm", ">7", ">8"], "Not recorded")

        with st.expander("Medications", expanded=False):
            change, diabetes_med, insulin, metformin = st.columns(4)
            change_value = change.selectbox("Medication change", learned_levels["change"], key="input_change")
            diabetes_value = diabetes_med.selectbox("Diabetes medication", learned_levels["diabetesMed"], key="input_diabetesMed")
            insulin_value = insulin.selectbox("Insulin", learned_levels["insulin"], key="input_insulin")
            metformin_value = metformin.selectbox("Metformin", learned_levels["metformin"], key="input_metformin")
            st.caption("Other medication settings calculate the training feature `num_med_changes` against each medication's training-set dominant value.")
            medication_values: dict[str, Any] = {}
            for start in range(0, len(med_options), 4):
                cols = st.columns(4)
                for widget_column, medication in zip(cols, list(med_options)[start:start + 4]):
                    medication_values[medication] = widget_column.selectbox(medication, med_options[medication], key=f"input_{medication}")

        with st.expander("Diagnosis codes", expanded=False):
            st.caption("Codes are mapped to their training-data frequency encodings before prediction.")
            diag1, diag2, diag3 = st.columns(3)
            diagnosis_1 = diag1.selectbox("Primary diagnosis (diag_1)", diagnosis_options["diag_1"], key="input_diag_1")
            diagnosis_2 = diag2.selectbox("Secondary diagnosis (diag_2)", diagnosis_options["diag_2"], key="input_diag_2")
            diagnosis_3 = diag3.selectbox("Tertiary diagnosis (diag_3)", diagnosis_options["diag_3"], key="input_diag_3")

        submitted = st.form_submit_button("Assess readmission risk", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "age": age, "gender": gender, "race": race,
            "admission_type_id": admission_type, "discharge_disposition_id": discharge,
            "admission_source_id": admission_source, "payer_code": payer_code, "medical_specialty": specialty,
            "max_glu_serum": max_glu, "A1Cresult": a1c, "change": change_value,
            "diabetesMed": diabetes_value, "insulin": insulin_value, "metformin": metformin_value,
            "diag_1": diagnosis_1, "diag_2": diagnosis_2, "diag_3": diagnosis_3,
            **numeric_inputs, **medication_values,
        }
        feature_row = build_feature_row(inputs, dominant, diag_freq, schema)
        transformed = preprocessor.transform(feature_row)
        probability = float(model_pipeline.named_steps["model"].predict_proba(transformed)[0, 1])
        tier = assign_risk_tier(probability)
        color = {"Low Risk": "#16803c", "Moderate Risk": "#b7791f", "High Risk": "#c05621", "Very High Risk": "#c53030"}[tier]
        st.divider()
        st.metric("Predicted 30-day readmission probability", f"{probability:.1%}")
        st.markdown(f"<h2 style='color:{color}; margin-bottom:0'>{tier}</h2>", unsafe_allow_html=True)
        st.write(get_intervention(tier))
        st.caption(f"The binary high-risk flag uses the deployed threshold of {DEPLOYMENT_THRESHOLD:.2f}; this patient is {'high-risk' if probability >= DEPLOYMENT_THRESHOLD else 'not high-risk'}.")


if __name__ == "__main__":
    main()
