# Deployment Architecture Proposal

## 1. Data Ingestion

In production, patient encounter information would enter the readmission-risk workflow through HL7 FHIR, the interoperability standard commonly used by hospital EHR systems. At discharge, an ETL service would retrieve the relevant FHIR resources—`Encounter`, `Condition`, `MedicationRequest`, and `Observation`—and validate that they are complete, current, and associated with the correct encounter. The service would map those resources into the exact feature schema used during model development, including diagnosis, medication, utilization, laboratory, demographic, and encounter-history fields. Missing or unmapped information would be handled according to the training-time preprocessing rules and logged for data-quality review.

## 2. Preprocessing Pipeline

The serialized training preprocessor, [`results/preprocessing_pipeline.joblib`](../results/preprocessing_pipeline.joblib), would be loaded once when the prediction service starts. Its `ColumnTransformer` applies the same imputation, scaling, and one-hot encoding used in training. Reusing this artifact guarantees that a live encounter is transformed identically to historical training data, preventing training-serving skew caused by hand-maintained or duplicated feature-engineering logic. The service should validate the incoming feature contract before transformation and record schema-validation failures for investigation.

## 3. Model Serving

The trained [`results/xgboost.joblib`](../results/xgboost.joblib) pipeline would be exposed through a FastAPI prediction service. The service would return the predicted probability, the deployed high-risk flag at the 0.12 threshold, the intervention tier, and any necessary metadata such as model version and scoring timestamp. It would be containerized with Docker and deployed behind a load balancer or through a managed serving platform such as AWS SageMaker. Model artifacts would be versioned and registered so that candidate models can be evaluated through controlled A/B testing and the current model can be rolled back immediately if quality, safety, or reliability concerns arise.

## 4. Clinician Dashboard

The score should appear in the existing clinician workflow through a SMART on FHIR application rather than a separate dashboard that requires another login. For each patient, the app would show the risk probability, risk tier, and recommended transition-of-care intervention. It would also display the leading patient-specific SHAP factors so that the score arrives with an explanation, not merely a number. The clinician remains responsible for interpreting this information in context and can document an override when information unavailable to the model changes the care plan.

The dashboard must be accessible and usable in routine clinical work. Risk-tier colors should be distinguishable for color-blind users and must never be the sole indicator of risk; each tier should pair its color with a clear text label and icon. Risk scores and SHAP charts should expose appropriate ARIA labels and alternative descriptions so they are usable with screen readers. The SMART on FHIR integration should keep the tool inside the clinician's existing workflow, without requiring a separate login or additional clicks beyond the workflow already described.

## 5. Alerting

Alerts should be deliberately selective. Only the High Risk and Very High Risk tiers would create an active work item or alert for the care-coordination team; Low and Moderate Risk results would remain visible passively in the SMART on FHIR view. This design reduces alert fatigue while reserving intensive transition resources for patients with the largest observed risk. In the held-out test cohort, only about 260 of 14,997 patients fell in the Very High Risk tier, illustrating that the most disruptive alert pathway can remain comparatively rare.

Notification channels should match urgency. A High Risk result should create an EHR task notification assigned to the care coordinator. A Very High Risk result should create that EHR task and also send a direct page or SMS to the on-call case manager. Email should not be the primary channel for Very High Risk alerts because it is not sufficiently real-time for this urgency level.

## 6. Monitoring and Feedback

Deployment requires continuous monitoring rather than a one-time launch. **Data drift** monitoring compares incoming feature distributions with training distributions to identify shifts in coding, workflow, patient mix, or data quality. **Concept drift** monitoring evaluates whether the relationship between features and real 30-day outcomes is changing as outcome labels become available, using rolling performance and calibration checks. **Fairness drift** monitoring reruns the Stage 5 subgroup analysis on a rolling basis to identify widening differences in recall, precision, or outcome rates across demographic groups. Material drift, performance degradation, or fairness gaps should trigger clinical and technical review, with a retraining signal sent through the model-governance process rather than automatically replacing the deployed model.

Every prediction should create an audit-trail record containing the timestamp, model version, an input-feature hash rather than raw protected health information, predicted probability, assigned tier, and any clinician override action with its reason code. This produces a complete, queryable record for HIPAA compliance audits and for retrospective review if an adverse outcome occurs.

**Architecture diagram annotation:** The SMART on FHIR dashboard branch should be understood as accessible, labeled decision support; the alerting branch distinguishes EHR-task escalation for High Risk from EHR task plus page/SMS escalation for Very High Risk; and the monitoring branch includes the prediction-level audit trail described above.
