# Employee Attrition Prediction

End-to-end ML solution that analyzes HR data to identify drivers of employee
attrition, compares classification models, and serves the best model through
a FastAPI REST API.

## Business Objective

XYZ Company sees ~15% annual attrition, driving up recruitment, onboarding and
project-delay costs. This project identifies the key factors behind attrition
and exposes a trained model via an API so HR systems can score employees and
target retention efforts.

## Project Structure

```
data/
  raw/            source files (general_data, employee_survey_data,
                   manager_survey_data, in_time, out_time, data_dictionary)
  merged_data/    employee_merged.csv — 5 files joined on EmployeeID (gitignored, regenerate from notebook)
  processed/      X_train / X_test / y_train / y_test after cleaning, feature
                   engineering, encoding and scaling
notebooks/
  data_understanding.ipynb   dataset study, merge strategy, data quality audit
  data_preprocessing.ipynb   cleaning, imputation, outlier check, EDA,
                              feature engineering, encoding, scaling
  model_training.ipynb       5 baseline models + hyperparameter tuning,
                              evaluation, SHAP, best-model export
models/
  best_model.pkl                trained model bundle (model, name, feature order, metrics)
  preprocessing_objects.pkl     fill values, encoders, scaler, column order — fit on TRAIN only
reports/           EDA charts, model comparison CSV, confusion matrix, SHAP plot
app/
  main.py            FastAPI application (/health, /predict)
  schemas.py         Pydantic request/response models
  preprocessing.py   reproduces the notebook's transform pipeline for one record
docs/
  sample_request.json / sample_response.json
postman/
  Employee_Attrition_API.postman_collection.json
run.py              convenience entrypoint (python run.py)
requirement.txt      dependencies
```

## 1–5. Data Understanding, Preprocessing, EDA, Feature Engineering, Modeling

Fully documented in the notebooks, run in this order:

1. `notebooks/data_understanding.ipynb` — loads the 5 raw files, documents each
   dataset's purpose, and merges them on `EmployeeID` into
   `data/merged_data/employee_merged.csv` (in_time/out_time collapse into a
   derived `avg_work_hours` feature).
2. `notebooks/data_preprocessing.ipynb` — dedups, does a stratified 80/20
   train/test split (split first, then all statistics — fill values, encoders,
   scaler — are fit on TRAIN only to avoid leakage), drops constant/ID columns,
   imputes missing values (mode for satisfaction scores, median for
   `NumCompaniesWorked`/`TotalWorkingYears`), checks IQR outliers (kept — they
   are genuine long-tenure employees, not data errors), runs EDA, engineers 5
   features (`TenureRatio`, `PromotionGap`, `IncomePerYear`, `IsOverworked`,
   `ManagerStability`), encodes categoricals (label encoding for the binary
   `Gender`, one-hot for the rest), scales continuous columns with
   `StandardScaler`, and saves `data/processed/*.csv` plus
   `models/preprocessing_objects.pkl`.
3. `notebooks/model_training.ipynb` — trains 5 baseline classifiers (Logistic
   Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) with
   class-imbalance handling (`class_weight='balanced'` / `scale_pos_weight`,
   since attrition is ~16% positive), evaluates each on Accuracy, Precision,
   Recall, F1, ROC-AUC and a confusion matrix, tunes the top-2 (by F1) with
   `RandomizedSearchCV`, and selects the best model by F1 on the test set.

**Model comparison** (`reports/model_comparison_report.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.988 | 0.975 | 0.952 | **0.964** | 0.987 |
| XGBoost (Tuned) | 0.988 | 1.000 | 0.928 | 0.963 | 0.984 |
| Random Forest | 0.979 | 0.950 | 0.920 | 0.935 | 0.987 |
| Random Forest (Tuned) | 0.979 | 0.966 | 0.904 | 0.934 | 0.989 |
| Decision Tree | 0.975 | 0.914 | 0.936 | 0.925 | 0.959 |
| Gradient Boosting | 0.883 | 0.595 | 0.880 | 0.710 | 0.944 |
| Logistic Regression | 0.757 | 0.378 | 0.760 | 0.505 | 0.833 |

**Selected model: XGBoost** — highest F1 (0.964) with strong recall
(0.952, i.e. it catches 95% of employees who actually leave — the metric that
matters most for a retention use case), saved as `models/best_model.pkl`
along with its exact training feature order and metrics. SHAP analysis
(`reports/shap_top10_features.png`) shows top drivers of attrition.

## 6. FastAPI Deployment

`app/` exposes the saved XGBoost model:

- `app/schemas.py` — `EmployeeFeatures` (request) with field-level validation
  (ranges/enums taken from the training data) and `PredictionResponse` /
  `HealthResponse` (response) Pydantic models.
- `app/preprocessing.py` — replays the exact notebook pipeline (impute →
  engineer features → encode → scale → reindex to training column order)
  using the objects saved in `models/preprocessing_objects.pkl`, so a raw
  JSON record is transformed identically to how training data was.
- `app/main.py` — loads `models/best_model.pkl` and
  `models/preprocessing_objects.pkl` once at startup (FastAPI `lifespan`),
  and exposes:
  - `GET /health` — model name + expected feature count
  - `POST /predict` — takes an `EmployeeFeatures` JSON body, returns
    `attrition_prediction` (`Yes`/`No`), `attrition_probability`,
    `risk_level` (`Low` < 0.3 ≤ `Medium` < 0.6 ≤ `High`), and `model_name`.
    Validation errors return `422` with field-level detail; unexpected
    prediction errors are caught and returned as `422` rather than a raw 500.

Fields imputed during training when missing
(`EnvironmentSatisfaction`, `JobSatisfaction`, `WorkLifeBalance`,
`NumCompaniesWorked`, `TotalWorkingYears`) are optional in the request — send
`null` or omit them and the API applies the same train-set fill values.

### Setup

```bash
python -m venv employeevenv
employeevenv\Scripts\activate          # Windows PowerShell: employeevenv\Scripts\Activate.ps1
pip install -r requirement.txt
```

Run the notebooks in order (data_understanding → data_preprocessing →
model_training) if `models/best_model.pkl` and
`models/preprocessing_objects.pkl` don't exist yet or you've changed the data.

### Run the API

```bash
python run.py
# or
uvicorn app.main:app --reload
```

API available at `http://127.0.0.1:8000`.

## 7. API Testing

### Swagger UI

1. Start the server, open `http://127.0.0.1:8000/docs`.
2. Expand `POST /predict` → **Try it out**, paste the contents of
   `docs/sample_request.json`, **Execute**.
3. Confirm a `200` response shaped like `docs/sample_response.json`.
4. Capture a screenshot of the request/response for the deliverable.

### Postman

1. Import `postman/Employee_Attrition_API.postman_collection.json` into Postman.
2. It includes: `Health Check`, three `/predict` happy-path requests
   (high-risk employee, low-risk employee, missing-survey-field imputation)
   and one intentionally invalid request that demonstrates the `422`
   validation error path.
3. Run each request against the running server
   (`baseUrl` collection variable defaults to `http://127.0.0.1:8000`) and
   screenshot the response body for the deliverable.

### Quick curl smoke test

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/sample_request.json
```

Expected: `{"attrition_prediction":"Yes","attrition_probability":0.9968,"risk_level":"High","model_name":"XGBoost"}`
