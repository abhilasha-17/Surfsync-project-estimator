# SurfSync Project Estimator

A project-estimation application that uses software requirements to estimate development effort and project duration. It also calculates an indicative cost and a rule-based complexity level.

> **Data limitation:** The models were trained on synthetic project records because real completed-project data was unavailable. Metrics describe results on a synthetic holdout set and do not establish accuracy on real client projects. Estimates are preliminary planning support, not guaranteed quotations.

## What it does

The Streamlit interface collects project requirements and sends them to a FastAPI service. The backend validates the input and uses saved model pipelines to estimate:

- Development effort in hours
- Project duration in calendar days
- Indicative cost, calculated as predicted hours × ₹1,200 per hour
- Complexity level from a transparent rule-based score
- Global permutation-importance values for overall model sensitivity

Cost is derived from predicted hours; it is not a separately trained model. Complexity is rule-based, not a trained classifier.

## Project structure

```text
SurfSync Project/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── requirements.txt
├── data/
│   ├── cross_validation_results.csv
│   ├── data_quality_issue_log.csv
│   ├── derived_cost_metrics.csv
│   ├── feature_importance_development_hours.csv
│   ├── feature_importance_duration_days.csv
│   ├── generation_assumptions.md
│   ├── holdout_test_results.csv
│   ├── preprocessing_quality_report.csv
│   └── synthetic_projects_raw.csv
├── docs/
│   ├── API.md
├── frontend/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── .env.example
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── models/
│   ├── development_hours_model.joblib
│   └── duration_days_model.joblib
├── generate_dataset.py
├── preprocessing_modeling.ipynb
└── README.md
```

## Dataset and preprocessing

The project uses 400 synthetic records. The raw CSV contains intentionally introduced data-quality issues, documented in `data/data_quality_issue_log.csv`. The raw file is preserved; the issue log records the generated examples.

The modeling workflow includes category and range checks, feature engineering, preprocessing pipelines, cross-validation, and holdout evaluation. Missing numeric values are imputed with the median; missing categorical values are imputed with the most frequent category. Categorical inputs are one-hot encoded. Preprocessing is fitted within training folds to help prevent data leakage.

Engineered features:

- `required_feature_count`: count of “Yes” values across the four feature requirement fields
- `integrations_per_module`: number of integrations divided by number of modules

Actual target values are excluded from the model inputs. The project name is used for display and is not a model feature.

## Models and evaluation

Candidate models were compared using repeated cross-validation on the training data. The selected model is the one with the lowest mean cross-validation MAE. The holdout set is reserved for final evaluation and is not used for model selection.

| Target | Selected model | Training CV MAE | Holdout MAE | Holdout RMSE | Holdout R² |
|---|---|---:|---:|---:|---:|
| Development hours | Linear Regression | 97.09 hours | 97.26 hours | 144.07 hours | 0.649 |
| Duration | XGBoost Regressor | 10.10 days | 8.58 days | 14.34 days | 0.889 |

MAE and RMSE are measured in each target’s units. Lower values indicate smaller errors. R² describes fit relative to a mean-based baseline and can be negative.

The cost holdout MAE is ₹116,717.60. Complexity labels are produced by a rule-based score. Agreement with the synthetic labels checks that the rule reproduces the generator’s labels; it is not classifier accuracy.

## Run locally on Windows

Run these commands from the project root. Python 3.13 is the version used for this project.

Create and activate a virtual environment, then install both dependency sets:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements.txt -r .\frontend\requirements.txt
```

Start the backend in the first terminal:

```powershell
python -m uvicorn backend.main:app --reload
```

Start the frontend in a second terminal, also from the project root:

```powershell
python -m streamlit run .\frontend\app.py
```

Open the app at `http://localhost:8501`.

The API runs at `http://127.0.0.1:8000`. Its interactive documentation is at `http://127.0.0.1:8000/docs`.

## API

| Method and route | Purpose |
|---|---|
| `GET /health` | Checks API status and whether both models loaded |
| `GET /model-info` | Shows selected models, features, assumptions, and available metrics |
| `POST /predict` | Validates input and returns an estimate |

See [docs/API.md](docs/API.md) for endpoint details and request/response examples.

## Explainability

The dashboard displays global permutation importance separately for hours and duration. A larger positive value means shuffling that feature increased holdout MAE more. These values describe overall model sensitivity; they are not project-specific explanations or proof of causation.

## Limitations and next steps

- The dataset and labels are synthetic and may not reflect real project-estimation patterns.
- The holdout evaluation uses a small synthetic test set.
- Cost uses a fixed rate of ₹1,200 per person-hour.
- Complexity is rule-based, not a trained classifier.
- Collect real completed-project outcomes and validate the estimates before operational use.

## Project links

- GitHub repository: https://github.com/abhilasha-17/Surfsync-project-estimator
- Live application: https://surfsync-project-estimator.streamlit.app
- API documentation: https://surfsync-project-estimator.onrender.com/docs

## Demo Video

[Watch the 4:41 project demo] (https://drive.google.com/file/d/1abj5PM9kC3y5Ylt1OijefrQvkFqgUdwT/view?usp=sharing)
