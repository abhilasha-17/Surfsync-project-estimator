# SurfSync Streamlit Frontend

This frontend collects the project inputs required by the SurfSync assignment and sends them to a FastAPI backend. It does not generate fake predictions when the backend is unavailable.

## Run locally

From the `frontend` directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

By default, the frontend expects the API at `http://127.0.0.1:8000`. Set `SURFSYNC_API_URL` if the backend runs elsewhere. In PowerShell:

```powershell
$env:SURFSYNC_API_URL = "http://127.0.0.1:8000"
streamlit run app.py
```

## API contract for the backend

The frontend checks `GET /health` and submits the form to `POST /predict`.

### Request

```json
{
  "project_name": "Customer support portal",
  "project_type": "Web Application",
  "num_modules": 6,
  "num_user_roles": 3,
  "num_integrations": 2,
  "payment_gateway_required": "No",
  "admin_panel_required": "Yes",
  "mobile_app_required": "No",
  "ai_ml_required": "No",
  "ui_ux_complexity": "Medium",
  "backend_complexity": "Medium",
  "num_developers": 3,
  "developer_experience": "Mid-level"
}
```

The backend should derive the two engineered inputs exactly as the notebook does before calling the saved model pipelines:

- `required_feature_count`: count the `"Yes"` values across payment gateway, admin panel, mobile app, and AI/ML requirements.
- `integrations_per_module`: `num_integrations / num_modules`.

It should exclude `project_name` from model inputs.

### Response

```json
{
  "predicted_hours": 486.4,
  "predicted_days": 31.6,
  "predicted_cost_inr": 583680,
  "predicted_complexity": "Medium",
  "factors": [
    {"feature": "num_modules", "hours_importance": 87.01, "days_importance": 7.24},
    {"feature": "num_developers", "hours_importance": 0.62, "days_importance": 25.27}
  ]
}
```

## Notes

- The numeric input limits match the ranges recorded in the notebook: modules 2–15, roles 1–8, integrations 0–6, and developers 1–8.
- The UI values for categories match the cleaned dataset categories.
- Predictions are preliminary because the training data is synthetic; do not present them as guaranteed quotations.
