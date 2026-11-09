
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("SURFSYNC_MODELS_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("SURFSYNC_DATA_DIR", PROJECT_ROOT / "data"))
RATE_INR_PER_HOUR = 1_200

MODEL_FILES = {
    "development_hours": "development_hours_model.joblib",
    "duration_days": "duration_days_model.joblib",
}

FEATURE_COLUMNS = [
    "project_type",
    "num_modules",
    "num_user_roles",
    "num_integrations",
    "payment_gateway_required",
    "admin_panel_required",
    "mobile_app_required",
    "ai_ml_required",
    "ui_ux_complexity",
    "backend_complexity",
    "num_developers",
    "developer_experience",
    "required_feature_count",
    "integrations_per_module",
]

PROJECT_TYPES = (
    "Web Application",
    "Mobile Application",
    "SaaS",
    "E-commerce",
    "CRM",
    "ERP",
    "AI/ML",
    "Other",
)
EXPERIENCE_LEVELS = ("Junior", "Mid-level", "Senior", "Mixed")
COMPLEXITY_LEVELS = ("Low", "Medium", "High")
YES_NO = ("Yes", "No")


class PredictionRequest(BaseModel):
    """Validated raw form fields; engineered columns are calculated server-side."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_name: str = Field(min_length=1, max_length=100)
    project_type: Literal[
        "Web Application",
        "Mobile Application",
        "SaaS",
        "E-commerce",
        "CRM",
        "ERP",
        "AI/ML",
        "Other",
    ]
    num_modules: int = Field(ge=2, le=15)
    num_user_roles: int = Field(ge=1, le=8)
    num_integrations: int = Field(ge=0, le=6)
    payment_gateway_required: Literal["Yes", "No"]
    admin_panel_required: Literal["Yes", "No"]
    mobile_app_required: Literal["Yes", "No"]
    ai_ml_required: Literal["Yes", "No"]
    ui_ux_complexity: Literal["Low", "Medium", "High"]
    backend_complexity: Literal["Low", "Medium", "High"]
    num_developers: int = Field(ge=1, le=8)
    developer_experience: Literal["Junior", "Mid-level", "Senior", "Mixed"]

    @field_validator("project_name")
    @classmethod
    def project_name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be blank")
        return value

    def model_features(self) -> dict[str, Any]:
        flags = [
            self.payment_gateway_required,
            self.admin_panel_required,
            self.mobile_app_required,
            self.ai_ml_required,
        ]
        row = {
            "project_type": self.project_type,
            "num_modules": self.num_modules,
            "num_user_roles": self.num_user_roles,
            "num_integrations": self.num_integrations,
            "payment_gateway_required": self.payment_gateway_required,
            "admin_panel_required": self.admin_panel_required,
            "mobile_app_required": self.mobile_app_required,
            "ai_ml_required": self.ai_ml_required,
            "ui_ux_complexity": self.ui_ux_complexity,
            "backend_complexity": self.backend_complexity,
            "num_developers": self.num_developers,
            "developer_experience": self.developer_experience,
            "required_feature_count": sum(value == "Yes" for value in flags),
            "integrations_per_module": self.num_integrations / self.num_modules,
        }
        return {column: row[column] for column in FEATURE_COLUMNS}


def _load_permutation_importance() -> list[dict[str, Any]]:
    files = {
        "development_hours": DATA_DIR / "feature_importance_development_hours.csv",
        "duration_days": DATA_DIR / "feature_importance_duration_days.csv",
    }
    by_feature: dict[str, dict[str, Any]] = {}
    for target, path in files.items():
        if not path.exists():
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            continue
        if not {"feature", "importance_mean"}.issubset(table.columns):
            continue
        key = "hours_importance" if target == "development_hours" else "days_importance"
        for row in table[["feature", "importance_mean"]].itertuples(index=False):
            if pd.isna(row.importance_mean):
                continue
            feature = str(row.feature)
            by_feature.setdefault(feature, {"feature": feature})[key] = float(row.importance_mean)

    factors = list(by_feature.values())
    factors.sort(
        key=lambda row: max(
            abs(row.get("hours_importance", 0.0)),
            abs(row.get("days_importance", 0.0)),
        ),
        reverse=True,
    )
    return factors[:6]


def _load_metric_summary() -> dict[str, dict[str, Any]]:
    """Load selected-model CV and holdout metrics when notebook CSVs are available."""
    summary: dict[str, dict[str, Any]] = {}
    cv_path = DATA_DIR / "cross_validation_results.csv"
    holdout_path = DATA_DIR / "holdout_test_results.csv"

    try:
        cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()
        holdout = pd.read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()
    except (OSError, pd.errors.ParserError, UnicodeError):
        return summary

    for target in MODEL_FILES:
        target_metrics: dict[str, Any] = {}
        if not cv.empty and {"target", "model", "CV_MAE_mean"}.issubset(cv.columns):
            subset = cv.loc[cv["target"] == target]
            if not subset.empty:
                winner = subset.sort_values("CV_MAE_mean").iloc[0]
                target_metrics["selected_model"] = str(winner["model"])
                target_metrics["cv_mae"] = float(winner["CV_MAE_mean"])
                if "CV_MAE_std" in winner and pd.notna(winner["CV_MAE_std"]):
                    target_metrics["cv_mae_std"] = float(winner["CV_MAE_std"])

        if not holdout.empty and {"target", "model", "test_MAE"}.issubset(holdout.columns):
            subset = holdout.loc[holdout["target"] == target]
            if "selected_by_training_CV" in subset.columns:
                selected = subset.loc[subset["selected_by_training_CV"].astype(str).str.lower() == "true"]
            else:
                selected = pd.DataFrame()
            if not selected.empty:
                winner = selected.iloc[0]
                target_metrics["test_mae"] = float(winner["test_MAE"])
                for source, dest in (("test_RMSE", "test_rmse"), ("test_R2", "test_r2")):
                    if source in winner and pd.notna(winner[source]):
                        target_metrics[dest] = float(winner[source])
        if target_metrics:
            summary[target] = target_metrics
    return summary


def _complexity_label(data: PredictionRequest) -> str:
    """Apply the same transparent rule used to generate the synthetic labels."""
    complexity_points = {"Low": 0, "Medium": 1, "High": 2}
    score = (
        data.num_modules / 3
        + data.num_user_roles / 2
        + data.num_integrations * 0.8
        + (data.payment_gateway_required == "Yes") * 1.0
        + (data.admin_panel_required == "Yes") * 0.5
        + (data.mobile_app_required == "Yes") * 1.5
        + (data.ai_ml_required == "Yes") * 2.0
        + complexity_points[data.ui_ux_complexity] * 1.0
        + complexity_points[data.backend_complexity] * 1.2
    )
    if score <= 7:
        return "Low"
    if score <= 12:
        return "Medium"
    if score <= 14:
        return "High"
    return "Very High"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.model_load_error = None
    app.state.feature_importance = _load_permutation_importance()
    app.state.metrics = _load_metric_summary()
    try:
        missing = [
            str(MODELS_DIR / filename)
            for filename in MODEL_FILES.values()
            if not (MODELS_DIR / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Model files not found: " + ", ".join(missing)
            )
        app.state.models = {
            target: joblib.load(MODELS_DIR / filename)
            for target, filename in MODEL_FILES.items()
        }
    except Exception as exc:  # Keep health/model-info endpoints available for diagnosis.
        app.state.models = {}
        app.state.model_load_error = str(exc)
    yield


app = FastAPI(
    title="SurfSync Project Estimator API",
    description=(
        "Predicts development hours and calendar duration from project requirements. "
        "Cost is derived using the configured hourly-rate assumption."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = bool(app.state.models)
    return {
        "status": "ok",
        "models_loaded": ready,
        "message": "Models ready" if ready else "API running; model artifacts need attention",
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    loaded = bool(app.state.models)
    selected_models: dict[str, str | None] = {}
    for target in MODEL_FILES:
        pipeline = app.state.models.get(target)
        estimator = getattr(pipeline, "named_steps", {}).get("model") if pipeline else None
        selected_models[target] = type(estimator).__name__ if estimator is not None else None
    return {
        "service": "SurfSync Project Estimator",
        "models_loaded": loaded,
        "model_load_error": app.state.model_load_error,
        "selected_models": selected_models,
        "model_features": FEATURE_COLUMNS,
        "cost_rate_inr_per_hour": RATE_INR_PER_HOUR,
        "complexity_method": "rule-based scoring",
        "dataset_note": "Models were trained on synthetic project records.",
        "metrics": app.state.metrics,
    }


@app.post("/predict")
def predict(data: PredictionRequest) -> dict[str, Any]:
    if not app.state.models:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction models are not loaded. Place development_hours_model.joblib "
                "and duration_days_model.joblib in the models folder, then restart the API. "
                f"Details: {app.state.model_load_error}"
            ),
        )

    model_input = pd.DataFrame([data.model_features()], columns=FEATURE_COLUMNS)
    try:
        hours = float(np.asarray(app.state.models["development_hours"].predict(model_input)).ravel()[0])
        days = float(np.asarray(app.state.models["duration_days"].predict(model_input)).ravel()[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    if not np.isfinite(hours) or not np.isfinite(days):
        raise HTTPException(status_code=500, detail="The model returned a non-finite prediction")

    hours = max(0.0, hours)
    days = max(0.0, days)
    return {
        "project_name": data.project_name,
        "predicted_hours": round(hours, 2),
        "predicted_days": round(days, 2),
        "predicted_cost_inr": round(hours * RATE_INR_PER_HOUR, 2),
        "hourly_rate_inr": RATE_INR_PER_HOUR,
        "predicted_complexity": _complexity_label(data),
        "factors": app.state.feature_importance,
        "explanation_type": "global permutation importance",
        "selected_models": {
            target: type(
                getattr(app.state.models[target], "named_steps", {}).get("model")
            ).__name__
            for target in MODEL_FILES
        },
        "estimate_note": "Preliminary estimate from models trained on synthetic data.",
    }
