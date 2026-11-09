from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="SurfSync | Project Estimator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("SURFSYNC_API_URL", "http://127.0.0.1:8000").rstrip("/")
RATE_INR_PER_HOUR = 1_200
PROJECT_TYPES = [
    "Web Application",
    "Mobile Application",
    "SaaS",
    "E-commerce",
    "CRM",
    "ERP",
    "AI/ML",
    "Other",
]
EXPERIENCE_LEVELS = ["Junior", "Mid-level", "Senior", "Mixed"]
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]


st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
      .hero { padding: 1.6rem 1.8rem; border-radius: 20px;
        background: linear-gradient(115deg, #102a43 0%, #174c68 58%, #147d78 100%);
        color: #fff; margin-bottom: 1.2rem; }
      .hero h1 { font-size: 2.15rem; margin: 0 0 .35rem 0; color: #fff; }
      .hero p { color: #d7e7ee; font-size: 1rem; margin: 0; }
      .section-label { font-size: 1.12rem; font-weight: 700; color: #17324d;
        margin: .7rem 0 .35rem 0; }
      div[data-testid="stMetric"] { background: #fff; border: 1px solid #e5ebf1;
        padding: 1rem; border-radius: 14px; box-shadow: 0 3px 12px #17324d0b; }
      .note { color: #536579; font-size: .9rem; }
      .stButton > button[kind="primary"] { border-radius: 10px; min-height: 3rem;
        font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read the first available response key, supporting nested API responses."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _format_inr(value: Any) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _format_number(value: Any, suffix: str) -> str:
    try:
        return f"{float(value):,.1f} {suffix}"
    except (TypeError, ValueError):
        return "—"


def _check_api() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "ok")).lower()
        if status not in {"ok", "healthy", "ready"}:
            return False, f"API responded with status: {status}"
        if payload.get("models_loaded") is False:
            return False, payload.get(
                "message", "API is running, but the saved prediction models are not loaded"
            )
        return True, "Prediction service connected"
    except requests.RequestException:
        return False, "Prediction service is not connected yet"
    except ValueError:
        return False, "Prediction service returned an unreadable health response"


def _render_factors(result: dict[str, Any]) -> None:
    factors = _get(result, "factors", "explanation", "top_factors", default=[])
    if isinstance(factors, dict):
        factors = [
            {"feature": feature, "importance": importance}
            for feature, importance in factors.items()
        ]
    if not isinstance(factors, list) or not factors:
        st.info(
            "The prediction service did not return feature-impact details. "
            "The backend can add these from the saved permutation-importance results."
        )
        return

    rows: list[dict[str, Any]] = []
    for item in factors[:6]:
        if isinstance(item, dict):
            rows.append({
                "Factor": item.get("feature", item.get("name", "Feature")),
                "Hours importance": item.get(
                    "hours_importance", item.get("effect", item.get("direction", "—"))
                ),
                "Days importance": item.get("days_importance", "—"),
            })
        else:
            rows.append({"Factor": str(item), "Hours importance": "—", "Days importance": "—"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Global permutation importance: a larger positive value means shuffling that input "
        "increased holdout MAE more. This describes overall model sensitivity, not a "
        "project-specific causal explanation."
    )


st.markdown(
    """
    <div class="hero">
      <h1>Project Estimator</h1>
      <p>Turn project requirements into an early effort, timeline, and budget estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_ok, api_message = _check_api()
status_col, assumption_col = st.columns([1, 2])
with status_col:
    if api_ok:
        st.success(api_message)
    else:
        st.warning(f"{api_message}. Start the FastAPI backend to generate real predictions.")
with assumption_col:
    st.caption(
        f"Cost is derived from estimated hours at the current assumption of "
        f"₹{RATE_INR_PER_HOUR:,} per hour. Estimates are preliminary."
    )

st.markdown('<div class="section-label">Project details</div>', unsafe_allow_html=True)

with st.form("project_estimation_form", clear_on_submit=False):
    project_name = st.text_input(
        "Project name",
        placeholder="e.g., Customer support portal",
        max_chars=100,
        help="Used to label this estimate; it is not a model input.",
    )

    a, b, c = st.columns(3)
    with a:
        project_type = st.selectbox("Project type", PROJECT_TYPES)
        num_modules = st.number_input(
            "Number of modules", min_value=2, max_value=15, value=6, step=1
        )
        num_user_roles = st.number_input(
            "Number of user roles", min_value=1, max_value=8, value=3, step=1
        )
        num_integrations = st.number_input(
            "Third-party integrations", min_value=0, max_value=6, value=2, step=1
        )
    with b:
        payment_gateway_required = st.selectbox("Payment gateway", ["No", "Yes"])
        admin_panel_required = st.selectbox("Admin panel", ["Yes", "No"])
        mobile_app_required = st.selectbox("Mobile app required", ["No", "Yes"])
        ai_ml_required = st.selectbox("AI/ML functionality", ["No", "Yes"])
    with c:
        ui_ux_complexity = st.selectbox("UI/UX complexity", COMPLEXITY_LEVELS)
        backend_complexity = st.selectbox("Backend complexity", COMPLEXITY_LEVELS)
        num_developers = st.number_input(
            "Number of developers", min_value=1, max_value=8, value=3, step=1
        )
        developer_experience = st.selectbox("Developer experience", EXPERIENCE_LEVELS)

    st.caption(
        "The numeric limits follow the ranges used for this project dataset. "
        "The backend derives engineered features from these form inputs."
    )
    submitted = st.form_submit_button(
        "Generate estimate", type="primary", use_container_width=True
    )


if submitted:
    if not project_name.strip():
        st.error("Enter a project name so the estimate can be identified.")
    else:
        request_payload = {
            "project_name": project_name.strip(),
            "project_type": project_type,
            "num_modules": int(num_modules),
            "num_user_roles": int(num_user_roles),
            "num_integrations": int(num_integrations),
            "payment_gateway_required": payment_gateway_required,
            "admin_panel_required": admin_panel_required,
            "mobile_app_required": mobile_app_required,
            "ai_ml_required": ai_ml_required,
            "ui_ux_complexity": ui_ux_complexity,
            "backend_complexity": backend_complexity,
            "num_developers": int(num_developers),
            "developer_experience": developer_experience,
        }

        try:
            with st.spinner("Estimating project effort and timeline…"):
                response = requests.post(
                    f"{API_URL}/predict", json=request_payload, timeout=60
                )
            if not response.ok:
                detail = response.text[:500] or f"HTTP {response.status_code}"
                st.error(f"The prediction request failed: {detail}")
            else:
                body = response.json()
                result = body.get("prediction", body)

                hours = _get(result, "predicted_hours", "development_hours")
                days = _get(result, "predicted_days", "duration_days")
                cost = _get(
                    result,
                    "predicted_cost_inr",
                    "predicted_cost_INR",
                    "development_cost_inr",
                )
                if cost is None and hours is not None:
                    cost = float(hours) * RATE_INR_PER_HOUR
                complexity = _get(
                    result, "predicted_complexity", "complexity_level", default="—"
                )

                st.markdown("### Estimate")
                st.caption(f"Estimate for **{project_name.strip()}**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Development effort", _format_number(hours, "hours"))
                m2.metric("Estimated timeline", _format_number(days, "days"))
                m3.metric("Indicative cost", _format_inr(cost))
                m4.metric("Complexity", str(complexity))

                st.markdown("### Which inputs influence the models?")
                _render_factors(result)
                selected_models = result.get("selected_models", {})
                if selected_models:
                    st.caption(
                        "Selected models — hours: "
                        f"{selected_models.get('development_hours', '—')}; duration: "
                        f"{selected_models.get('duration_days', '—')}"
                    )

                st.info(
                    "This is a preliminary estimate based on a model trained with synthetic "
                    "project data. Review it with a project or engineering lead before using "
                    "it for a commitment or quotation."
                )
        except requests.RequestException as exc:
            st.error(
                "Could not reach the prediction API. Confirm that the FastAPI service is "
                f"running at {API_URL}. Details: {exc}"
            )
        except ValueError:
            st.error("The prediction API returned invalid JSON. Check the backend response.")

with st.expander("How this estimate is produced"):
    st.write(
        "Development hours and duration are produced by the selected regression pipelines. "
        "Cost is calculated from predicted hours using the displayed hourly-rate assumption. "
        "Complexity is a transparent rule-based estimate. The model influence section should "
        "be populated by the backend using the saved feature-importance results."
    )

st.divider()
st.caption("SurfSync Project Estimator · Planning support, not a guaranteed quotation")
