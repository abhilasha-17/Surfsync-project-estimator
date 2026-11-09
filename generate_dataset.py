from pathlib import Path

import numpy as np
import pandas as pd


# Reproducible output
SEED = 42
N_ROWS = 400
ASSUMED_HOURLY_RATE_INR = 1_200

rng = np.random.default_rng(SEED)
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)


def choose(values, probabilities):
    """Choose one value using the supplied probabilities."""
    return rng.choice(values, p=probabilities)


# ---------- Generate clean project inputs ----------

project_types = [
    "Web Application",
    "Mobile Application",
    "SaaS",
    "E-commerce",
    "CRM",
    "ERP",
    "AI/ML",
    "Other",
]

records = []

for i in range(N_ROWS):
    project_type = choose(
        project_types,
        [0.24, 0.15, 0.16, 0.14, 0.10, 0.08, 0.08, 0.05],
    )

    num_modules = int(rng.integers(2, 16))
    num_user_roles = int(rng.integers(1, min(9, num_modules + 2)))
    num_integrations = int(rng.integers(0, 7))

    if project_type == "E-commerce":
        payment_gateway = "Yes" if rng.random() < 0.80 else "No"
    else:
        payment_gateway = "Yes" if rng.random() < 0.25 else "No"

    admin_panel = "Yes" if rng.random() < 0.75 else "No"

    if project_type == "Mobile Application":
        mobile_app = "Yes"
    else:
        mobile_app = "Yes" if rng.random() < 0.22 else "No"

    if project_type == "AI/ML":
        ai_ml = "Yes"
    else:
        ai_ml = "Yes" if rng.random() < 0.15 else "No"

    ui_ux_complexity = choose(
        ["Low", "Medium", "High"],
        [0.25, 0.50, 0.25],
    )
    backend_complexity = choose(
        ["Low", "Medium", "High"],
        [0.20, 0.50, 0.30],
    )

    num_developers = int(rng.integers(1, 9))
    developer_experience = choose(
        ["Junior", "Mid-level", "Senior", "Mixed"],
        [0.20, 0.40, 0.20, 0.20],
    )

    records.append(
        {
            "project_id": f"PRJ-{i + 1:04d}",
            "project_name": f"Synthetic Project {i + 1:04d}",
            "project_type": project_type,
            "num_modules": num_modules,
            "num_user_roles": num_user_roles,
            "num_integrations": num_integrations,
            "payment_gateway_required": payment_gateway,
            "admin_panel_required": admin_panel,
            "mobile_app_required": mobile_app,
            "ai_ml_required": ai_ml,
            "ui_ux_complexity": ui_ux_complexity,
            "backend_complexity": backend_complexity,
            "num_developers": num_developers,
            "developer_experience": developer_experience,
        }
    )

df = pd.DataFrame(records)



complexity_points = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

experience_effort_factor = {
    "Junior": 1.15,
    "Mid-level": 1.00,
    "Senior": 0.90,
    "Mixed": 0.98,
}

base_hours = (
    45
    + df["num_modules"] * 32
    + df["num_user_roles"] * 11
    + df["num_integrations"] * 24
    + df["payment_gateway_required"].eq("Yes") * 38
    + df["admin_panel_required"].eq("Yes") * 32
    + df["mobile_app_required"].eq("Yes") * 100
    + df["ai_ml_required"].eq("Yes") * 125
    + df["ui_ux_complexity"].map(complexity_points) * 55
    + df["backend_complexity"].map(complexity_points) * 75
)

experience_factor = df["developer_experience"].map(experience_effort_factor)

effort_noise = rng.lognormal(mean=0.0, sigma=0.16, size=N_ROWS)

df["development_hours"] = np.maximum(
    40,
    np.round(base_hours * experience_factor * effort_noise),
).astype(int)

# Assumption: each developer contributes 6 productive hours per calendar day.
# Team capacity grows sub-linearly to reflect coordination overhead.
team_size = df["num_developers"].to_numpy()
effective_team_capacity = 0.65 * team_size + 0.35 * np.sqrt(team_size)

duration_noise = rng.lognormal(mean=0.0, sigma=0.10, size=N_ROWS)

df["duration_days"] = np.maximum(
    5,
    np.ceil(
        df["development_hours"]
        / (6 * effective_team_capacity)
        * duration_noise
    ),
).astype(int)

# Cost uses a fixed assumed blended rate. The rate is not a model input.
df["development_cost_inr"] = (
    df["development_hours"] * ASSUMED_HOURLY_RATE_INR
).astype(int)

complexity_score = (
    df["num_modules"] / 3
    + df["num_user_roles"] / 2
    + df["num_integrations"] * 0.8
    + df["payment_gateway_required"].eq("Yes") * 1.0
    + df["admin_panel_required"].eq("Yes") * 0.5
    + df["mobile_app_required"].eq("Yes") * 1.5
    + df["ai_ml_required"].eq("Yes") * 2.0
    + df["ui_ux_complexity"].map(complexity_points) * 1.0
    + df["backend_complexity"].map(complexity_points) * 1.2
)

df["complexity_level"] = pd.cut(
    complexity_score,
    bins=[-np.inf, 7, 12, 14, np.inf],
    labels=["Low", "Medium", "High", "Very High"],
).astype(str)


column_order = [
    "project_id",
    "project_name",
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
    "development_hours",
    "duration_days",
    "development_cost_inr",
    "complexity_level",
]
df = df[column_order]



issue_records = []


numeric_input_columns = [
    "num_modules",
    "num_user_roles",
    "num_integrations",
    "num_developers",
]
for column in numeric_input_columns:
    df[column] = df[column].astype(object)


missing_columns = [
    "project_type",
    "num_modules",
    "num_user_roles",
    "num_integrations",
    "ui_ux_complexity",
    "backend_complexity",
    "num_developers",
    "developer_experience",
]

missing_per_column = round(N_ROWS * 0.03)

for column in missing_columns:
    selected_indices = rng.choice(
        df.index.to_numpy(),
        size=missing_per_column,
        replace=False,
    )

    for row_index in selected_indices:
        original_value = df.at[row_index, column]

        issue_records.append(
            {
                "project_id": df.at[row_index, "project_id"],
                "column": column,
                "issue": "missing_value",
                "original_value": str(original_value),
                "modified_value": "",
            }
        )
        df.at[row_index, column] = np.nan


category_variants = {
    "ui_ux_complexity": {
        "Low": ["low", "LOW", "Low "],
        "Medium": ["medium", "MEDIUM", "Medium "],
        "High": ["high", "HIGH", "High "],
    },
    "backend_complexity": {
        "Low": ["low", "LOW", "Low "],
        "Medium": ["medium", "MEDIUM", "Medium "],
        "High": ["high", "HIGH", "High "],
    },
    "developer_experience": {
        "Junior": ["junior", "JUNIOR", "Junior "],
        "Mid-level": ["Mid Level", "mid-level", "Mid-level "],
        "Senior": ["senior", "SENIOR", "Senior "],
        "Mixed": ["mixed", "MIXED", "Mixed "],
    },
}

for column, variants_by_value in category_variants.items():
    eligible_indices = df.index[df[column].notna()].to_numpy()
    selected_indices = rng.choice(
        eligible_indices,
        size=min(12, len(eligible_indices)),
        replace=False,
    )

    for row_index in selected_indices:
        original_value = df.at[row_index, column]
        modified_value = choose(variants_by_value[original_value], None)

        issue_records.append(
            {
                "project_id": df.at[row_index, "project_id"],
                "column": column,
                "issue": "inconsistent_category",
                "original_value": original_value,
                "modified_value": modified_value,
            }
        )
        df.at[row_index, column] = modified_value



numeric_string_examples = {
    "num_modules": " modules",
    "num_integrations": " integrations",
    "num_developers": " developers",
}

for column, suffix in numeric_string_examples.items():
    eligible_indices = df.index[df[column].notna()].to_numpy()
    selected_indices = rng.choice(
        eligible_indices,
        size=3,
        replace=False,
    )

    for row_index in selected_indices:
        original_value = int(df.at[row_index, column])
        modified_value = f"{original_value}{suffix}"

        issue_records.append(
            {
                "project_id": df.at[row_index, "project_id"],
                "column": column,
                "issue": "numeric_value_with_text_suffix",
                "original_value": str(original_value),
                "modified_value": modified_value,
            }
        )
        df.at[row_index, column] = modified_value



raw_file = output_dir / "synthetic_projects_raw.csv"
issue_file = output_dir / "data_quality_issue_log.csv"
assumptions_file = output_dir / "generation_assumptions.md"

df.to_csv(raw_file, index=False)
pd.DataFrame(issue_records).to_csv(issue_file, index=False)

assumptions_text = f"""# Synthetic dataset assumptions

- Rows: {N_ROWS}
- Random seed: {SEED}; rerunning this script reproduces the dataset.
- The dataset is synthetic educational data, not historical company data.
- Effort is generated from documented weights for modules, roles,
  integrations, optional features, complexity, and developer experience.
- A lognormal noise factor adds variation between projects with similar inputs.
- Duration assumes 6 productive hours per developer per day.
- Team capacity grows sub-linearly to represent coordination overhead.
- Cost is development hours multiplied by a fixed assumed blended rate of
  INR {ASSUMED_HOURLY_RATE_INR:,} per person-hour.
- Complexity level is assigned from a feature score and thresholds in the
  generation script.
- Controlled issues are added after targets are generated: missing input
  values, inconsistent category labels, and numeric fields with text suffixes.
- Target columns, IDs, and project names are not intentionally corrupted.
- Model results on this dataset measure fit to these assumptions; they do not
  prove real-world prediction accuracy.
- The issue-by-issue record is in `{issue_file.as_posix()}`.
"""

assumptions_file.write_text(assumptions_text, encoding="utf-8")

print(f"Saved raw dataset: {raw_file}")
print(f"Saved issue log: {issue_file}")
print(f"Saved assumptions: {assumptions_file}")
print(f"Rows: {len(df)}")
print(f"Logged data-quality issues: {len(issue_records)}")
print("\nComplexity label counts:")
print(df["complexity_level"].value_counts(dropna=False))