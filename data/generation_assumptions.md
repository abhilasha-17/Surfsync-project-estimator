# Synthetic dataset assumptions

- Rows: 400
- Random seed: 42; rerunning this script reproduces the dataset.
- The dataset is synthetic educational data, not historical company data.
- Effort is generated from documented weights for modules, roles,
  integrations, optional features, complexity, and developer experience.
- A lognormal noise factor adds variation between projects with similar inputs.
- Duration assumes 6 productive hours per developer per day.
- Team capacity grows sub-linearly to represent coordination overhead.
- Cost is development hours multiplied by a fixed assumed blended rate of
  INR 1,200 per person-hour.
- Complexity level is assigned from a feature score and thresholds in the
  generation script.
- Controlled issues are added after targets are generated: missing input
  values, inconsistent category labels, and numeric fields with text suffixes.
- Target columns, IDs, and project names are not intentionally corrupted.
- Model results on this dataset measure fit to these assumptions; they do not
  prove real-world prediction accuracy.
- The issue-by-issue record is in `data/data_quality_issue_log.csv`.
