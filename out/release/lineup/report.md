## Saved ODI forecast comparison

Evaluation years: 2023, 2024, 2025. Models are fitted separately by competition,
with past-only tuning and a fixed forecast model within each evaluation year.
Lower log loss and Brier scores are better. Each match has equal weight.

This is a retrospective pilot restricted to decisive matches with fixed XIs.
It is not a published ranking of current player ability.

### 5-year window (primary)

| Competition | Model | Matches | Log loss | Brier |
| --- | --- | ---: | ---: | ---: |
| Men | 50:50 reference | 364 | 0.693 | 0.250 |
| Men | Team only | 364 | 0.689 | 0.247 |
| Men | Players only | 364 | 0.687 | 0.247 |
| Men | Teams + players | 364 | 0.686 | 0.246 |
| Women | 50:50 reference | 194 | 0.693 | 0.250 |
| Women | Team only | 194 | 0.639 | 0.225 |
| Women | Players only | 194 | 0.622 | 0.219 |
| Women | Teams + players | 194 | 0.622 | 0.219 |

Paired differences below are model minus team-only log loss.
Negative differences favor the model containing player identities.

| Competition | Model | Difference | 95% block interval | Quarters |
| --- | --- | ---: | --- | ---: |
| Women | Players only | -0.017 | [-0.044, +0.013] | 12 |
| Women | Teams + players | -0.016 | [-0.045, +0.015] | 12 |
| Men | Players only | -0.001 | [-0.042, +0.030] | 11 |
| Men | Teams + players | -0.003 | [-0.043, +0.030] | 11 |

### 3-year window (sensitivity)

| Competition | Model | Matches | Log loss | Brier |
| --- | --- | ---: | ---: | ---: |
| Men | 50:50 reference | 364 | 0.693 | 0.250 |
| Men | Team only | 364 | 0.695 | 0.250 |
| Men | Players only | 364 | 0.701 | 0.252 |
| Men | Teams + players | 364 | 0.701 | 0.253 |
| Women | 50:50 reference | 194 | 0.693 | 0.250 |
| Women | Team only | 194 | 0.667 | 0.237 |
| Women | Players only | 194 | 0.630 | 0.221 |
| Women | Teams + players | 194 | 0.629 | 0.221 |

Paired differences below are model minus team-only log loss.
Negative differences favor the model containing player identities.

| Competition | Model | Difference | 95% block interval | Quarters |
| --- | --- | ---: | --- | ---: |
| Women | Players only | -0.036 | [-0.066, -0.004] | 12 |
| Women | Teams + players | -0.037 | [-0.068, -0.004] | 12 |
| Men | Players only | +0.006 | [-0.026, +0.031] | 11 |
| Men | Teams + players | +0.006 | [-0.025, +0.032] | 11 |

Intervals resample quarters, retaining every match in each sampled quarter.
Few blocks, dependence across quarters, and uncertainty from training
and tuning limit their interpretation. They are conditional, exploratory
summaries, not evidence that individual player coefficients are identified.

The team baseline omits venue and dynamic strength. Player identities can
proxy those changes; the combined model also changes team shrinkage.

### Download the saved experiment

- [calibration.csv](../out/release/lineup/calibration.csv)
- [coefficients.csv](../out/release/lineup/coefficients.csv)
- [comparisons.csv](../out/release/lineup/comparisons.csv)
- [diagnostics.csv](../out/release/lineup/diagnostics.csv)
- [metrics.csv](../out/release/lineup/metrics.csv)
- [paired losses.csv](../out/release/lineup/paired_losses.csv)
- [predictions.csv](../out/release/lineup/predictions.csv)
- [preparation manifest.json](../out/release/lineup/preparation_manifest.json)
- [tuning.csv](../out/release/lineup/tuning.csv)
- [Analysis provenance](../out/release/lineup/analysis_manifest.json)
- [Publication checksums](../out/release/lineup/manifest.json)

The coefficient table contains static associations in log-odds units,
with exposure and co-occurrence diagnostics. It contains no WAR,
player confidence intervals, or current-strength ranks. The saved analysis
manifest records splits, dependencies, code hashes, and the input snapshot.
