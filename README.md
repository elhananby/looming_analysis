# Looming Analysis

Analyze angular velocity and turn responses to looming visual stimuli in `.braidz`
recordings.

## Install

```bash
uv sync --dev
```

## Quickstart

For routine runs, use config files:

```bash
looming-analysis \
  --files examples/files.example.json \
  --analysis examples/analysis.example.toml \
  --output-root outputs
```

This creates a folder like `outputs/20260427_153000-CS_Empty-Split/` containing:

- `trials.csv`
- `traces.parquet`
- Standard plot PNGs
- Copies of the files and analysis config files used for the run

For custom notebook analysis, import the package directly:

```python
from looming_analysis import (
    AnalysisConfig,
    ResponsivenessConfig,
    find_braidz,
    run_analysis,
)

files = find_braidz("/mnt/data/experiments")

result = run_analysis(
    {"experiment": files},
    analysis=AnalysisConfig(pre_ms=100, post_ms=500, include_sham=False),
    responsiveness=ResponsivenessConfig(
        method="combined",
        threshold_deg_s=500,
        heading_threshold_deg=45,
        baseline_window_ms=(-90, -10),
        window_ms=(100, 200),
    ),
)

trials = result.to_dataframe(kind="scalar")
traces = result.to_dataframe(kind="long")

fig = result.plot_traces(hue_by="stimulus_offset_deg")
heading_fig = result.plot_heading_traces(hue_by="stimulus_offset_deg")
```

## Comparing Groups

```python
file_groups = {
    "CS": find_braidz("/mnt/data/experiments/cs"),
    "Empty-Split": find_braidz("/mnt/data/experiments/empty-split"),
}

result = run_analysis(file_groups)
fig = result.plot_responsiveness_rates(col_by="stimulus_offset_deg", hue_by="group")
heading_fig = result.plot_heading_traces(col_by="stimulus_offset_deg", hue_by="group")
```

## Config Files

`files.json`:

```json
{
  "groups": {
    "CS": ["/path/to/cs_1.braidz", "/path/to/cs_2.braidz"],
    "Empty-Split": ["/path/to/empty_split_1.braidz"]
  }
}
```

`analysis.toml`:

```toml
[analysis]
pre_ms = 100
post_ms = 500
max_gap_ms = 50
heading_ref_ms = 100
include_sham = true
cache_dir = ".braidz_cache"

[responsiveness]
method = "combined"
threshold_deg_s = 500
heading_threshold_deg = 45
baseline_window_ms = [-90, -10]
window_ms = [100, 200]

[plots]
col_by = "stimulus_offset_deg"
hue_by = "group"
```

## Responsiveness Methods

`classify_responsiveness` computes all supported methods for each trial. The
`method` setting controls only the primary `is_responsive` field.

The default `combined` method is the recommended response definition: a signed
saccade peak detected by `scipy.signal.find_peaks` in the reaction window plus
`abs(heading_change) >= heading_threshold_deg`.

Supported methods:

- `peak`: peak absolute angular velocity around expansion end.
- `zscore`: peak normalized by pre-stimulus baseline.
- `heading`: absolute heading change.
- `saccade`: signed local peak detected by `find_peaks` in the reaction window.
- `impulse`: integrated angular velocity in the detection window.
- `combined`: saccade detected by signed peak plus heading change above threshold.

## Common Outputs

```python
result.to_dataframe(kind="scalar").write_csv("trials.csv")
result.to_dataframe(kind="long").write_parquet("traces.parquet")
```

## Standard Plots

```python
result.plot_traces(col_by="stimulus_offset_deg", hue_by="group")
result.plot_heading_traces(col_by="stimulus_offset_deg", hue_by="group")
result.plot_heading_changes(col_by="stimulus_offset_deg", hue_by="group", absolute=True)
result.plot_responsiveness_rates(col_by="stimulus_offset_deg", hue_by="group")
result.plot_peak_velocity(col_by="stimulus_offset_deg", hue_by="group")
result.plot_turn_proportions(x_by="stimulus_offset_deg", col_by="group")
```

## Troubleshooting

- If a plot says to call `classify_responsiveness`, use `run_analysis` or call
  `classify_responsiveness(responses)` before plotting rates.
- If a plot says to call `compute_turn_direction`, use `run_analysis` with
  `compute_turns=True` or call `compute_turn_direction(responses)` before plotting
  turn proportions.
- If many trials are skipped, rerun with `debug=True` in `process_file_groups` to
  see per-trial skip reasons.
- If loading is slow, keep `cache_dir=".braidz_cache"` enabled.
- Use the `looming-analysis` command for repeatable runs and custom notebooks for
  exploratory one-off analyses.
