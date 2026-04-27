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

This creates a timestamped folder like `outputs/20260427_153000-CS_Empty-Split/`
containing:

| File | Contents |
|---|---|
| `trials.csv` | One row per trial — all scalar fields + responsiveness flags |
| `traces.parquet` | Long-format angular velocity traces |
| `average-angular-velocity.png` | Mean \|ω\| traces, split by responsiveness |
| `average-heading.png` | Mean heading traces, split by responsiveness |
| `heading-change-distribution.png` | Heading change violins, split by responsiveness |
| `heading-change-comparison.png` | All five heading metrics across groups |
| `responsiveness-rates.png` | Fraction responsive per condition |
| `peak-angular-velocity.png` | Peak \|ω\| distribution, split by responsiveness |
| `turn-proportions.png` | Left/right turn proportions per condition |
| `*.json` / `*.toml` | Copies of the config files used |

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
        window_ms=200,
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
fig = result.plot_heading_change_comparison(group_by="group")
```

## Config Files

### `files.json`

Maps group names to lists of `.braidz` file paths.

```json
{
  "groups": {
    "CS": ["/path/to/cs_1.braidz", "/path/to/cs_2.braidz"],
    "Empty-Split": ["/path/to/empty_split_1.braidz"]
  }
}
```

Use `"files"` instead of `"groups"` for a single unlabelled condition:

```json
{ "files": ["/path/to/recording.braidz"] }
```

### `analysis.toml`

```toml
# ── Trial extraction ──────────────────────────────────────────────────────────
[analysis]

# Milliseconds of data to include before stimulus onset (must be > 0).
pre_ms = 100

# Milliseconds of data to include after stimulus onset.
# Should cover the full expansion + hold period plus your expected response window.
post_ms = 500

# Maximum gap (ms) in Kalman tracking allowed within a trial window.
# Trials with longer gaps are discarded.
max_gap_ms = 50

# Duration (ms) of the pre- and post-expansion heading averages used to
# compute the heading_change scalar.  Longer = more stable, but risks
# including early turns for post.
heading_ref_ms = 100

# Set to true to include sham trials (stimulus column flagged as sham)
# alongside real looming trials.
include_sham = false

# Directory for caching parsed .braidz data.  Subsequent runs skip re-parsing
# the same files.  Delete or set to "" to force a full reload.
cache_dir = ".braidz_cache"


# ── Responsiveness classification ─────────────────────────────────────────────
[responsiveness]

# Which criterion writes the primary is_responsive field.
# Options: "peak" | "zscore" | "heading" | "saccade" | "impulse" | "combined"
# "combined" (default) requires both a find_peaks saccade AND a heading change
# above heading_threshold_deg.
method = "combined"

# Angular velocity threshold for saccade detection (deg/s).
# A local peak in the detection window must exceed this to count.
# Typical wild-type: ~1000 deg/s.  Reduce for weaker genetic lines.
threshold_deg_s = 300

# Minimum net heading change (degrees) required by the heading and combined
# methods.  Captures the directional outcome of the turn.
heading_threshold_deg = 45

# Pre-stimulus window [start_ms, end_ms] relative to stimulus onset used to
# compute the baseline mean and SD of angular velocity.
# Avoids the 10 ms immediately before onset to exclude anticipatory movement.
baseline_window_ms = [-90, -10]

# Reaction window around end_expansion_time (ms).
# A scalar gives a symmetric ±window (e.g. 200 → ±200 ms = 400 ms total).
# A two-element list [before_ms, after_ms] gives an asymmetric window.
# At 100 Hz, ±200 ms = ±20 samples.
window_ms = 200


# ── Plot layout ───────────────────────────────────────────────────────────────
[plots]

# Stimulus parameter used as the x-axis / column facet in all plots.
col_by = "stimulus_offset_deg"

# Response field used for color grouping (side-by-side violins / colored lines).
# Typically "group" when comparing experimental conditions.
hue_by = "group"
```

## Responsiveness Methods

`classify_responsiveness` computes all methods unconditionally for every trial.
The `method` setting controls only which one writes the primary `is_responsive`
field; the per-method flags remain available for comparison.

| Method | Field | Description |
|---|---|---|
| `peak` | `is_responsive_peak` | First `find_peaks` hit above `threshold_deg_s` in the detection window |
| `zscore` | `is_responsive_zscore` | Same peak normalised by pre-stimulus baseline SD |
| `heading` | `is_responsive_heading` | `\|heading_change\| ≥ heading_threshold_deg` |
| `saccade` | `is_responsive_saccade` | Signed peak detected by `find_peaks` (positive or negative) |
| `impulse` | `is_responsive_impulse` | Integrated `\|ω\|` over the detection window |
| `combined` | `is_responsive_combined` | Saccade **and** heading change (default) |

**Saccade detection parameters** (applied inside `find_peaks`):

- `height = threshold_deg_s` — minimum peak amplitude
- `prominence = 300 deg/s` — minimum rise above local surroundings
- `width = (3, 8) samples` — 30–80 ms at 100 Hz; excludes noise spikes and sustained turns
- `distance = 5 samples` — 50 ms minimum between peaks

## Heading Change Metrics

Five heading change metrics are computed per trial and stored for comparison:

| Field | Description |
|---|---|
| `heading_change` | Net change: pre-stim mean → post-expansion mean (computed at extract time) |
| `heading_change_window_net` | Net change: pre-stim mean → end of detection window |
| `heading_change_max_dev` | Max circular deviation from pre-stim baseline within the window |
| `heading_change_post_saccade` | Net change 50 ms before → 50 ms after saccade peak (NaN if no saccade) |
| `heading_change_path_length` | Total `\|Δheading\|` over the detection window (all rotation accumulated) |

Use `result.plot_heading_change_comparison()` to visualise all five side-by-side,
split by group and responsiveness.

## Standard Plots

```python
# All plot functions accept row_by, col_by, hue_by for faceting.
result.plot_traces(col_by="stimulus_offset_deg", hue_by="group", row_by="is_responsive")
result.plot_heading_traces(col_by="stimulus_offset_deg", hue_by="group", row_by="is_responsive")
result.plot_heading_changes(col_by="stimulus_offset_deg", hue_by="group", row_by="is_responsive")
result.plot_responsiveness_rates(col_by="stimulus_offset_deg", hue_by="group")
result.plot_peak_velocity(col_by="stimulus_offset_deg", hue_by="group", row_by="is_responsive")
result.plot_turn_proportions(x_by="stimulus_offset_deg", col_by="group")
result.plot_heading_change_comparison(group_by="group")
```

## Troubleshooting

- **Plot requires `classify_responsiveness`** — use `run_analysis` or call
  `classify_responsiveness(responses)` before plotting.
- **Plot requires `compute_turn_direction`** — use `run_analysis` with
  `compute_turns=True` or call `compute_turn_direction(responses)`.  Call it
  *after* `classify_responsiveness` so turn direction is derived from the
  saccade-detected peak.
- **Many trials skipped** — rerun with `debug=True` in `process_file_groups` to
  see per-trial skip reasons.
- **Slow loading** — keep `cache_dir = ".braidz_cache"` enabled; delete the
  cache directory to force a full reload after changing extraction parameters.
- **Low saccade detection rate** — try reducing `threshold_deg_s` or widening
  `window_ms`.  Check `heading_change_post_saccade` counts against total
  responsive counts to gauge how many detected saccades fall outside the
  measurement window.
