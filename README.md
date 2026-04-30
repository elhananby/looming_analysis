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

Append a label to the output folder name:

```bash
looming-analysis \
  --files examples/CS_vs_J64xKir2.1_all.json \
  --analysis examples/analysis.toml \
  --output-root outputs \
  --suffix responsive_only
```

This creates a timestamped folder like `outputs/20260427_153000-CS_J64xKir2.1(w+)_responsive_only/`
containing:

| File | Contents |
|---|---|
| `trials.csv` | One row per trial — all scalar fields + responsiveness flags |
| `traces.parquet` | Long-format angular velocity traces |
| `average-angular-velocity.png` | Mean \|ω\| traces, faceted by `row_by` |
| `average-heading.png` | Mean heading traces, faceted by `row_by` |
| `heading-change-distribution.png` | Heading change violins, faceted by `row_by` |
| `heading-change-comparison.png` | All five heading metrics across groups |
| `responsiveness-rates.png` | Fraction responsive per condition |
| `peak-angular-velocity.png` | Peak \|ω\| distribution, faceted by `row_by` |
| `turn-proportions.png` | Left/right turn proportions per condition |
| `inter-trigger-interval.png` | Histogram of time between consecutive triggers |
| `screen-position-effect.png` | Peak and mean \|ω\| vs within-screen x position |
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
fig = result.plot_inter_trigger_interval(hue_by="group")
fig = result.plot_screen_position_effect(hue_by="group")
```

## Comparing Groups

```python
file_groups = {
    "CS": find_braidz("/mnt/data/experiments/cs"),
    "J64xKir2.1(w+)": find_braidz("/mnt/data/experiments/j64"),
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
    "J64xKir2.1(w+)": ["/path/to/j64_1.braidz"]
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
pre_ms = 100
post_ms = 500
max_gap_ms = 50
heading_ref_ms = 100
include_sham = false
cache_dir = ".braidz_cache"

# ── Responsiveness classification ─────────────────────────────────────────────
[responsiveness]
method = "combined"
threshold_deg_s = 300
heading_threshold_deg = 45
baseline_window_ms = [-90, -10]
window_ms = 200

# ── Plot layout ───────────────────────────────────────────────────────────────
[plots]
col_by = "stimulus_offset_deg"
hue_by = "group"

# "is_responsive" splits rows into responsive/non-responsive (default).
# Any trigger or stimulus field works, e.g. "refractory_period".
row_by = "is_responsive"

# Restrict all plots to responsive trials only.
# Combine with row_by = "refractory_period" to compare refractory conditions
# without the responsive/non-responsive split.
responsive_only = false

# Drop the top N% of ITI values from the histogram (e.g. 99 removes the top 1%).
# iti_percentile_cutoff = 99
```

See `examples/analysis.example.toml` for the full annotated reference.

## Trigger Metadata

Each `.braidz` archive may contain a `config.toml` with trigger handling
parameters. These are automatically extracted and attached to every trial as
scalar fields, making them available for faceting:

| Field | Description |
|---|---|
| `refractory_period` | Minimum time between triggers (s) |
| `z_min` / `z_max` | Z-height bounds of the trigger zone (m) |
| `heading_cone_deg` | Angular tolerance for heading alignment (°) |
| `min_velocity` / `max_velocity` | Velocity bounds for trigger eligibility (m/s) |
| `min_tracking_age` | Minimum object age before triggering (s) |
| `zone_timeout` | Auto zone-exit timeout (s) |
| `pre_zone_expansion` | FOV expansion for pre-trigger camera/lens (m) |

Example — facet plots by refractory period to check for cooldown effects:

```toml
[plots]
row_by = "refractory_period"
responsive_only = true
```

## Responsiveness Methods

`classify_responsiveness` computes all methods unconditionally for every trial.
The `method` setting controls only which one writes the primary `is_responsive`
field; the per-method flags remain available for comparison.

| Method | Field | Description |
|---|---|
| `peak` | `is_responsive_peak` | First `find_peaks` hit above `threshold_deg_s` in the detection window |
| `zscore` | `is_responsive_zscore` | Same peak normalised by pre-stimulus baseline SD |
| `heading` | `is_responsive_heading` | `\|heading_change\| ≥ heading_threshold_deg` |
| `saccade` | `is_responsive_saccade` | Signed peak detected by `find_peaks` (positive or negative) |
| `impulse` | `is_responsive_impulse` | Integrated `\|ω\|` over the detection window |
| `combined` | `is_responsive_combined` | Saccade **and** heading change (default) |

**Detection window:** symmetric `±window_ms` around `end_expansion_time`
(the moment the looming circle reaches its final size, i.e. `expansion_duration_ms`
after stimulus onset). The hold period is not included in the window offset.

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

# Diagnostic plots
result.plot_inter_trigger_interval(hue_by="group", percentile_cutoff=99)
result.plot_screen_position_effect(hue_by="group", n_bins=10, responsive_only=True)
```

## Troubleshooting

- **Plot requires `classify_responsiveness`** — use `run_analysis` or call
  `classify_responsiveness(responses)` before plotting.
- **Plot requires `compute_turn_direction`** — use `run_analysis` with
  `compute_turns=True` or call `compute_turn_direction(responses)`. Call it
  *after* `classify_responsiveness` so turn direction is derived from the
  saccade-detected peak.
- **Many trials skipped** — rerun with `debug=True` in `process_file_groups` to
  see per-trial skip reasons.
- **Slow loading** — keep `cache_dir = ".braidz_cache"` enabled; delete the
  cache directory to force a full reload after changing extraction parameters.
- **Low saccade detection rate** — try reducing `threshold_deg_s` or widening
  `window_ms`. Check `heading_change_post_saccade` counts against total
  responsive counts to gauge how many detected saccades fall outside the
  measurement window.
- **ITI histogram dominated by long gaps** — set `iti_percentile_cutoff = 99`
  in `[plots]` to clip the top 1% of intervals.
- **No screen position data** — `plot_screen_position_effect` silently skips
  responses without a `pixel_x` field. Recordings from older software versions
  may not have logged pixel coordinates.
