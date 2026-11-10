# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predictive maintenance on EDFA & SFP telemetry
#
# Commtel Networks, AI/ML Engineer II take-home.
#
# This notebook runs top to bottom from the three raw files. `python run.py` is the same code and writes the same outputs.
# It follows the brief's parts:
#
# * **A. Wrangling:** load the files, join sensor names, parse the SFP sensor names, align readings onto a 1-minute grid, build the wide tables and the sensor inventory.
# * **B. Preprocessing:** sentinels and junk values, outages and gaps, SFP reporting-window choice, resampling, feature set and scaling, plus the preprocessing log.
# * **C. Insights:** the findings, each with the numbers and plots behind it.
# * **D. Anomaly detection:** rule layer, physics-residual detector, trend analysis, injected-fault evaluation and the flagged periods.
# * **E. Decision log:** the final section.
#
# Every number quoted in a markdown cell is printed by a code cell above it.

# %%
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

IN_NOTEBOOK = "ipykernel" in sys.modules
import matplotlib  # noqa: E402

if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 11})

T_START = time.time()
SEED = 7
IST = "Asia/Kolkata"


def find_data_dir() -> Path:
    """Folder holding the raw files: $PDM_DATA_DIR, Kaggle's input mount, or a local copy."""
    roots = [os.environ.get("PDM_DATA_DIR"), "/kaggle/input", "data", "../ML Test Data"]
    for root in filter(None, roots):
        root_path = Path(root)
        if root_path.exists():
            hits = sorted(root_path.rglob("Sensor_mapping_edfa_sfp.json"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("Raw files not found: set PDM_DATA_DIR to the folder with the two CSVs and the mapping JSON")


def find_csv(folder: Path, keyword: str) -> Path:
    """The brief calls them EDFA_data.csv / SFP_data.csv; the delivered files are *_anomalies_f.csv. Match by keyword."""
    hits = [p for p in folder.iterdir() if p.suffix.lower() == ".csv" and keyword in p.name.lower()]
    if not hits:  # e.g. Kaggle mounts the CSVs and the JSON in sibling folders
        hits = [p for p in folder.parent.rglob("*.csv") if keyword in p.name.lower()]
    assert len(hits) == 1, f"expected one {keyword} CSV near {folder}, found {hits}"
    return hits[0]


DATA_DIR = find_data_dir()
EDFA_CSV, SFP_CSV = find_csv(DATA_DIR, "edfa"), find_csv(DATA_DIR, "sfp")
MAPPING_JSON = DATA_DIR / "Sensor_mapping_edfa_sfp.json"
OUT = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path("outputs")
for sub in ("tables", "figures", "scores"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
print("data:", DATA_DIR)
print("files:", EDFA_CSV.name, SFP_CSV.name, MAPPING_JSON.name)
print("outputs:", OUT.resolve())


def show(obj, n: int | None = None) -> None:
    """Display in a notebook, print in a script."""
    if isinstance(obj, (pd.DataFrame, pd.Series)) and n is not None:
        obj = obj.head(n)
    if IN_NOTEBOOK:
        from IPython.display import display

        display(obj)
    else:
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=110, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids a dependency on tabulate)."""
    cells = df.astype(str).apply(lambda col: col.str.replace("|", "/", regex=False))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


PREP_LOG: list[dict] = []


def log_step(step: str, what: str, why: str, affected: str) -> None:
    """One row of the preprocessing log (printed at the end of Part B and written to preprocessing_log.md)."""
    PREP_LOG.append({"step": step, "what": what, "why": why, "affected": affected})


def short(asset_id: str) -> str:
    """Assets are UUIDs; the first 8 characters are unique in this dataset and readable in plots."""
    return str(asset_id)[:8]


# %% [markdown]
# ## Part A: Data wrangling and transformation
#
# ### A1. Load the files and report rows, assets, sensors and time span
#
# * Timestamps arrive as `2026-01-02 23:59:33+00`, with a two-digit UTC offset and between 0 and 6 fractional digits. They are parsed as UTC.
# * Blank `value` cells are kept as NaN and counted, not dropped.

# %%
def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        engine="pyarrow",
        dtype={"id": "string[pyarrow]", "asset_id": "string[pyarrow]", "value": "float64", "timestamp": "string[pyarrow]"},
    )
    df["ts"] = pd.to_datetime(df["timestamp"].str.replace(r"\+00$", "+00:00", regex=True), format="ISO8601", utc=True)
    assert df["ts"].notna().all(), "unparsed timestamps"
    return df.drop(columns="timestamp")


t0 = time.time()
raw = {"EDFA": load_long(EDFA_CSV), "SFP": load_long(SFP_CSV)}
mapping = pd.DataFrame(json.loads(MAPPING_JSON.read_text(encoding="utf-8")))
print(f"loaded in {time.time() - t0:.1f}s")

summary = pd.DataFrame(
    {
        kind: {
            "rows": len(df),
            "blank values": int(df["value"].isna().sum()),
            "assets": df["asset_id"].nunique(),
            "sensor ids": df["id"].nunique(),
            "first reading (UTC)": df["ts"].min(),
            "last reading (UTC)": df["ts"].max(),
        }
        for kind, df in raw.items()
    }
)
show(summary)
print(f"mapping file: {len(mapping):,} entries, {mapping['assetId'].nunique()} assets, ids unique: {mapping['id'].is_unique}")

per_asset = pd.concat(
    [
        df.groupby("asset_id")
        .agg(rows=("value", "size"), sensor_ids=("id", "nunique"), first=("ts", "min"), last=("ts", "max"))
        .assign(asset_type=kind)
        for kind, df in raw.items()
    ]
)
per_asset["span_days"] = ((per_asset["last"] - per_asset["first"]).dt.total_seconds() / 86400).round(1)
show(per_asset)

# %% [markdown]
# ### A2. Join sensor names and handle unmapped ids
#
# * The mapping file is the only source of sensor names. Each `id` is checked to be unique in it, and its `assetId` is checked against the reading's `asset_id`.
# * Unmapped ids are **kept**. They are labelled `UNMAPPED_<first 8 chars of id>` so they survive the pivot. Their probable role is inferred from their values in the A5 inventory.
# * Exact duplicate rows (same id, timestamp and value) are counted and then removed.

# %%
assert mapping["id"].is_unique
name_map = mapping.rename(columns={"assetId": "map_asset_id", "sensorName": "sensor_name"})

long = {}
join_report = []
for kind, df in raw.items():
    df = df.merge(name_map, on="id", how="left", validate="many_to_one")
    mapped = df["sensor_name"].notna()
    asset_mismatch = int((mapped & (df["map_asset_id"] != df["asset_id"])).sum())
    unmapped_ids = df.loc[~mapped, "id"].unique()
    labels = pd.Series(unmapped_ids).str[:8]
    assert labels.is_unique, "8-char prefixes collide; use longer labels"
    df["sensor"] = df["sensor_name"].fillna("UNMAPPED_" + df["id"].str[:8])
    df["mapped"] = mapped
    dup_exact = df.duplicated(["id", "ts", "value"])
    join_report.append(
        {
            "asset type": kind,
            "readings": len(df),
            "unmapped readings": int((~mapped).sum()),
            "unmapped share": f"{(~mapped).mean():.1%}",
            "distinct unmapped ids": len(unmapped_ids),
            "asset id mismatches": asset_mismatch,
            "exact duplicate rows removed": int(dup_exact.sum()),
        }
    )
    df = df.loc[~dup_exact, ["asset_id", "id", "sensor", "mapped", "value", "ts"]].copy()
    df["asset_id"] = df["asset_id"].astype("category")
    df["sensor"] = df["sensor"].astype("category")
    long[kind] = df
del raw
show(pd.DataFrame(join_report).set_index("asset type"))
for r in join_report:
    if r["distinct unmapped ids"]:
        log_step("A2 join", f"{r['asset type']}: keep unmapped sensor ids as UNMAPPED_<id8>", "the brief left them in deliberately; dropping them would silently lose data",
                 f"{r['unmapped readings']:,} readings, {r['distinct unmapped ids']} ids")
    log_step("A2 duplicates", f"{r['asset type']}: drop exact duplicate rows (same id, timestamp, value)", "pure repeats; one unmapped SFP id arrives 5x per poll",
             f"{r['exact duplicate rows removed']:,} rows")

dup_by_sensor = (
    pd.concat(long.values())
    .pipe(lambda d: d[d.duplicated(["id", "ts"], keep=False)])
    .groupby("sensor", observed=True)
    .size()
    .sort_values(ascending=False)
)
print("readings that share (id, timestamp) with another reading, after exact-duplicate removal:", int(dup_by_sensor.sum()))
show(dup_by_sensor, 5)

# %% [markdown]
# ### A3. Parse the SFP sensor names into structured fields
#
# Pattern: `<card>-<shelf>-<slot>-<port>[ <window>]: <metric>`, for example `STM4-1-4-3 Previous 15min: Laser Bias`.
# Device-level sensors (`ICMP loss`, `Valid Status`) and unmapped ids have no port fields.

# %%
SFP_NAME = re.compile(
    r"^(?P<card>[A-Z0-9]+)-(?P<shelf>\d+)-(?P<slot>\d+)-(?P<port>\d+)"
    r"(?: (?P<window>Current Day|Current|Previous 15min|Previous Day))?: (?P<metric>.+)$"
)


def parse_sfp_names(names: pd.Series) -> pd.DataFrame:
    parsed = names.str.extract(SFP_NAME)
    parsed.insert(0, "sensor", names.values)
    parsed["interface_id"] = parsed["card"] + "-" + parsed["shelf"] + "-" + parsed["slot"] + "-" + parsed["port"]
    device_level = parsed["card"].isna()
    parsed.loc[device_level, "metric"] = parsed.loc[device_level, "sensor"]
    parsed["window"] = parsed["window"].fillna("(none)")
    return parsed


sfp_names = parse_sfp_names(pd.Series(long["SFP"]["sensor"].cat.categories))
print(f"{len(sfp_names)} SFP sensor names; parsed into port fields: {sfp_names['card'].notna().sum()}")
show(sfp_names.groupby(["card", "window"], dropna=False).size().unstack(fill_value=0))
show(sfp_names[sfp_names["sensor"].str.startswith("STM4-1-4-3")].sample(8, random_state=SEED).sort_values("sensor"))
sfp_names.to_csv(OUT / "tables" / "sfp_sensor_names_parsed.csv", index=False)

# %% [markdown]
# ### A4. Pivot to wide format: one row per asset per polling slot
#
# **How timestamps are aligned.** Each device is polled once a minute in several groups.
#
# * On the EDFA, SNMP availability is polled around second :13, uptime around :21–:33, the optical block around :51 and ICMP around :53.
# * Sensors in the same group share an identical timestamp. Groups are offset by seconds, and the offset drifts slowly.
#
# Every reading is assigned to its **1-minute slot** (`floor(ts, 1 min)`), which matches the nominal 60 s polling interval.
#
# * A clustering approach that merges timestamps closer than 30 s would chain consecutive groups (:13 → :33 → :51 → :13 ...) into one endless poll, so it was rejected.
# * When a sensor has two readings in the same slot, the later non-blank one is kept. Such collisions mostly come from the uptime counters, which are polled twice a minute. They are counted below, not hidden.
#
# Checks: the index is unique, and the number of non-empty cells equals the number of (asset, slot, sensor) groups with at least one value, so there is no row explosion.

# %%
def to_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.assign(slot=df["ts"].dt.floor("1min")).sort_values("ts")
    key = ["asset_id", "slot", "sensor"]
    group_sizes = df.groupby(key, observed=True).size()
    collided = group_sizes[group_sizes > 1]
    differing = df[df.set_index(key).index.isin(collided.index)].groupby(key, observed=True)["value"].nunique().gt(1).sum()
    wide = df.groupby(key, observed=True)["value"].last().unstack("sensor")
    wide.columns = wide.columns.astype(str)
    expected_cells = int(df.dropna(subset=["value"]).groupby(key, observed=True).ngroups)
    assert wide.index.is_unique
    assert int(wide.notna().sum().sum()) == expected_cells, "pivot lost or invented values"
    stats = {
        "readings": len(df),
        "rows (asset x slot)": len(wide),
        "sensor columns": wide.shape[1],
        "slots with 2+ readings of one sensor": len(collided),
        "of which values differed": int(differing),
        "readings merged away": int((collided - 1).sum()),
    }
    return wide, stats


t0 = time.time()
wide, pivot_stats = {}, {}
for kind in ("EDFA", "SFP"):
    wide[kind], pivot_stats[kind] = to_wide(long[kind])
    wide[kind].to_parquet(OUT / "tables" / f"{kind.lower()}_wide_1min.parquet")
print(f"pivot in {time.time() - t0:.1f}s")
show(pd.DataFrame(pivot_stats))
for kind, st in pivot_stats.items():
    log_step("A4 pivot", f"{kind}: align readings to 1-min slots; keep the last non-blank reading per sensor and slot",
             "sensor groups are polled seconds apart within each minute; the uptime counter is polled twice a minute",
             f"{st['readings merged away']:,} readings merged ({st['of which values differed']:,} slots had differing values)")
collision_sensors = (
    long["EDFA"].assign(slot=lambda d: d["ts"].dt.floor("1min"))
    .groupby(["asset_id", "slot", "sensor"], observed=True).size().loc[lambda s: s > 1]
    .groupby(level="sensor", observed=True).size().sort_values(ascending=False)
)
print("EDFA sensors with same-slot collisions:")
show(collision_sensors, 6)

# %% [markdown]
# ### A5. Sensor inventory and classification
#
# For each asset and sensor the inventory gives:
#
# * the number of readings;
# * coverage: the fraction of the asset's polled minutes in which the sensor has a value, over all slots and over the sensor's own active span;
# * min, max and the number of distinct values;
# * a class: `telemetry`, `config`, `status`, `network` or `junk`.
#
# The class comes from the name together with the data, not from the name alone:
#
# * Pump2 sensors are `junk` because the data shows both EDFAs are single-pump. Pump2 Itec is 0 at every reading, and Pump2 current and power swing around zero, including negative values.
# * `Uptime (hardware)` is `junk` because it is 0 at every reading.
# * `Current: Timestamp` is `junk` because it is a constant-0 sentinel.
# * The other window timestamps are `status`: they record when a window closed, which is useful for spotting stale windows.
# * Unmapped SFP ids are classified from their value signature, and every such inference is marked "(inferred)".

# %%
EDFA_CLASSES = {
    "Input Power (PIN)": ("telemetry", "optical input, dBm"),
    "Output Power (POUT)": ("telemetry", "optical output, dBm; set-point defines operating class"),
    "Pump1 Current": ("telemetry", "pump drive current, mA"),
    "Pump1 Power": ("telemetry", "pump optical power, mW"),
    "Pump1 Temperature": ("telemetry", "pump chip temperature, degC (TEC-held)"),
    "Pump1 Itec": ("telemetry", "pump cooler (TEC) current, mA"),
    "EDFA Case Temperature": ("telemetry", "enclosure temperature, degC"),
    "Pump2 Current": ("junk", "single-pump unit: values swing around 0 incl. negatives"),
    "Pump2 Power": ("junk", "single-pump unit: values around 0 incl. negatives"),
    "Pump2 Temperature": ("junk", "single-pump unit: values around 0 incl. negatives"),
    "Pump2 Itec": ("junk", "single-pump unit: constant 0"),
    "Input Power Threshold (PIN)": ("config", "alarm threshold"),
    "Output Power Threshold (PIN)": ("config", "alarm threshold (misnamed: it is the POUT threshold)"),
    "EDFA Index": ("config", "identity"),
    "EDFA Address": ("config", "identity; flips between 2 and 255"),
    "Uptime (network)": ("status", "SNMP uptime counter, seconds; used to detect reboots"),
    "Uptime (hardware)": ("junk", "constant 0"),
    "ICMP ping": ("network", "reachability 0/1"),
    "ICMP loss": ("network", "packet loss %"),
    "ICMP response time": ("network", "round-trip time, s"),
    "SNMP agent availability": ("network", "SNMP reachability 0/1"),
}
SFP_METRIC_CLASSES = {
    "Laser Bias": "telemetry", "Laser Temperature": "telemetry", "Laser Voltage": "telemetry",
    "Received Power": "telemetry", "Transmit Power": "telemetry",
    "Interval Valid": "status", "Present Status": "status", "Operational Status": "status", "Valid Status": "status",
    "Interface Number": "config", "ICMP loss": "network", "ICMP ping": "network", "ICMP response time": "network",
}


def classify_unmapped(values: pd.Series) -> tuple[str, str]:
    """Best guess for an unmapped SFP sensor from its value signature. Inferences, not facts."""
    v = values.dropna()
    lo, hi, nd = v.min(), v.max(), v.nunique()
    if nd == 1:
        x = lo
        if 100000 <= x < 200000:
            return "config", f"(inferred) interface number, constant {x:.0f}: same value as an STM4 port's Interface Number"
        if -60 <= x <= -40:
            return "config", f"(inferred) -48 V DC alarm threshold or nominal, constant {x:g} V"
        if x in (0, 1, 2):
            return "status", f"(inferred) status flag, constant {x:g}"
        return "config", f"(inferred) constant {x:g}: configuration or identity"
    if -60 <= lo and hi <= -40:
        return "telemetry", f"(inferred) -48 V DC supply voltage, {lo:g} to {hi:g} V"
    if 0 <= lo and hi <= 90 and (v == v.round()).all():
        return "telemetry", f"(inferred) temperature in whole degC, {lo:g} to {hi:g}"
    return "telemetry", f"(inferred) varying, {lo:g} to {hi:g}"


def inventory(kind: str) -> pd.DataFrame:
    df, w = long[kind], wide[kind]
    rows = []
    for (asset, sensor), g in df.groupby(["asset_id", "sensor"], observed=True):
        col = w.loc[asset, sensor] if sensor in w.columns else pd.Series(dtype=float)
        active = col.loc[g["ts"].min().floor("1min"): g["ts"].max().floor("1min")]
        if kind == "EDFA":
            cls, note = EDFA_CLASSES.get(sensor, ("unknown", "not in rule table"))
        elif sensor.startswith("UNMAPPED_"):
            cls, note = classify_unmapped(g["value"])
        else:
            metric = sfp_names.set_index("sensor").at[sensor, "metric"]
            window = sfp_names.set_index("sensor").at[sensor, "window"]
            cls = "junk" if (metric == "Timestamp" and window == "Current") else SFP_METRIC_CLASSES.get(metric, "status" if metric == "Timestamp" else "unknown")
            note = "constant 0 sentinel" if cls == "junk" else ("window close time (epoch s)" if metric == "Timestamp" else "")
        rows.append(
            {
                "asset": short(asset), "sensor": sensor, "readings": len(g), "blank": int(g["value"].isna().sum()),
                "coverage_all_slots": col.notna().mean() if len(col) else 0.0,
                "coverage_active_span": active.notna().mean() if len(active) else 0.0,
                "min": g["value"].min(), "max": g["value"].max(), "distinct": g["value"].nunique(),
                "first": g["ts"].min(), "last": g["ts"].max(), "class": cls, "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["asset", "class", "sensor"]).reset_index(drop=True)


inv = {kind: inventory(kind) for kind in ("EDFA", "SFP")}
for kind, table in inv.items():
    table.to_csv(OUT / "tables" / f"sensor_inventory_{kind.lower()}.csv", index=False)
    print(kind, "classes:", table["class"].value_counts().to_dict())
show(inv["EDFA"].drop(columns=["first", "last"]).round(3))
show(inv["SFP"][inv["SFP"]["sensor"].str.startswith("UNMAPPED_")]["note"].str.replace(r"[-\d., ]+(to [-\d.]+)?( V)?$", "", regex=True).value_counts())
show(inv["SFP"][inv["SFP"]["class"] == "telemetry"].drop(columns=["first", "last"]).round(3), 70)

# %% [markdown]
# ## Part B: Preprocessing
#
# Order of operations. Each step is logged in `PREP_LOG`, which is printed at the end of Part B as the preprocessing log.
#
# 1. **Validity rules** on the 1-minute tables. Physically impossible readings and sentinels are written to a flag table, since they are Part D's point anomalies, and then masked.
# 2. **Outages.** Optical readings are masked while ICMP loss is 100% and for a short guard period after recovery.
# 3. **Junk and identity columns** are dropped, each with a reason.
# 4. **SFP reporting window.** One window is chosen per metric, with the evidence for the choice.
# 5. **Resampling** onto a regular grid: EDFA 5 min, SFP 15 min. Short polling misses are filled, long gaps are left missing.
# 6. **Feature set and scaling.**

# %%
# Physical envelopes. Outside these a reading cannot come from a working device.
# EDFA limits are generous engineering bounds; SFP limits follow SFF-8472 DDM ranges and typical STM-4 optics.
EDFA_VALID = {
    "Input Power (PIN)": (-30, 10), "Output Power (POUT)": (-10, 25),
    "Pump1 Current": (5, 1000), "Pump1 Power": (5, 1000), "Pump1 Itec": (5, 2000),
    "Pump1 Temperature": (0, 80), "EDFA Case Temperature": (-5, 75),
    "ICMP loss": (0, 100), "ICMP response time": (0, 10),
}
SFP_VALID = {
    "Laser Bias": (0.5, 131), "Laser Temperature": (-5, 85), "Laser Voltage": (3.0, 3.6),
    "Transmit Power": (0.01, 5), "Received Power": (-40, 3),
}
SFP_NETWORK_VALID = {"ICMP loss": (0, 100), "ICMP ping": (0, 1), "ICMP response time": (0, 10)}


def validity_flags(kind: str) -> pd.DataFrame:
    """Every reading outside its physical envelope, as a long table (asset, slot, sensor, value, rule)."""
    w = wide[kind]
    flags = []
    if kind == "EDFA":
        rules = {col: EDFA_VALID[col] for col in w.columns if col in EDFA_VALID}
    else:
        meta = sfp_names.set_index("sensor")
        rules = {}
        for col in w.columns:
            if col in SFP_NETWORK_VALID:
                rules[col] = SFP_NETWORK_VALID[col]
            elif col in meta.index and meta.at[col, "metric"] in SFP_VALID:
                rules[col] = SFP_VALID[meta.at[col, "metric"]]
    for col, (lo, hi) in rules.items():
        x = w[col]
        bad = x.notna() & ((x < lo) | (x > hi))
        if bad.any():
            f = x[bad].rename("value").reset_index()
            f["sensor"] = col
            f["rule"] = f"outside [{lo}, {hi}]"
            flags.append(f)
    if kind == "SFP":  # integer-valued config that arrives with a fractional part
        for col in [c for c in w.columns if c.endswith("Interface Number")]:
            x = w[col]
            bad = x.notna() & (x != x.round())
            f = x[bad].rename("value").reset_index()
            f["sensor"] = col
            f["rule"] = "non-integer interface number"
            flags.append(f)
    out = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame(columns=["asset_id", "slot", "value", "sensor", "rule"])
    return out[["asset_id", "slot", "sensor", "value", "rule"]]


point_flags = {kind: validity_flags(kind) for kind in ("EDFA", "SFP")}
clean_1min = {}
for kind, flags in point_flags.items():
    w = wide[kind].copy()
    for sensor, g in flags.groupby("sensor"):
        w.loc[pd.MultiIndex.from_frame(g[["asset_id", "slot"]]), sensor] = np.nan
    clean_1min[kind] = w
    pd.concat([flags]).to_csv(OUT / "tables" / f"point_flags_{kind.lower()}.csv", index=False)
    log_step("B3 sentinels/validity", f"{kind}: mask readings outside physical envelope", "impossible for a working device (sentinel, glitch or corrupted value); logged as point anomalies for Part D",
             f"{len(flags):,} readings in {flags['sensor'].nunique()} sensors")
    print(kind, f"{len(flags):,} readings outside physical envelope")
    if kind == "SFP":
        volt = flags[flags["sensor"].str.endswith("Laser Voltage")]
        print(f"  of which Laser Voltage: {len(volt):,} (exactly zero: {(volt['value'] == 0).sum():,}, negative: {(volt['value'] < 0).sum():,})")
    show(flags.groupby(["sensor", "rule"]).agg(n=("value", "size"), min=("value", "min"), max=("value", "max")).sort_values("n", ascending=False), 25)

# %% [markdown]
# ### B2. Outages, polling misses and missing sensors
#
# These are three different kinds of gap, and they are treated differently.
#
# * **Device outage (masked).** ICMP loss is 100%, so the management port was unreachable. Readings then are missing or stale and say nothing about the optics.
#   * ICMP is itself present in only about a third of minutes, so the outage state is carried forward from each ICMP reading until the next one (capped at 10 min).
#   * Optical readings in outage minutes, plus a 5-minute guard after recovery, are masked. They are never filled.
#   * Partial loss (33% or 67%, meaning 1 or 2 of 3 pings lost) counts as degraded reachability, not an outage, and is only flagged.
# * **Polling miss (filled).** The device is reachable but a sensor is absent from a slot. EDFA optical sensors are absent from 20–30% of polled minutes and SFP sensors from about 50%, so a single reading is never the unit of analysis. Resampling takes the median of whatever readings fall in a 5- or 15-minute bin. A bin that is still empty is interpolated only if the gap is 2 bins or fewer.
# * **Genuinely missing sensor (left missing).** Examples are the EDFA optical sensors before 16 Feb and the 20 Jan to 13 Feb collection gap. These stay NaN and are excluded from training and scoring.

# %%
OUTAGE_CARRY_MIN = 10  # trust an ICMP reading for at most 10 minutes
OUTAGE_GUARD_MIN = 5  # stale-value guard after recovery


def outage_minutes(w: pd.DataFrame, loss_col: str = "ICMP loss") -> pd.Series:
    """True per (asset, slot) while the most recent ICMP reading (at most OUTAGE_CARRY_MIN old) said 100% loss,
    extended by OUTAGE_GUARD_MIN minutes after recovery. Computed on a regular 1-minute grid per asset."""
    parts = []
    for asset in w.index.get_level_values("asset_id").unique():
        slots = w.loc[asset].index
        grid = pd.date_range(slots.min(), slots.max(), freq="1min")
        loss = w.loc[asset, loss_col].dropna()
        state = (loss >= 100).astype(float).reindex(grid).ffill(limit=OUTAGE_CARRY_MIN).fillna(0)
        state = state.rolling(OUTAGE_GUARD_MIN + 1, min_periods=1).max().astype(bool)
        parts.append(pd.Series(state.reindex(slots).values, index=pd.MultiIndex.from_product([[asset], slots], names=["asset_id", "slot"])))
    return pd.concat(parts)


def mask_outages(kind: str, optical_cols: list[str]) -> pd.DataFrame:
    w = clean_1min[kind]
    guarded = outage_minutes(w).reindex(w.index).fillna(False).astype(bool)
    cols = [c for c in optical_cols if c in w.columns]
    n_masked = int(w.loc[guarded, cols].notna().sum().sum())
    w.loc[guarded, cols] = np.nan
    w["outage"] = guarded
    clean_1min[kind] = w
    return pd.DataFrame(
        {"outage minutes (incl. guard)": guarded.groupby(level="asset_id", observed=True).sum(),
         "optical readings masked": n_masked}
    )


EDFA_OPTICAL = [c for c, (cls, _) in EDFA_CLASSES.items() if cls == "telemetry"]
SFP_OPTICAL = sfp_names.loc[sfp_names["metric"].isin(SFP_VALID), "sensor"].tolist()
edfa_out = mask_outages("EDFA", EDFA_OPTICAL)
sfp_out = mask_outages("SFP", SFP_OPTICAL)
show(pd.concat([edfa_out, sfp_out]))
for kind, table in (("EDFA", edfa_out), ("SFP", sfp_out)):
    log_step("B2 outages", f"{kind}: mask optical readings while ICMP loss = 100% (+5 min guard)",
             "unreachable device: optical values missing or stale; must not train or score on them",
             f"{int(table.iloc[:, 0].sum()):,} asset-minutes; {int(table.iloc[0, 1]):,} readings masked")


def outage_episodes(kind: str) -> pd.DataFrame:
    """Contiguous outage periods per asset (minute slots)."""
    eps = []
    for asset, s in clean_1min[kind]["outage"].groupby(level="asset_id", observed=True):
        s = s.droplevel("asset_id")
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="1min"), fill_value=False)
        run_id = (s != s.shift()).cumsum()
        for _, r in s[s].groupby(run_id[s]):
            eps.append({"asset": short(asset), "start": r.index[0], "end": r.index[-1], "minutes": len(r)})
    return pd.DataFrame(eps)


episodes = {kind: outage_episodes(kind) for kind in ("EDFA", "SFP")}
for kind, ep in episodes.items():
    print(kind, "outage episodes >= 30 min:")
    show(ep[ep["minutes"] >= 30].assign(hours=lambda d: (d["minutes"] / 60).round(1)), 20)

# %% [markdown]
# ### B4. SFP: which reporting window to use
#
# Each SFP metric is reported in four windows:
#
# * `Current`: instantaneous;
# * `Previous 15min`: the last closed 15-minute window;
# * `Current Day`: today so far;
# * `Previous Day`: the last closed day.
#
# These are four views of one physical quantity, so before choosing, the code checks how they relate:
#
# 1. The share of physically impossible readings in each window.
# 2. The device's window clock. The `Previous 15min: Timestamp` should be at most 15 minutes before the poll.
# 3. Medians per period of each window, per port and metric. They should agree, and they should follow the physics: to hold transmit power constant, laser bias rises with laser temperature.

# %%
SFP_ASSET = long["SFP"]["asset_id"].cat.categories[0]
SFP_PORTS = sorted(sfp_names.loc[sfp_names["card"] == "STM4", "interface_id"].unique())
SFP_METRICS = list(SFP_VALID)
WINDOWS = ["Current", "Previous 15min", "Current Day", "Previous Day"]
sfp_w = clean_1min["SFP"].loc[SFP_ASSET]

# 1. impossible-value rate per window
flag_counts = point_flags["SFP"].merge(sfp_names, on="sensor").groupby(["window", "metric"]).size()
total_counts = long["SFP"].merge(sfp_names, on="sensor").query("metric in @SFP_METRICS").groupby(["window", "metric"]).size()
invalid_rate = (flag_counts.reindex(total_counts.index, fill_value=0) / total_counts).unstack("window")[WINDOWS]
print("share of readings outside the physical envelope, by window:")
show((invalid_rate * 100).round(2).astype(str) + "%")

# 2. device window clock vs poll time
clock = []
for port in SFP_PORTS:
    for win in ["Previous 15min", "Current Day", "Previous Day"]:
        ts = sfp_w[f"{port} {win}: Timestamp"].dropna()
        lag_min = (ts.index.to_series() - pd.to_datetime(ts, unit="s", utc=True)).dt.total_seconds() / 60
        clock.append({"port": port, "window": win, "poll minus window timestamp, median (min)": lag_min.median(),
                      "p05": lag_min.quantile(0.05), "p95": lag_min.quantile(0.95),
                      "timestamps on a 15-min boundary": f"{(ts % 900 == 0).mean():.0%}"})
show(pd.DataFrame(clock).round(1))

# 3. per-period medians of every window
PERIODS = {"02-08 Jan": ("2026-01-02", "2026-01-08"), "16-22 Feb": ("2026-02-16", "2026-02-22"),
           "23 Feb-01 Mar": ("2026-02-23", "2026-03-01"), "23-31 Mar": ("2026-03-23", "2026-03-31")}
period_medians = pd.DataFrame(
    {(port[-3:], metric, win): {k: sfp_w[f"{port} {win}: {metric}"].loc[a:b].median() for k, (a, b) in PERIODS.items()}
     for metric in ["Laser Temperature", "Laser Bias", "Transmit Power", "Received Power"] for port in SFP_PORTS for win in WINDOWS}
).T
period_medians.index.names = ["port", "metric", "window"]
show(period_medians.round(2))
period_medians.round(3).to_csv(OUT / "tables" / "sfp_window_period_medians.csv")

# %% [markdown]
# **What the comparison shows, and the choice made:**
#
# * **Temperature:** all four windows agree. Both lasers warm by about 8 °C from early January to late March.
# * **Day windows:** `Current Day` and `Previous Day` show bias rising about 2.5 mA (+7%) over the same period. That is what a laser held at constant Tx does as it warms (roughly 0.3 mA/°C), so it is the physically consistent story.
# * **`Current Day` is corrupted.** It holds almost all impossible values: 24% of its Laser Voltage readings and 1.8% of its Laser Bias readings. On port 1-4-3 its Received Power sits 2–6 dB away from the other three windows.
# * **`Previous 15min` has two problems:**
#   * Its window timestamps are **49–94 minutes ahead of the poll**. The module's clock runs ahead of the poller, so the label "previous 15 minutes" cannot be trusted for alignment.
#   * It carries constant bias offsets against every other window: +5.3 mA on 1-4-3, and −4 to −7 mA on 1-4-4 after a step on 9 Jan.
# * **`Current` has no impossible bias, voltage, Tx or Rx readings, and it is the only window aligned to the poll time.** It does show **level shifts that no other window shows**: Tx on 1-4-3 −10% from 6 Jan, bias on 1-4-4 +15% from 18 Feb, bias on 1-4-3 −12% from 27 Feb. These are kept as signal and examined in Parts C and D, not cleaned away.
#
# **Choice:** use the `Current` window for all five metrics, resampled to 15-minute medians.
#
# * It is the only window with intra-day resolution aligned to real time.
# * It has no impossible readings apart from a handful of temperature spikes (removed by the B3 rules).
# * The day windows are kept **as a cross-check**, not as features. Each one changes only once a day, so it cannot add information at the 15-minute scale. They are what Part D uses to tell a change in one reporting path from a change in the laser itself.
#
# Nothing needed later is lost: temperature agrees across all windows, and the slow trends visible in the day windows are equally visible in the daily medians of `Current` except where `Current` itself shifts. Those shifts are the anomalies Part D reports.

# %% [markdown]
# ### B5. Resample to a regular grid
#
# * **EDFA: 5-minute bins**, median of the readings in each bin. Readings arrive about every 60 s, but each optical sensor is missing from 20–30% of polled minutes (A5 coverage), so a 5-minute bin usually holds 3 or 4 readings. The failure modes run from minutes (an upstream fibre loss shows as a PIN step) to weeks (pump ageing). Five minutes keeps a fibre event visible within one bin, and the median removes single-poll noise. Cost: about 12.6k bins per asset, trivial on a CPU.
# * **SFP: 15-minute bins**, aligned to the module's own 15-minute windows (the `Previous 15min: Timestamp` values are multiples of 900 s). SFP failure modes (laser ageing, connector degradation) are slower, and the device already aggregates to 15 minutes.
# * **Gap filling:** polling misses of at most 2 bins (10 min EDFA, 30 min SFP) are filled by time interpolation. Longer gaps stay NaN, and outage bins are never filled. Every bin carries `n_readings` and `filled` columns, so filled data is traceable.

# %%
def fill_short_gaps(s: pd.Series, max_bins: int) -> tuple[pd.Series, pd.Series]:
    """Interpolate only NaN runs of length <= max_bins that have data on both sides."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    fillable = isna & (run_len <= max_bins)
    interp = s.interpolate(method="time", limit_area="inside")
    out = s.where(~fillable, interp)
    return out, fillable & out.notna()


def resample_asset(kind: str, asset, cols: list[str], freq: str, max_fill_bins: int = 2) -> pd.DataFrame:
    w = clean_1min[kind].loc[asset]
    present = [c for c in cols if c in w.columns]
    first = w[present].dropna(how="all").index.min()
    w = w.loc[first:]
    binned = w[present].resample(freq).median()
    counts = w[present].notna().resample(freq).sum()
    outage = w["outage"].resample(freq).max().fillna(False).astype(bool)
    out = pd.DataFrame(index=binned.index)
    for c in present:
        filled, was_filled = fill_short_gaps(binned[c].where(~outage), max_fill_bins)
        out[c] = filled
        out[f"{c}__filled"] = was_filled
    out["n_readings"] = counts.sum(axis=1)
    out["outage"] = outage
    out.index.name = "bin"
    return out


EDFA_FEATURES = ["Input Power (PIN)", "Output Power (POUT)", "Pump1 Current", "Pump1 Power", "Pump1 Itec", "Pump1 Temperature", "EDFA Case Temperature"]
EDFA_ASSETS = list(long["EDFA"]["asset_id"].cat.categories)
edfa5 = {asset: resample_asset("EDFA", asset, EDFA_FEATURES, "5min") for asset in EDFA_ASSETS}
for asset, d in edfa5.items():
    d["gain_db"] = d["Output Power (POUT)"] - d["Input Power (PIN)"]
    d["hour_ist"] = d.index.tz_convert(IST).hour + d.index.tz_convert(IST).minute / 60
    d.to_parquet(OUT / "tables" / f"edfa_{short(asset)}_5min.parquet")
edfa_resample_report = pd.DataFrame(
    {
        short(a): {
            "bins": len(d), "from": d.index.min(), "to": d.index.max(),
            "bins with no reading (after fill)": int(d[EDFA_FEATURES].isna().all(axis=1).sum()),
            "outage bins": int(d["outage"].sum()),
            "values filled": int(d[[f"{c}__filled" for c in EDFA_FEATURES]].sum().sum()),
        }
        for a, d in edfa5.items()
    }
)
show(edfa_resample_report)
log_step("B1 resample", "EDFA: 5-min median bins from first optical reading", "keeps minute-scale fibre events visible, removes single-poll noise; 3-4 readings per bin",
         f"{sum(len(d) for d in edfa5.values()):,} bins over 2 assets")
log_step("B2 gaps", "fill polling misses <= 2 bins by time interpolation; never fill outages or longer gaps",
         "slow physical quantities; longer gaps would invent data", f"{int(edfa_resample_report.loc['values filled'].sum()):,} EDFA values filled")

SFP_CHASSIS_TEMP = "UNMAPPED_3d9bf6d0"  # inferred whole-degree temperature; Part C shows it tracks the laser temperatures
SFP_FEATURES = [f"{p} Current: {m}" for p in SFP_PORTS for m in SFP_METRICS] + [SFP_CHASSIS_TEMP]
SFP_CROSSCHECK = [f"{p} {win}: {m}" for p in SFP_PORTS for win in ("Previous 15min", "Current Day", "Previous Day")
                  for m in ("Laser Bias", "Transmit Power", "Received Power")]
sfp15 = resample_asset("SFP", SFP_ASSET, SFP_FEATURES + SFP_CROSSCHECK, "15min")
sfp15.to_parquet(OUT / "tables" / "sfp_15min.parquet")
print(f"SFP 15-min table: {len(sfp15):,} bins, {sfp15['outage'].sum()} outage bins, "
      f"{int(sfp15[[f'{c}__filled' for c in SFP_FEATURES]].sum().sum()):,} values filled, "
      f"bins with no feature value: {int(sfp15[SFP_FEATURES].isna().all(axis=1).sum()):,}")
log_step("B1 resample", "SFP: 15-min median bins aligned to the module's 15-min windows", "slower failure modes; matches device aggregation",
         f"{len(sfp15):,} bins")
log_step("B4 SFP window", "use the Current window for Bias/Temp/Voltage/Tx/Rx; keep day windows only as a cross-check",
         "Current: 0% impossible values and poll-aligned; Current Day up to 24% impossible; Previous 15min clock 49-94 min ahead with constant bias offsets",
         f"{len(SFP_FEATURES) - 1} of 40 window x metric x port series kept as features")

# %% [markdown]
# ### B6. Feature set and scaling
#
# Every sensor not carried into Part D is listed with a one-line reason in `outputs/tables/feature_decisions_*.csv`. The summary is below.
#
# **Scaling: per asset, never global.**
#
# * The detector models (Part D) work in physical units: mA, °C, dBm.
# * Residuals are scaled per asset by a robust spread (1.4826 × MAD) estimated on that asset's training period. The spread is floored at the sensor's quantisation step, so a coarse sensor cannot produce a zero spread.
# * The Isolation Forest cross-check uses per-asset median/IQR scaling.
#
# Global scaling would mix the two EDFA operating classes. 010147e2 holds POUT at 15 dBm on 167 mA of pump current, while 7a1d09b1 holds 7 dBm on 67 mA. A global scale would make one unit look permanently anomalous relative to the other. Scaling per asset is the same thing as normalising by operating class here, since each unit has a single, constant set-point.

# %%
EDFA_DROP_REASON = {
    "junk": "not physical (single-pump unit or constant-0 counter)",
    "config": "configuration/identity, not a measurement",
    "network": "used to detect outages (B2), not a model input",
    "status": "uptime counter: monotonic, no reboot in the optical period; used only as a reboot check",
}


def feature_decisions(kind: str) -> pd.DataFrame:
    table = inv[kind][["asset", "sensor", "class"]].copy()
    if kind == "EDFA":
        table["kept"] = table["sensor"].isin(EDFA_FEATURES)
        table["reason"] = np.where(table["kept"], "physical telemetry used by the detectors",
                                   table["class"].map(EDFA_DROP_REASON).fillna("not needed"))
        bimodal = (table["asset"] == "7a1d09b1") & (table["sensor"] == "Pump1 Temperature")
        table.loc[bimodal, "reason"] = "kept, but only 2 distinct values (25.0 / 34.68): used as a stuck-reading check, not a model input"
    else:
        meta = sfp_names.set_index("sensor")
        table["kept"] = table["sensor"].isin(SFP_FEATURES)

        def reason(row):
            if row["kept"]:
                return "physical telemetry (Current window)" if row["sensor"] != SFP_CHASSIS_TEMP else "context: inferred chassis temperature"
            if row["sensor"] in SFP_CROSSCHECK:
                return "same quantity as the Current window; kept only as a cross-check"
            if row["sensor"].startswith("UNMAPPED_"):
                return "unmapped and constant (or a -48 V feed with no variation that matters): no information for the lasers"
            m = meta.loc[row["sensor"]]
            if m["metric"] == "Timestamp":
                return "window close time: used for the clock check in B4"
            if m["metric"] in SFP_VALID:
                return f"{m['window']} window: same quantity as Current, less reliable (B4)"
            if m["metric"] in ("Present Status", "Operational Status", "Valid Status", "Interval Valid"):
                return "status flag: constant for 88 days (no port flaps, every interval valid)"
            if m["metric"] == "Interface Number":
                return "configuration; corrupted with fractional values on 1-4-3"
            return "network reachability: used for outage masking"

        table["reason"] = table.apply(reason, axis=1)
    return table


log_step("B3 junk", "exclude Pump2 * (both units single-pump), Uptime (hardware) (constant 0), SFP 'Current: Timestamp' (constant-0 sentinel)",
         "not physical measurements; the zero timestamp is a sentinel, the other windows' timestamps are real epochs", "8 + 2 EDFA columns, 2 SFP columns")
feat = {kind: feature_decisions(kind) for kind in ("EDFA", "SFP")}
for kind, table in feat.items():
    table.to_csv(OUT / "tables" / f"feature_decisions_{kind.lower()}.csv", index=False)
    log_step("B5 features", f"{kind}: keep {int(table['kept'].sum())} of {len(table)} asset-sensor columns",
             "physical telemetry tied to the failure modes; every drop has a one-line reason in feature_decisions", f"{int((~table['kept']).sum())} dropped")
show(feat["EDFA"].sort_values(["kept", "asset"], ascending=[False, True]))
show(feat["SFP"].groupby(["kept", "reason"]).size().rename("sensors"))
log_step("B6 scaling", "per-asset scaling; residuals in physical units scaled by training-period MAD floored at sensor resolution",
         "two EDFA operating classes (15 vs 7 dBm); quantised sensors would otherwise give zero spread", "all model inputs")

# %% [markdown]
# ### Preprocessing log
#
# The log is generated by the code above, so it cannot drift from what actually ran. It is also written to `outputs/preprocessing_log.md`.

# %%
prep_log = pd.DataFrame(PREP_LOG)
show(prep_log)
(OUT / "preprocessing_log.md").write_text("# Preprocessing log (generated by run.py)\n\n" + md_table(prep_log) + "\n", encoding="utf-8")

# %% [markdown]
# ## Part C: Exploratory analysis and insights
#
# Each insight is one claim, the numbers or plot that support it, and why it matters operationally.
# The cell below computes the evidence. The findings are summarised in markdown after the plots and in `insights.md`.

# %%
EDFA_LABEL = {a: f"EDFA {short(a)}" for a in EDFA_ASSETS}
COLORS = {EDFA_ASSETS[0]: "#1f77b4", EDFA_ASSETS[1]: "#d62728"}
evidence: dict[str, object] = {}


def shade_outages(ax, outage: pd.Series, label: bool = True) -> None:
    on = outage.fillna(False).astype(bool)
    run_id = (on != on.shift()).cumsum()
    first = True
    for _, r in on[on].groupby(run_id[on]):
        ax.axvspan(r.index[0], r.index[-1], color="grey", alpha=0.25, lw=0, label="ICMP outage" if (first and label) else None)
        first = False


# --- C1. Coverage: what the monitoring system did -----------------------------------------
def daily_state(kind: str, optical_cols: list[str]) -> pd.DataFrame:
    w = clean_1min[kind]
    rows = []
    for asset in w.index.get_level_values("asset_id").unique():
        x = w.loc[asset]
        days = pd.date_range("2026-01-02", "2026-03-31", freq="1D", tz="UTC")
        polled = x.index.floor("1D").value_counts().reindex(days, fill_value=0) / 1440
        optical = x[[c for c in optical_cols if c in x.columns]].notna().any(axis=1)
        optical_days = optical.groupby(x.index.floor("1D")).mean().reindex(days, fill_value=0)
        outage = x["outage"].groupby(x.index.floor("1D")).mean().reindex(days, fill_value=0)
        rows.append(pd.DataFrame({"asset": f"{kind} {short(asset)}", "polled": polled, "optical": optical_days, "outage": outage}))
    return pd.concat(rows)


cov = pd.concat([daily_state("EDFA", EDFA_OPTICAL), daily_state("SFP", SFP_OPTICAL)])
no_data_days = cov.groupby(level=0)["polled"].max().eq(0)
evidence["collector_gap_days"] = int(no_data_days.sum())
evidence["collector_gaps"] = [(str(g.index[0].date()), str(g.index[-1].date()), len(g))
                              for _, g in no_data_days[no_data_days].groupby((~no_data_days).cumsum()[no_data_days])]
edfa_jan = clean_1min["EDFA"].loc[(slice(None), slice("2026-01-02", "2026-01-20")), "outage"]
evidence["edfa_jan_unreachable_share"] = round(float(edfa_jan.mean()), 3)
evidence["edfa_optical_first"] = str(long["EDFA"].loc[long["EDFA"]["sensor"].isin(EDFA_OPTICAL), "ts"].min())
evidence["edfa_optical_days"] = round(float(cov[cov["asset"].str.startswith("EDFA")].groupby("asset")["optical"].apply(lambda s: (s > 0.5).sum()).mean()), 1)
print(evidence)

fig, ax = plt.subplots(figsize=(12, 3.2))
for i, (asset, g) in enumerate(cov.groupby("asset", sort=False)):
    state = np.select([g["polled"] == 0, g["outage"] > 0.5, g["optical"] > 0.5], [0, 1, 3], default=2)
    cmap = matplotlib.colors.ListedColormap(["white", "#d62728", "#ffbf00", "#2ca02c"])
    ax.imshow(state[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3, extent=[0, len(g), i + 0.1, i + 0.9])
ax.set_yticks([0.5, 1.5, 2.5], [f"{k}" for k in cov["asset"].unique()])
ax.set_xticks(range(0, 89, 7), [d.strftime("%d %b") for d in pd.date_range("2026-01-02", periods=13, freq="7D")])
ax.set_xlabel("date (UTC)")
ax.set_title("C1. Daily monitoring state per asset: optical telemetry exists on only 22 of 88 days", fontsize=10)
handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
           [("#2ca02c", "optical telemetry"), ("#d62728", "unreachable"), ("#ffbf00", "reachable, no optics"), ("white", "no data")]]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=False)
save_fig(fig, "C1_monitoring_coverage")

# %%
# --- C2 + C3. Site thermal event, and the cooler current explained by case temperature ----------
THERMAL_EVENT = ("2026-02-16 10:00", "2026-02-18 06:30")
BASELINE = ("2026-02-18 12:00", "2026-03-31 23:59")
c2 = []
for asset, d in edfa5.items():
    ev, base = d.loc[THERMAL_EVENT[0]:THERMAL_EVENT[1]], d.loc[BASELINE[0]:BASELINE[1]]
    x, y = base["EDFA Case Temperature"], base["Pump1 Itec"]
    ok = x.notna() & y.notna()
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    hourly = base.groupby(base["hour_ist"].astype(int))["EDFA Case Temperature"].median()
    c2.append({
        "asset": short(asset),
        "case temp, event median (degC)": ev["EDFA Case Temperature"].median(),
        "case temp, rest median (degC)": base["EDFA Case Temperature"].median(),
        "Itec, event median (mA)": ev["Pump1 Itec"].median(),
        "Itec, rest median (mA)": base["Pump1 Itec"].median(),
        "pump chip temp during event (degC)": ev["Pump1 Temperature"].median(),
        "Itec slope (mA per degC)": slope,
        "Itec vs case temp r": np.corrcoef(x[ok], y[ok])[0, 1],
        "warmest IST hour": int(hourly.idxmax()), "coolest IST hour": int(hourly.idxmin()),
        "diurnal swing (degC)": hourly.max() - hourly.min(),
    })
c2 = pd.DataFrame(c2).set_index("asset").round(2)
both = pd.concat({short(a): d["EDFA Case Temperature"] for a, d in edfa5.items()}, axis=1).dropna()
evidence["case_temp_corr_between_units"] = round(float(both.corr().iloc[0, 1]), 3)
show(c2)
print("correlation of the two units' case temperatures (5-min):", evidence["case_temp_corr_between_units"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for asset, d in edfa5.items():
    seg = d.loc["2026-02-16":"2026-02-21"]
    axes[0].plot(seg.index, seg["EDFA Case Temperature"], color=COLORS[asset], label=f"{EDFA_LABEL[asset]} case temp")
    ax2 = axes[0].twinx() if asset == EDFA_ASSETS[0] else ax2
    ax2.plot(seg.index, seg["Pump1 Itec"], color=COLORS[asset], ls=":", lw=1, label=f"{EDFA_LABEL[asset]} Itec")
axes[0].axvspan(pd.Timestamp(THERMAL_EVENT[0], tz="UTC"), pd.Timestamp(THERMAL_EVENT[1], tz="UTC"), color="orange", alpha=0.15, label="thermal event")
axes[0].set_ylabel("case temperature (degC), solid")
ax2.set_ylabel("TEC current Itec (mA), dotted")
axes[0].set_title("C2. Both EDFAs ran 7-9 degC hot for 44 h after monitoring started")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
for asset, d in edfa5.items():
    base = d.loc[BASELINE[0]:BASELINE[1]]
    ev = d.loc[THERMAL_EVENT[0]:THERMAL_EVENT[1]]
    axes[1].scatter(base["EDFA Case Temperature"], base["Pump1 Itec"], s=3, alpha=0.3, color=COLORS[asset], label=f"{EDFA_LABEL[asset]} normal")
    axes[1].scatter(ev["EDFA Case Temperature"], ev["Pump1 Itec"], s=6, marker="x", color=COLORS[asset], label=f"{EDFA_LABEL[asset]} thermal event")
axes[1].set_xlabel("EDFA case temperature (degC)")
axes[1].set_ylabel("pump TEC current Itec (mA)")
axes[1].set_title("C3. Cooler current is set by case temperature (same line during the event)")
axes[1].legend(fontsize=8)
save_fig(fig, "C2_C3_thermal")

# %%
# --- C4. Operating classes ------------------------------------------------------------------
c4 = pd.DataFrame({
    short(a): {
        "POUT set-point (dBm)": d["Output Power (POUT)"].median(),
        "PIN median (dBm)": d["Input Power (PIN)"].median(),
        "gain median (dB)": d["gain_db"].median(),
        "pump current median (mA)": d["Pump1 Current"].median(),
        "pump power median (mW)": d["Pump1 Power"].median(),
        "POUT alarm threshold (dBm)": wide["EDFA"].loc[a, "Output Power Threshold (PIN)"].median(),
        "PIN alarm threshold (dBm)": wide["EDFA"].loc[a, "Input Power Threshold (PIN)"].median(),
    } for a, d in edfa5.items()
})
c4.loc["PIN margin to alarm (dB)"] = c4.loc["PIN median (dBm)"] - c4.loc["PIN alarm threshold (dBm)"]
c4.loc["POUT margin to alarm (dB)"] = c4.loc["POUT set-point (dBm)"] - c4.loc["POUT alarm threshold (dBm)"]
ratio_out = 10 ** ((c4.iloc[0, 0] - c4.iloc[0, 1]) / 10)
ratio_pump = c4.loc["pump current median (mA)"].iloc[0] / c4.loc["pump current median (mA)"].iloc[1]
evidence["output_power_ratio"] = round(float(ratio_out), 2)
evidence["pump_current_ratio"] = round(float(ratio_pump), 2)
show(c4.round(2))
print(f"output power ratio {ratio_out:.1f}x for pump-current ratio {ratio_pump:.2f}x")

# %%
# --- C5. 7a1d09b1: pump power fell 28% with no change in drive current or output --------------
a7 = next(a for a in EDFA_ASSETS if short(a) == "7a1d09b1")
a0 = next(a for a in EDFA_ASSETS if short(a) == "010147e2")
d7 = edfa5[a7]
C5_COLS = ["Pump1 Power", "Pump1 Current", "Output Power (POUT)", "Input Power (PIN)", "EDFA Case Temperature"]
before, after = d7.loc["2026-02-18 12:00":"2026-02-21 23:55"], d7.loc["2026-02-24":"2026-03-31"]
c5 = pd.DataFrame({"18-21 Feb": before[C5_COLS].median(), "24 Feb-31 Mar": after[C5_COLS].median()})
c5["change"] = c5["24 Feb-31 Mar"] - c5["18-21 Feb"]
c5["change %"] = 100 * c5["change"] / c5["18-21 Feb"]
show(c5.round(2))
evidence["pump_power_drop_pct"] = round(float(c5.loc["Pump1 Power", "change %"]), 1)

fig, ax = plt.subplots(figsize=(12, 3.6))
seg = d7.loc["2026-02-18":"2026-03-01"]
ax.plot(seg.index, seg["Pump1 Power"], color="#d62728", label="Pump1 Power (mW)")
ax.plot(seg.index, seg["Pump1 Current"] / 2.5, color="#1f77b4", label="Pump1 Current / 2.5 (mA)")
ax.plot(seg.index, seg["Output Power (POUT)"] * 3, color="#2ca02c", label="POUT x 3 (dBm)")
ax.set_ylabel("scaled units (see legend)")
ax.set_title("C5. EDFA 7a1d09b1: pump optical power -28% (26.0 -> 18.6 mW) while drive current and POUT stay flat")
ax.legend(loc="lower left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save_fig(fig, "C5_pump_power_7a1d09b1")

# %%
# --- C6. Monitoring integrity: values that leave the sensor's own quantisation grid ------------
def quantisation_breaks(kind: str, asset, cols: list[str], grid: float = 0.1, ref_hours: int = 48) -> pd.DataFrame:
    """A sensor that reported only multiples of `grid` in its first `ref_hours` of data is 'quantised'.
    Later readings off that grid did not come through the same measurement path."""
    w = wide[kind].loc[asset]
    rows = []
    for c in cols:
        x = w[c].dropna()
        if x.empty:
            continue
        ref = x.loc[: x.index[0] + pd.Timedelta(hours=ref_hours)]
        if (((ref / grid) - (ref / grid).round()).abs() < 1e-6).mean() < 0.99:
            continue
        off = x[((x / grid) - (x / grid).round()).abs() >= 1e-6]
        if off.empty:
            continue
        run = off.index.to_series().diff().gt(pd.Timedelta("6h")).cumsum()
        for _, g in off.groupby(run.values):
            rows.append({"asset": short(asset), "sensor": c, "first": g.index[0], "last": g.index[-1], "readings": len(g),
                         "typical value before": x.loc[: g.index[0]].iloc[:-1].median(), "median off-grid value": g.median()})
    return pd.DataFrame(rows)


qbreaks = pd.concat([quantisation_breaks("EDFA", a, EDFA_OPTICAL) for a in EDFA_ASSETS], ignore_index=True)
show(qbreaks)
qbreaks.to_csv(OUT / "tables" / "quantisation_breaks_edfa.csv", index=False)
stuck = wide["EDFA"].loc[a7, "Pump1 Temperature"].dropna()
stuck_hot = stuck[stuck > 30]
evidence["pump_temp_stuck"] = {"value": round(float(stuck_hot.median()), 3), "readings": len(stuck_hot),
                               "first": str(stuck_hot.index.min()), "last": str(stuck_hot.index.max()),
                               "itec_median_during": round(float(wide["EDFA"].loc[a7, "Pump1 Itec"].reindex(stuck_hot.index).median()), 1),
                               "itec_median_26feb_1mar": round(float(wide["EDFA"].loc[a7, "Pump1 Itec"].loc["2026-02-26":"2026-03-01"].median()), 1)}
print("7a1d09b1 Pump1 Temperature off set-point:", evidence["pump_temp_stuck"])

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
x = wide["EDFA"].loc[a0, "Pump1 Current"].loc["2026-03-23":"2026-03-31"].dropna()
axes[0].plot(x.index, x, ".", ms=2, color="#1f77b4")
axes[0].set_ylabel("Pump1 Current (mA), 1-min readings")
axes[0].set_title("C6a. 010147e2 pump current: -2.7% step at 00:00 UTC 25 Mar,\nnew values off the usual 0.1 mA grid", fontsize=10)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
y = wide["EDFA"].loc[a7, ["Pump1 Temperature", "Pump1 Itec"]].loc["2026-02-26":"2026-03-02"]
axes[1].plot(y.index, y["Pump1 Temperature"], ".", ms=2, color="#d62728", label="Pump1 Temperature (degC)")
ax2 = axes[1].twinx()
ax2.plot(y.index, y["Pump1 Itec"], ".", ms=1, color="grey", alpha=0.5, label="Pump1 Itec (mA)")
axes[1].set_ylabel("pump chip temperature (degC), red")
ax2.set_ylabel("TEC current (mA), grey")
axes[1].set_title("C6b. 7a1d09b1 pump temperature jumps to a fixed 34.68 degC\nwhile the cooler current does not react", fontsize=10)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save_fig(fig, "C6_monitoring_integrity")

# %%
# --- C7. SFP: the four reporting windows disagree; shifts confined to one window -----------
bias_daily = pd.DataFrame({(p[-3:], win): sfp_w[f"{p} {win}: Laser Bias"].resample("1D").median()
                           for p in SFP_PORTS for win in WINDOWS}).dropna(how="all")
shift_table = []
for p in SFP_PORTS:
    for metric in ("Laser Bias", "Transmit Power"):
        cur = sfp_w[f"{p} Current: {metric}"].resample("1D").median()
        ref = pd.concat([sfp_w[f"{p} {win}: {metric}"].resample("1D").median() for win in ("Previous 15min", "Current Day", "Previous Day")], axis=1).median(axis=1)
        gap = (cur - ref).dropna()
        shift_table.append({"port": p, "metric": metric,
                            "Current minus other-window median, 2-5 Jan": round(gap.loc["2026-01-02":"2026-01-05"].median(), 3),
                            "same, 23-31 Mar": round(gap.loc["2026-03-23":].median(), 3),
                            "largest day-to-day jump": round(gap.diff().abs().max(), 3), "on": gap.diff().abs().idxmax().date()})
show(pd.DataFrame(shift_table))
evidence["current_day_voltage_invalid_pct"] = round(float(invalid_rate.loc["Laser Voltage", "Current Day"] * 100), 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for ax, p in zip(axes, SFP_PORTS):
    for win, style in zip(WINDOWS, ["-", "--", ":", "-."]):
        s_ = bias_daily[(p[-3:], win)]
        ax.plot(s_.index, s_, style, marker="o", ms=3, label=win)
    ax.set_title(f"C7. {p} laser bias, daily median by reporting window", fontsize=10)
    ax.set_ylabel("laser bias (mA)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[0].legend(fontsize=8)
save_fig(fig, "C7_sfp_windows")

# %%
# --- C8. SFP lasers warm with the shelf -----------------------------------------------------
temp_cols = [f"{p} Current: Laser Temperature" for p in SFP_PORTS]
daily_t = sfp15[temp_cols + [SFP_CHASSIS_TEMP]].resample("1D").agg(["median", "max"]).dropna(how="all")
first_wk, last_wk = sfp15.loc["2026-01-02":"2026-01-08"], sfp15.loc["2026-03-23":"2026-03-31"]
weeks = (last_wk.index.mean() - first_wk.index.mean()).days / 7
c8 = pd.DataFrame({
    "2-8 Jan median": first_wk[temp_cols + [SFP_CHASSIS_TEMP]].median(),
    "23-31 Mar median": last_wk[temp_cols + [SFP_CHASSIS_TEMP]].median(),
    "23-31 Mar daily max (median)": last_wk[temp_cols + [SFP_CHASSIS_TEMP]].resample("1D").max().median(),
})
c8["rise (degC)"] = c8["23-31 Mar median"] - c8["2-8 Jan median"]
c8["rate (degC/week)"] = c8["rise (degC)"] / weeks
ASSUMED_SFP_CASE_LIMIT = 70.0  # commercial-temperature SFP case limit; an assumption, the module part number is unknown
c8["weeks until daily max reaches 70 degC at this rate"] = (ASSUMED_SFP_CASE_LIMIT - c8["23-31 Mar daily max (median)"]) / c8["rate (degC/week)"]
corr_chassis = sfp15[temp_cols + [SFP_CHASSIS_TEMP]].corr()[SFP_CHASSIS_TEMP].drop(SFP_CHASSIS_TEMP)
show(c8.round(2))
print("correlation of laser temperatures with the inferred chassis temperature:", corr_chassis.round(3).to_dict())
print("inferred -48 V feeds (min, max):", {c: (sfp_w[c].min(), sfp_w[c].max()) for c in ("UNMAPPED_56c10543", "UNMAPPED_f9d48b36")})

fig, ax = plt.subplots(figsize=(12, 3.6))
for c, color in zip(temp_cols, ["#1f77b4", "#d62728"]):
    ax.plot(daily_t.index, daily_t[(c, "median")], "o-", ms=3, color=color, label=f"{c.split(' ')[0]} laser temp, daily median")
    ax.plot(daily_t.index, daily_t[(c, "max")], ".", ms=4, color=color, alpha=0.5, label=f"{c.split(' ')[0]} daily max")
ax.plot(daily_t.index, daily_t[(SFP_CHASSIS_TEMP, "median")], "s-", ms=3, color="black", label="chassis temp (inferred, unmapped id 3d9bf6d0)")
ax.axhline(ASSUMED_SFP_CASE_LIMIT, color="orange", ls="--", label="70 degC (assumed commercial SFP limit)")
ax.set_ylabel("temperature (degC)")
ax.set_title("C8. SFP lasers warmed ~8 degC from January to late March, tracking the shelf temperature")
ax.legend(fontsize=7, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save_fig(fig, "C8_sfp_temperature")


# %% [markdown]
# ### Insights
#
# Every number below is printed by a cell in this part.
#
# **1. Only 22 of the 88 days in the files contain EDFA optical telemetry. The gaps are the monitoring system's, not the devices'.**
#
# * From 2 to 20 Jan both EDFAs were unreachable 93.5% of the time (ICMP loss 100%) and reported no optical sensors at all. Optical telemetry first appears on 16 Feb at 10:41 UTC.
# * On 46 days there are **no rows from any sensor on any of the three assets**: 21 Jan–12 Feb (23 days), 14–15 Feb (2 days) and 2–22 Mar (21 days).
# * All assets drop out on the same days, so this is a collector or export gap, not three devices failing at once.
# * The uptime counters never reset (both EDFAs were up 354 → 397 days), which confirms the devices kept running through every gap.
#
# *Why it matters:* any baseline or trend for these EDFAs rests on two blocks of 13 and 9 days. A 3-week hole in the history of a live amplifier is itself a maintenance-visibility problem worth raising with the NMS team.
#
# **2. Both EDFAs ran 7–9 °C hot for the first 44 hours of monitoring (16 Feb 10:41 to 18 Feb ~06:30 UTC). It was the site, not the amplifiers.**
#
# * Case temperature medians were 37.7 °C on 010147e2 and 40.4 °C on 7a1d09b1, against 30.8 and 31.6 °C afterwards.
# * The cooler (TEC) current doubled, to 349 and 386 mA from 173 and 161 mA.
# * Both units moved together: their case temperatures correlate at r = 0.995. The pump chips stayed at 25.0–25.1 °C throughout, so the coolers had headroom and held regulation.
#
# *Why it matters:* this was a site cooling event, probably HVAC, coinciding with the restoration of monitoring. A naive Itec threshold would have paged twice, once per unit, for a problem that belongs to facilities.
#
# **3. Cooler current is almost entirely explained by case temperature, about 25 mA per °C on both units (r = 0.997). The EDFA room is warmest at night.**
#
# * Fitted slopes are 24.5 and 24.9 mA/°C, and the 44-hour hot spell sits on the same line (plot C3).
# * The daily temperature swing is small (1.7–1.8 °C) and **inverted**: warmest 02:00–05:00 IST, coolest around 17:00 IST. The SFP site follows the normal pattern (laser temperatures peak 16:00–18:00 IST).
#
# *Why it matters:* the cooler's health check should be "Itec given case temperature", never Itec alone. The inverted cycle suggests the EDFA room's air conditioning is scheduled down at night. That is plausible but unverified, and worth checking against the site's HVAC settings before summer.
#
# **4. The two amplifiers sit in different operating classes, and that fully explains their different pump drive.**
#
# | | 010147e2 | 7a1d09b1 |
# |---|---|---|
# | POUT set-point | 15.0 dBm | 7.1 dBm |
# | Gain | 14.8 dB | 5.4 dB |
# | Pump current | 167 mA | 67 mA |
#
# * Output power differs 6.2× while pump current differs only 2.5×.
# * Both have their POUT alarm 3 dB below the set-point, and 9.2 and 10.7 dB of PIN margin above the −9 dBm input alarm.
# * PIN never moved more than 0.3 dB on either unit, so no upstream fibre event occurred in the data.
#
# *Why it matters:* scaling, thresholds and spare-part planning must be done per operating class. A shared threshold would mark 010147e2 as permanently "over-driven".
#
# **5. On 7a1d09b1 the reported pump power fell 28% (26.0 → 18.6 mW, in two steps on 22 and 24 Feb), with pump current (66.8 mA), POUT (7.1 dBm) and case temperature unchanged.**
#
# * By the primer, falling pump power at constant current is the pump-degradation signature.
# * But a real 28% loss of pump light would cut the gain, so POUT would sag or the control loop would raise the current. Neither happened.
#
# *Why it matters:* the physics points to the **pump-power monitor or its reporting path**, not the pump. Recommended action: verify pump power locally before scheduling a pump swap. Treating it as pump end-of-life would replace a working amplifier.
#
# **6. Three EDFA "events" carry a monitoring signature: they start exactly at 00:00 UTC and introduce values the sensor never produced before.**
#
# * **010147e2 pump current** reads 162.67 mA, −2.7%, from 00:00 UTC on 25 Mar (on and off until the end of the data). Before that the sensor only ever reported values on a 0.1 mA grid, such as 166.4 and 167.2.
# * **7a1d09b1 pump power** steps fall at 00:00 on 22 and 24 Feb, to exactly 85% and 72% of 26.0 mW.
# * **7a1d09b1 pump temperature** jumps to a fixed 34.681 °C, 9.7 °C above its set-point, from 00:00 on 28 Feb and again on 1 Mar. Meanwhile the cooler current does not react (194 mA during these readings, against 178 mA that week). A pump really 9.7 °C over its set-point would drive the TEC to its limit.
#
# *Why it matters:* this separates what the devices did from what the monitoring did. Real pump changes are not scheduled for UTC midnight and do not change a sensor's resolution. These are processing artefacts, or deliberately injected test anomalies (the delivered files are named `*_anomalies_f.csv`). Either way, they should be routed to the monitoring team, not to field maintenance. The detector in Part D flags all three anyway.
#
# **7. The four SFP reporting windows disagree. Every step change in the SFP data is confined to a single window.**
#
# * **`Current Day`** carries almost all impossible values: 23.8% of its Laser Voltage readings, plus negative bias and Tx.
# * **`Previous 15min`** is stamped 49–94 minutes **ahead** of the poll time, because the module clock is fast. Its bias also sits 5.3 mA above every other window on 1-4-3 throughout.
# * **`Current`** steps on its own:
#   * Tx on 1-4-3 −12% (0.83 → 0.73 mW) on 6 Jan;
#   * bias on 1-4-4 +5.5 mA (+16%) on 18 Feb;
#   * bias on 1-4-3 −4.4 mA on 27 Feb.
#
#   In all three cases the day windows, computed by the same module, show no step.
# * Status flags (present, operational, interval valid) were constant for all 88 days, so there were no port flaps.
#
# *Why it matters:* a single-window step is a reporting-path problem. A laser that really aged 16% would move every window. Alerting on any one window would send technicians to swap healthy optics. Comparing windows against each other is a cheap, label-free integrity check.
#
# **8. Both SFP lasers warmed by about 8 °C between early January and late March (46.6 → 55.0 °C and 45.6 → 53.9 °C), tracking the shelf temperature.**
#
# * One of the unmapped ids (`3d9bf6d0`) behaves like a chassis temperature: 42 → 49 °C, correlating with the laser temperatures at r = 0.81 and 0.92.
# * Laser bias in the day windows rose about 2.5 mA, consistent with the fitted 0.29–0.31 mA/°C thermal coefficient. That is normal physics, not ageing.
# * At the observed 0.73 °C/week, late-March daily peaks (58.1 and 56.9 °C) reach 70 °C in about 16–18 weeks, that is by mid-July. 70 °C is an *assumed* commercial-grade SFP limit; the part numbers are unknown.
# * Two other unmapped ids behave like the shelf's −48 V DC feeds (−46.8 to −48.1 V), sitting well inside the constant −43.2 / −56 V values that look like their alarm limits.
#
# *Why it matters:* the Indian summer (April–June) will push these lasers toward their thermal limit before any ageing does. Check the shelf's fans and cooling at this site now, and confirm the SFP temperature grade.

# %% [markdown]
# ## Part D: Anomaly detection
#
# ### D1. Framing
#
# **Primary target: the EDFAs. Framing: contextual anomalies, with persistence turning them into collective (level-shift) alerts. Point anomalies are handled by the B3 rule layer.**
#
# Most of the variance in the EDFA telemetry is explained by context:
#
# * The cooler current tracks case temperature at r = 0.997 (C3).
# * POUT is pinned by the control loop.
#
# So a reading is only abnormal **given** its context. A raw threshold on Itec would fire every warm night and throughout the 16–18 Feb site event, which C3 shows the cooler handled normally.
#
# The primer's failure modes are themselves relationships, and each maps to the residual of a small physics model:
#
# * pump current at constant POUT (ageing, control-loop saturation);
# * pump power for the same current (pump degradation);
# * cooler current for the same case temperature (cooler or thermal fault);
# * pump temperature against its set-point (lost thermal regulation).
#
# What the data cannot support:
#
# * **Degradation trend with a lead time.** Each unit has about 22 days of optical data in two blocks separated by a 3-week collection gap, and the shifts we see are steps, not ramps (D6 checks this).
# * **A learned multivariate model.** There are only 2 units and no labels. The Isolation Forest is kept only as an independent cross-check.
#
# **Secondary target: SFP (D7).** The SFP changes found in C7 are confined to one reporting window. They are framed as **collective anomalies in the reporting path**, detected by comparing the `Current` window against the two day windows. A contextual bias-vs-temperature residual is added on top.

# %%
TRAIN = ("2026-02-18 12:00", "2026-02-21 23:55")
PERSIST_BINS = 3        # an alert needs 3 consecutive 5-min bins (15 min) above threshold
MIN_THRESHOLD = 4.0     # never alert below 4 robust standard deviations
EDFA_MODELS = {
    # name: (target, inputs, minimum meaningful deviation in target units)
    "cooler": ("Pump1 Itec", ["EDFA Case Temperature"], 3.0),
    "pump_drive": ("Pump1 Current", ["Input Power (PIN)", "EDFA Case Temperature"], 0.8),
    "pump_efficiency": ("pump_eff", [], 0.01),
    "pump_temp": ("Pump1 Temperature", [], 0.25),
}


def robust_scale(r: np.ndarray, floor: float) -> float:
    """1.4826 x MAD of the training residuals, floored at the smallest deviation that matters physically."""
    r = r[~np.isnan(r)]
    return max(1.4826 * float(np.median(np.abs(r - np.median(r)))), floor)


def fit_linear(df: pd.DataFrame, target: str, inputs: list[str], floor: float) -> dict:
    """Ordinary least squares in physical units: target = b0 + sum(b_i * input_i). With no inputs it is the training median."""
    d = df[[target] + inputs].dropna()
    if inputs:
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in inputs])
        coef, *_ = np.linalg.lstsq(X, d[target].to_numpy(), rcond=None)
    else:
        coef = np.array([d[target].median()])
    model = {"target": target, "inputs": inputs, "coef": coef, "n_train": len(d)}
    resid = d[target].to_numpy() - predict(model, d).to_numpy()
    model["scale"] = robust_scale(resid, floor)
    model["r2"] = 1 - resid.var() / d[target].var() if inputs and d[target].var() > 0 else np.nan
    return model


def predict(model: dict, df: pd.DataFrame) -> pd.Series:
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in model["inputs"]])
    return pd.Series(X @ model["coef"], index=df.index)


def persistent(score: pd.Series, bins: int = PERSIST_BINS) -> pd.Series:
    """|score| that has been exceeded for `bins` consecutive bins: rolling minimum of |score|."""
    return score.abs().rolling(bins, min_periods=bins).min()


MERGE_GAP_BINS = 72     # an incident stays open until the condition has been clear for 72 bins (6 h at 5 min)


def alert_episodes(flag: pd.Series, merge_gap: int = 0) -> pd.DataFrame:
    """Runs of True bins; runs separated by at most `merge_gap` bins are merged into one episode."""
    flag = flag.fillna(False).astype(bool)
    run_id = (flag != flag.shift()).cumsum()
    runs = [(r.index[0], r.index[-1]) for _, r in flag[flag].groupby(run_id[flag])]
    step = flag.index[1] - flag.index[0] if len(flag) > 1 else pd.Timedelta("5min")
    merged: list[list] = []
    for start, end in runs:
        if merged and (start - merged[-1][1]) <= step * (merge_gap + 1):
            merged[-1][1] = end
        else:
            merged.append([start, end])
    rows = [{"start": a, "end": b, "bins": int((b - a) / step) + 1} for a, b in merged]
    return pd.DataFrame(rows, columns=["start", "end", "bins"])


def prepare_edfa(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pump_eff"] = d["Pump1 Power"] / d["Pump1 Current"]
    return d


edfa_fit, edfa_scored, model_table = {}, {}, []
for asset in EDFA_ASSETS:
    d = prepare_edfa(edfa5[asset])
    train = d.loc[TRAIN[0]:TRAIN[1]]
    train = train[~train["outage"]]
    fits = {name: fit_linear(train, tgt, inp, floor) for name, (tgt, inp, floor) in EDFA_MODELS.items()}
    for name, m in fits.items():
        d[f"pred_{name}"] = predict(m, d)
        d[f"z_{name}"] = (d[m["target"]] - d[f"pred_{name}"]) / m["scale"]
        train_z = d.loc[train.index, f"z_{name}"]
        m["threshold"] = max(MIN_THRESHOLD, float(np.nanquantile(persistent(train_z), 0.999)))
        d[f"alert_{name}"] = persistent(d[f"z_{name}"]) > m["threshold"]
        model_table.append({"asset": short(asset), "model": name, "target": m["target"], "inputs": ", ".join(m["inputs"]) or "(training median)",
                            "coef": np.round(m["coef"], 3).tolist(), "R2 train": round(m["r2"], 3) if not np.isnan(m["r2"]) else None,
                            "scale (target units)": round(m["scale"], 3), "threshold (z)": round(m["threshold"], 2), "train bins": m["n_train"]})
    d["alert"] = d[[f"alert_{n}" for n in EDFA_MODELS]].any(axis=1)
    edfa_fit[asset], edfa_scored[asset] = fits, d
model_table = pd.DataFrame(model_table)
show(model_table)

# %% [markdown]
# ### D2. Method and why
#
# Each EDFA signal gets a small model fitted on that asset's training window. The models are ordinary least squares, or a training median where the context does not vary. Each residual is divided by a robust spread to give a z-score.
#
# * **Fit to the framing.** Each residual answers one physical question, such as "is the cooler drawing more current than this case temperature explains?". That is exactly a contextual anomaly.
# * **Data volume.** The models have 1–3 coefficients, fitted on about 1,000 five-minute bins per unit. Nothing here needs more data than we have.
# * **Interpretability.** An alert reads as, for example, "Itec 38 mA above what 31.5 °C explains". The coefficients are physical: about 25 mA of TEC current per °C of case temperature, the same on both units.
# * **Cost.** Scoring one bin takes a few multiply-adds, so it runs on any on-site server in microseconds.
#
# Rejected alternatives:
#
# * **Autoencoder or LSTM:** no labels, about 22 days per unit, and not interpretable.
# * **One-class SVM:** needs scaling choices and gives nothing an engineer can read.
# * **Raw control charts on each sensor:** fire on every warm night.
#
# **Training window: 18 Feb 12:00 to 21 Feb 23:55 (3.5 days).** It is the longest stretch that is:
#
# * after the 16–18 Feb site thermal event (C2);
# * free of outages;
# * before the first level shift Part C found in any EDFA signal (22 Feb).
#
# It is short, so the thresholds are conservative. It covers 3.5 daily temperature cycles, which is what the cooler model needs.
#
# ### D3–D4. Threshold and expected alert rate
#
# Each model's alert threshold is the larger of two numbers:
#
# * 4 robust standard deviations;
# * the 99.9th percentile of its 15-minute persistent score in training.
#
# An alert also needs **3 consecutive 5-minute bins** (15 minutes) above threshold. This suppresses single-poll glitches, which are already caught by the B3 rule layer.
#
# Each deviation has a physical floor, below which it is never treated as significant: 3 mA of TEC current, 0.8 mA of pump current (one quantisation step), 0.01 mW/mA of pump efficiency, and 0.25 °C of pump temperature. This stops a quantised sensor, whose training spread can be near 0, from alerting on a single rounding step.

# %%
def score_summary(d: pd.DataFrame, asset) -> dict:
    scored = d[~d["outage"] & d[[f"z_{n}" for n in EDFA_MODELS]].notna().any(axis=1)]
    out = {"asset": short(asset)}
    for name in EDFA_MODELS:
        tr = scored.loc[TRAIN[0]:TRAIN[1], f"alert_{name}"]
        rest = scored.drop(index=scored.loc[TRAIN[0]:TRAIN[1]].index)[f"alert_{name}"]
        eps = alert_episodes(d[f"alert_{name}"], MERGE_GAP_BINS)
        out[f"{name}: alert episodes (all data)"] = len(eps)
        out[f"{name}: % bins alerted outside training"] = round(100 * rest.mean(), 1)
        out[f"{name}: alert episodes in training"] = int(alert_episodes(tr, MERGE_GAP_BINS).shape[0])
    days = scored.index.normalize().nunique()
    out["scored days"] = days
    out["alert episodes per week (all models)"] = round(len(alert_episodes(d["alert"], MERGE_GAP_BINS)) / days * 7, 2)
    return out


alert_rate = pd.DataFrame([score_summary(d, a) for a, d in edfa_scored.items()]).set_index("asset").T
show(alert_rate)


def plot_scored_edfa(asset) -> None:
    d = edfa_scored[asset]
    panels = [("Pump1 Itec", "pred_cooler", "Itec (mA)"), ("Pump1 Current", "pred_pump_drive", "pump current (mA)"),
              ("pump_eff", "pred_pump_efficiency", "pump power / current (mW/mA)"), ("Pump1 Temperature", "pred_pump_temp", "pump temp (degC)")]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(13, 11), sharex=True)
    for ax, (col, pred, label) in zip(axes, panels):
        ax.plot(d.index, d[col], lw=0.8, color="black", label="observed")
        ax.plot(d.index, d[pred], lw=0.8, color="#ff7f0e", label="model expectation")
        ax.set_ylabel(label, fontsize=8)
        ax.ticklabel_format(axis="y", useOffset=False)
        lo, hi = d[col].quantile(0.001), d[col].quantile(0.999)
        pad = max(hi - lo, 0.02 * abs(d[col].median())) * 0.3
        ax.set_ylim(lo - pad, hi + pad)
        shade_outages(ax, d["outage"], label=False)
    for name, color in zip(EDFA_MODELS, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]):
        axes[-1].plot(d.index, persistent(d[f"z_{name}"]).clip(upper=30), lw=0.8, color=color, label=f"{name} (threshold {edfa_fit[asset][name]['threshold']:.1f})")
    axes[-1].axhline(MIN_THRESHOLD, color="grey", ls="--", lw=0.8)
    axes[-1].set_ylabel("15-min persistent |z| (clipped at 30)", fontsize=8)
    axes[-1].legend(fontsize=7, ncol=4, loc="upper left")
    for ax in axes:
        ax.axvspan(pd.Timestamp(TRAIN[0], tz="UTC"), pd.Timestamp(TRAIN[1], tz="UTC"), color="#2ca02c", alpha=0.08)
        for _, e in alert_episodes(d["alert"], MERGE_GAP_BINS).iterrows():
            ax.axvspan(e["start"], e["end"], color="red", alpha=0.12, lw=0)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"D3. Scored timeline, EDFA {short(asset)}: green = training window, red = alert episodes, grey = outages")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save_fig(fig, f"D3_scored_timeline_{short(asset)}")


for asset in EDFA_ASSETS:
    plot_scored_edfa(asset)
    keep = [c for c in edfa_scored[asset].columns if not c.endswith("__filled")]
    edfa_scored[asset][keep].to_csv(OUT / "scores" / f"edfa_{short(asset)}_scored_5min.csv")

# %% [markdown]
# ### D5. Evaluation without labels
#
# Three independent checks.
#
# 1. **Injected synthetic faults.** Faults are injected into clean stretches outside training, the frozen models re-score them, and detection rate and delay are measured. Five fault types, 12 random placements each, per unit:
#    * **F1 pump ageing:** pump current ramps up 6% over 72 h, POUT unchanged.
#    * **F2 cooler fault:** Itec +30 mA for 6 h, case temperature unchanged.
#    * **F3 hot afternoon (negative control):** case temperature +4 °C for 6 h, with Itec rising by the fitted 25 mA/°C. This is explained variation and should **not** alert.
#    * **F4 pump power loss:** pump power −15% for 12 h at unchanged current.
#    * **F5 single-bin spike:** pump current +10% for one 5-minute bin. By design this should **not** alert, because persistence suppresses it.
# 2. **Agreement with an independent method.** An Isolation Forest on the same 6 raw signals, scaled per asset, trained on the same window and with no physics.
# 3. **Expert-rule validation.** Do the alerts cover the integrity breaks found in C6? Those are values leaving the sensor's own quantisation grid, found by a rule that knows nothing about the models.

# %%
rng = np.random.default_rng(SEED)
FAULTS = {
    "F1 pump ageing (+6% over 72 h)": ("Pump1 Current", "ramp", 0.06, 72 * 12),
    "F2 cooler fault (Itec +30 mA, 6 h)": ("Pump1 Itec", "add", 30.0, 6 * 12),
    "F3 hot afternoon, explained (negative control)": ("EDFA Case Temperature", "thermal", 4.0, 6 * 12),
    "F4 pump power -15% (12 h)": ("Pump1 Power", "scale", -0.15, 12 * 12),
    "F5 single-bin spike (+10%)": ("Pump1 Current", "scale", 0.10, 1),
}
IF_FEATURES = ["Input Power (PIN)", "Pump1 Current", "Pump1 Power", "Pump1 Itec", "EDFA Case Temperature", "Pump1 Temperature"]


def rescore(d: pd.DataFrame, asset) -> pd.Series:
    d = prepare_edfa(d)
    alert = pd.Series(False, index=d.index)
    for name, m in edfa_fit[asset].items():
        z = (d[m["target"]] - predict(m, d)) / m["scale"]
        alert |= persistent(z) > m["threshold"]
    return alert


def inject(d: pd.DataFrame, start: int, spec, asset) -> pd.DataFrame:
    col, kind, size, n = spec
    d = d.copy()
    idx = d.index[start:start + n]
    if kind == "ramp":
        d.loc[idx, col] *= 1 + size * np.linspace(0, 1, len(idx))
    elif kind == "add":
        d.loc[idx, col] += size
    elif kind == "scale":
        d.loc[idx, col] *= 1 + size
    elif kind == "thermal":
        d.loc[idx, col] += size
        d.loc[idx, "Pump1 Itec"] += size * edfa_fit[asset]["cooler"]["coef"][1]
    return d


# Isolation Forest cross-check (independent of the physics models)
iforest = {}
for asset in EDFA_ASSETS:
    d = edfa_scored[asset]
    tr = d.loc[TRAIN[0]:TRAIN[1]].dropna(subset=IF_FEATURES)
    med, iqr = tr[IF_FEATURES].median(), (tr[IF_FEATURES].quantile(0.75) - tr[IF_FEATURES].quantile(0.25)).replace(0, 1.0)
    clf = IsolationForest(n_estimators=200, random_state=SEED).fit((tr[IF_FEATURES] - med) / iqr)
    ok = d[IF_FEATURES].notna().all(axis=1)
    s = pd.Series(np.nan, index=d.index)
    s[ok] = -clf.score_samples((d.loc[ok, IF_FEATURES] - med) / iqr)
    thr = float(np.quantile(-clf.score_samples((tr[IF_FEATURES] - med) / iqr), 0.999))
    d["if_score"] = s
    d["if_alert"] = s.rolling(PERSIST_BINS, min_periods=PERSIST_BINS).min() > thr
    iforest[asset] = (clf, med, iqr, thr)


def if_rescore(d: pd.DataFrame, asset) -> pd.Series:
    clf, med, iqr, thr = iforest[asset]
    ok = d[IF_FEATURES].notna().all(axis=1)
    s = pd.Series(np.nan, index=d.index)
    s[ok] = -clf.score_samples((d.loc[ok, IF_FEATURES] - med) / iqr)
    return s.rolling(PERSIST_BINS, min_periods=PERSIST_BINS).min() > thr


FAULT_TARGET = {  # which model a fault should trip (None = negative control, nothing should trip)
    "F1 pump ageing (+6% over 72 h)": "pump_drive", "F2 cooler fault (Itec +30 mA, 6 h)": "cooler",
    "F3 hot afternoon, explained (negative control)": None, "F4 pump power -15% (12 h)": "pump_efficiency",
    "F5 single-bin spike (+10%)": "pump_drive",
}
eval_rows = []
for asset in EDFA_ASSETS:
    d = edfa_scored[asset]
    base = edfa5[asset]
    base_alert = rescore(base, asset)
    base_if = if_rescore(base, asset)
    outside_train = d.index > pd.Timestamp(TRAIN[1], tz="UTC")
    for fname, spec in FAULTS.items():
        col, _, _, n = spec
        target = FAULT_TARGET[fname]
        span = n + PERSIST_BINS + MERGE_GAP_BINS
        # a placement is valid if the window is outside training, has no outage, has the injected signal
        # in >= 80% of bins, and the model that should fire is quiet there without the injection
        quiet = ~d[f"alert_{target}"] if target else ~d["alert"]
        valid = (outside_train & ~d["outage"] & quiet & d[col].notna()).astype(float)
        share = valid.rolling(span, min_periods=span).mean().shift(-(span - 1))
        starts = np.flatnonzero((share >= 0.8).to_numpy())
        picks = rng.choice(starts, size=min(12, len(starts)), replace=False) if len(starts) else []
        hits, if_hits, delays = 0, 0, []
        for st in picks:
            dd = inject(base, int(st), spec, asset)
            window = dd.index[st:st + span]
            new_alert = (rescore(dd, asset) & ~base_alert).reindex(window)
            new_if = (if_rescore(dd, asset) & ~base_if).reindex(window)
            if new_alert.any():
                hits += 1
                delays.append((new_alert.idxmax() - window[0]).total_seconds() / 3600)
            if_hits += int(new_if.any())
        expect = "no alert" if target is None or fname.startswith("F5") else "alert"
        eval_rows.append({"asset": short(asset), "fault": fname, "expected": expect, "placements": len(picks),
                          "physics detector: alerted": f"{hits}/{len(picks)}",
                          "median delay (h)": round(float(np.median(delays)), 1) if delays else None,
                          "Isolation Forest: alerted": f"{if_hits}/{len(picks)}"})
evaluation = pd.DataFrame(eval_rows)
show(evaluation)
evaluation.to_csv(OUT / "scores" / "edfa_injection_evaluation.csv", index=False)

# agreement between the two methods on the real data (outside training)
agree = []
for asset in EDFA_ASSETS:
    d = edfa_scored[asset]
    d = d[(d.index > pd.Timestamp(TRAIN[1], tz="UTC")) & ~d["outage"] & d[IF_FEATURES].notna().all(axis=1)]
    a, b = d["alert"], d["if_alert"]
    agree.append({"asset": short(asset), "bins": len(d), "physics alerts": int(a.sum()), "IF alerts": int(b.sum()),
                  "both": int((a & b).sum()), "Jaccard": round((a & b).sum() / max((a | b).sum(), 1), 2),
                  "share of physics alerts IF also flags": round((a & b).sum() / max(a.sum(), 1), 2)})
show(pd.DataFrame(agree))

# expert-rule validation: do alerts cover the quantisation breaks of C6 (outside training)?
rule_rows = []
for _, q in qbreaks.iterrows():
    asset = next(a for a in EDFA_ASSETS if short(a) == q["asset"])
    d = edfa_scored[asset].loc[q["first"]:q["last"]]
    d = d[~d["outage"]]
    rule_rows.append({"asset": q["asset"], "sensor": q["sensor"], "from": q["first"], "to": q["last"],
                      "shift": round(q["median off-grid value"] - q["typical value before"], 3),
                      "share of bins alerted by physics detector": round(float(d["alert"].mean()), 2) if len(d) else None})
show(pd.DataFrame(rule_rows))

# %% [markdown]
# ### D6. What was flagged, and is it the device or the monitoring?
#
# The table lists every alert episode longer than one hour on the real data, with the model that fired and what the sensors were doing. Under the table, each one is characterised as a device problem, a monitoring artefact or a false positive.

# %%
episode_rows = []
for asset in EDFA_ASSETS:
    d = edfa_scored[asset]
    for name, m in edfa_fit[asset].items():
        for _, e in alert_episodes(d[f"alert_{name}"], MERGE_GAP_BINS).iterrows():
            seg = d.loc[e["start"]:e["end"]]
            hours = e["bins"] * 5 / 60
            if hours < 1:
                continue
            episode_rows.append({"asset": short(asset), "model": name, "start": e["start"], "end": e["end"], "hours": round(hours, 1),
                                 "observed (median)": round(seg[m["target"]].median(), 3), "expected (median)": round(seg[f"pred_{name}"].median(), 3),
                                 "median z": round(seg[f"z_{name}"].median(), 1), "case temp": round(seg["EDFA Case Temperature"].median(), 1)})
flagged = pd.DataFrame(episode_rows).sort_values(["asset", "start"])
show(flagged)
flagged.to_csv(OUT / "scores" / "edfa_alert_episodes.csv", index=False)

trend = []
for asset in EDFA_ASSETS:
    daily = prepare_edfa(edfa5[asset])[["Pump1 Current", "pump_eff"]].resample("1D").median().dropna()
    t = (daily.index - daily.index[0]).days.to_numpy()
    for col in ["Pump1 Current", "pump_eff"]:
        slope, intercept = np.polyfit(t, daily[col], 1)
        resid = daily[col] - (slope * t + intercept)
        se = np.sqrt(resid.var(ddof=2) / ((t - t.mean()) ** 2).sum())
        trend.append({"asset": short(asset), "signal": col, "days": len(daily), "slope per week": round(7 * slope, 4),
                      "95% CI per week": f"[{7 * (slope - 2 * se):.4f}, {7 * (slope + 2 * se):.4f}]",
                      "largest day-to-day step": round(daily[col].diff().abs().max(), 4), "step on": daily[col].diff().abs().idxmax().date()})
show(pd.DataFrame(trend))


# %%
# Expert rule for the one failure mode the models do not cover: an upstream fibre event shows as a PIN drop.
pin_events = []
for asset in EDFA_ASSETS:
    pin = edfa5[asset]["Input Power (PIN)"]
    drop = pin.rolling("24h", min_periods=12).median() - pin
    flag = drop > 3.0  # more than 3 dB below the trailing 24 h median
    pin_events.append({"asset": short(asset), "PIN range (dBm)": f"{pin.min():.1f} to {pin.max():.1f}",
                       "largest drop below 24 h median (dB)": round(float(drop.max()), 2),
                       "PIN-drop events (> 3 dB)": len(alert_episodes(flag, MERGE_GAP_BINS))})
show(pd.DataFrame(pin_events))

# %% [markdown]
# **Characterisation of the flagged periods.**
#
# Every alert longer than one hour traces back to one of three root causes. There were none in training and none on the cooler model, including the 16–18 Feb hot spell. The fourth row shows a shift the detector correctly ignored.
#
# | Asset | Period (UTC) | Model(s) | What the sensors did | Verdict |
# |---|---|---|---|---|
# | 7a1d09b1 | 22 Feb 00:40 to 1 Mar, and 24–31 Mar | pump_efficiency, z ≈ −6 then −11 | Pump power 26.0 → 22.1 → 18.6 mW at a fixed 66.8 mA; POUT, PIN and case temperature flat | **Monitoring artefact, probably the pump-power monitor** (insight 5): physically inconsistent with a real pump loss. Verify locally. |
# | 7a1d09b1 | 28 Feb 00:35–06:55 and 1 Mar 14:45–19:15 | pump_temp, z ≈ +39 | Pump temperature fixed at 34.681 °C; cooler current unchanged | **Monitoring artefact** (a stuck value): a real +9.7 °C excursion would saturate the TEC. |
# | 010147e2 | 25 Mar 00:10 to 31 Mar, on and off | pump_drive (z ≈ −5.7) and pump_efficiency (z ≈ +6.3), one root cause | Pump current 167.2 → 162.67 mA (−2.7%) at constant POUT 15.0 dBm and PIN | **Monitoring artefact, most likely**: starts at 00:00 UTC with off-grid values, and the current drops while the output holds. A real pump would need *more* current as it ages, not less. Low severity either way. |
# | 010147e2 | (none) | (pump_temp) | Pump temperature 25.100 → 25.093 °C, off-grid from 22 Feb | **Correctly not alerted:** 0.007 °C is below the 0.25 °C physical floor. |
#
# There are **no confirmed device faults** in the EDFA data. The detector is behaving as designed: every incident it raised has a visible, specific cause in the raw sensors. Whether those causes live in the device or the monitoring path is a separate judgement, made above from physics, and labels would be needed to confirm it.
#
# **Degradation trend.** The trend table above shows statistically non-zero slopes, for example −0.63 mA/week of pump current on 010147e2. Every one of them is produced by a single step (largest day-to-day jump on 26 Mar and 22 Feb), not by a ramp. With 22 days of data in two blocks there is no gradual ageing to extrapolate, so **no lead-time estimate is given**. Fitting a line through a step and projecting it would be false precision.

# %% [markdown]
# ### D7. SFP: reporting-window consistency, and bias against temperature
#
# Two detectors run on the 15-minute SFP table.
#
# * **Window consistency.** For each port and metric, compute the daily difference between the `Current` window and the median of the other three windows (a median of three tolerates one corrupted window). Its 2–5 Jan median, before any shift, is the baseline. An alert is a deviation from that baseline of more than 4 robust standard deviations, floored at 0.5 mA of bias or 0.02 mW of Tx, for at least 2 consecutive days. A change here is confined to one reporting path: a real laser change would move every window.
# * **Bias given temperature.** `Current` bias ~ `Current` laser temperature, fitted on 2–5 Jan with a floor of 0.5 mA. It separates the expected thermal rise in bias (about 0.3 mA/°C) from unexplained shifts.

# %%
sfp_alerts = []
for p in SFP_PORTS:
    for metric, floor in (("Laser Bias", 0.5), ("Transmit Power", 0.02), ("Received Power", 0.2)):
        cur = sfp_w[f"{p} Current: {metric}"].resample("1D").median()
        ref = pd.concat([sfp_w[f"{p} {w_}: {metric}"].resample("1D").median() for w_ in ("Previous 15min", "Current Day", "Previous Day")], axis=1).median(axis=1)
        gap = (cur - ref).dropna()
        base = gap.loc["2026-01-02":"2026-01-05"]
        z = (gap - base.median()) / robust_scale(base.to_numpy(), floor)
        flag = z.abs().rolling(2, min_periods=2).min() > MIN_THRESHOLD
        for _, e in alert_episodes(flag, merge_gap=2).iterrows():
            sfp_alerts.append({"port": p, "metric": metric, "detector": "window consistency", "first alert day": e["start"].date(),
                               "last day": e["end"].date(), "days": e["bins"], "Current minus day windows": round(gap.loc[e["start"]:e["end"]].median(), 3)})
    m = fit_linear(sfp15.loc["2026-01-02":"2026-01-05"], f"{p} Current: Laser Bias", [f"{p} Current: Laser Temperature"], 0.5)
    sfp15[f"z_bias_{p}"] = (sfp15[f"{p} Current: Laser Bias"] - predict(m, sfp15)) / m["scale"]
    sfp15[f"alert_bias_{p}"] = persistent(sfp15[f"z_bias_{p}"], 4) > MIN_THRESHOLD
    print(f"{p}: bias = {m['coef'][0]:.2f} + {m['coef'][1]:.3f} x temperature (mA, degC); scale {m['scale']:.2f} mA")
    for _, e in alert_episodes(sfp15[f"alert_bias_{p}"], merge_gap=96).iterrows():
        if e["bins"] >= 8:
            sfp_alerts.append({"port": p, "metric": "Laser Bias", "detector": "bias given temperature", "first alert day": e["start"].date(),
                               "last day": e["end"].date(), "days": round(e["bins"] / 96, 1),
                               "Current minus day windows": round(sfp15.loc[e["start"]:e["end"], f"z_bias_{p}"].median() * m["scale"], 2)})
sfp_alerts = pd.DataFrame(sfp_alerts)
show(sfp_alerts)
sfp_alerts.to_csv(OUT / "scores" / "sfp_alerts.csv", index=False)
sfp15[[c for c in sfp15.columns if c.startswith(("z_", "alert_")) or c in SFP_FEATURES]].to_csv(OUT / "scores" / "sfp_scored_15min.csv")

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for ax, p in zip(axes, SFP_PORTS):
    ax.plot(sfp15.index, sfp15[f"{p} Current: Laser Bias"], lw=0.6, color="black", label="Current bias (mA)")
    ax.plot(sfp15.index, sfp15[f"{p} Current Day: Laser Bias"], lw=0.6, color="#2ca02c", label="Current Day bias (mA)")
    ax2 = ax.twinx()
    ax2.plot(sfp15.index, sfp15[f"{p} Current: Laser Temperature"], lw=0.5, color="#ff7f0e", alpha=0.6, label="laser temp (degC)")
    ax2.set_ylabel("laser temp (degC)", fontsize=8)
    for _, e in alert_episodes(sfp15[f"alert_bias_{p}"], merge_gap=96).iterrows():
        ax.axvspan(e["start"], e["end"], color="red", alpha=0.12, lw=0)
    ax.set_ylabel("laser bias (mA)", fontsize=8)
    ax.set_title(f"D7. SFP {p}: bias vs the day windows and temperature; red = bias-given-temperature alerts", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save_fig(fig, "D7_sfp_scored")

# %% [markdown]
# ### D8. Summary: what was found, and how much to trust it
#
# **What was found.**
#
# * **EDFAs:** the physics-residual detector raised alerts on three root causes across 23 scored days per unit. That is about one alert episode per unit per week, and none during training.
#   * Each root cause is a level shift in pump telemetry with no physical corroboration, and all three start at exactly 00:00 UTC with values off the sensor's grid.
#   * The cooler model raised nothing, correctly treating the 44-hour site heat event as explained by case temperature.
# * **SFP:** the window-consistency and bias-given-temperature detectors agree on three single-window steps:
#   * 1-4-3 Tx from 6 Jan;
#   * 1-4-4 bias from 18 Feb;
#   * 1-4-3 bias from 27 Feb.
#
#   The day windows do not move with them, so these are reporting-path problems, not laser ageing.
# * **No device fault is confirmed anywhere in the data.** The most actionable maintenance finding is thermal, not a fault: the SFP lasers' warming trend ahead of summer (insight 8).
#
# **How much to trust it.**
#
# * **High** for the claim that the detector catches persistent relational changes of a few percent:
#   * 12/12 injected pump-ageing ramps, alerting at about 2% drift on 010147e2;
#   * 23/24 cooler faults, within about 1 h;
#   * 12/12 pump-power drops on 010147e2. There was no clean placement on 7a1d09b1, whose efficiency model was already alerting.
#
#   It also stays quiet where it should: 0/24 alerts on explained hot afternoons and 0/24 on single-bin spikes. The context-free Isolation Forest, given the same data, alerted on none of the injected faults and on 8/12 explained hot afternoons on 010147e2. It also cannot see a change in a signal that was constant in training (7a1d09b1's pump power). That contrast is the evidence that the contextual framing matters.
# * **Low** for anything the evaluation cannot show:
#   * whether real faults look like the injected ones;
#   * the false-alarm rate over a full year;
#   * whether a flagged shift is the device or the monitoring path. The verdicts above rest on physics and timing, not labels.
#
# **Main limits.**
#
# * Training is 3.5 days, so the cooler line is extrapolated to case temperatures it never saw in training (it held from 28 to 42 °C).
# * Sensor quantisation limits sensitivity on 7a1d09b1. One 0.8 mA pump-current step is 1.2%, so a ramp is caught only after about 5%.

# %%
print(f"end-to-end runtime (everything above): {time.time() - T_START:.0f} s")

# %% [markdown]
# ## Part E: Decision log
#
# The log was written as the work progressed and tidied at the end. It is also saved as `decision_log.md`.
#
# **1. Alternatives rejected**
#
# | Decision | Options considered | Chosen | The one reason that decided it |
# |---|---|---|---|
# | Aligning timestamps that differ by seconds | (a) cluster timestamps closer than 30 s into one poll; (b) exact-second join; (c) floor to the 1-minute slot | (c) | Sensor groups are polled at :13, :21–33, :51 and :53 each minute, so (a) chains them into one endless poll and (b) never joins anything. Collisions within a slot (mostly the twice-a-minute uptime counter) are counted, not hidden. |
# | EDFA sampling interval | 1 min, 5 min, 15 min, 1 h | 5 min | Each optical sensor is missing from 20–30% of minutes, so 1 min is mostly gaps. 15 min or more would smear a minute-scale fibre event; 5 min gives 3–4 readings per median. |
# | SFP reporting window | Current; Previous 15min; Current Day / Previous Day; a consensus of all four | Current, with the day windows as a cross-check | It is the only window with no impossible values that is aligned to poll time. Previous 15min's clock is 49–94 min fast and Current Day holds 24% impossible voltages, so neither can anchor a 15-min model. A consensus would hide the single-window steps that turned out to be the main SFP finding. |
# | Outage handling | drop the rows; impute through them; mask with a guard | Mask optical values while ICMP loss = 100%, plus a 5-min guard, and never fill | Imputation invents data for an unreachable device, and dropping rows loses the timeline needed to tell outages from polling misses. |
# | Detection method | Isolation Forest; autoencoder or LSTM; raw control charts; physics-informed residuals | Residuals of 1–3-coefficient models, with Isolation Forest only as a cross-check | Evaluated, not assumed: Isolation Forest caught none of the injected faults, alerted on 8/12 explained hot afternoons, and is blind to signals that were constant in training. |
# | Threshold | fixed 3σ; a quantile of training scores; an alert budget | max(4σ, 99.9th percentile of the training persistent score), physical floors per signal, 15 min of persistence, 6 h incident merge | Quantised sensors have a near-zero spread, so without floors a single rounding step would alert. The floors encode "the smallest change an engineer cares about". |
# | Unmapped SFP ids (41% of SFP rows) | drop; keep as unknown; infer and model | Keep, label, infer the role from values, do not model | The value signatures (−48 V-like feeds, a temperature correlating with the lasers at r ≈ 0.8–0.9) support an inference, but an unverified identity should not drive alerts. |
#
# **2. Assumptions**
#
# | Assumption | Status |
# |---|---|
# | Timestamps are UTC, as the brief says. Consistent with the SFP day windows closing at 18:30 UTC, which is midnight IST. | verified |
# | Polling is nominally every 60 s (median inter-arrival 60 s). | verified |
# | Both EDFAs are single-pump (Pump2 Itec is 0 at every reading). | verified |
# | ICMP loss = 100% means unreachable, and optical values then are stale. 33% or 67% loss is reachable. | plausible |
# | Unmapped SFP ids are shelf-level sensors (a chassis temperature, two −48 V feeds, their thresholds, status flags). | plausible |
# | SFP Transmit Power is in mW (0.7–1.1 mW fits STM-4 optics at −1.5 to +0.4 dBm). | plausible |
# | 18–21 Feb is normal for training: no outage, after the thermal event, before every level shift seen in Part C. | plausible |
# | The 16–18 Feb heat was site-level (both units, r = 0.995, pump temperature held). | plausible |
# | Steps that start at 00:00 UTC with off-grid values come from the monitoring path or are injected, not from the device. | plausible |
# | The EDFA room's inverted daily cycle comes from scheduled HVAC. | guess |
# | 70 °C is the SFP case limit (part numbers unknown). | guess |
#
# **3. What surprised me**
#
# * **Coverage.** Only 22 of 88 days carry EDFA optics, with two multi-week holes across all assets. I rebuilt the plan around two short blocks and a 3.5-day training window.
# * **Unmapped ids.** The 60 unmapped SFP ids are not the two missing STM4 ports another asset's mapping suggested. Their values look like shelf power and temperature, so they were kept and inferred rather than dropped.
# * **Window clock and corruption.** The SFP windows disagree, the 15-minute window's clock runs over an hour fast, and `Current Day` is corrupted. This turned "pick a window" into a consistency check that became a detector.
# * **Step timing.** Several step changes start at exactly 00:00 UTC with values off the sensor's own resolution. That contradicts "values untouched" in the brief, so each one was judged on physics before being called a fault.
# * **Cooler linearity.** Itec follows case temperature at r = 0.997. That is cleaner than a real thermo-electric cooler usually is, so I used it but noted it.
# * **Naming and files.** `Output Power Threshold (PIN)` is really the POUT threshold. The delivered files are `*_anomalies_f.csv`, not `EDFA_data.csv` / `SFP_data.csv`. One unmapped id arrives 5 times per poll (130k exact duplicates). SFP ICMP loss contains values from −309 to +366.
#
# **4. Where the time went**
#
# *[Candidate: replace with your actual hours.]*
#
# | Part | Hours |
# |---|---|
# | A. wrangling and inventory | _ |
# | B. preprocessing | _ |
# | C. insights | _ |
# | D. detection and evaluation | _ |
# | E. write-up | _ |
#
# Part B and the SFP window question took the most time, because the windows had to be understood before they could be chosen.
#
# **5. With ten times the data and two more weeks, in order of expected impact**
#
# 1. **A full year per unit, without collector gaps.** This would let the cooler model see the whole seasonal range, show real ageing slopes with honest lead times, and measure the false-alarm rate per season.
# 2. **Vendor limits.** Pump end-of-life current, TEC maximum current, and the SFP part numbers with their DDM alarm thresholds. These turn z-scores into a "margin to failure" that an engineer can plan around.
# 3. **Maintenance tickets as weak labels.** Use them to calibrate the floors and thresholds, and to check whether flagged shifts were devices or monitoring.
# 4. **Pooling across many units of the same class.** A hierarchical version of the same residual models, so a new unit inherits a prior instead of needing 3.5 clean days.
# 5. **A data-quality monitor upstream of the detector.** Window consistency, quantisation breaks and collector gaps, reported separately from device health.
# 6. Only then, a learned multivariate model, if the residuals show structure that the physics models miss.
#
# **6. Deployment sketch**
#
# * **How it runs.** A Python job, using pandas and numpy only, runs every 5 minutes on the site's NMS server. It reads the last 24 hours from the poller's database, applies the validity rules, resamples, and scores against per-asset coefficients stored as a small JSON file. It then writes incidents to the NMS alarm table, labelled "device" or "monitoring".
# * **Retraining.** Coefficients are refitted monthly on the last 14 clean days (no outage, no open incident). A refit is also triggered immediately by a hardware swap, a firmware change or a set-point change, which the config and identity sensors reveal.
# * **False alarms.** An engineer closes each incident with a reason code: real, explained or false. False windows are excluded from the next refit, and counts per model are reviewed monthly to adjust the floors.
# * **Compute.** 1 vCPU, 1 GB of RAM and about 100 MB of disk per 100 assets. No GPU, no internet. The model is the smallest part of this system; the collector, the data-quality checks and the feedback loop matter more.

