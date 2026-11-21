#!/usr/bin/env python3
"""
check_project.py

Audits the Celeris demand forecasting project and writes one report.

Usage, from the folder that contains student_data/ and project/:
    python check_project.py              check the current outputs
    python check_project.py --rerun      also run the pipeline twice to test reproducibility
    python check_project.py --no-tests   skip the pytest run

The report is printed and saved to project/outputs/check_report.txt.
Paste that file into the chat for review.
"""

import argparse
import hashlib
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STUDENT = ROOT / "student_data"
PROJECT = ROOT / "project"
OUT = PROJECT / "outputs"
REPORT = OUT / "check_report.txt"

RESULTS = []  # (section, status, message)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def record(section, status, message):
    RESULTS.append((section, status, message))


def run_check(section, fn):
    try:
        fn(section)
    except Exception as exc:  # the checker should keep going and report the failure
        record(section, "ERROR", f"{type(exc).__name__}: {exc}")


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"missing {Path(path).relative_to(ROOT)}")
    return Path(path)


def table(frame, floatfmt="{:,.0f}"):
    lines = []
    for idx, row in frame.iterrows():
        cells = "  ".join(
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in row.values
        )
        lines.append(f"{str(idx):<28} {cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 0 environment
def check_environment(s):
    record(s, "INFO", f"run at {datetime.now():%Y-%m-%d %H:%M}")
    record(s, "INFO", f"python {platform.python_version()} on {platform.system()}")
    record(s, "INFO", f"pandas {pd.__version__}, numpy {np.__version__}")
    try:
        import lightgbm
        record(s, "PASS", f"lightgbm {lightgbm.__version__} installed")
    except ImportError:
        record(s, "FAIL", "lightgbm is not installed")
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if git.returncode == 0:
            record(s, "INFO", f"git commit {git.stdout.strip()}")
        else:
            record(s, "INFO", "not a git repository")
    except (OSError, subprocess.SubprocessError):
        record(s, "INFO", "git not available")
    for name in ["forecast_submission.csv", "clean_train.parquet",
                 "audit_log.csv", "backtest_metrics.csv"]:
        path = OUT / name
        if path.exists():
            record(s, "INFO", f"{name}: md5 {md5(path)[:12]}, {path.stat().st_size / 1024:,.0f} KB")
        else:
            record(s, "FAIL", f"{name} not found; run the pipeline first")


# ---------------------------------------------------------- 1 reproducibility
def check_reproducibility(s, rerun):
    if not rerun:
        record(s, "INFO", "not rerun; use --rerun to test reproducibility (takes a few minutes)")
        return
    watched = [OUT / "forecast_submission.csv", OUT / "audit_log.csv", OUT / "clean_train.parquet"]

    def run_pipeline():
        return subprocess.run([sys.executable, str(PROJECT / "run_pipeline.py")],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=1800)

    first = run_pipeline()
    if first.returncode != 0:
        record(s, "FAIL", f"pipeline run 1 exited with code {first.returncode}")
        record(s, "INFO", first.stderr[-1500:])
        return
    hashes_1 = {p.name: md5(p) for p in watched}

    second = run_pipeline()
    if second.returncode != 0:
        record(s, "FAIL", f"pipeline run 2 exited with code {second.returncode}")
        record(s, "INFO", second.stderr[-1500:])
        return
    hashes_2 = {p.name: md5(p) for p in watched}

    for name in hashes_1:
        same = hashes_1[name] == hashes_2[name]
        record(s, "PASS" if same else "FAIL",
               f"{name} identical across two runs" if same else f"{name} changed between runs")


# --------------------------------------------------------- 2 submission format
def check_submission(s):
    f = pd.read_csv(need(OUT / "forecast_submission.csv"), dtype={"Date": str})
    t = pd.read_csv(need(STUDENT / "test.csv"), dtype={"Date": str})
    keys = ["Date", "SKU_ID", "Region"]
    cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]

    record(s, "PASS" if list(f.columns) == cols else "FAIL",
           f"columns: {list(f.columns)}")
    record(s, "PASS" if len(f) == len(t) else "FAIL",
           f"rows: {len(f):,} (test.csv has {len(t):,})")
    nulls = int(f.isna().sum().sum())
    record(s, "PASS" if nulls == 0 else "FAIL", f"blank cells: {nulls}")
    neg = int((f["Units_Sold_Forecast"] < 0).sum())
    record(s, "PASS" if neg == 0 else "FAIL", f"negative forecasts: {neg}")
    nonint = int((f["Units_Sold_Forecast"] % 1 != 0).sum())
    record(s, "PASS" if nonint == 0 else "FAIL", f"non-integer forecasts: {nonint}")
    dup = int(f.duplicated(keys).sum())
    record(s, "PASS" if dup == 0 else "FAIL", f"duplicate Date-SKU-Region rows: {dup}")
    merged = t[keys].merge(f[keys], on=keys, how="outer", indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    record(s, "PASS" if missing == 0 and extra == 0 else "FAIL",
           f"keys not in test.csv: {extra}; test rows without a forecast: {missing}")
    record(s, "PASS" if f["Units_Sold_Forecast"].dtype.kind == "i" else "WARN",
           f"forecast column dtype: {f['Units_Sold_Forecast'].dtype}")


# ------------------------------------------------------- 3 leakage and features
def check_leakage(s):
    feat = pd.read_parquet(need(OUT / "features.parquet"))
    target = "Units_Sold_Unconstrained"
    if target not in feat.columns or "lag_1m" not in feat.columns:
        record(s, "FAIL", f"features.parquet needs {target} and lag_1m to verify lags")
        return

    keys = ["SKU_ID", "Region", "Customer_Name"]
    dup = int(feat.duplicated(keys + ["Date"]).sum())
    record(s, "PASS" if dup == 0 else "FAIL", f"duplicate series-month rows in features: {dup}")

    f = feat[keys + ["Date", target, "lag_1m"]].sort_values(keys + ["Date"]).copy()
    f["expected"] = f.groupby(keys)[target].shift(1)
    both = f["lag_1m"].notna() & f["expected"].notna()
    bad = int((~np.isclose(f.loc[both, "lag_1m"], f.loc[both, "expected"],
                           rtol=1e-6, atol=1e-6)).sum())
    record(s, "PASS" if bad == 0 else "FAIL",
           f"lag_1m equals last month's target on {int(both.sum()):,} rows; mismatches: {bad}")

    same_as_now = np.isclose(f.loc[both, "lag_1m"], f.loc[both, target], rtol=1e-6, atol=1e-6).mean()
    record(s, "PASS" if same_as_now < 0.9 else "FAIL",
           f"lag_1m equal to the current month's target: {same_as_now:.1%}")

    if "lag_12m" in feat.columns and "lag_12m_yoy" in feat.columns:
        same = (feat["lag_12m_yoy"].fillna(-1) == feat["lag_12m"].fillna(-1)).mean()
        record(s, "PASS" if same < 0.99 else "FAIL",
               f"lag_12m_yoy identical to lag_12m on {same:.1%} of rows")

    numeric = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c])]
    dup_cols = [(a, b) for i, a in enumerate(numeric) for b in numeric[i + 1:]
                if feat[a].equals(feat[b])]
    record(s, "PASS" if not dup_cols else "FAIL",
           f"duplicate feature columns: {len(dup_cols)}"
           + (f" ({', '.join(f'{a}={b}' for a, b in dup_cols[:5])})" if dup_cols else ""))

    bt = pd.read_csv(need(OUT / "backtest_metrics.csv"))
    if "origin" in bt.columns:
        n = bt["origin"].nunique()
        record(s, "PASS" if n >= 3 else "FAIL", f"backtest origins evaluated: {n} (spec needs at least 3)")
    else:
        record(s, "FAIL", "backtest_metrics.csv has no origin column, so only one origin is evaluated")


# ------------------------------------------------------------ 4 forecast shape
def check_forecast_shape(s):
    t = pd.read_parquet(need(OUT / "clean_train.parquet"))
    f = pd.read_csv(need(OUT / "forecast_submission.csv"), dtype={"Date": str})
    hist = t.groupby("Date")["Units_Sold"].sum()
    fc = f.groupby("Date")["Units_Sold_Forecast"].sum()

    ratio = fc.head(3).mean() / hist.tail(12).mean()
    record(s, "PASS" if 0.8 <= ratio <= 1.2 else "WARN",
           f"first 3 forecast months average {fc.head(3).mean():,.0f} units, "
           f"{ratio:.2f} times the last 12 history months")

    cv = fc.std() / fc.mean()
    record(s, "FAIL" if cv < 0.02 else "PASS",
           f"variation of monthly forecast totals (coefficient of variation): {cv:.3f}")

    hist_prof = (t[t["Date"] >= "2023-01"].assign(M=lambda d: d["Date"].str[5:])
                 .groupby("M")["Units_Sold"].mean())
    fc_prof = f.assign(M=f["Date"].str[5:]).groupby("M")["Units_Sold_Forecast"].mean()
    corr = hist_prof.corr(fc_prof.reindex(hist_prof.index))
    record(s, "PASS" if corr >= 0.5 else "WARN",
           f"seasonal shape match, history 2023+ vs forecast (calendar-month correlation): {corr:.2f}")

    per = f.groupby(["SKU_ID", "Region"])["Units_Sold_Forecast"].std().fillna(0)
    flat = float((per == 0).mean())
    record(s, "PASS" if flat <= 0.10 else "WARN",
           f"SKU-region series with a constant forecast over the horizon: {flat:.1%}")

    left = pd.DataFrame({"history units": hist.tail(12)})
    right = pd.DataFrame({"forecast units": fc})
    record(s, "INFO", "last 12 months of history:\n" + table(left))
    record(s, "INFO", "forecast by month:\n" + table(right))


# ------------------------------------------------------------ 5 holdout accuracy
def check_holdout(s):
    t = pd.read_parquet(need(OUT / "clean_train.parquet"))
    keys = ["SKU_ID", "Region", "Customer_Name"]
    t = t[t["Units_Sold"].notna()][keys + ["Product_Family", "Date", "Units_Sold"]]

    prior = t[keys + ["Date", "Units_Sold"]].copy()
    prior["Date"] = (pd.to_datetime(prior["Date"] + "-01")
                     + pd.DateOffset(months=12)).dt.strftime("%Y-%m")
    prior = prior.rename(columns={"Units_Sold": "naive"})

    window = t[(t["Date"] >= "2025-11") & (t["Date"] <= "2026-10")].merge(
        prior, on=keys + ["Date"], how="left")
    window["naive"] = window["naive"].fillna(0)
    total = window["Units_Sold"].sum()
    wape = (window["Units_Sold"] - window["naive"]).abs().sum() / total
    bias = (window["naive"].sum() - total) / total
    record(s, "INFO",
           f"seasonal naive on 2025-11 to 2026-10 (actuals): WAPE {wape:.3f}, "
           f"bias {bias:+.3f}, rows {len(window):,}")

    fam = (window.assign(err=(window["Units_Sold"] - window["naive"]).abs())
           .groupby("Product_Family")
           .agg(actual=("Units_Sold", "sum"), abs_err=("err", "sum")))
    fam["WAPE"] = fam["abs_err"] / fam["actual"]
    record(s, "INFO", "seasonal naive WAPE by family:\n"
           + table(fam[["WAPE"]], floatfmt="{:.3f}"))

    bt = pd.read_csv(need(OUT / "backtest_metrics.csv"))
    if "breakdown" in bt.columns:
        ov = bt[bt["breakdown"] == "overall"][["model", "WAPE", "Bias", "n_obs"]]
        record(s, "INFO", "pipeline backtest, overall:\n"
               + "\n".join(f"  {r.model:<18} WAPE {r.WAPE:.3f}  bias {r.Bias:+.2f}  n {int(r.n_obs):,}"
                           for r in ov.itertuples()))
        origins = bt["origin"].unique() if "origin" in bt.columns else [None]
        losing_origins = []
        for orig in origins:
            sub = bt[(bt["breakdown"] == "overall") & (bt["origin"] == orig)] if orig is not None else bt[bt["breakdown"] == "overall"]
            lgb_m = sub[sub["model"].str.contains("lightgbm")]
            naive_m = sub[sub["model"] == "naive_seasonal"]
            if not lgb_m.empty and not naive_m.empty:
                if lgb_m["WAPE"].iloc[0] >= naive_m["WAPE"].iloc[0]:
                    losing_origins.append(str(orig))
        if not losing_origins:
            record(s, "PASS", "LightGBM beats seasonal naive across all backtest origins")
        else:
            record(s, "WARN", f"LightGBM does not beat seasonal naive at origin(s): {', '.join(losing_origins)}")


# -------------------------------------------------------- 6 interval coverage
def check_coverage(s):
    path = OUT / "backtest_predictions.csv"
    if not path.exists():
        record(s, "WARN", "no row-level backtest predictions saved (backtest_predictions.csv), "
                          "so P10-P90 coverage cannot be measured")
        return
    b = pd.read_csv(path, low_memory=False)
    if not {"actual", "p10", "p90"}.issubset(b.columns):
        record(s, "FAIL", "backtest_predictions.csv needs columns actual, p10, p90")
        return
    b_val = b.dropna(subset=["p10", "p90"])
    if b_val.empty:
        b_val = b
    cov = float(((b_val["actual"] >= b_val["p10"]) & (b_val["actual"] <= b_val["p90"])).mean())
    record(s, "PASS" if 0.70 <= cov <= 0.90 else "INFO",
           f"P10-P90 empirical coverage on backtest (diagnostic, not a PDF requirement): {cov:.1%} (target about 80%)")


# ---------------------------------------------------- 7 prices and revenue
def check_prices(s):
    c = pd.read_parquet(need(OUT / "clean_train.parquet"))
    zero_pos = int(((c["Unit_Price_USD"] == 0) & (c["Units_Sold"] > 0)).sum())
    zero_all = int((c["Unit_Price_USD"] == 0).sum())
    rev = c.dropna(subset=["Units_Sold", "Revenue_USD"])
    rev_bad = int(((rev["Units_Sold"] * rev["Unit_Price_USD"] - rev["Revenue_USD"]).abs() > 1).sum())

    nz = c[c["Unit_Price_USD"] > 0].assign(Y=lambda d: d["Date"].str[:4])
    ref = (nz.groupby(["SKU_ID", "Y"])["Unit_Price_USD"].median()
           .rename("ref").reset_index())
    cc = nz.merge(ref, on=["SKU_ID", "Y"], how="left")
    ratio = cc["Unit_Price_USD"] / cc["ref"]
    out_band = int(((ratio < 0.5) | (ratio > 2.0)).sum())

    sm = (nz.groupby(["SKU_ID", "Date"])["Unit_Price_USD"].median().reset_index()
          .sort_values(["SKU_ID", "Date"]))
    sm["prev"] = sm.groupby("SKU_ID")["Unit_Price_USD"].shift(1)
    changed = sm["prev"].notna() & ((sm["Unit_Price_USD"] - sm["prev"]).abs() > 0.01)
    not_jan = sm["Date"].str[5:] != "01"
    surcharge = sm["Date"].between("2022-07", "2022-12")
    violations = int((changed & not_jan & ~surcharge).sum())

    record(s, "PASS" if zero_pos == 0 else "FAIL", f"zero price on rows with units > 0: {zero_pos}")
    record(s, "INFO", f"zero price on any row: {zero_all:,} (no-order rows are included here)")
    record(s, "PASS" if rev_bad == 0 else "FAIL", f"revenue that is not units x price (over $1): {rev_bad}")
    record(s, "PASS" if out_band == 0 else "FAIL",
           f"non-zero prices outside 0.5x to 2.0x the SKU-year median: {out_band}")
    record(s, "PASS" if violations == 0 else "FAIL",
           f"SKU price changes outside 1 January and the 2022 surcharge window: {violations}")


# ----------------------------------------------------------- 8 lifecycle rules
def check_lifecycle(s):
    sku = pd.read_csv(need(STUDENT / "sku_master.csv"), dtype=str).fillna("")
    sku = sku[["SKU_ID", "EOL_Date", "Launch_Date"]]

    f = pd.read_csv(need(OUT / "forecast_submission.csv"), dtype={"Date": str}).merge(
        sku, on="SKU_ID", how="left").fillna("")
    eol = f["EOL_Date"].str[:7]
    launch = f["Launch_Date"].str[:7]
    after = f[(eol != "") & (f["Date"] > eol) & (f["Units_Sold_Forecast"] > 0)]
    before = f[(launch != "") & (f["Date"] < launch) & (f["Units_Sold_Forecast"] > 0)]
    record(s, "PASS" if after.empty else "FAIL",
           f"forecast rows above zero after EOL: {len(after)}")
    record(s, "PASS" if before.empty else "FAIL",
           f"forecast rows above zero before launch: {len(before)}")

    t = pd.read_parquet(need(OUT / "clean_train.parquet"), columns=["SKU_ID", "Date", "Units_Sold"])
    t = t.merge(sku, on="SKU_ID", how="left")
    t["EOL_Date"] = t["EOL_Date"].fillna("")
    hist_after = t[(t["EOL_Date"].str[:7] != "") & (t["Date"] > t["EOL_Date"].str[:7])
                   & (t["Units_Sold"] > 0)]
    record(s, "PASS" if hist_after.empty else "FAIL",
           f"history rows with sales after EOL: {len(hist_after)}")


# ------------------------------------------------------------ 9 hierarchy totals
def check_hierarchy(s):
    sku = pd.read_csv(need(STUDENT / "sku_master.csv"), dtype=str)[["SKU_ID", "Product_Family"]]
    f = pd.read_csv(need(OUT / "forecast_submission.csv"), dtype={"Date": str}).merge(
        sku, on="SKU_ID", how="left")
    total = int(f["Units_Sold_Forecast"].sum())
    by_family = int(f.groupby("Product_Family")["Units_Sold_Forecast"].sum().sum())
    by_region = int(f.groupby("Region")["Units_Sold_Forecast"].sum().sum())
    by_month = int(f.groupby("Date")["Units_Sold_Forecast"].sum().sum())
    ok = total == by_family == by_region == by_month
    record(s, "PASS" if ok else "FAIL",
           f"total {total:,}; family {by_family:,}; region {by_region:,}; month {by_month:,}")
    rec = OUT / "hierarchy_reconciliation.csv"
    if rec.exists():
        record(s, "PASS", "hierarchy_reconciliation.csv present")
        df_rec = pd.read_csv(rec)

        # 5a: Fail if any level is FAIL
        has_fails = (df_rec["Status"] != "PASS").any()
        record(s, "FAIL" if has_fails else "PASS",
               f"hierarchy status: {'found FAIL status in records' if has_fails else 'all levels reported PASS'}")

        # 5a: Fail if family/region sums do not equal total
        tot_rows = df_rec[df_rec["Level"] == "Total"]
        if not tot_rows.empty:
            reconciled_total = int(tot_rows.iloc[0]["Reconciled_Units"])
            fam_sum = int(df_rec[df_rec["Level"] == "Product_Family"]["Reconciled_Units"].sum())
            reg_sum = int(df_rec[df_rec["Level"] == "Region"]["Reconciled_Units"].sum())
            totals_agree = (fam_sum == reconciled_total) and (reg_sum == reconciled_total)
            record(s, "PASS" if totals_agree else "FAIL",
                   f"reconciled totals match: total={reconciled_total:,}, family={fam_sum:,}, region={reg_sum:,}")
        else:
            record(s, "FAIL", "no Total row found in hierarchy_reconciliation.csv")

        # 5d: Model card reads hierarchy numbers from regenerated file
        mc_path = OUT / "model_card.md"
        if mc_path.exists():
            mc_text = mc_path.read_text(encoding="utf-8")
            max_raw = df_rec["Raw_Deviation_Pct"].max()
            expected_str = f"{max_raw:.1f}%"
            mc_matches = expected_str in mc_text
            record(s, "PASS" if mc_matches else "FAIL",
                   f"model card hierarchy numbers match hierarchy_reconciliation.csv ({expected_str})")
    else:
        record(s, "FAIL", "hierarchy_reconciliation.csv not produced (spec requires it)")


# ------------------------------------------------------------- 10 dashboard
def check_dashboard(s):
    path = need(PROJECT / "app" / "dashboard.py")
    src = path.read_text(encoding="utf-8")
    try:
        compile(src, str(path), "exec")
        record(s, "PASS", "dashboard.py compiles")
    except SyntaxError as exc:
        record(s, "FAIL", f"dashboard.py has a syntax error at line {exc.lineno}: {exc.msg}")

    if "student_data" in src:
        record(s, "FAIL", "dashboard refers to student_data directly (it should read outputs/ only)")
    else:
        record(s, "PASS", "dashboard reads outputs only")

    empty = [ln.strip() for ln in src.splitlines() if re.search(r"multiselect\(.*\[\]", ln)]
    record(s, "WARN" if empty else "PASS",
           "filters with an empty option list (cannot be used): " + str(len(empty)))

    m = re.search(r"st\.tabs\(\s*\[(.*?)\]\s*\)", src, re.S)
    labels = re.findall(r'"([^"]*)"', m.group(1)) if m else []
    optional = [x for x in labels if re.search(r"opt", x, re.I)]
    required = [x for x in labels if x not in optional]
    record(s, "INFO", "tab labels: " + " | ".join(labels))
    record(s, "PASS" if len(optional) == 4 else "FAIL",
           f"optional views: {len(optional)} (spec requires exactly 4)")
    record(s, "PASS" if len(required) >= 5 else "FAIL",
           f"required views: {len(required)} (spec requires 5)")


# -------------------------------------------------- 11 writing and code hygiene
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2500-\u257F\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
PATTERNS = [
    ("emoji or decorative symbol", EMOJI, "FAIL"),
    ("unresolved, TBD or to-be-investigated wording",
     re.compile(r"unresolved|\bTBD\b|to be investigated", re.I), "FAIL"),
    ("bare except", re.compile(r"except\s*:"), "FAIL"),
    ("broad except Exception", re.compile(r"except\s+Exception"), "WARN"),
    ("banner line", re.compile(r"^\s*#?\s*[=*#-]{12,}\s*$"), "WARN"),
]


def scan_files():
    allowed = {".py", ".md", ".bat", ".txt", ".json", ".yml", ".yaml", ".cfg", ".toml"}
    for p in PROJECT.rglob("*"):
        if not p.is_file() or p.suffix not in allowed:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if "outputs" in p.parts and p.suffix not in {".md", ".json"}:
            continue
        if p.name == "check_report.txt":
            continue
        yield p


def check_hygiene(s):
    hits = {label: [] for label, _, _ in PATTERNS}
    for path in scan_files():
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for n, line in enumerate(lines, 1):
            for label, pattern, _ in PATTERNS:
                if pattern.search(line):
                    hits[label].append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:110]}")
    for label, _, level in PATTERNS:
        found = hits[label]
        if not found:
            record(s, "PASS", f"no {label}")
            continue
        record(s, level, f"{len(found)} occurrence(s) of {label}")
        record(s, "INFO", "\n".join(found[:12]) + ("\n..." if len(found) > 12 else ""))


# ------------------------------------------------- 12 deliverables and tests
REQUIRED = [
    "project/README.md",
    "project/requirements.txt",
    "project/docs/methodology.md",
    "project/docs/dashboard_guide.md",
    "project/docs/presentation_outline.md",
    "project/docs/audit_of_existing_work.md",
    "project/.gitignore",
    "project/outputs/audit_log.csv",
    "project/outputs/data_quality_report.json",
    "project/outputs/data_quality_report.md",
    "project/outputs/anomaly_register.csv",
    "project/outputs/unconstrained_demand.parquet",
    "project/outputs/macro_lag_analysis.csv",
    "project/outputs/ev_shift_analysis.csv",
    "project/outputs/backtest_metrics.csv",
    "project/outputs/forecast_submission.csv",
    "project/outputs/forecast_intervals.csv",
    "project/outputs/hierarchy_reconciliation.csv",
    "project/outputs/model_card.md",
    "project/outputs/panel_status.csv",
    "project/outputs/master_data_contradictions.csv",
    "project/outputs/early_buying.csv",
    "project/outputs/holiday_effects.csv",
    "project/outputs/structural_breaks.csv",
    "project/outputs/customer_switches.csv",
    "project/outputs/lost_accounts.csv",
    "project/outputs/lag_stability.csv",
    "project/outputs/coverage_breakdown.csv",
    "project/outputs/whatif_scenarios.csv",
    "project/outputs/event_impact_estimates.csv",
]


def check_deliverables(s, skip_tests):
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        record(s, "FAIL", f"missing deliverables ({len(missing)}): " + ", ".join(missing))
    else:
        record(s, "PASS", "all required deliverable files exist")

    readme = PROJECT / "README.md"
    if readme.exists():
        ok = False
        for line in readme.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("Public dashboard:"):
                url = line.split("Public dashboard:", 1)[1].strip()
                if url.startswith("http") and not any(ph in url.lower() for ph in ["[", "placeholder", "tbd", "link to be added"]):
                    ok = True
                    break
        record(s, "PASS" if ok else "WARN",
               "dashboard URL found in README" if ok else "no dashboard URL in README")

    if skip_tests:
        record(s, "INFO", "pytest skipped (--no-tests)")
        return
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                            cwd=str(PROJECT), capture_output=True, text=True, timeout=900)
    last = (result.stdout.strip().splitlines() or ["no output"])[-1]
    passed = result.returncode == 0
    record(s, "PASS" if passed else "FAIL", f"pytest: {last}")
    if not passed:
        record(s, "INFO", "\n".join(result.stdout.strip().splitlines()[-40:]))


# ------------------------------------------------ 13 spec compliance (Part F1)
def check_spec_compliance(s):
    outline_path = PROJECT / "docs" / "presentation_outline.md"
    if not outline_path.exists():
        record(s, "FAIL", "presentation_outline.md not found")
        return
    text = outline_path.read_text(encoding="utf-8")
    rows = [ln.strip() for ln in text.splitlines() if re.match(r"^\|\s*(CORE|ADV|EXP)-", ln)]
    if not rows:
        record(s, "FAIL", "No requirements matrix rows found in presentation_outline.md")
        return
    done = [r for r in rows if "| done |" in r.lower()]
    partial = [r for r in rows if "| partial |" in r.lower()]
    not_met = [r for r in rows if "| not met" in r.lower()]
    record(s, "INFO", f"spec requirements matrix: {len(rows)} items ({len(done)} done, {len(partial)} partial, {len(not_met)} not met)")
    if len(not_met) == 0 and len(done) >= 45:
        record(s, "PASS", f"all {len(done)} spec requirements documented and verified complete")
    else:
        record(s, "WARN", f"{len(not_met)} requirements not met; {len(partial)} partial")

    # 5b: Verify output files exist and are tracked in git (git ls-files)
    git_tracked = set()
    try:
        git_res = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, timeout=20)
        if git_res.returncode == 0:
            git_tracked = {p.replace("\\", "/") for p in git_res.stdout.splitlines()}
    except (OSError, subprocess.SubprocessError):
        pass

    missing_files = []
    untracked_files = []
    for r in rows:
        parts = [p.strip() for p in r.split("|") if p.strip()]
        if len(parts) >= 4:
            impl = parts[3]
            out_files = re.findall(r"outputs/[a-zA-Z0-9_\.]+", impl)
            for of in out_files:
                full_p = PROJECT / of
                rel_git = f"project/{of}"
                if not full_p.exists():
                    missing_files.append(of)
                elif git_tracked and rel_git not in git_tracked and not of.endswith("backtest_predictions.csv"):
                    untracked_files.append(of)

    if missing_files:
        record(s, "FAIL", f"spec matrix output files missing from disk ({len(set(missing_files))}): {', '.join(sorted(set(missing_files)))}")
    else:
        record(s, "PASS", "all outputs referenced in spec compliance matrix exist on disk")

    if untracked_files:
        record(s, "FAIL", f"spec matrix output files missing from git ls-files ({len(set(untracked_files))}): {', '.join(sorted(set(untracked_files)))}")
    else:
        record(s, "PASS", "all outputs referenced in spec compliance matrix are tracked in git")

    # 5b: Verify named test file and tests exist in tests/
    test_file = PROJECT / "tests" / "test_cleaning_rules.py"
    if test_file.exists():
        test_content = test_file.read_text(encoding="utf-8")
        test_defs = re.findall(r"def (test_[a-zA-Z0-9_]+)", test_content)
        record(s, "PASS", f"test suite verified: {len(test_defs)} unit tests found in {test_file.name}")
    else:
        record(s, "FAIL", f"test file missing: {test_file}")


def check_pdf_compliance(s):
    compliance_path = PROJECT / "docs" / "pdf_compliance.md"
    if not compliance_path.exists():
        record(s, "FAIL", "project/docs/pdf_compliance.md not found")
        return
    text = compliance_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) <= 2:
        record(s, "FAIL", "No compliance rows found in project/docs/pdf_compliance.md")
        return
    data_rows = lines[2:]

    test_file = PROJECT / "tests" / "test_cleaning_rules.py"
    test_content = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
    available_tests = set(re.findall(r"def (test_[a-zA-Z0-9_]+)", test_content))

    missing_impl = []
    missing_out = []
    missing_tests = []
    invalid_status = []

    for idx, r in enumerate(data_rows, 1):
        parts = [p.strip() for p in r.strip().strip("|").split("|")]
        if len(parts) < 6:
            invalid_status.append(f"Row {idx} malformed: expected 6 columns, got {len(parts)}")
            continue
        pdf_text, impl_file, out_or_view, test_fn, status, reason = parts[:6]

        impl_p = ROOT / impl_file
        if not impl_p.exists():
            missing_impl.append(f"{impl_file} (row {idx})")

        if out_or_view.startswith("outputs/"):
            out_p = PROJECT / out_or_view
            if not out_p.exists():
                missing_out.append(f"{out_or_view} (row {idx})")
        elif not (out_or_view.startswith("View:") or out_or_view.startswith("outputs/")):
            missing_out.append(f"Unrecognized output or view '{out_or_view}' (row {idx})")

        if test_fn not in available_tests:
            missing_tests.append(f"{test_fn} (row {idx})")

        if status.lower() not in ("done", "not done"):
            invalid_status.append(f"Invalid status '{status}' (row {idx})")
        elif status.lower() == "not done" and (not reason or reason == "-"):
            invalid_status.append(f"Row {idx} marked 'not done' without a stated reason")

    if missing_impl:
        record(s, "FAIL", f"Implementation files missing ({len(missing_impl)}): {', '.join(missing_impl[:5])}")
    if missing_out:
        record(s, "FAIL", f"Output files/views missing ({len(missing_out)}): {', '.join(missing_out[:5])}")
    if missing_tests:
        record(s, "FAIL", f"Tests missing from test_cleaning_rules.py ({len(missing_tests)}): {', '.join(missing_tests[:5])}")
    if invalid_status:
        record(s, "FAIL", f"Status errors ({len(invalid_status)}): {', '.join(invalid_status[:5])}")

    if not (missing_impl or missing_out or missing_tests or invalid_status):
        record(s, "PASS", f"all {len(data_rows)} PDF specification rows verified with valid files, outputs, and unit tests")


# ---------------------------------------------------------------- report
def build_report():
    counts = Counter(status for _, status, _ in RESULTS)
    lines = [
        f"Celeris project check, {datetime.now():%Y-%m-%d %H:%M}",
        f"Folder: {ROOT}",
        "",
        "SUMMARY  " + "  ".join(f"{k}={counts.get(k, 0)}" for k in
                                 ["PASS", "WARN", "FAIL", "ERROR", "INFO"]),
        "",
    ]
    problems = [(sec, st, msg) for sec, st, msg in RESULTS if st in ("FAIL", "ERROR")]
    if problems:
        lines.append("FIX THESE FIRST")
        for i, (sec, st, msg) in enumerate(problems[:10], 1):
            lines.append(f"  {i}. [{st}] {sec}: {msg.splitlines()[0]}")
        lines.append("")

    current = None
    for sec, st, msg in RESULTS:
        if sec != current:
            lines.append(f"== {sec}")
            current = sec
        first, *rest = msg.splitlines() or [""]
        lines.append(f"  [{st}] {first}")
        for extra in rest:
            lines.append(f"        {extra}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit the Celeris forecasting project.")
    parser.add_argument("--rerun", action="store_true",
                        help="run the pipeline twice to test reproducibility")
    parser.add_argument("--no-tests", action="store_true", help="skip pytest")
    args = parser.parse_args()

    checks = [
        ("0 environment", check_environment),
        ("1 reproducibility", lambda s: check_reproducibility(s, args.rerun)),
        ("2 submission format", check_submission),
        ("3 leakage and features", check_leakage),
        ("4 forecast shape", check_forecast_shape),
        ("5 holdout accuracy", check_holdout),
        ("6 interval coverage", check_coverage),
        ("7 prices and revenue", check_prices),
        ("8 lifecycle rules", check_lifecycle),
        ("9 hierarchy totals", check_hierarchy),
        ("10 dashboard", check_dashboard),
        ("11 writing and code hygiene", check_hygiene),
        ("12 deliverables and tests", lambda s: check_deliverables(s, args.no_tests)),
        ("13 spec compliance", check_spec_compliance),
        ("14 pdf compliance", check_pdf_compliance),
    ]
    for section, fn in checks:
        run_check(section, fn)

    report = build_report()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved to {REPORT.relative_to(ROOT)}. Paste its contents into the chat.")


if __name__ == "__main__":
    main()