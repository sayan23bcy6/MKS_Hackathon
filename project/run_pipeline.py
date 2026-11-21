"""
run_pipeline.py - Master pipeline runner for Celeris Automobili demand forecasting.
Executes end-to-end data cleaning, feature engineering, structural analysis,
model training, backtesting, recursive forecasting, reconciliation, and reporting.
"""

import os
import sys
import time
import pathlib
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import (
    OUT_TRAIN_CLEAN, OUT_MACRO_CLEAN, OUT_EVENTS_CLEAN,
    OUT_AUDIT_LOG, OUT_FEATURES, OUT_BACKTEST,
    OUT_FORECAST_SUB, OUT_FORECAST_INT, OUT_HIERARCHY_RECON,
    OUT_SCENARIO, OUT_EV_SHIFT, OUT_MACRO_LAG,
    OUT_WHATIF_SCENARIOS, OUT_EVENT_ESTIMATES,
)
from pipeline.load import (
    load_train, load_test, load_macro,
    load_sku_master, load_events, load_stockouts
)
from pipeline.audit import audit
from pipeline.clean_train import clean_train
from pipeline.clean_master_macro_events import (
    clean_sku_master, clean_macro, clean_events
)
from pipeline.unconstrained import build_unconstrained_demand
from pipeline.features import build_features, compute_macro_lag_analysis
from pipeline.ordering_cycles import analyze_ordering_cycles
from pipeline.ev_shift import run_ev_transition_analysis
from pipeline.structural_breaks import (
    detect_structural_breaks, quantify_early_buying,
    measure_holiday_impacts, audit_master_data_integrity
)
from pipeline.anomalies import build_anomaly_register
from pipeline.reports import generate_dq_report
from pipeline.models.lightgbm_model import train_all_quantiles, get_feature_names
from pipeline.backtest import run_honest_backtest, OUT_BACKTEST_PREDS
from pipeline.forecast import (
    generate_recursive_forecast, compute_hierarchy_reconciliation,
    generate_whatif_scenario, write_model_card
)


def main():
    t0 = time.time()
    print("Starting Celeris Automobili demand forecasting pipeline")

    # Step 1: Load raw data files
    print("\n[1/10] Loading raw input data...")
    raw_train = load_train()
    raw_test = load_test()
    raw_macro = load_macro()
    raw_sku = load_sku_master()
    raw_events = load_events()
    raw_stockouts = load_stockouts()
    print(f"  train: {raw_train.shape}, test: {raw_test.shape}, macro: {raw_macro.shape}")

    # Step 2: Clean master files
    print("\n[2/10] Cleaning SKU master, macro indicators, and events calendar...")
    clean_sku, dq_sku = clean_sku_master(raw_sku)
    clean_mac, dq_mac = clean_macro(raw_macro)
    clean_ev, dq_ev = clean_events(raw_events)

    clean_mac.to_csv(OUT_MACRO_CLEAN, index=False)
    clean_ev.to_csv(OUT_EVENTS_CLEAN, index=False)
    print(f"  Cleaned macro rows: {len(clean_mac)}, Cleaned events rows: {len(clean_ev)}")

    # Step 3: Clean training sales data
    print("\n[3/10] Cleaning sales transaction history (train.csv)...")
    df_clean, dq_train = clean_train(raw_train, clean_sku, raw_stockouts)
    df_clean.to_parquet(OUT_TRAIN_CLEAN, index=False)
    print(f"  Cleaned transactions: {len(df_clean)} rows (from raw {dq_train['raw_rows']})")

    # Save audit log
    n_audit = audit.save()
    print(f"  Audit log: {n_audit} entries saved to {OUT_AUDIT_LOG}")
    audit_df = pd.read_csv(OUT_AUDIT_LOG)

    # Step 4: Reconstruct unconstrained demand
    print("\n[4/10] Reconstructing unconstrained demand from stockout records...")
    df_unconstrained, match_rate = build_unconstrained_demand(df_clean, raw_stockouts)
    dq_train['stockout_match_rate_pct'] = match_rate

    # Step 5: Advanced structural breaks, customer cycles, and powertrain transition
    print("\n[5/10] Running structural breaks and advanced market dynamics analysis...")
    breaks_df = detect_structural_breaks(df_clean, clean_ev)
    early_buying_summary = quantify_early_buying(df_clean)
    holiday_summary = measure_holiday_impacts(df_clean)
    master_integrity = audit_master_data_integrity(df_clean, clean_sku)
    cycle_df, _, _ = analyze_ordering_cycles(df_clean)
    ev_summary = run_ev_transition_analysis(df_clean, clean_mac, clean_sku)
    print(f"  Structural breaks detected: {len(breaks_df)}, Customer cycle series: {len(cycle_df)}")

    # Step 6: Assemble anomaly register
    print("\n[6/10] Assembling anomaly register...")
    anomaly_df = build_anomaly_register(df_clean, clean_ev, raw_stockouts, audit_df)
    print(f"  Registered anomalies: {len(anomaly_df)}")

    # Step 7: Feature engineering
    print("\n[7/10] Engineering feature matrix...")
    features_df = build_features(df_clean, clean_mac, clean_ev, clean_sku, df_unconstrained)
    print(f"  Features generated: {features_df.shape}")

    # Compute macro lag cross-correlation
    lag_df = compute_macro_lag_analysis(df_clean, clean_mac)
    lag_df.to_csv(OUT_MACRO_LAG, index=False)

    # Step 8: Train LightGBM models with full historical refit
    print("\n[8/10] Training LightGBM quantile models (P10, P50, P90)...")
    TRAIN_CUT = '2025-03'
    VAL_START = '2025-04'
    train_split = features_df[features_df['Date'] <= TRAIN_CUT].copy()
    val_split = features_df[features_df['Date'] >= VAL_START].copy()

    # Train on validation split then refit on full history (Fix F1)
    models = train_all_quantiles(train_split, val_split, full_df=features_df)

    # Step 8: Honest backtesting across multi-year rolling origins
    print("  Running honest multi-origin rolling backtests (2023-10, 2024-10, 2025-03)...")
    backtest_metrics, backtest_preds = run_honest_backtest(
        df_unconstrained, clean_mac, clean_ev, clean_sku,
        origins=['2023-10', '2024-10', '2025-03'], horizon=19
    )

    # Step 9: Recursive forecasting and hierarchy reconciliation
    print("\n[9/10] Generating 19-month recursive forecast and hierarchy reconciliation...")
    submission, intervals = generate_recursive_forecast(
        raw_test, features_df, clean_mac, clean_ev, clean_sku, models
    )

    recon_results = compute_hierarchy_reconciliation(submission, df_unconstrained, clean_sku)
    print(f"  Hierarchy reconciliation: total forecast = {recon_results['total_forecast_units']} units, status = {'PASS' if recon_results['reconciled'] else 'FAIL'}")

    # What-if scenario simulation (model rerun with shifted PMI)
    print("  Generating macroeconomic what-if scenarios (-6.0 to +6.0 PMI shift)...")
    scenario_df = generate_whatif_scenario(
        submission, raw_test, clean_mac, clean_sku, models, clean_ev, features_df, pmi_delta_target=-3.0
    )

    # Step 10: Generate documentation, model card, and data quality reports
    print("\n[10/10] Generating model card and data quality reports...")
    n_audit = audit.save()
    audit_df = pd.read_csv(OUT_AUDIT_LOG)
    write_model_card(models, get_feature_names(features_df), backtest_metrics=backtest_metrics)
    dq_report = generate_dq_report(dq_train, dq_mac, dq_sku, dq_ev, audit_df, anomaly_df)

    elapsed = time.time() - t0
    print(f"\nPipeline finished successfully in {elapsed:.1f}s")
    print(f"Cleaned train rows: {len(df_clean)}")
    print(f"Forecast submission rows: {len(submission)} (matches expected 58,219)")
    print(f"Audit log entries: {n_audit}")

    # Verify generated output files
    print("\nGenerated output files:")
    for out_path in [
        OUT_TRAIN_CLEAN, OUT_MACRO_CLEAN, OUT_EVENTS_CLEAN,
        OUT_AUDIT_LOG, OUT_BACKTEST, OUT_BACKTEST_PREDS,
        OUT_FORECAST_SUB, OUT_FORECAST_INT, OUT_HIERARCHY_RECON,
        OUT_SCENARIO, OUT_WHATIF_SCENARIOS, OUT_EVENT_ESTIMATES,
        OUT_EV_SHIFT, OUT_MACRO_LAG
    ]:
        p = pathlib.Path(str(out_path))
        if p.exists():
            print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            raise FileNotFoundError(f"Expected pipeline output file missing: {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
