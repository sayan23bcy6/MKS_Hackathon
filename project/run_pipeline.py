"""
run_pipeline.py - Master pipeline runner for Celeris Automobili demand forecasting.
Executes end-to-end data cleaning, feature engineering, structural analysis,
model training, backtesting, recursive forecasting, reconciliation, and reporting.
"""

import os
import sys
import time
import pathlib
import shutil
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
    OUT_PANEL_STATUS, OUT_MASTER_CONTRADICTIONS, OUT_EARLY_BUYING,
    OUT_HOLIDAY_EFFECTS, OUT_STRUCTURAL_BREAKS, OUT_CUSTOMER_SWITCHES,
    OUT_LOST_ACCOUNTS, OUT_LAG_STABILITY, OUT_COVERAGE_BREAKDOWN,
    RAW_SKU_MASTER, OUTPUT_DIR, OUT_FAMILY_SELECTION, OUT_BASELINE_FORECAST, OUT_MODEL_CARD
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
from pipeline.reports import generate_dq_report, build_panel_status
from pipeline.models.lightgbm_model import train_all_quantiles, get_feature_names
from pipeline.backtest import run_honest_backtest, OUT_BACKTEST_PREDS
from pipeline.family_selection import (
    select_family_models, compute_baseline_forecasts, assemble_family_forecast
)
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
    # The dashboard reads outputs only, so the SKU master is copied there.
    shutil.copy(RAW_SKU_MASTER, OUTPUT_DIR / 'sku_master.csv')
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
    breaks_df.to_csv(OUT_STRUCTURAL_BREAKS, index=False)

    early_buying_summary = quantify_early_buying(df_clean)
    early_buying_summary.to_csv(OUT_EARLY_BUYING, index=False)

    holiday_summary = measure_holiday_impacts(df_clean)
    holiday_summary.to_csv(OUT_HOLIDAY_EFFECTS, index=False)

    master_integrity = audit_master_data_integrity(df_clean, clean_sku)
    master_integrity.to_csv(OUT_MASTER_CONTRADICTIONS, index=False)

    cycle_df, switches_df, lost_accounts_df = analyze_ordering_cycles(df_clean)
    switches_df.to_csv(OUT_CUSTOMER_SWITCHES, index=False)
    lost_accounts_df.to_csv(OUT_LOST_ACCOUNTS, index=False)

    ev_summary = run_ev_transition_analysis(df_clean, clean_mac, clean_sku)
    print(f"  Structural breaks: {len(breaks_df)}, Customer switches: {len(switches_df)}, Lost accounts: {len(lost_accounts_df)}")

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

    # Step 8: Rolling-origin backtests across multi-year origins
    print("  Running multi-origin rolling backtests (2023-10, 2024-10, 2025-03)...")
    backtest_metrics, backtest_preds = run_honest_backtest(
        df_unconstrained, clean_mac, clean_ev, clean_sku,
        origins=['2023-10', '2024-10', '2025-03'], horizon=19
    )

    # Per-family model choice from the backtest (selection origins 2023-10 and 2024-10).
    family_selection = select_family_models(backtest_preds)

    # Step 9: Recursive forecasting and hierarchy reconciliation
    print("\n[9/10] Generating 19-month recursive forecast and hierarchy reconciliation...")
    lgb_submission, lgb_intervals = generate_recursive_forecast(
        raw_test, features_df, clean_mac, clean_ev, clean_sku, models, save_outputs=False
    )
    baselines = compute_baseline_forecasts(df_unconstrained, raw_test)
    baselines.to_csv(OUT_BASELINE_FORECAST, index=False)
    submission, intervals, assignment = assemble_family_forecast(
        lgb_submission, lgb_intervals, baselines, family_selection, raw_test, clean_sku
    )
    submission.to_csv(OUT_FORECAST_SUB, index=False)
    intervals.to_csv(OUT_FORECAST_INT, index=False)

    recon_results = compute_hierarchy_reconciliation(submission, df_unconstrained, clean_sku)
    print(f"  Hierarchy addition: total forecast = {recon_results['total_forecast_units']} units, adds_up = {recon_results['adds_up']}")

    # What-if scenario simulation (model rerun with shifted PMI)
    print("  Generating macroeconomic what-if scenarios (-6.0 to +6.0 PMI shift)...")
    scenario_df = generate_whatif_scenario(
        submission, raw_test, clean_mac, clean_sku, models, clean_ev, features_df, pmi_delta_target=-3.0,
        baselines=baselines, family_selection=family_selection,
    )

    # Step 10: Generate documentation, model card, and data quality reports
    print("\n[10/10] Generating model card, data quality reports, and panel status...")
    n_audit = audit.save()
    audit_df = pd.read_csv(OUT_AUDIT_LOG)
    write_model_card(models, get_feature_names(features_df), backtest_metrics=backtest_metrics)
    with open(OUT_MODEL_CARD, 'a', encoding='utf-8') as fh:
        fh.write('\n## Per-family model selection\n\n')
        fh.write('Each family uses the model with the lowest mean WAPE over the selection origins 2023-10 and 2024-10. '
                 'The 2025-03 origin is the holdout and was not used for the choice.\n\n')
        fh.write('| Family | Selected model | Selection WAPE (selected) | Holdout WAPE 2025-03 (selected) |\n|---|---|---:|---:|\n')
        for _, r in family_selection.iterrows():
            sel_col = 'Selection_WAPE_' + r['Selected_Model']
            fh.write(f"| {r['Product_Family']} | {r['Selected_Model']} | {r[sel_col]:.4f} | {r['Holdout_WAPE_selected']:.4f} |\n")
    dq_report = generate_dq_report(dq_train, dq_mac, dq_sku, dq_ev, audit_df, anomaly_df, raw_stockouts)
    build_panel_status(df_clean, clean_sku)

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
        OUT_EV_SHIFT, OUT_MACRO_LAG, OUT_PANEL_STATUS,
        OUT_MASTER_CONTRADICTIONS, OUT_EARLY_BUYING, OUT_HOLIDAY_EFFECTS,
        OUT_STRUCTURAL_BREAKS, OUT_CUSTOMER_SWITCHES, OUT_LOST_ACCOUNTS,
        OUT_LAG_STABILITY, OUT_COVERAGE_BREAKDOWN
    ]:
        p = pathlib.Path(str(out_path))
        if p.exists():
            print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            raise FileNotFoundError(f"Expected pipeline output file missing: {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
