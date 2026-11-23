import pathlib
"""Tests for per-family model selection and the assembled forecast."""

import numpy as np
import pandas as pd
import pytest

from pipeline.family_selection import (
    select_family_models, assemble_family_forecast, family_wape_table, SELECTION_ORIGINS, HOLDOUT_ORIGIN,
)


def _preds(rows):
    """rows: (origin, family, model, actual, p50)."""
    df = pd.DataFrame(rows, columns=['origin', 'Product_Family', 'model', 'actual', 'p50'])
    df['Date'] = '2024-01'
    df['SKU_ID'] = 'X-001'
    df['Region'] = 'Europe'
    df['Customer_Name'] = 'C'
    return df


def test_selection_uses_only_selection_origins(tmp_path):
    # naive is exact on the selection origins; lightgbm is exact only on the holdout.
    rows = []
    for origin in SELECTION_ORIGINS:
        rows += [(origin, 'F1', 'naive_seasonal', 100, 100), (origin, 'F1', 'lightgbm_p50', 100, 150)]
    rows += [(HOLDOUT_ORIGIN, 'F1', 'naive_seasonal', 100, 150), (HOLDOUT_ORIGIN, 'F1', 'lightgbm_p50', 100, 100)]
    sel = select_family_models(_preds(rows), out_path=tmp_path / 'sel.csv')
    assert sel.loc[sel.Product_Family == 'F1', 'Selected_Model'].iloc[0] == 'naive_seasonal'


def test_holdout_cannot_change_the_choice(tmp_path):
    rows = []
    for origin in SELECTION_ORIGINS:
        rows += [(origin, 'F1', 'moving_average', 100, 100), (origin, 'F1', 'lightgbm_p50', 100, 120)]
    rows += [(HOLDOUT_ORIGIN, 'F1', 'moving_average', 100, 0), (HOLDOUT_ORIGIN, 'F1', 'lightgbm_p50', 100, 100)]
    sel = select_family_models(_preds(rows), out_path=tmp_path / 'sel.csv')
    assert sel.loc[sel.Product_Family == 'F1', 'Selected_Model'].iloc[0] == 'moving_average'


def test_ties_go_to_the_simpler_model(tmp_path):
    rows = []
    for origin in SELECTION_ORIGINS:
        rows += [(origin, 'F1', 'naive_seasonal', 100, 90), (origin, 'F1', 'lightgbm_p50', 100, 90)]
    rows += [(HOLDOUT_ORIGIN, 'F1', 'naive_seasonal', 100, 90), (HOLDOUT_ORIGIN, 'F1', 'lightgbm_p50', 100, 90)]
    sel = select_family_models(_preds(rows), out_path=tmp_path / 'sel.csv')
    assert sel.loc[sel.Product_Family == 'F1', 'Selected_Model'].iloc[0] == 'naive_seasonal'


def test_wape_ignores_null_actuals():
    rows = [('2023-10', 'F1', 'lightgbm_p50', np.nan, 500), ('2023-10', 'F1', 'lightgbm_p50', 100, 110)]
    t = family_wape_table(_preds(rows))
    assert t.loc[0, 'WAPE'] == pytest.approx(0.10)


def _fixture_forecast():
    dates = ['2026-11', '2026-12']
    test_df = pd.DataFrame({
        'Date': dates * 2, 'SKU_ID': ['A-001'] * 2 + ['B-001'] * 2, 'Region': ['Europe'] * 4,
        'Customer_Name': ['C1'] * 4, 'Product_Family': ['Alpha'] * 2 + ['Beta'] * 2,
    })
    lgb_sub = test_df[['Date', 'SKU_ID', 'Region']].copy()
    lgb_sub['Units_Sold_Forecast'] = [50, 60, 70, 80]
    lgb_int = lgb_sub.copy()
    lgb_int['Units_Sold_Forecast_p10'] = [40, 50, 60, 70]
    lgb_int['Units_Sold_Forecast_p50'] = [50, 60, 70, 80]
    lgb_int['Units_Sold_Forecast_p90'] = [60, 70, 80, 90]
    baselines = test_df[['Date', 'SKU_ID', 'Region', 'Customer_Name']].copy()
    baselines['Units_Sold_Forecast_naive'] = [5, 6, 7, 8]
    baselines['Units_Sold_Forecast_moving'] = [1, 1, 1, 1]
    baselines['Units_Sold_Forecast_croston'] = [2, 2, 2, 2]
    sel = pd.DataFrame({'Product_Family': ['Alpha', 'Beta'],
                        'Selected_Model': ['naive_seasonal', 'lightgbm_p50']})
    sku_master = pd.DataFrame({'SKU_ID': ['A-001', 'B-001'], 'Product_Family': ['Alpha', 'Beta'],
                               'Launch_Date': ['', ''], 'EOL_Date': ['', '']})
    return lgb_sub, lgb_int, baselines, sel, test_df, sku_master


def test_assembly_takes_the_selected_model_per_family():
    lgb_sub, lgb_int, baselines, sel, test_df, sku_master = _fixture_forecast()
    sub, _, _ = assemble_family_forecast(lgb_sub, lgb_int, baselines, sel, test_df, sku_master)
    got = sub.set_index(['Date', 'SKU_ID'])['Units_Sold_Forecast']
    assert got.loc[('2026-11', 'A-001')] == 5      # Alpha uses naive
    assert got.loc[('2026-11', 'B-001')] == 70     # Beta uses LightGBM


def test_baseline_families_have_no_bands_and_lightgbm_families_do():
    lgb_sub, lgb_int, baselines, sel, test_df, sku_master = _fixture_forecast()
    _, intervals, _ = assemble_family_forecast(lgb_sub, lgb_int, baselines, sel, test_df, sku_master)
    alpha = intervals[intervals.SKU_ID == 'A-001']
    beta = intervals[intervals.SKU_ID == 'B-001']
    assert alpha['Units_Sold_Forecast_p10'].isna().all() and alpha['Units_Sold_Forecast_p90'].isna().all()
    assert beta['Units_Sold_Forecast_p10'].notna().all() and beta['Units_Sold_Forecast_p90'].notna().all()


def test_assembled_quantiles_stay_ordered():
    lgb_sub, lgb_int, baselines, sel, test_df, sku_master = _fixture_forecast()
    _, intervals, _ = assemble_family_forecast(lgb_sub, lgb_int, baselines, sel, test_df, sku_master)
    ok = intervals.dropna(subset=['Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p90'])
    assert (ok['Units_Sold_Forecast_p10'] <= ok['Units_Sold_Forecast_p50']).all()
    assert (ok['Units_Sold_Forecast_p50'] <= ok['Units_Sold_Forecast_p90']).all()


def test_submission_rows_match_test_rows():
    lgb_sub, lgb_int, baselines, sel, test_df, sku_master = _fixture_forecast()
    sub, _, _ = assemble_family_forecast(lgb_sub, lgb_int, baselines, sel, test_df, sku_master)
    keys = ['Date', 'SKU_ID', 'Region']
    assert len(sub) == len(test_df)
    assert not sub.duplicated(keys).any()
    assert (sub['Units_Sold_Forecast'] >= 0).all()


def test_dashboard_band_contains_p50_under_filters():
    """The summed P10-P90 band in the Executive Overview must contain the summed P50 line."""
    import base64
    import json
    import numpy as np
    from streamlit.testing.v1 import AppTest

    def decode(v):
        if isinstance(v, dict) and 'bdata' in v:
            return np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype'])).astype(float)
        return np.asarray(v, float)

    def bands_ok(app):
        checked = 0
        for el in app.get('plotly_chart'):
            try:
                fig = json.loads(el.proto.spec)
            except Exception:
                continue
            data = fig.get('data', [])
            p50 = [t for t in data if 'P50' in t.get('name', '')]
            band = [t for t in data if 'Interval' in t.get('name', '')]
            if not p50 or not band:
                continue
            y50 = decode(p50[0]['y'])
            yb = decode(band[0]['y'])
            n = len(y50)
            upper, lower = yb[:n], yb[n:][::-1]
            assert np.all((lower - 1e-6 <= y50) & (y50 <= upper + 1e-6))
            checked += 1
        return checked

    dash = str(pathlib.Path(__file__).resolve().parent.parent / 'app' / 'dashboard.py')
    app = AppTest.from_file(dash, default_timeout=200).run()
    assert len(app.exception) == 0
    assert bands_ok(app) >= 1
    app = AppTest.from_file(dash, default_timeout=200).run()
    app.sidebar.multiselect[0].set_value(['Electrical Assemblies'])
    app.run()
    assert len(app.exception) == 0
    assert bands_ok(app) >= 1
