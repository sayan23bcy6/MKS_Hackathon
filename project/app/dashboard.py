"""
dashboard.py - Celeris Automobili Demand Control Tower.
Streamlit application providing 5 required control views plus 4 advanced analytical views.
Reads exclusively from project/outputs/ (Rule 7).
"""

import os
import sys
import json
import pathlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Celeris Automobili - Demand Control Tower",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = pathlib.Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# Data Loaders (strictly reading from outputs/)
@st.cache_data
def load_clean_train():
    path = OUTPUTS_DIR / "clean_train.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df['Units_Sold_num'] = pd.to_numeric(df.get('Units_Sold', 0), errors='coerce').fillna(0)
        df['Revenue_USD_num'] = pd.to_numeric(df.get('Revenue_USD', 0), errors='coerce').fillna(0)
        return df
    return pd.DataFrame()


@st.cache_data
def load_forecast():
    sub_path = OUTPUTS_DIR / "forecast_submission.csv"
    int_path = OUTPUTS_DIR / "forecast_intervals.csv"
    sub = pd.read_csv(sub_path, dtype={'Date': str}) if sub_path.exists() else pd.DataFrame()
    intervals = pd.read_csv(int_path, dtype={'Date': str}) if int_path.exists() else pd.DataFrame()
    sku_path = OUTPUTS_DIR / "sku_master.csv"
    if sku_path.exists():
        sku = pd.read_csv(sku_path, usecols=['SKU_ID', 'Product_Family'])
        if not sub.empty and 'Product_Family' not in sub.columns:
            sub = sub.merge(sku, on='SKU_ID', how='left')
            sub['Product_Family'] = sub['Product_Family'].fillna('Unknown')
        if not intervals.empty and 'Product_Family' not in intervals.columns:
            intervals = intervals.merge(sku, on='SKU_ID', how='left')
            intervals['Product_Family'] = intervals['Product_Family'].fillna('Unknown')
    return sub, intervals


@st.cache_data
def load_dq_report():
    path = OUTPUTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data
def load_anomaly_register():
    path = OUTPUTS_DIR / "anomaly_register.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_backtest_metrics():
    path = OUTPUTS_DIR / "backtest_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_macro_lag():
    path = OUTPUTS_DIR / "macro_lag_analysis.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_stockouts():
    uc_path = OUTPUTS_DIR / "unconstrained_demand.parquet"
    unconstrained = pd.read_parquet(uc_path) if uc_path.exists() else pd.DataFrame()
    return pd.DataFrame(), unconstrained


@st.cache_data
def load_ev_shift():
    path = OUTPUTS_DIR / "ev_shift_analysis.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_scenario():
    path = OUTPUTS_DIR / "whatif_scenario_forecast.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_all_scenarios():
    path = OUTPUTS_DIR / "whatif_scenarios.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_family_selection():
    path = OUTPUTS_DIR / "family_model_selection.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_coverage_breakdown():
    path = OUTPUTS_DIR / "coverage_breakdown.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_panel_status():
    path = OUTPUTS_DIR / "panel_status.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# Shared Filters
def sidebar_filters(df_train: pd.DataFrame, df_forecast: pd.DataFrame) -> dict:
    st.sidebar.header("Global Filters")
    st.sidebar.caption("Filters apply uniformly across all dashboard views")

    all_families = sorted(set(
        list(df_train['Product_Family'].dropna().unique()) +
        list(df_forecast['Product_Family'].dropna().unique() if 'Product_Family' in df_forecast.columns else [])
    ))
    all_skus = sorted(set(
        list(df_train['SKU_ID'].dropna().unique()) +
        list(df_forecast['SKU_ID'].dropna().unique())
    ))
    all_regions = sorted(set(
        list(df_train['Region'].dropna().unique()) +
        list(df_forecast['Region'].dropna().unique())
    ))
    all_channels = sorted(set(
        list(df_train['Sales_Channel'].dropna().unique()) +
        list(df_forecast['Sales_Channel'].dropna().unique() if 'Sales_Channel' in df_forecast.columns else [])
    ))
    all_customers = sorted(set(
        list(df_train['Customer_Name'].dropna().unique()) +
        list(df_forecast['Customer_Name'].dropna().unique() if 'Customer_Name' in df_forecast.columns else [])
    ))

    sel_families = st.sidebar.multiselect("Product Family", all_families, default=all_families, key='family')
    sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions, key='region')

    # Filter available SKUs and customers by selected families and regions
    meta_cols = ['Product_Family', 'Region', 'SKU_ID', 'Customer_Name']
    meta_source = pd.concat(
        [df_train[[c for c in meta_cols if c in df_train.columns]],
         df_forecast[[c for c in meta_cols if c in df_forecast.columns]]],
        ignore_index=True,
    )
    df_filtered_meta = meta_source[meta_source['Product_Family'].isin(sel_families or all_families) &
                                   meta_source['Region'].isin(sel_regions or all_regions)]
    filtered_skus = sorted(df_filtered_meta['SKU_ID'].dropna().unique()) if not df_filtered_meta.empty else all_skus
    filtered_customers = sorted(df_filtered_meta['Customer_Name'].dropna().unique()) if not df_filtered_meta.empty else all_customers

    sel_skus = st.sidebar.multiselect("SKU ID", options=filtered_skus, default=filtered_skus, key='sku', help="Select specific SKUs")
    sel_channels = st.sidebar.multiselect("Sales Channel", all_channels, default=all_channels, key='channel')
    sel_customers = st.sidebar.multiselect("Customer Name", options=filtered_customers, default=filtered_customers, key='customer', help="Select specific customers")

    return {
        'families': sel_families or all_families,
        'skus': sel_skus or all_skus,
        'regions': sel_regions or all_regions,
        'channels': sel_channels or all_channels,
        'customers': sel_customers or all_customers,
    }


def band_frame(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Rows selected as a baseline have no P10 or P90. Their band is the point forecast, so the
    summed band always contains the summed P50 line."""
    d = intervals.copy()
    for c in ['p10', 'p90']:
        col = f'Units_Sold_Forecast_{c}'
        d[col] = d[col].fillna(d['Units_Sold_Forecast_p50'])
    return d


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df.empty:
        return df
    res = df.copy()
    if 'Product_Family' in res.columns:
        res = res[res['Product_Family'].isin(filters['families'])]
    if 'SKU_ID' in res.columns and filters['skus']:
        res = res[res['SKU_ID'].isin(filters['skus'])]
    if 'Region' in res.columns:
        res = res[res['Region'].isin(filters['regions'])]
    if 'Sales_Channel' in res.columns and filters['channels']:
        res = res[res['Sales_Channel'].isin(filters['channels'])]
    if 'Customer_Name' in res.columns and filters['customers']:
        res = res[res['Customer_Name'].isin(filters['customers'])]
    return res


# 1. Executive Overview
def view_executive_overview(df_train, df_forecast, df_intervals, filters):
    st.header("1. Executive Overview")
    st.caption("P10-P90 bands are shown only for families forecast by LightGBM. Families forecast by a baseline show a point forecast.")
    st.caption("Source: outputs/clean_train.parquet, outputs/forecast_submission.csv, outputs/forecast_intervals.csv")
    st.caption("Historical monthly demand and 19-month forward forecast across units and revenue on a continuous timeline")

    df_t = apply_filters(df_train, filters)
    df_f = apply_filters(df_forecast, filters)
    unfiltered_forecast_total = int(df_forecast['Units_Sold_Forecast'].sum()) if not df_forecast.empty else 0
    def _universe(col):
        values = set(df_train[col].dropna().unique()) if col in df_train.columns else set()
        if col in df_forecast.columns:
            values |= set(df_forecast[col].dropna().unique())
        return values
    all_filters = (
        set(filters.get('families', [])) == _universe('Product_Family')
        and set(filters.get('regions', [])) == _universe('Region')
        and set(filters.get('skus', [])) == _universe('SKU_ID')
    )
    headline_label = "All data" if all_filters else "Filtered"
    st.caption(f"{headline_label} forecast total: {int(df_f['Units_Sold_Forecast'].sum()) if not df_f.empty else 0:,} units")
    if not all_filters:
        st.caption(f"Unfiltered forecast total: {unfiltered_forecast_total:,} units")

    if df_t.empty and df_f.empty:
        st.warning("No data found matching the selected filter criteria.")
        return

    # Price map for projected revenue calculation
    latest_prices = (
        df_train.assign(_dt=pd.to_datetime(df_train['Date'] + '-01'))
        .sort_values(['SKU_ID', '_dt'])
        .drop_duplicates('SKU_ID', keep='last')
        .set_index('SKU_ID')['Unit_Price_USD']
        .to_dict()
        if not df_train.empty
        else {}
    )
    sku_price_map = latest_prices
    df_f_rev = df_f.copy()
    df_f_rev['Est_Unit_Price'] = df_f_rev['SKU_ID'].map(sku_price_map).fillna(100.0)
    df_f_rev['Est_Revenue'] = df_f_rev['Units_Sold_Forecast'] * df_f_rev['Est_Unit_Price']

    metric_mode = st.radio("Display Metric", ["Units Sold", "Revenue (USD)"], horizontal=True)

    if metric_mode == "Units Sold":
        hist = df_t.groupby('Date').agg(Val=('Units_Sold_num', 'sum')).reset_index().sort_values('Date') if not df_t.empty else pd.DataFrame(columns=['Date', 'Val'])
        fc = df_f.groupby('Date')['Units_Sold_Forecast'].sum().reset_index() if not df_f.empty else pd.DataFrame(columns=['Date', 'Val'])
        fc.columns = ['Date', 'Val']
        y_label = "Units"
    else:
        hist = df_t.groupby('Date').agg(Val=('Revenue_USD_num', 'sum')).reset_index().sort_values('Date') if not df_t.empty else pd.DataFrame(columns=['Date', 'Val'])
        fc = df_f_rev.groupby('Date')['Est_Revenue'].sum().reset_index() if not df_f_rev.empty else pd.DataFrame(columns=['Date', 'Val'])
        fc.columns = ['Date', 'Val']
        y_label = "Revenue (USD)"

    hist['Type'] = 'History'
    fc['Type'] = 'Forecast'
    combined = pd.concat([hist, fc], ignore_index=True).sort_values('Date')

    fig = go.Figure()
    hist_pts = combined[combined['Type'] == 'History']
    fc_pts = combined[combined['Type'] == 'Forecast']

    fig.add_trace(go.Scatter(
        x=hist_pts['Date'], y=hist_pts['Val'],
        mode='lines', name=f'Actual {metric_mode}',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fc_pts['Date'], y=fc_pts['Val'],
        mode='lines', name=f'Forecast (P50 {metric_mode})',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    if metric_mode == "Units Sold" and not df_intervals.empty:
        df_int_filtered = apply_filters(band_frame(df_intervals), filters)
        if not df_int_filtered.empty:
            int_monthly = df_int_filtered.groupby('Date').agg(
                p10=('Units_Sold_Forecast_p10', 'sum'),
                p90=('Units_Sold_Forecast_p90', 'sum')
            ).reset_index().sort_values('Date')
            fig.add_trace(go.Scatter(
                x=list(int_monthly['Date']) + list(reversed(int_monthly['Date'])),
                y=list(int_monthly['p90']) + list(reversed(int_monthly['p10'])),
                fill='toself', fillcolor='rgba(255,127,14,0.15)',
                line=dict(color='rgba(255,127,14,0)'),
                name='Forecast Interval (P10-P90)'
            ))
    elif metric_mode == "Revenue (USD)" and not df_intervals.empty:
        df_int_filtered = apply_filters(band_frame(df_intervals), filters).copy()
        if not df_int_filtered.empty:
            df_int_filtered['Est_P10_Price'] = df_int_filtered['SKU_ID'].map(sku_price_map).fillna(100.0)
            df_int_filtered['Est_P90_Price'] = df_int_filtered['Est_P10_Price']
            int_monthly = df_int_filtered.groupby('Date').agg(
                p10=('Units_Sold_Forecast_p10', 'sum'),
                p90=('Units_Sold_Forecast_p90', 'sum')
            ).reset_index()
            int_monthly['p10'] = int_monthly['p10'] * df_int_filtered.groupby('Date')['Est_P10_Price'].mean().reindex(int_monthly['Date']).fillna(100.0).values
            int_monthly['p90'] = int_monthly['p90'] * df_int_filtered.groupby('Date')['Est_P90_Price'].mean().reindex(int_monthly['Date']).fillna(100.0).values
            fig.add_trace(go.Scatter(
                x=list(int_monthly['Date']) + list(reversed(int_monthly['Date'])),
                y=list(int_monthly['p90']) + list(reversed(int_monthly['p10'])),
                fill='toself', fillcolor='rgba(255,127,14,0.15)',
                line=dict(color='rgba(255,127,14,0)'), name='Revenue Interval'
            ))

    fig.add_vline(x='2026-10', line_dash='dot', line_color='gray')
    fig.add_annotation(
        x='2026-10', y=1, yref='paper',
        text='Forecast Origin (2026-10)', showarrow=False,
        xanchor='left', font=dict(color='gray', size=11)
    )
    fig.update_layout(title=f'Monthly {metric_mode} Timeline (History & 19-Month Forecast)', xaxis_title='Date', yaxis_title=y_label, height=420)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"Historical Total ({metric_mode})", f"{int(hist_pts['Val'].sum()):,}")
    with col2:
        forecast_headline = unfiltered_forecast_total if all_filters else int(fc_pts['Val'].sum())
        st.metric(f"19-Month Forecast Total", f"{forecast_headline:,}")
    with col3:
        last12 = hist_pts.tail(12)['Val'].sum()
        st.metric(f"Trailing 12-Month Total", f"{int(last12):,}")
    with col4:
        pct_growth = ((fc_pts['Val'].sum() / (last12 * (19 / 12))) - 1) * 100 if last12 > 0 else 0
        st.metric("Forecast vs Trailing Run-Rate", f"{pct_growth:+.1f}%")

    st.subheader("Commercial Mix Breakdown (Forecast vs History)")
    mix_col1, mix_col2, mix_col3 = st.columns(3)
    with mix_col1:
        if 'Product_Family' in df_t.columns:
            fam_mix = df_t.groupby('Product_Family')['Units_Sold_num'].sum().reset_index()
            fig_fam = px.bar(fam_mix, x='Product_Family', y='Units_Sold_num', title='Historical Volume by Family')
            fig_fam.update_layout(xaxis_title='Product Family', yaxis_title='Historical Units Sold')
            st.plotly_chart(fig_fam, use_container_width=True)
    with mix_col2:
        if 'Region' in df_f.columns:
            reg_mix = df_f.groupby('Region')['Units_Sold_Forecast'].sum().reset_index()
            fig_reg = px.bar(reg_mix, x='Region', y='Units_Sold_Forecast', title='Forecast Volume by Region')
            fig_reg.update_layout(xaxis_title='Region', yaxis_title='Forecast Units')
            st.plotly_chart(fig_reg, use_container_width=True)
    with mix_col3:
        if 'Sales_Channel' in df_t.columns:
            ch_mix = df_t.groupby('Sales_Channel')['Units_Sold_num'].sum().reset_index()
            fig_ch = px.pie(ch_mix, names='Sales_Channel', values='Units_Sold_num', title='Channel Distribution')
            st.plotly_chart(fig_ch, use_container_width=True)


# 2. Data Health and Audit
def view_data_health(dq_report):
    st.header("2. Data Health and Audit Log")
    st.caption("Source: outputs/data_quality_report.json, outputs/data_quality_report.md, outputs/audit_log.csv")
    st.caption("Verification of automated cleaning rules, data transformations, and decision tracking")

    if not dq_report:
        st.warning("Data quality report not available.")
        return

    summary = dq_report.get('summary', {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Audit Log Records", f"{summary.get('total_audit_entries', 0):,}")
    with col2:
        st.metric("Registered Anomalies", f"{summary.get('total_anomalies', 0):,}")

    st.subheader("Cleaning Rules Applied to Sales History")
    train_issues = dq_report.get('train_issues', {})
    rows = []
    for key, val in train_issues.items():
        rows.append({
            'Rule ID': val.get('rule', key),
            'Description': val.get('description', ''),
            'Details': str(val),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    rule_counts = dq_report.get('rules', {})
    if rule_counts:
        st.subheader("Rule Coverage")
        rule_rows = [
            {
                'Rule ID': rule_id,
                'Before': info.get('before_count', 0),
                'After': info.get('after_count', 0),
                'Audit Entries': info.get('audit_entries', 0),
            }
            for rule_id, info in rule_counts.items()
        ]
        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

    st.subheader("Documented Decision Rules")
    decisions = dq_report.get('decision_rules_applied', {})
    for k, v in decisions.items():
        with st.expander(f"Rule {v.get('rule_id', '')}: {k}"):
            st.write(f"**Records affected:** {v.get('count', 'N/A')}")
            st.write(f"**Decision rule:** {v.get('decision', 'N/A')}")
            st.write(f"**Justification:** {v.get('reason', 'N/A')}")


# 3. Demand Explorer
def view_demand_explorer(df_train, filters):
    st.header("3. Demand Explorer")
    st.caption("Source: outputs/clean_train.parquet")
    st.caption("Multidimensional slice-and-dice across families, regions, and customers")

    df = apply_filters(df_train, filters)
    if df.empty:
        st.warning("No data matching selection.")
        return

    dimension_options = {
        'Product family': 'Product_Family',
        'Region': 'Region',
        'Sales channel': 'Sales_Channel',
        'Customer name': 'Customer_Name',
    }
    display_dimension = st.selectbox("Group by", list(dimension_options.keys()))
    if display_dimension in dimension_options:
        group_col = dimension_options[display_dimension]
    elif display_dimension in dimension_options.values():
        group_col = display_dimension
    else:
        group_col = 'Product_Family'
    df = df.rename(columns={'Units_Sold_num': 'Units sold'})
    display_group = 'Product family' if group_col == 'Product_Family' else group_col
    if display_group == 'Product family':
        df['Product family'] = df['Product_Family']
        group_col = 'Product family'
    grouped = df.groupby(['Date', group_col])['Units sold'].sum().reset_index()

    fig = px.line(grouped, x='Date', y='Units sold', color=group_col,
                  title=f'Historical Demand by {group_col}', height=450)
    st.plotly_chart(fig, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Seasonality Profile")
        df['Month_Num'] = df['Date'].str[5:7].astype(int)
        seas = df.groupby('Month_Num')['Units sold'].mean().reset_index()
        fig_seas = px.bar(seas, x='Month_Num', y='Units sold', title='Average Demand by Calendar Month')
        fig_seas.update_layout(xaxis_title='Month of Year (1-12)', yaxis_title='Average Units Sold')
        st.plotly_chart(fig_seas, use_container_width=True)

    with col_s2:
        st.subheader("Zero-Demand Transaction Profile")
        zero_df = df[df['Units sold'] == 0].groupby('Date')['SKU_ID'].count().reset_index()
        if not zero_df.empty:
            fig_zeros = px.bar(zero_df, x='Date', y='SKU_ID', labels={'SKU_ID': 'Zero-order rows'}, title='Zero-order rows over time')
            fig_zeros.update_layout(xaxis_title='Date', yaxis_title='Zero-order rows')
            st.plotly_chart(fig_zeros, use_container_width=True)


# 4. Anomaly Monitor
def view_anomaly_monitor(df_train, anomaly_df, filters):
    st.header("4. Anomaly Monitor")
    st.caption("Source: outputs/anomaly_register.csv, outputs/clean_train.parquet")
    st.caption("Historical demand overlaid with structural breaks, market shocks, and high-volume events")

    df = apply_filters(df_train, filters)
    if anomaly_df.empty:
        st.info("No anomalies registered.")
        return

    type_counts = anomaly_df['anomaly_type'].value_counts().reset_index()
    type_counts.columns = ['Anomaly Type', 'Count']

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_pie = px.pie(type_counts, values='Count', names='Anomaly Type', title='Distribution of Anomaly Types')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        monthly = df.groupby('Date')['Units_Sold_num'].sum().reset_index()
        fig_line = px.line(monthly, x='Date', y='Units_Sold_num', title='Monthly Demand Trend with Anomaly Overlay')
        fig_line.update_layout(xaxis_title='Date', yaxis_title='Units Sold')

        # Add anomaly markers
        anom_dates = anomaly_df[anomaly_df['Date'].isin(monthly['Date'])].groupby('Date').first().reset_index()
        anom_merged = anom_dates.merge(monthly, on='Date', how='inner')
        if not anom_merged.empty:
            fig_line.add_trace(go.Scatter(
                x=anom_merged['Date'], y=anom_merged['Units_Sold_num'],
                mode='markers', name='Flagged Event / Anomaly',
                marker=dict(color='red', size=8, symbol='diamond'),
                hovertext=anom_merged['reason']
            ))
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Anomaly Register Records")
    display_cols = [c for c in ['Date', 'SKU_ID', 'Region', 'anomaly_type', 'field', 'value', 'action', 'reason'] if c in anomaly_df.columns]
    st.dataframe(anomaly_df[display_cols].head(300), use_container_width=True, hide_index=True)


# 5. Forecast and Accuracy
def view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters):
    st.header("5. Forecast and Model Accuracy")
    st.caption("Source: outputs/backtest_metrics.csv, outputs/coverage_breakdown.csv")
    st.caption("Rolling-origin backtest metrics across 19-month horizons, model comparisons, empirical coverage, and prediction intervals")

    if backtest_df.empty:
        st.warning("Backtest metrics not found.")
        return

    st.subheader("Model used for each product family")
    st.caption("Source: outputs/family_model_selection.csv. Each family uses the model with the lowest mean WAPE "
               "over the selection origins 2023-10 and 2024-10. The 2025-03 origin is the holdout and was not used for the choice.")
    fam_sel = load_family_selection()
    if not fam_sel.empty:
        show = fam_sel[['Product_Family', 'Selected_Model']].copy()
        show['Selection WAPE'] = [r[f"Selection_WAPE_{r['Selected_Model']}"] for _, r in fam_sel.iterrows()]
        show['Holdout WAPE 2025-03'] = fam_sel['Holdout_WAPE_selected'].values
        show['Holdout WAPE, seasonal naive'] = fam_sel['Holdout_WAPE_naive_seasonal'].values
        show['Holdout WAPE, LightGBM'] = fam_sel['Holdout_WAPE_lightgbm_p50'].values
        st.dataframe(show.round(4), use_container_width=True, hide_index=True)
        st.caption("Families forecast by a baseline have no P10 or P90 band. Coverage figures below apply to LightGBM rows only.")

    st.subheader("Rolling-Origin Backtest Performance Summary")
    overall = backtest_df[backtest_df['breakdown'] == 'overall']
    display_ov = [c for c in ['origin', 'model', 'WAPE', 'Bias', 'coverage_raw', 'coverage_calibrated', 'n_obs'] if c in overall.columns]
    st.dataframe(overall[display_ov], use_container_width=True, hide_index=True)

    # Coverage breakdown table
    cov_df = load_coverage_breakdown()
    if not cov_df.empty:
        st.subheader("Empirical Interval Coverage Breakdown (P10 - P90)")
        st.caption("Raw quantile coverage vs out-of-sample calibrated coverage by origin, horizon, and family")
        cov_df = cov_df.copy()
        cov_df.loc[cov_df['breakdown'] == 'overall', 'horizon_month'] = 'all'
        st.dataframe(cov_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forecast Error Degradation Across Horizon")
        horizons = backtest_df[(backtest_df['breakdown'] == 'horizon') & (backtest_df['model'] == 'lightgbm_p50')].sort_values('horizon_month')
        if not horizons.empty:
            fig_hor = px.line(horizons, x='horizon_month', y='WAPE', color='origin', title='WAPE by Horizon Month')
            fig_hor.update_layout(xaxis_title='Forecast Horizon (Month)', yaxis_title='WAPE')
            st.plotly_chart(fig_hor, use_container_width=True)

    with col2:
        st.subheader("LightGBM P50 mean bias (predicted minus actual), units")
        st.caption("Negative bias means under-forecasting; positive bias means over-forecasting.")
        fam_bias = backtest_df[(backtest_df['breakdown'] == 'family') & (backtest_df['model'] == 'lightgbm_p50')]
        if not fam_bias.empty:
            fig_bias = px.bar(fam_bias, x='group', y='Bias', color='origin', barmode='group', title='Mean Bias (Pred - Actual) by Family')
            fig_bias.update_layout(xaxis_title='Product Family', yaxis_title='Mean Bias (Units)')
            st.plotly_chart(fig_bias, use_container_width=True)

    reg_metrics = backtest_df[(backtest_df['breakdown'] == 'region') & (backtest_df['model'] == 'lightgbm_p50')]
    if not reg_metrics.empty:
        st.subheader("Model Accuracy (WAPE) by Region")
        fig_reg = px.bar(reg_metrics, x='group', y='WAPE', color='origin', barmode='group', title='WAPE by Geographic Region')
        fig_reg.update_layout(xaxis_title='Geographic Region', yaxis_title='WAPE')
        st.plotly_chart(fig_reg, use_container_width=True)
        unassigned = reg_metrics[reg_metrics['group'] == 'Unassigned']
        if not unassigned.empty:
            wape_text = ", ".join(
                f"{row.origin}: {row.WAPE:.4f}" for row in unassigned.itertuples()
            )
            st.caption(
                "Unassigned rows are sales with no region in the source data. "
                f"Their forecast error is higher than the named regions. WAPE by origin: {wape_text}"
            )


# 6. Macro Lead/Lag Heatmap (Opt 1)
def view_macro_lag_heatmap(macro_lag_df):
    st.header("Opt 1: Macro Lead/Lag Analysis")
    st.caption("Source: outputs/macro_lag_analysis.csv")
    st.caption("Cross-correlation analysis between leading macroeconomic indicators and product family volumes at lags 0-6 months")

    if macro_lag_df.empty:
        st.warning("Macro lag analysis data not found.")
        return

    analysis_type = st.radio("Correlation view", ["Levels", "Changes"], horizontal=True)
    if analysis_type not in ("Levels", "Changes"):
        analysis_type = "Changes"
    if analysis_type == "Levels":
        st.caption("Levels correlations can be high because both series trend over time. Use the Changes view for a like-for-like comparison.")
    else:
        st.caption("First differences remove the shared trend, so this view compares month-to-month movements.")
    if 'Analysis_Type' in macro_lag_df.columns:
        filtered_lag = macro_lag_df[macro_lag_df['Analysis_Type'] == analysis_type]
        if not filtered_lag.empty:
            macro_lag_df = filtered_lag

    lags = sorted(macro_lag_df['Lag_Months'].unique())
    if not lags:
        st.dataframe(macro_lag_df.head(30), use_container_width=True, hide_index=True)
        return
    sel_lag = st.select_slider("Macro Lag Horizon (Months)", options=lags, value=0)

    sub = macro_lag_df[macro_lag_df['Lag_Months'] == sel_lag]
    if sub.empty:
        st.dataframe(macro_lag_df.head(30), use_container_width=True, hide_index=True)
        return
    pivot = sub.pivot(index='Product_Family', columns='MacroIndicator', values='Correlation')

    fig = px.imshow(pivot, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto', text_auto='.2f',
                    title=f'Macro Indicator Cross-Correlation Matrix (Lag = {sel_lag} Months)')
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


# 7. Stockouts and Lost Demand (Opt 2)
def view_stockouts(stockouts_df, unconstrained_df, filters):
    st.header("Opt 2: Stockouts and Unconstrained Demand")
    st.caption("Source: outputs/unconstrained_demand.parquet and outputs/data_quality_report.json")
    st.caption("Reconstruction of lost demand from inventory stockouts and unlisted disruption events")

    dq_path = OUTPUTS_DIR / 'data_quality_report.json'
    dq = {}
    if dq_path.exists():
        with open(dq_path, encoding='utf-8') as fh:
            dq = json.load(fh)
    stockout_summary = dq.get('stockouts', {})
    documented_stockouts = int(stockout_summary.get('documented_stockouts', 0))
    lost_units = int(stockout_summary.get('lost_units', 0))
    fill_rate = float(stockout_summary.get('fill_rate_pct', 0.0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documented Stockouts", f"{documented_stockouts:,}")
    with col2:
        st.metric("Total Lost Volume (Units)", f"{lost_units:,}")
    with col3:
        st.metric("Historical Order Fill Rate", f"{fill_rate:.1f}%")

    if not unconstrained_df.empty:
        uc = unconstrained_df.copy()
        uc['Units_Sold'] = pd.to_numeric(uc.get('Units_Sold', 0), errors='coerce').fillna(0)
        uc['Units_Sold_Unconstrained'] = pd.to_numeric(uc.get('Units_Sold_Unconstrained', 0), errors='coerce').fillna(0)
        so_monthly = uc.groupby('Date').agg(
            Shipped=('Units_Sold', 'sum'),
            Lost=('Units_Sold_Unconstrained', 'sum')
        ).reset_index()
        so_monthly['Lost'] = (so_monthly['Lost'] - so_monthly['Shipped']).clip(lower=0)
        fig_so = go.Figure()
        fig_so.add_bar(x=so_monthly['Date'], y=so_monthly['Shipped'], name='Shipped Demand', marker_color='#1f77b4')
        fig_so.add_bar(x=so_monthly['Date'], y=so_monthly['Lost'], name='Lost Demand (Stockout)', marker_color='#d62728')
        fig_so.update_layout(barmode='stack', title='Shipped vs Lost Units by Month', height=400, xaxis_title='Date', yaxis_title='Units')
        st.plotly_chart(fig_so, use_container_width=True)


# 8. What-If Macro Scenario (Opt 3)
def view_scenario(scenario_df, df_forecast, filters=None):
    st.header("Opt 3: Macroeconomic What-If Scenario Generator")
    st.caption("Source: outputs/whatif_scenarios.csv")
    st.caption("Interactive sensitivity analysis: simulate macroeconomic shocks and observe family-level demand shifts from model reruns. Only families forecast by LightGBM respond to the shock; baseline families stay fixed")

    all_scen = load_all_scenarios()
    if all_scen.empty:
        st.warning("Scenario analysis data not found (outputs/whatif_scenarios.csv).")
        return

    pmi_options = [-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0]
    pmi_shift = st.select_slider("ISM Manufacturing PMI Shock (Points)", options=pmi_options, value=-3.0)

    match_scen = all_scen[np.isclose(all_scen['Scenario_PMI_Delta'], pmi_shift)]
    if match_scen.empty:
        st.warning("No data found for selected scenario delta.")
        return

    fam_rows = match_scen[match_scen['Level'] == 'Product_Family'].copy()
    fam_filter = (filters or {}).get('families')
    all_families_selected = fam_filter is None or set(fam_filter) >= set(fam_rows['Group'].unique())
    if fam_filter is not None and not all_families_selected:
        fam_rows = fam_rows[fam_rows['Group'].isin(fam_filter)].copy()
    if not fam_rows.empty:
        delta_units = int(fam_rows['Shift_Units'].sum())
        base_units = int(fam_rows['Baseline_Units'].sum())
        pct_impact = round(delta_units / max(base_units, 1) * 100, 2)
        st.metric("Net Projected Impact on 19-Month Volume", f"{delta_units:+,} units ({pct_impact:+.2f}%)")
        st.caption("Net unconstrained volume impact across the 19-month horizon (2026-11 to 2028-05) from rerunning the forecasting model with shifted ISM Manufacturing PMI trajectory. The scenario is by product family, so only the family filter applies here.")

    if not fam_rows.empty:
        st.subheader(f"Family-Level Demand Shifts for {pmi_shift:+.1f} PMI Shock")
        fig_fam = px.bar(
            fam_rows, x='Group', y='Shift_Pct',
            labels={'Group': 'Product Family', 'Shift_Pct': 'Volume Shift (%)'},
            title=f'Projected Volume Shift (%) by Product Family ({pmi_shift:+.1f} PMI Delta)'
        )
        st.plotly_chart(fig_fam, use_container_width=True)

        display_cols = ['Group', 'Baseline_Units', 'Forecast_Units', 'Shift_Units', 'Shift_Pct']
        st.dataframe(fam_rows[display_cols].rename(columns={'Group': 'Product Family'}), use_container_width=True, hide_index=True)


# 9. Lifecycle and Powertrain Transition View (Opt 4)
def view_powertrain_lifecycle(ev_df):
    st.header("Opt 4: Lifecycle and Powertrain Transition View")
    st.caption("Source: outputs/ev_shift_analysis.csv")
    st.caption("Evolution of internal combustion (ICE) vs electric vehicle (EV) demand and technological transition rate")

    if ev_df.empty:
        st.warning("Powertrain transition analysis not found.")
        return

    sku_file = OUTPUTS_DIR / 'sku_master.csv'
    forecast_file = OUTPUTS_DIR / 'forecast_submission.csv'
    sku_master = pd.read_csv(sku_file) if sku_file.exists() else pd.DataFrame()
    forecast = pd.read_csv(forecast_file) if forecast_file.exists() else pd.DataFrame()

    st.subheader("Launch and end-of-life timeline")
    n_timeline = int(((sku_master['Launch_Date'].notna()) | (sku_master['EOL_Date'].notna())).sum()) if not sku_master.empty else 0
    st.caption(f"Timeline rows: {n_timeline} SKUs with a Launch_Date or EOL_Date in sku_master.csv.")
    if not sku_master.empty:
        timeline = sku_master[
            sku_master['Launch_Date'].fillna('').ne('') | sku_master['EOL_Date'].fillna('').ne('')
        ].copy()
        families = sorted(timeline['Product_Family'].dropna().unique())
        selected_family = st.selectbox("Family", families, key='lifecycle_family') if families else None
        if selected_family:
            timeline = timeline[timeline['Product_Family'] == selected_family]
        if not timeline.empty:
            timeline_rows = []
            timeline_fig = go.Figure()
            for _, row in timeline.iterrows():
                start = str(row['Launch_Date'])[:10] if pd.notna(row.get('Launch_Date')) else ''
                end = str(row['EOL_Date'])[:10] if pd.notna(row.get('EOL_Date')) else ''
                timeline_rows.append({
                    'SKU_ID': row['SKU_ID'],
                    'Launch_Date': start,
                    'EOL_Date': end,
                    'Product_Family': row.get('Product_Family', ''),
                })
                if start and end:
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                    timeline_fig.add_trace(go.Bar(
                        x=[(end_dt - start_dt).days],
                        y=[row['SKU_ID']],
                        orientation='h',
                        base=[start_dt],
                        name=row['SKU_ID'],
                        showlegend=False,
                    ))
            if len(timeline_fig.data) > 0:
                timeline_fig.update_layout(title='SKU lifecycle timeline', xaxis_title='Date', yaxis_title='SKU')
                st.plotly_chart(timeline_fig, use_container_width=True)
            st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)

    st.subheader("Replaces_SKU chains")
    if not sku_master.empty and 'Replaces_SKU' in sku_master.columns:
        chains = sku_master[
            sku_master['Replaces_SKU'].fillna('').astype(str).str.strip().ne('')
        ][['SKU_ID', 'Replaces_SKU', 'Launch_Date', 'EOL_Date']]
        if not chains.empty:
            predecessor = sku_master.set_index('SKU_ID')
            chain_rows = []
            for _, row in chains.iterrows():
                pred = predecessor.loc[row['Replaces_SKU']] if row['Replaces_SKU'] in predecessor.index else None
                chain_rows.append({
                    'Predecessor': row['Replaces_SKU'],
                    'Predecessor_Launch_Date': pred.get('Launch_Date', '') if pred is not None else '',
                    'Successor': row['SKU_ID'],
                    'Successor_Launch_Date': row['Launch_Date'],
                })
            st.dataframe(pd.DataFrame(chain_rows), use_container_width=True, hide_index=True)

    st.subheader("Forecast before and after EOL")
    if not forecast.empty and not sku_master.empty:
        sku_options = sorted(sku_master[sku_master['EOL_Date'].fillna('').ne('')]['SKU_ID'].unique())
        if sku_options:
            selected_sku = st.selectbox("SKU", sku_options, key='lifecycle_sku')
            sku_forecast = forecast[forecast['SKU_ID'] == selected_sku].copy()
            sku_forecast = sku_forecast.sort_values('Date')
            if not sku_forecast.empty:
                eol_date = sku_master.loc[sku_master['SKU_ID'] == selected_sku, 'EOL_Date'].iloc[0]
                fig_eol = px.line(sku_forecast, x='Date', y='Units_Sold_Forecast', title=f'Forecast for {selected_sku}')
                if pd.notna(eol_date):
                    fig_eol.add_vline(x=str(eol_date)[:7], line_dash='dot', line_color='gray')
                st.plotly_chart(fig_eol, use_container_width=True)

    st.subheader("EV Adoption Index Correlation Analysis")
    level = ev_df[ev_df['Metric_Type'] == 'EV_Adoption_vs_ICE_Regression']
    diff = ev_df[ev_df['Metric_Type'].astype(str).str.contains('First_Difference', na=False)]
    if not level.empty:
        level_row = level.iloc[0]
        st.caption(
            f"Level result, inflated by shared trend: coefficient {level_row.get('Coefficient', '')}, "
            f"p-value {level_row.get('P_Value', '')}, lag {level_row.get('Lag_Months', '')}"
        )
        st.dataframe(level, use_container_width=True, hide_index=True)
    if not diff.empty:
        st.subheader("First-difference EV adoption and ICE volume")
        diff_columns = [c for c in ['Metric_Type', 'Coefficient', 'P_Value', 'Lag_Months', 'Observations', 'Significant_At_5Pct'] if c in diff.columns]
        st.dataframe(diff[diff_columns], use_container_width=True, hide_index=True)
        if 'Significant_At_5Pct' in diff.columns:
            n_sig = int(diff['Significant_At_5Pct'].astype(str).str.lower().isin(['true', '1', 'yes']).sum())
            st.caption(f"{n_sig} of {len(diff)} lags are significant at 5% in the first-difference table.")


# Main Entry Point
def main():
    st.title("Celeris Automobili Demand Control Tower")
    st.caption("Operational Planning & Demand Intelligence Platform")

    df_train = load_clean_train()
    df_forecast, df_intervals = load_forecast()
    dq_report = load_dq_report()
    anomaly_df = load_anomaly_register()
    backtest_df = load_backtest_metrics()
    macro_lag_df = load_macro_lag()
    stockouts_df, unconstrained_df = load_stockouts()
    ev_df = load_ev_shift()
    scenario_df = load_scenario()

    if df_train.empty:
        st.error("Pipeline outputs not found. Run 'python project/run_pipeline.py' to generate artifacts.")
        st.stop()

    filters = sidebar_filters(df_train, df_forecast)

    tabs = st.tabs([
        "1. Executive Overview",
        "2. Data Health & Audit",
        "3. Demand Explorer",
        "4. Anomaly Monitor",
        "5. Forecast & Accuracy",
        "Opt 1: Macro Lead/Lag",
        "Opt 2: Stockouts & Lost Demand",
        "Opt 3: What-If Scenario",
        "Opt 4: Lifecycle & Powertrain"
    ])

    with tabs[0]:
        view_executive_overview(df_train, df_forecast, df_intervals, filters)
    with tabs[1]:
        view_data_health(dq_report)
    with tabs[2]:
        view_demand_explorer(df_train, filters)
    with tabs[3]:
        view_anomaly_monitor(df_train, anomaly_df, filters)
    with tabs[4]:
        view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters)
    with tabs[5]:
        view_macro_lag_heatmap(macro_lag_df)
    with tabs[6]:
        view_stockouts(stockouts_df, unconstrained_df, filters)
    with tabs[7]:
        view_scenario(scenario_df, df_forecast, filters)
    with tabs[8]:
        view_powertrain_lifecycle(ev_df)


if __name__ == '__main__':
    main()