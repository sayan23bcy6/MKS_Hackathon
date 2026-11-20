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
    sub = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()
    intervals = pd.read_csv(int_path) if int_path.exists() else pd.DataFrame()
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
    so_path = OUTPUTS_DIR / "inventory_stockouts.csv"
    uc_path = OUTPUTS_DIR / "unconstrained_demand.parquet"
    stockouts = pd.read_csv(so_path) if so_path.exists() else pd.DataFrame()
    unconstrained = pd.read_parquet(uc_path) if uc_path.exists() else pd.DataFrame()
    return stockouts, unconstrained


@st.cache_data
def load_ev_shift():
    path = OUTPUTS_DIR / "ev_shift_analysis.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_scenario():
    path = OUTPUTS_DIR / "whatif_scenario_forecast.csv"
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
    sel_skus = st.sidebar.multiselect("SKU ID", options=all_skus, default=[], key='sku', help="Select specific SKUs or leave empty for all")
    sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions, key='region')
    sel_channels = st.sidebar.multiselect("Sales Channel", all_channels, default=all_channels, key='channel')
    sel_customers = st.sidebar.multiselect("Customer Name", options=all_customers, default=[], key='customer', help="Select specific customers or leave empty for all")

    return {
        'families': sel_families or all_families,
        'skus': sel_skus or all_skus,
        'regions': sel_regions or all_regions,
        'channels': sel_channels or all_channels,
        'customers': sel_customers or all_customers,
    }


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
    st.caption("Historical monthly demand and 19-month forward forecast on a consolidated timeline")

    df_t = apply_filters(df_train, filters)
    df_f = apply_filters(df_forecast, filters)

    if df_t.empty and df_f.empty:
        st.warning("No data found matching the selected filter criteria.")
        return

    hist = df_t.groupby('Date').agg(Units=('Units_Sold_num', 'sum')).reset_index().sort_values('Date') if not df_t.empty else pd.DataFrame(columns=['Date', 'Units'])
    hist['Type'] = 'History'

    fc = df_f.groupby('Date')['Units_Sold_Forecast'].sum().reset_index() if not df_f.empty else pd.DataFrame(columns=['Date', 'Units'])
    fc.columns = ['Date', 'Units']
    fc['Type'] = 'Forecast'

    combined = pd.concat([hist, fc], ignore_index=True).sort_values('Date')

    fig = go.Figure()
    hist_pts = combined[combined['Type'] == 'History']
    fc_pts = combined[combined['Type'] == 'Forecast']

    fig.add_trace(go.Scatter(
        x=hist_pts['Date'], y=hist_pts['Units'],
        mode='lines', name='Actual Shipments',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fc_pts['Date'], y=fc_pts['Units'],
        mode='lines', name='Forecast (P50 Point Estimate)',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    if not df_intervals.empty:
        df_int_filtered = df_intervals[df_intervals['Region'].isin(filters['regions'])]
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

    fig.add_vline(x='2026-10', line_dash='dot', line_color='gray')
    fig.add_annotation(
        x='2026-10', y=1, yref='paper',
        text='Forecast Origin (2026-10)', showarrow=False,
        xanchor='left', font=dict(color='gray', size=11)
    )
    fig.update_layout(title='Monthly Units Demand (History and 19-Month Forecast)', xaxis_title='Date', yaxis_title='Units', height=420)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Volume", f"{int(hist_pts['Units'].sum()):,}")
    with col2:
        st.metric("19-Month Forecast Total", f"{int(fc_pts['Units'].sum()):,}")
    with col3:
        last12 = hist_pts.tail(12)['Units'].sum()
        st.metric("Trailing 12-Month Volume", f"{int(last12):,}")
    with col4:
        pct_growth = ((fc_pts['Units'].sum() / (last12 * (19 / 12))) - 1) * 100 if last12 > 0 else 0
        st.metric("Annualized Run-Rate Shift", f"{pct_growth:+.1f}%")


# 2. Data Health and Audit
def view_data_health(dq_report):
    st.header("2. Data Health and Audit Log")
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
    st.caption("Multidimensional slice-and-dice across families, regions, and customers")

    df = apply_filters(df_train, filters)
    if df.empty:
        st.warning("No data matching selection.")
        return

    group_col = st.selectbox("Group By Dimension", ['Product_Family', 'Region', 'Sales_Channel', 'Customer_Name'])
    grouped = df.groupby(['Date', group_col])['Units_Sold_num'].sum().reset_index()

    fig = px.line(grouped, x='Date', y='Units_Sold_num', color=group_col,
                  title=f'Historical Demand by {group_col}', height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonality Profile")
    df['Month_Num'] = df['Date'].str[5:7].astype(int)
    seas = df.groupby('Month_Num')['Units_Sold_num'].mean().reset_index()
    fig_seas = px.bar(seas, x='Month_Num', y='Units_Sold_num', title='Average Demand by Calendar Month')
    st.plotly_chart(fig_seas, use_container_width=True)


# 4. Anomaly Monitor
def view_anomaly_monitor(df_train, anomaly_df, filters):
    st.header("4. Anomaly Monitor")
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
        fig_line = px.line(monthly, x='Date', y='Units_Sold_num', title='Monthly Demand Trend')
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Anomaly Register Records")
    display_cols = [c for c in ['Date', 'SKU_ID', 'Region', 'anomaly_type', 'field', 'value', 'action', 'reason'] if c in anomaly_df.columns]
    st.dataframe(anomaly_df[display_cols].head(300), use_container_width=True, hide_index=True)


# 5. Forecast and Accuracy
def view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters):
    st.header("5. Forecast and Model Accuracy")
    st.caption("Rolling-origin backtest metrics across 19-month horizons, model comparisons, and prediction intervals")

    if backtest_df.empty:
        st.warning("Backtest metrics not found.")
        return

    st.subheader("Rolling-Origin Backtest Performance Summary")
    overall = backtest_df[backtest_df['breakdown'] == 'overall']
    st.dataframe(overall[['origin', 'model', 'WAPE', 'Bias', 'n_obs']], use_container_width=True, hide_index=True)

    st.subheader("Forecast Error Degradation Across Horizon (Months 1-19)")
    horizons = backtest_df[(backtest_df['breakdown'] == 'horizon') & (backtest_df['model'] == 'lightgbm_p50')].sort_values('horizon_month')
    if not horizons.empty:
        fig_hor = px.line(horizons, x='horizon_month', y='WAPE', color='origin', title='WAPE by Horizon Month across Evaluation Origins')
        st.plotly_chart(fig_hor, use_container_width=True)

    st.subheader("Model Performance by Product Family")
    fam_metrics = backtest_df[(backtest_df['breakdown'] == 'family') & (backtest_df['model'] == 'lightgbm_p50')]
    if not fam_metrics.empty:
        fig_fam = px.bar(fam_metrics, x='group', y='WAPE', color='origin', barmode='group', title='WAPE by Product Family')
        st.plotly_chart(fig_fam, use_container_width=True)


# 6. Macro Lead/Lag Heatmap
def view_macro_lag_heatmap(macro_lag_df):
    st.header("6. Macroeconomic Lead/Lag Analysis")
    st.caption("Cross-correlation analysis between leading macroeconomic indicators and product family volumes at lags 0-6 months")

    if macro_lag_df.empty:
        st.warning("Macro lag analysis data not found.")
        return

    lags = sorted(macro_lag_df['Lag_Months'].unique())
    sel_lag = st.select_slider("Macro Lag Horizon (Months)", options=lags, value=0)

    sub = macro_lag_df[macro_lag_df['Lag_Months'] == sel_lag]
    pivot = sub.pivot(index='Product_Family', columns='MacroIndicator', values='Correlation')

    fig = px.imshow(pivot, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto', text_auto='.2f',
                    title=f'Macro Indicator Cross-Correlation Matrix (Lag = {sel_lag} Months)')
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


# 7. Stockouts and Lost Demand
def view_stockouts(stockouts_df, unconstrained_df, filters):
    st.header("7. Stockouts and Unconstrained Demand")
    st.caption("Reconstruction of lost demand from inventory stockouts and unlisted disruption events")

    if stockouts_df.empty:
        st.warning("Stockout data not available.")
        return

    so = stockouts_df.copy()
    for col in ['Requested_Units', 'Shipped_Units', 'Lost_Units', 'Fill_Rate_Pct']:
        so[col] = pd.to_numeric(so[col], errors='coerce').fillna(0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documented Stockouts", f"{len(so):,}")
    with col2:
        st.metric("Total Lost Volume (Units)", f"{int(so['Lost_Units'].sum()):,}")
    with col3:
        st.metric("Historical Order Fill Rate", f"{so['Fill_Rate_Pct'].mean():.1f}%")

    so_monthly = so.groupby('Date').agg(Shipped=('Shipped_Units', 'sum'), Lost=('Lost_Units', 'sum')).reset_index()
    fig_so = go.Figure()
    fig_so.add_bar(x=so_monthly['Date'], y=so_monthly['Shipped'], name='Shipped Demand', marker_color='#1f77b4')
    fig_so.add_bar(x=so_monthly['Date'], y=so_monthly['Lost'], name='Lost Demand (Stockout)', marker_color='#d62728')
    fig_so.update_layout(barmode='stack', title='Shipped vs Lost Units by Month', height=400)
    st.plotly_chart(fig_so, use_container_width=True)


# 8. What-If Macro Scenario
def view_scenario(scenario_df, df_forecast):
    st.header("8. Macroeconomic What-If Scenario Generator")
    st.caption("Interactive sensitivity analysis: simulate macroeconomic shocks and observe family-level demand shifts")

    pmi_shift = st.slider("ISM Manufacturing PMI Shock (Points)", min_value=-6.0, max_value=6.0, value=-3.0, step=0.5)

    if scenario_df.empty:
        st.warning("Scenario forecast data not found.")
        return

    scen = scenario_df.copy()
    scen_monthly = scen.groupby('Date').agg(
        Base=('Units_Sold_Forecast', 'sum'),
        Shifted=('Scenario_Units_Sold_Forecast', 'sum')
    ).reset_index()

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=scen_monthly['Date'], y=scen_monthly['Base'], mode='lines', name='Baseline Forecast', line=dict(color='#1f77b4', width=2)))
    fig_scen.add_trace(go.Scatter(x=scen_monthly['Date'], y=scen_monthly['Shifted'], mode='lines', name=f'Shifted Forecast ({pmi_shift:+.1f} PMI)', line=dict(color='#d62728', width=2, dash='dot')))
    fig_scen.update_layout(title='Baseline vs Scenario Forecast', xaxis_title='Date', yaxis_title='Units', height=420)
    st.plotly_chart(fig_scen, use_container_width=True)

    delta_units = scen_monthly['Shifted'].sum() - scen_monthly['Base'].sum()
    pct_impact = (delta_units / scen_monthly['Base'].sum()) * 100
    st.metric("Net Projected Impact on 19-Month Volume", f"{int(delta_units):+,} units ({pct_impact:+.2f}%)")


# 9. Lifecycle & Powertrain Transition View
def view_powertrain_lifecycle(ev_df):
    st.header("9. Lifecycle and Powertrain Transition View")
    st.caption("Evolution of internal combustion (ICE) vs electric vehicle (EV) demand and technological transition rate")

    if ev_df.empty:
        st.warning("Powertrain transition analysis not found.")
        return

    st.subheader("EV Adoption Index Correlation Analysis")
    st.dataframe(ev_df, use_container_width=True, hide_index=True)


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
        "6. Macro Lead/Lag",
        "7. Stockouts & Lost Demand",
        "8. What-If Scenario",
        "9. Lifecycle & Powertrain"
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
        view_scenario(scenario_df, df_forecast)
    with tabs[8]:
        view_powertrain_lifecycle(ev_df)


if __name__ == '__main__':
    main()
