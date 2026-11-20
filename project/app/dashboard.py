"""
app/dashboard.py - Celeris Automobili Demand Forecasting Control Tower
Streamlit dashboard with 5 required views + 2 optional (Advanced tier).
Reads ONLY from project/outputs/.
All views share the same filters: family, SKU, region, channel, customer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pathlib

# --- Page config ---
st.set_page_config(
    page_title="Celeris Automobili - Demand Control Tower",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Paths ---
PROJECT_DIR = pathlib.Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# --- Data loaders ---
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
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
    else:
        sub = pd.DataFrame()
    if int_path.exists():
        intervals = pd.read_csv(int_path)
    else:
        intervals = pd.DataFrame()
    return sub, intervals


@st.cache_data
def load_dq_report():
    path = OUTPUTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_anomaly_register():
    path = OUTPUTS_DIR / "anomaly_register.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_backtest_metrics():
    path = OUTPUTS_DIR / "backtest_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_macro_lag():
    path = OUTPUTS_DIR / "macro_lag_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_stockouts():
    # Read raw stockouts
    so_path = PROJECT_DIR.parent / "student_data" / "inventory_stockouts.csv"
    uc_path = OUTPUTS_DIR / "unconstrained_demand.parquet"
    stockouts = pd.read_csv(so_path, low_memory=False) if so_path.exists() else pd.DataFrame()
    unconstrained = pd.read_parquet(uc_path) if uc_path.exists() else pd.DataFrame()
    return stockouts, unconstrained


# --- Shared filters ---
def _sidebar_filters(df_train: pd.DataFrame, df_forecast: pd.DataFrame) -> dict:
    """Return a dict of selected filter values."""
    st.sidebar.header("Filters")
    st.sidebar.caption("All views use these filters")

    # Combine both datasets for filter options
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
    sel_skus = st.sidebar.multiselect("SKU (leave blank for all)", [], key='sku')
    sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions, key='region')
    sel_channels = st.sidebar.multiselect("Channel", all_channels, default=all_channels, key='channel')
    sel_customers = st.sidebar.multiselect("Customer (leave blank for all)", [], key='customer')

    return {
        'families': sel_families or all_families,
        'skus': sel_skus or all_skus,
        'regions': sel_regions or all_regions,
        'channels': sel_channels or all_channels,
        'customers': sel_customers or all_customers,
    }


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply shared filters to any DataFrame."""
    if df.empty:
        return df
    result = df.copy()
    if 'Product_Family' in result.columns:
        result = result[result['Product_Family'].isin(filters['families'])]
    if 'SKU_ID' in result.columns and filters['skus']:
        result = result[result['SKU_ID'].isin(filters['skus'])]
    if 'Region' in result.columns:
        result = result[result['Region'].isin(filters['regions'])]
    if 'Sales_Channel' in result.columns and filters['channels']:
        result = result[result['Sales_Channel'].isin(filters['channels'])]
    if 'Customer_Name' in result.columns and filters['customers']:
        result = result[result['Customer_Name'].isin(filters['customers'])]
    return result


# --- View 1: Executive Overview ---
def view_executive_overview(df_train, df_forecast, df_intervals, filters):
    st.header("1. Executive Overview")
    st.caption("Historical performance and 19-month forward forecast on a single timeline")

    df_t = _apply_filters(df_train, filters)
    df_f = _apply_filters(df_forecast, filters)

    if df_t.empty and df_f.empty:
        st.warning("No data for selected filters.")
        return

    # Aggregate history monthly
    if not df_t.empty:
        hist = df_t.groupby('Date').agg(
            Units=('Units_Sold_num', 'sum'),
            Revenue=('Revenue_USD_num', 'sum')
        ).reset_index().sort_values('Date')
        hist['Type'] = 'History'
    else:
        hist = pd.DataFrame(columns=['Date', 'Units', 'Revenue', 'Type'])

    # Aggregate forecast monthly
    if not df_f.empty:
        fc = df_f.groupby('Date')['Units_Sold_Forecast'].sum().reset_index()
        fc.columns = ['Date', 'Units']
        fc['Revenue'] = np.nan
        fc['Type'] = 'Forecast'
    else:
        fc = pd.DataFrame(columns=['Date', 'Units', 'Revenue', 'Type'])

    combined = pd.concat([hist, fc], ignore_index=True).sort_values('Date')

    # Main timeline chart
    fig = go.Figure()
    hist_data = combined[combined['Type'] == 'History']
    fc_data = combined[combined['Type'] == 'Forecast']

    fig.add_trace(go.Scatter(
        x=hist_data['Date'], y=hist_data['Units'],
        mode='lines', name='History (Actual)',
        line=dict(color='#2196F3', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fc_data['Date'], y=fc_data['Units'],
        mode='lines', name='Forecast (P50)',
        line=dict(color='#FF9800', width=2, dash='dash')
    ))

    # Add P10/P90 bands if available
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
                fill='toself', fillcolor='rgba(255,152,0,0.15)',
                line=dict(color='rgba(255,152,0,0)'),
                name='P10-P90 Band'
            ))

    fig.add_vline(x='2026-10', line_dash='dot', line_color='gray')
    fig.add_annotation(
        x='2026-10', y=1, yref='paper',
        text='Forecast start ->', showarrow=False,
        xanchor='left', font=dict(color='gray', size=11)
    )
    fig.update_layout(
        title='Units Sold (History + Forecast)',
        xaxis_title='Month', yaxis_title='Units',
        height=400, legend=dict(orientation='h')
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_hist = int(hist_data['Units'].sum())
        st.metric("Total Historical Units (all history)", f"{total_hist:,}")
    with col2:
        total_fc = int(fc_data['Units'].sum()) if not fc_data.empty else 0
        st.metric("19-Month Forecast Total", f"{total_fc:,}")
    with col3:
        last12 = hist_data.tail(12)['Units'].sum()
        st.metric("Last 12 Months Actual", f"{int(last12):,}")
    with col4:
        pct_chg = ((total_fc / last12) - 1) * 100 if last12 > 0 else 0
        st.metric("Forecast vs Last 12M", f"{pct_chg:+.1f}%")

    # Mix charts
    st.subheader("Mix Analysis")
    col_a, col_b, col_c = st.columns(3)

    if not df_t.empty:
        with col_a:
            fam_mix = df_t.groupby('Product_Family')['Units_Sold_num'].sum().reset_index()
            fig_fam = px.pie(fam_mix, values='Units_Sold_num', names='Product_Family',
                              title='Units by Family (History)')
            st.plotly_chart(fig_fam, use_container_width=True)

        with col_b:
            reg_mix = df_t.groupby('Region')['Units_Sold_num'].sum().reset_index()
            fig_reg = px.bar(reg_mix.sort_values('Units_Sold_num', ascending=True),
                              x='Units_Sold_num', y='Region', orientation='h',
                              title='Units by Region (History)', color='Units_Sold_num',
                              color_continuous_scale='Blues')
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_c:
            ch_mix = df_t.groupby('Sales_Channel')['Units_Sold_num'].sum().reset_index()
            fig_ch = px.pie(ch_mix, values='Units_Sold_num', names='Sales_Channel',
                             title='Units by Channel (History)')
            st.plotly_chart(fig_ch, use_container_width=True)


# --- View 2: Data Health ---
def view_data_health(dq_report, filters):
    st.header("2. Data Health and Audit")
    st.caption("Data quality issues detected, before vs. after cleaning")

    if not dq_report:
        st.warning("Data quality report not found. Run the pipeline first.")
        return

    summary = dq_report.get('summary', {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Audit Log Entries", f"{summary.get('total_audit_entries', 0):,}")
    with col2:
        st.metric("Anomalies Flagged", f"{summary.get('total_anomalies', 0):,}")

    st.subheader("Train.csv Issues")
    train_issues = dq_report.get('train_issues', {})
    rows = []
    for key, val in train_issues.items():
        rows.append({
            'Issue': key.upper(),
            'Description': val.get('description', ''),
            'Rule': val.get('rule', ''),
            'Count / Action': _summarise_dq_issue(val),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Visual: issues by count
    counts = {}
    for key, val in train_issues.items():
        # Try to find a numeric count
        for field in ['count', 'formats_fixed', 'rows_dropped', 'blank_count', 'normalised',
                       'bad_before', 'blank', 'dash', 'comma_formatted', 'negative',
                       'filled_zero_price_rows', 'discrepancy_before', 'eol_violations']:
            n = val.get(field, 0)
            if isinstance(n, int) and n > 0:
                counts[key] = n
                break

    if counts:
        fig = px.bar(
            pd.DataFrame(list(counts.items()), columns=['Issue', 'Rows Affected']),
            x='Issue', y='Rows Affected',
            title='Rows Affected by Issue Type',
            color='Rows Affected', color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Macro & SKU issues
    st.subheader("Macro Indicators Issues")
    mac_issues = dq_report.get('macro_issues', {})
    for k, v in mac_issues.items():
        if v:
            st.write(f"**{k}**: {v}")

    st.subheader("Unresolved Items")
    unresolved = dq_report.get('unresolved_items', {})
    for key, val in unresolved.items():
        with st.expander(f"Item: {key}"):
            st.write(f"**Count:** {val.get('count', 'N/A')}")
            st.write(f"**Decision:** {val.get('decision', 'N/A')}")
            st.write(f"**Reason:** {val.get('reason', 'N/A')}")


def _summarise_dq_issue(val: dict) -> str:
    """Create a one-line summary string for a DQ issue dict."""
    parts = []
    for field in ['count', 'formats_fixed', 'rows_dropped', 'rows_after',
                   'blank_count', 'blank_resolved', 'blank_unresolved',
                   'normalised', 'bad_before', 'filled', 'blank']:
        n = val.get(field)
        if isinstance(n, int):
            parts.append(f"{field}={n:,}")
    return '; '.join(parts) if parts else '-'


# --- View 3: Demand Explorer ---
def view_demand_explorer(df_train, filters):
    st.header("3. Demand Explorer")
    st.caption("Drill-down: family -> SKU -> region -> customer")

    df = _apply_filters(df_train, filters)
    if df.empty:
        st.warning("No data for selected filters.")
        return

    drill = st.selectbox("Group by", ['Product_Family', 'SKU_ID', 'Region', 'Customer_Name'])

    monthly = df.groupby(['Date', drill])['Units_Sold_num'].sum().reset_index()
    monthly.columns = ['Date', drill, 'Units_Sold']

    fig = px.line(monthly, x='Date', y='Units_Sold', color=drill,
                   title=f'Monthly Units by {drill}', height=450)
    fig.add_vline(x='2026-10', line_dash='dot', line_color='gray')
    st.plotly_chart(fig, use_container_width=True)

    # Seasonality: average by month
    st.subheader("Seasonality Profile")
    df['month'] = df['Date'].str[5:7].astype(int)
    seas = df.groupby('month')['Units_Sold_num'].mean().reset_index()
    seas.columns = ['Month', 'Avg_Units']
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    seas['Month_Name'] = seas['Month'].map(month_names)
    fig_seas = px.bar(seas, x='Month_Name', y='Avg_Units',
                       title='Average Monthly Demand (Seasonality)')
    st.plotly_chart(fig_seas, use_container_width=True)

    # Zero months
    st.subheader("Zero-Demand Months")
    zero_pct = df.groupby(drill).apply(
        lambda g: (g['Units_Sold_num'] == 0).mean() * 100
    ).reset_index()
    zero_pct.columns = [drill, 'Zero_Month_Pct']
    zero_pct = zero_pct.sort_values('Zero_Month_Pct', ascending=False).head(20)
    fig_zero = px.bar(zero_pct, x=drill, y='Zero_Month_Pct',
                       title=f'% Zero-Demand Months by {drill} (top 20)',
                       labels={'Zero_Month_Pct': '% Zero Months'})
    fig_zero.update_xaxes(tickangle=45)
    st.plotly_chart(fig_zero, use_container_width=True)

    # Lifecycle view
    st.subheader("Active SKUs per Month")
    active = df[df['Units_Sold_num'] > 0].groupby('Date')['SKU_ID'].nunique().reset_index()
    active.columns = ['Date', 'Active_SKUs']
    fig_lc = px.area(active, x='Date', y='Active_SKUs',
                      title='Number of SKUs with Positive Demand per Month')
    st.plotly_chart(fig_lc, use_container_width=True)


# --- View 4: Anomaly Monitor ---
def view_anomaly_monitor(df_train, anomaly_df, filters):
    st.header("4. Anomaly Monitor")
    st.caption("Flagged points on the time series with type, reason, and action")

    df = _apply_filters(df_train, filters)

    if anomaly_df.empty:
        st.info("No anomaly register found. Run the pipeline first.")
        return

    # Anomaly type distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        type_counts = anomaly_df['anomaly_type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        fig_types = px.pie(type_counts, values='Count', names='Type',
                            title='Anomaly Types')
        st.plotly_chart(fig_types, use_container_width=True)

    with col2:
        # Monthly units with anomaly overlay
        monthly = df.groupby('Date')['Units_Sold_num'].sum().reset_index()
        monthly.columns = ['Date', 'Units']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['Date'], y=monthly['Units'],
            mode='lines', name='Units Sold',
            line=dict(color='#2196F3')
        ))

        # Overlay anomaly events
        for atype in ['stockout', 'covid_disruption', 'early_buying', 'large_units', 'mea_before_market_entry']:
            an = anomaly_df[anomaly_df['anomaly_type'] == atype].copy()
            if not an.empty and 'Date' in an.columns:
                an_dates = an[an['Date'].str.match(r'\d{4}-\d{2}', na=False)]['Date'].unique()
                if len(an_dates):
                    an_monthly = monthly[monthly['Date'].isin(an_dates)]
                    fig.add_trace(go.Scatter(
                        x=an_monthly['Date'], y=an_monthly['Units'],
                        mode='markers', name=atype,
                        marker=dict(size=10, symbol='x')
                    ))

        fig.update_layout(title='Demand with Anomaly Flags', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.subheader("Anomaly Register")
    display_cols = ['Date', 'SKU_ID', 'Region', 'anomaly_type', 'field', 'value', 'action', 'reason']
    display_cols = [c for c in display_cols if c in anomaly_df.columns]
    an_filtered = anomaly_df.copy()
    if 'Region' in anomaly_df.columns:
        an_filtered = an_filtered[
            an_filtered['Region'].isin(filters['regions']) |
            an_filtered['Region'].isin(['', 'ALL'])
        ]
    st.dataframe(an_filtered[display_cols].head(200), use_container_width=True, hide_index=True)


# --- View 5: Forecast & Accuracy ---
def view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters):
    st.header("5. Forecast and Model Accuracy")
    st.caption("Forecast vs history with P10/P50/P90; backtest WAPE and bias")

    df_t = _apply_filters(df_train, filters)
    df_f = _apply_filters(df_forecast, filters)

    # Combined timeline
    if not df_t.empty:
        hist = df_t.groupby('Date')['Units_Sold_num'].sum().reset_index()
        hist.columns = ['Date', 'Units']
    else:
        hist = pd.DataFrame(columns=['Date', 'Units'])

    if not df_f.empty:
        fc = df_f.groupby('Date')['Units_Sold_Forecast'].sum().reset_index()
        fc.columns = ['Date', 'Units']
    else:
        fc = pd.DataFrame(columns=['Date', 'Units'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist['Date'], y=hist['Units'],
        mode='lines', name='Actual (History)',
        line=dict(color='#2196F3', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fc['Date'], y=fc['Units'],
        mode='lines', name='Forecast (P50)',
        line=dict(color='#FF9800', width=2, dash='dash')
    ))

    # P10/P90
    if not df_intervals.empty:
        df_int_f = df_intervals[df_intervals['Region'].isin(filters['regions'])]
        if not df_int_f.empty:
            int_m = df_int_f.groupby('Date').agg(
                p10=('Units_Sold_Forecast_p10', 'sum'),
                p90=('Units_Sold_Forecast_p90', 'sum')
            ).reset_index()
            fig.add_trace(go.Scatter(
                x=list(int_m['Date']) + list(reversed(int_m['Date'])),
                y=list(int_m['p90']) + list(reversed(int_m['p10'])),
                fill='toself', fillcolor='rgba(255,152,0,0.2)',
                line=dict(color='rgba(0,0,0,0)'), name='P10-P90 Band'
            ))

    fig.add_vline(x='2026-10', line_dash='dot', line_color='gray')
    fig.add_annotation(
        x='2026-10', y=1, yref='paper',
        text='<- History | Forecast ->', showarrow=False,
        xanchor='left', font=dict(color='gray', size=11)
    )
    fig.update_layout(title='Forecast vs History', xaxis_title='Month',
                       yaxis_title='Units', height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Backtest metrics
    st.subheader("Backtest Performance (19-Month Horizon)")
    if backtest_df.empty:
        st.info("Backtest metrics not available. Run the pipeline.")
        return

    # Overall table
    overall = backtest_df[backtest_df['breakdown'] == 'overall']
    st.dataframe(overall[['model', 'WAPE', 'Bias', 'n_obs']].round(4),
                 use_container_width=True, hide_index=True)

    # WAPE by family
    st.subheader("WAPE by Product Family (LightGBM)")
    lgbm_fam = backtest_df[
        (backtest_df['breakdown'] == 'family') &
        (backtest_df['model'] == 'lightgbm_p50')
    ]
    if not lgbm_fam.empty:
        fig_wape = px.bar(lgbm_fam.sort_values('WAPE'),
                           x='group', y='WAPE',
                           title='WAPE by Family', labels={'group': 'Family'})
        st.plotly_chart(fig_wape, use_container_width=True)

    # WAPE by horizon month
    st.subheader("WAPE by Forecast Horizon Month")
    lgbm_hor = backtest_df[
        (backtest_df['breakdown'] == 'horizon') &
        (backtest_df['model'] == 'lightgbm_p50')
    ].sort_values('horizon_month')
    if not lgbm_hor.empty:
        fig_hor = px.line(lgbm_hor, x='horizon_month', y='WAPE',
                           title='WAPE Degradation over Forecast Horizon',
                           labels={'horizon_month': 'Months Ahead'})
        st.plotly_chart(fig_hor, use_container_width=True)


# --- Optional View 1: Macro Lead/Lag Heatmap ---
def view_macro_lag_heatmap(macro_lag_df, filters):
    st.header("Macro Lead/Lag Analysis")
    st.caption("Cross-correlation of macro indicators vs product families at lags 0-6 months")

    if macro_lag_df.empty:
        st.info("Macro lag analysis not available. Run the pipeline.")
        return

    families = sorted(macro_lag_df['Product_Family'].unique())
    indicators = sorted(macro_lag_df['MacroIndicator'].unique())

    sel_family = st.selectbox("Select Product Family", families)
    sel_lag = st.slider("Lag (months ahead macro leads demand)", 0, 6, 0)

    subset = macro_lag_df[macro_lag_df['Lag_Months'] == sel_lag]
    pivot = subset.pivot(index='Product_Family', columns='MacroIndicator', values='Correlation')
    pivot = pivot.reindex(families)

    fig = px.imshow(
        pivot,
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        zmin=-1, zmax=1,
        title=f'Correlation of Macro Indicators with Demand (Lag = {sel_lag} months)',
        aspect='auto',
        text_auto='.2f'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Best lag per family
    st.subheader(f"Best Macro Indicator for: {sel_family}")
    fam_data = macro_lag_df[macro_lag_df['Product_Family'] == sel_family]
    best = fam_data.loc[fam_data['Correlation'].abs().idxmax()] if not fam_data.empty else None
    if best is not None:
        st.metric("Best Indicator", best['MacroIndicator'])
        st.metric("Best Lag", f"{int(best['Lag_Months'])} months")
        st.metric("Correlation", f"{best['Correlation']:.3f}")

    fig2 = px.line(
        fam_data, x='Lag_Months', y='Correlation', color='MacroIndicator',
        title=f'Correlation vs Lag for {sel_family}',
        labels={'Lag_Months': 'Lag (months)', 'Correlation': 'Pearson Correlation'}
    )
    fig2.add_hline(y=0, line_dash='dash', line_color='gray')
    st.plotly_chart(fig2, use_container_width=True)


# --- Optional View 5: Stockouts & Lost Sales ---
def view_stockouts(stockouts_df, unconstrained_df, filters):
    st.header("Stockouts and Lost Demand Analysis")
    st.caption("Requested vs Shipped vs Lost units from inventory_stockouts.csv")

    if stockouts_df.empty:
        st.info("Stockout data not available.")
        return

    so = stockouts_df.copy()
    for col in ['Requested_Units', 'Shipped_Units', 'Lost_Units', 'Fill_Rate_Pct']:
        so[col] = pd.to_numeric(so[col], errors='coerce').fillna(0)

    # Apply filters
    if 'Region' in so.columns:
        so = so[so['Region'].isin(filters['regions'])]

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stockout Rows", f"{len(so):,}")
    with col2:
        st.metric("Total Lost Units", f"{int(so['Lost_Units'].sum()):,}")
    with col3:
        avg_fill = so['Fill_Rate_Pct'].mean()
        st.metric("Avg Fill Rate", f"{avg_fill:.1f}%")

    # Monthly view
    so_monthly = so.groupby('Date').agg(
        Requested=('Requested_Units', 'sum'),
        Shipped=('Shipped_Units', 'sum'),
        Lost=('Lost_Units', 'sum')
    ).reset_index().sort_values('Date')

    fig = go.Figure()
    fig.add_bar(x=so_monthly['Date'], y=so_monthly['Shipped'], name='Shipped', marker_color='#2196F3')
    fig.add_bar(x=so_monthly['Date'], y=so_monthly['Lost'], name='Lost (Stockout)', marker_color='#F44336')
    fig.update_layout(barmode='stack', title='Requested vs Shipped vs Lost Units',
                       xaxis_title='Month', yaxis_title='Units', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # By reason
    st.subheader("Stockout by Reason")
    reason_df = so.groupby('Reason')['Lost_Units'].sum().reset_index().sort_values('Lost_Units', ascending=False)
    fig_r = px.bar(reason_df, x='Reason', y='Lost_Units', title='Lost Units by Stockout Reason',
                    color='Lost_Units', color_continuous_scale='Reds')
    st.plotly_chart(fig_r, use_container_width=True)

    # By region
    st.subheader("Stockout by Region")
    reg_df = so.groupby('Region')[['Requested_Units', 'Lost_Units']].sum().reset_index()
    reg_df['Fill_Rate'] = (1 - reg_df['Lost_Units'] / reg_df['Requested_Units'].replace(0, np.nan)) * 100
    st.dataframe(reg_df.round(1), use_container_width=True, hide_index=True)


# --- Main App ---
def main():
    st.title("Celeris Automobili - Demand Forecasting Control Tower")
    st.caption("Hackathon Project | History: 2019-01 to 2026-10 | Forecast: 2026-11 to 2028-05")

    # Load data
    df_train = load_clean_train()
    df_forecast, df_intervals = load_forecast()
    dq_report = load_dq_report()
    anomaly_df = load_anomaly_register()
    backtest_df = load_backtest_metrics()
    macro_lag_df = load_macro_lag()
    stockouts_df, unconstrained_df = load_stockouts()

    # Pipeline status
    if df_train.empty:
        st.error("Pipeline outputs not found. Run python project/run_pipeline.py to generate artifacts.")
        st.stop()

    # Shared filters
    filters = _sidebar_filters(df_train, df_forecast if not df_forecast.empty else pd.DataFrame())

    # Navigation tabs
    tabs = st.tabs([
        "Executive Overview",
        "Data Health",
        "Demand Explorer",
        "Anomaly Monitor",
        "Forecast and Accuracy",
        "Macro Lead/Lag",
        "Stockouts and Lost Demand"
    ])

    with tabs[0]:
        view_executive_overview(df_train, df_forecast, df_intervals, filters)
    with tabs[1]:
        view_data_health(dq_report, filters)
    with tabs[2]:
        view_demand_explorer(df_train, filters)
    with tabs[3]:
        view_anomaly_monitor(df_train, anomaly_df, filters)
    with tabs[4]:
        view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters)
    with tabs[5]:
        view_macro_lag_heatmap(macro_lag_df, filters)
    with tabs[6]:
        view_stockouts(stockouts_df, unconstrained_df, filters)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("All numbers read from `project/outputs/`")
    st.sidebar.caption("Pipeline: `python project/run_pipeline.py`")


if __name__ == '__main__':
    main()

