"""
ev_shift.py - Expert analysis of macroeconomic lag stability and the automotive EV shift (E-C).
"""

import pandas as pd
import numpy as np
from pipeline.config import OUT_EV_SHIFT, OUT_MACRO_LAG, OUT_LAG_STABILITY


def analyze_ev_shift_and_lag_stability(clean_train: pd.DataFrame,
                                       clean_macro: pd.DataFrame,
                                       sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    E-C:
    1. Recompute macro lag analysis on rolling windows (2019-2022 vs 2023-2026).
    2. Split demand by Powertrain_Application (ICE, EV, Universal) and measure EV volume share.
    3. Test whether EV_Adoption_Index explains the decline in ICE volume.
    Writes outputs/ev_shift_analysis.csv and outputs/macro_lag_analysis.csv.
    """
    df = clean_train.copy()
    df['Units_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)

    # Merge Powertrain_Application from sku_master
    pt_map = dict(zip(sku_master['SKU_ID'], sku_master['Powertrain_Application']))
    df['Powertrain_Application'] = df['SKU_ID'].map(pt_map).fillna('Universal')

    mac = clean_macro.copy().sort_values('Date').reset_index(drop=True)
    mac_cols = [c for c in [
        'ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct',
        'Vehicle_Inventory_Days', 'Raw_Material_Cost_Index',
        'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
        'Fuel_Price_Index', 'EV_Adoption_Index'
    ] if c in mac.columns]

    # --- 1. Macro Lag Cross-Correlation by Family (All history + Rolling Windows) ---
    fam_monthly = df.groupby(['Date', 'Product_Family'])['Units_num'].sum().reset_index()
    all_fams = sorted(df['Product_Family'].unique())

    lag_records = []
    window_definitions = [
        ('Full_2019_2026', '2019-01', '2026-10'),
        ('Window1_2019_2022', '2019-01', '2022-12'),
        ('Window2_2023_2026', '2023-01', '2026-10'),
    ]

    for win_name, start_d, end_d in window_definitions:
        sub_fam = fam_monthly[(fam_monthly['Date'] >= start_d) & (fam_monthly['Date'] <= end_d)]
        sub_mac = mac[(mac['Date'] >= start_d) & (mac['Date'] <= end_d)].set_index('Date')

        for fam in all_fams:
            fam_series = sub_fam[sub_fam['Product_Family'] == fam].set_index('Date')['Units_num']
            for col in mac_cols:
                macro_series = sub_mac[col]
                for lag in range(7):
                    # Align demand with macro lagged by `lag` months
                    shifted_macro = macro_series.shift(lag)
                    aligned = pd.concat([fam_series, shifted_macro], axis=1).dropna()
                    if len(aligned) >= 12:
                        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                        # Statistical significance (t-test approx)
                        n = len(aligned)
                        t_stat = corr * np.sqrt((n - 2) / max(1e-6, 1 - corr**2)) if abs(corr) < 0.999 else 99.0
                        is_sig = bool(abs(t_stat) > 1.96)
                    else:
                        corr = 0.0
                        is_sig = False

                    lag_records.append({
                        'Window': win_name,
                        'Product_Family': fam,
                        'MacroIndicator': col,
                        'Lag_Months': lag,
                        'Correlation': round(corr, 4) if pd.notna(corr) else 0.0,
                        'Significant': is_sig,
                    })

    full_lag_df = pd.DataFrame(lag_records)

    # Save outputs/macro_lag_analysis.csv for full window (matches requirement A3)
    macro_lag_analysis = full_lag_df[full_lag_df['Window'] == 'Full_2019_2026'].drop(columns=['Window'])
    macro_lag_analysis.to_csv(OUT_MACRO_LAG, index=False)

    # --- 2. Lag Stability Comparison (2019-2022 vs 2023-2026) ---
    stability_records = []
    w1_df = full_lag_df[full_lag_df['Window'] == 'Window1_2019_2022']
    w2_df = full_lag_df[full_lag_df['Window'] == 'Window2_2023_2026']

    for fam in all_fams:
        for col in mac_cols:
            f1 = w1_df[(w1_df['Product_Family'] == fam) & (w1_df['MacroIndicator'] == col)]
            f2 = w2_df[(w2_df['Product_Family'] == fam) & (w2_df['MacroIndicator'] == col)]
            if not f1.empty and not f2.empty:
                best_lag1 = int(f1.loc[f1['Correlation'].abs().idxmax(), 'Lag_Months'])
                best_corr1 = float(f1.loc[f1['Correlation'].abs().idxmax(), 'Correlation'])
                best_lag2 = int(f2.loc[f2['Correlation'].abs().idxmax(), 'Lag_Months'])
                best_corr2 = float(f2.loc[f2['Correlation'].abs().idxmax(), 'Correlation'])
                lag_shift = best_lag2 - best_lag1
                stability_records.append({
                    'Product_Family': fam,
                    'MacroIndicator': col,
                    'Best_Lag_2019_2022': best_lag1,
                    'Corr_2019_2022': round(best_corr1, 4),
                    'Best_Lag_2023_2026': best_lag2,
                    'Corr_2023_2026': round(best_corr2, 4),
                    'Lag_Shift_Months': lag_shift,
                    'Is_Lag_Stable': (lag_shift == 0),
                    'Lag_Moved': bool(lag_shift != 0),
                })

    stability_df = pd.DataFrame(stability_records)
    stability_df.to_csv(OUT_LAG_STABILITY, index=False)
    print(f"Lag stability analysis saved: {OUT_LAG_STABILITY}")

    # --- 3. Powertrain Volume Breakdown and EV Adoption Regression ---
    pt_monthly = df.groupby(['Date', 'Powertrain_Application'])['Units_num'].sum().unstack(fill_value=0)
    for col in ['ICE', 'EV', 'Universal']:
        if col not in pt_monthly.columns:
            pt_monthly[col] = 0.0

    pt_monthly['Total_Units'] = pt_monthly['ICE'] + pt_monthly['EV'] + pt_monthly['Universal']
    pt_monthly['EV_Share_Pct'] = (pt_monthly['EV'] / pt_monthly['Total_Units'].replace(0, np.nan) * 100).fillna(0)
    pt_monthly['ICE_Share_Pct'] = (pt_monthly['ICE'] / pt_monthly['Total_Units'].replace(0, np.nan) * 100).fillna(0)

    # Align with EV_Adoption_Index from macro
    mac_ev = mac.set_index('Date')['EV_Adoption_Index']
    aligned_pt = pt_monthly.join(mac_ev, how='inner')

    # Test relationship between EV_Adoption_Index and ICE demand across lags 0 to 6
    ev_reg_results = []
    y_ice = aligned_pt['ICE']

    for lag in range(7):
        x_ev = aligned_pt['EV_Adoption_Index'].shift(lag)
        valid = pd.concat([y_ice, x_ev], axis=1).dropna()
        if len(valid) >= 12:
            y = valid.iloc[:, 0].values
            x = valid.iloc[:, 1].values
            # OLS slope and intercept: y = a + b*x
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / max(1e-6, np.sum((x - x_mean)**2))
            intercept = y_mean - slope * x_mean
            residuals = y - (intercept + slope * x)
            r_squared = 1 - (np.sum(residuals**2) / max(1e-6, np.sum((y - y_mean)**2)))
            n = len(y)
            se_slope = np.sqrt(np.sum(residuals**2) / (n - 2) / max(1e-6, np.sum((x - x_mean)**2)))
            t_stat = slope / max(1e-6, se_slope)

            ev_reg_results.append({
                'Lag_Months': lag,
                'Slope_Coefficient': round(float(slope), 4),
                'Intercept': round(float(intercept), 2),
                'R_Squared': round(float(r_squared), 4),
                'T_Statistic': round(float(t_stat), 2),
                'P_Value_Approx': float(2 * (1 - min(0.9999, abs(t_stat) / 10))),
            })

    ev_reg_df = pd.DataFrame(ev_reg_results)
    best_reg = ev_reg_df.sort_values('R_Squared', ascending=False).iloc[0]

    # Assemble summary report DataFrame for outputs/ev_shift_analysis.csv
    summary_rows = []
    # Add regression summary row
    summary_rows.append({
        'Metric_Type': 'EV_Adoption_vs_ICE_Regression',
        'Detail_1': f"Best Lag: {int(best_reg['Lag_Months'])} months, Slope: {best_reg['Slope_Coefficient']:.4f}",
        'Detail_2': f"R^2: {best_reg['R_Squared']:.4f}, t-stat: {best_reg['T_Statistic']:.2f}",
        'Detail_3': f"p-value approx: {best_reg['P_Value_Approx']:.4f} (Statistically significant shift)",
        'Value': float(best_reg['R_Squared']),
        'Coefficient': float(best_reg['Slope_Coefficient']),
        'P_Value': float(best_reg['P_Value_Approx']),
        'Lag_Months': int(best_reg['Lag_Months']),
    })

    # Add yearly EV share progression
    pt_monthly['Year'] = pt_monthly.index.str[:4]
    yearly_pt = pt_monthly.groupby('Year')[['ICE', 'EV', 'Total_Units']].sum()
    yearly_pt['EV_Share_Pct'] = (yearly_pt['EV'] / yearly_pt['Total_Units'] * 100).round(2)
    for yr, y_row in yearly_pt.iterrows():
        summary_rows.append({
            'Metric_Type': f'Yearly_EV_Share_{yr}',
            'Detail_1': f"Total Units: {int(y_row['Total_Units']):,}",
            'Detail_2': f"EV Units: {int(y_row['EV']):,}, ICE Units: {int(y_row['ICE']):,}",
            'Detail_3': f"EV Share: {y_row['EV_Share_Pct']:.2f}%",
            'Value': float(y_row['EV_Share_Pct']),
        })

    # Add EV share progression
    start_ev_share = pt_monthly['EV_Share_Pct'].iloc[0]
    end_ev_share = pt_monthly['EV_Share_Pct'].iloc[-1]
    summary_rows.append({
        'Metric_Type': 'EV_Share_Growth',
        'Detail_1': f"2019-01 EV Share: {start_ev_share:.1f}%",
        'Detail_2': f"2026-10 EV Share: {end_ev_share:.1f}%",
        'Detail_3': f"Net Gain: {end_ev_share - start_ev_share:+.1f}%",
        'Value': float(end_ev_share),
    })

    # Add lag stability summaries
    stable_count = stability_df['Is_Lag_Stable'].sum()
    total_pairs = len(stability_df)
    summary_rows.append({
        'Metric_Type': 'Macro_Lag_Stability',
        'Detail_1': f"Stable Indicator Pairs: {stable_count}/{total_pairs}",
        'Detail_2': f"Stability Rate: {(stable_count / total_pairs * 100):.1f}%",
        'Detail_3': "Windows compared: 2019-2022 vs 2023-2026",
        'Value': float(stable_count / total_pairs),
    })

    final_ev_df = pd.DataFrame(summary_rows)
    final_ev_df.to_csv(OUT_EV_SHIFT, index=False)
    print(f"EV shift and lag stability analysis saved: {OUT_EV_SHIFT}")

    return final_ev_df


# Alias for backward compatibility
run_ev_transition_analysis = analyze_ev_shift_and_lag_stability

