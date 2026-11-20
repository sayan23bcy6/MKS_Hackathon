# Data Quality Report – Celeris Automobili Demand Forecasting

**Total audit log entries:** 37595  
**Total anomalies flagged:** 1815

---

## Train.csv Issues

| Issue | Description | Before | After / Count | Rule |
|---|---|---|---|---|
| T1 | Mixed date formats in Date column | 248248 rows | 9967 fixed | T1 |
| T2 | Year/Month disagree with Date (R1: Date authoritative) | — | 246 rows | T2 |
| T3 | Impossible future dates (2062-xx typo for 2026-xx) | 30 found | 30 corrected | T3 |
| T4 | Duplicate Record_IDs (loaded twice) – kept latest timestamp | 248248 | 244566 (+3682 dropped) | T4 |
| T5 | Region labels inconsistent or blank | 942 blank | 697 unresolved | T5 |
| T6 | Sales_Channel label variants normalised | — | 3524 normalised | T6 |
| T7 | Product_Family blank – derived from SKU prefix | 1228 blank | 1228 filled | T7 |
| T8 | SKU_ID format errors – normalised to PREFIX-NNN | 3683 bad | 3683 fixed | T8 |
| T9 | SKU_Description blank – filled from sku_master | 982 blank | 982 filled | T9 |
| T10 | Customer_Name variants normalised | — | 2455 normalised | T10 |
| T11 | Units_Sold: blanks | 3327 | — | T11 |
| T11 | Units_Sold: dashes (NULL) | 217 | — | T11 |
| T11 | Units_Sold: comma-formatted | 44 | — | T11 |
| T11 | Units_Sold: negatives | 491 | abs() applied | T11 |
| T11 | Units_Sold: large (>5000) | 32 | flagged | T11 |
| T12 | Unit_Price_USD zero (with units>0) | — | 295 filled | T12 |
| T12 | Price ×100 band | 583 | reference price applied | T12 |
| T12 | Price ×85 band (INR hypothesis) | 31 | reference price applied | T12 |
| T12 | Price 5–20× band (unresolved) | 385 | flagged, kept for demand | T12 |
| T13 | Revenue_USD recomputed | 1956 discrepant | 1956 recomputed | T13 |
| R5 | Sales after EOL or before Launch | — | 0 EOL, 15 launch violations | R5 |
| R9/T4b | Duplicate Date-SKU-Region after cleaning | — | 494 | T4b |

## Macro Indicators Issues

- **M1** Duplicate 2022-06: 1 months affected. Resolution: Row B (52.7) is closer to neighbour interpolation (52.85)
- **M2** Auto_Production_YoY_Pct fraction→percent for 2019: 12 rows fixed
- **M3** PMI decimal shift (2023-05): 1 rows corrected
- **M4** Blank cells interpolated: {'Vehicle_Inventory_Days': ['2019-07'], 'New_Vehicle_Orders_Index': ['2023-08'], 'Industrial_Util_Rate_Pct': ['2021-02'], 'Fuel_Price_Index': ['2020-09'], 'EV_Adoption_Index': ['2024-11']}
- **M5** Real extreme values noted: Min Auto_Prod_YoY=-45.4%, Max=36.9% (COVID period – not errors)

## SKU Master Issues

- **S1** Precision Machined Parts BU conflict: 1 SKU(s): ['PMP-004']
- **S2** Planned launches: 3, EOL SKUs: 9

## Events Calendar Issues

- **E1** Date formats normalised: 48 rows
- **E2** Open-ended events (no End_Date): 1
- **E3** Customer|Region scope split: 3
- **E4** Event type data-impact annotations applied

## Unresolved Items

### price_5_20x_band
- **Count/affected:** 385
- **Decision:** Flagged in anomaly register; kept for demand analysis; excluded from price-based analysis
- **Reason:** No clear single-factor explanation; not auto-corrected per Rule 7

### price_85x_band_rupee_hypothesis
- **Count/affected:** 31
- **Decision:** Treated as currency denomination error (INR→USD) and replaced with SKU-year reference price. India region subset showed higher concentration of this band.
- **Reason:** Approximately 85× ratio consistent with INR/USD exchange rate in relevant period (~80-85 INR/USD). Hypothesis is plausible and documented.

### large_units_gt5000
- **Count/affected:** 32
- **Decision:** Flagged in anomaly register; retained as-is pending cross-check with inventory_stockouts Requested_Units
- **Reason:** Cannot determine if data error or real bulk order without stockout corroboration

### macro_2022_06_duplicate
- **Count/affected:** 1
- **Decision:** Row B (52.7) is closer to neighbour interpolation (52.85)
- **Reason:** Chosen row is closer to the linear interpolation of 2022-05 and 2022-07 neighbours

### macro_2023_05_pmi
- **Count/affected:** 1
- **Decision:** Corrected 4.79 → 47.9 (×10 decimal shift) based on neighbour confirmation
- **Reason:** 2023-04 PMI=47.2 and 2023-06 PMI=46.1; 47.9 fits the trend; 4.79 does not

### sku_bu_conflict
- **Count/affected:** 1
- **Decision:** Flagged for manual review; not auto-corrected; transaction data should be used to determine correct BU
- **Reason:** Cannot determine correct BU from master data alone without transaction evidence
