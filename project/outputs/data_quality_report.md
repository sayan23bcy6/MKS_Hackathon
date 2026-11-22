# Data Quality Report - Celeris Automobili Demand Forecasting

Total audit log entries recorded: 106332  
Total anomalies registered: 1605

---

## Transaction Sales History (train.csv)

| Rule | Issue Description | Before | After / Action |
|---|---|---|---|
| T1 | Mixed date formats in Date column converted to YYYY-MM | 248248 rows | 9967 fixed |
| T2 | Year/Month disagree with Date column (Date treated as authoritative) | - | 246 rows |
| T3 | Impossible future dates (e.g. 2062 transposed to 2026) | 30 found | 30 corrected |
| T4 | Duplicate Record_IDs resolved by keeping latest Load_Timestamp | 248248 | 244566 (3682 duplicates removed) |
| T5 | Region labels normalized and unresolvable blanks assigned to 'Unassigned' | 942 blank | 697 set to Unassigned |
| T6 | Sales channel label variants normalized to standard set | - | 3524 normalized |
| T7 | Missing product family inferred deterministically from SKU prefix | 1228 blank | 1228 filled from SKU prefix |
| T8 | SKU_ID format variations normalized to standard PREFIX-NNN | 3683 malformed | 3683 normalized to PREFIX-NNN |
| T9 | Missing SKU descriptions populated from sku_master reference | 982 blank | 982 populated from sku_master |
| T10 | Customer name casing and whitespace variants unified | - | 2455 normalized |
| T11 | Units_Sold formatting and negatives | 44 comma, 497 negative | Converted to numeric and absolute value applied |
| T12 | Unit_Price_USD scale errors & zero prices | 0 zero rows | Annual SKU median reference price applied |
| T13 | Revenue_USD recalculation | 18531 discrepant | Recalculated as Units_Sold * Unit_Price_USD |
| R5 | Sales outside launch/EOL lifecycle | - | Zeroed out 0 EOL and 15 pre-launch records |
| T4b | Duplicate customer series records across differing channels | 244488 | 244321 (167 duplicates removed; clean dataset = 244,321 rows) |

## Stockout Reconciliation and Unconstrained Demand (Rule A2 / T11)

- **Stockout Records:** Documented inventory shortage periods from inventory_stockouts.csv.
- **Transaction Match Rate:** 84 shortage occurrences aligned with sales transactions.
- **Customer Account Allocation:** Allocated stockouts proportionally across active customer accounts by non-negative sales volume, assigning remainder units to largest account.
- **Unconstrained Reconstruction:** Reconstructed actual demand using Fill_Rate adjustment to eliminate artificial supply-constrained dips before model training.

## Macroeconomic Indicators

- **M1** Duplicate 2022-06 month: 1 duplicates resolved via nearest linear neighbor interpolation.
- **M2** Auto Production YoY fraction-to-percentage: 12 rows in 2019 scaled x100.
- **M3** PMI decimal point correction: 1 rows corrected (4.79 to 47.9).
- **M4** Missing indicators: Linear time interpolation applied with Macro_Imputed indicator flag.
- **M5** COVID period structural break: Is_COVID_Period flag active for 2020-03 through 2021-06.

## SKU Master & Events Calendar

- **S1** Business Unit conflict on PMP-004: Resolved to 'Advanced Manufacturing' per Rule R8.
- **E1-E4** Events calendar normalized: start/end dates formatted, open-ended events bounded, customer-region scopes split.

## Decision Rules Applied

### price_out_of_band_scaling
- **Rule ID:** T12
- **Affected Records:** 37351
- **Decision:** Replaced 37,351 T12 price rows: 518 100x-scale entries, 73 85x-scale entries and 369 entries in the 5-20x band.
- **Reason:** Corrects cent-scale entry errors, currency conversion disparities, and manual data-entry slips while maintaining intra-year price stability.

### blank_region_categorization
- **Rule ID:** T5
- **Affected Records:** 697
- **Decision:** Assigned persistent unmatchable blank regions to canonical category 'Unassigned'.
- **Reason:** Preserves total company sales volume and transaction count without guessing geographic origins.

### high_volume_orders
- **Rule ID:** A3
- **Affected Records:** 104
- **Decision:** Replaced suspected entry errors above 20 times the SKU-year median.
- **Reason:** Uses the SKU-year median and records each replacement in the audit log.

### macro_2022_06_duplicate
- **Rule ID:** M1
- **Affected Records:** 1
- **Decision:** Row B (52.7) is closer to neighbour interpolation (52.85)
- **Reason:** Row B 52.7 is compared with interpolation 52.85 from May 2022 53.5 and July 2022 52.2.

### macro_2023_05_pmi_decimal
- **Rule ID:** M3
- **Affected Records:** 1
- **Decision:** Corrected 4.79 to 47.9 via decimal point shift.
- **Reason:** April 2023 was 47.2 and June 2023 was 46.1; value of 4.79 was a typographical single-decimal slip.

### sku_bu_conflict_resolution
- **Rule ID:** S1
- **Affected Records:** 1
- **Decision:** Assigned PMP-004 to Business Unit 'Advanced Manufacturing' per Rule R8 and transaction history majority.
- **Reason:** All transaction records for PMP-004 consistently map to Advanced Manufacturing.

### customer_series_deduplication
- **Rule ID:** T4b
- **Affected Records:** 167
- **Decision:** Deduplicated series strictly on Date, SKU_ID, Region, Customer_Name by retaining latest Load_Timestamp.
- **Reason:** Resolves multiple records loaded across differing sales channels for the same customer account; establishes authoritative 244,321 cleaned transaction rows.

### stockout_account_allocation
- **Rule ID:** A2
- **Affected Records:** 0
- **Decision:** Allocated inventory stockouts defined at Date-SKU-Region proportionally across active customer accounts by non-negative volume, with remainder to largest customer.
- **Reason:** Preserves individual customer demand history while ensuring exact conservation of reported stockout units.
