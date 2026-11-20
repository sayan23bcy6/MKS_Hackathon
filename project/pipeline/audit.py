"""
audit.py – Thread-safe audit log accumulator.
Every cleaning rule appends records here; the log is written at the end.
"""

import pandas as pd
from pipeline.config import AUDIT_COLS, OUT_AUDIT_LOG


class AuditLog:
    """Accumulates audit records and saves to CSV."""

    def __init__(self):
        self._records = []

    def log(self, file: str, row_key: str, field: str,
            raw_value, new_value, rule_id: str, reason: str):
        self._records.append({
            "file": file,
            "row_key": str(row_key),
            "field": field,
            "raw_value": str(raw_value),
            "new_value": str(new_value),
            "rule_id": rule_id,
            "reason": reason,
        })

    def log_many(self, records: list):
        """Extend with a list of dicts matching AUDIT_COLS."""
        self._records.extend(records)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=AUDIT_COLS)

    def save(self):
        df = self.as_dataframe()
        df.to_csv(OUT_AUDIT_LOG, index=False)
        return len(df)

    def __len__(self):
        return len(self._records)


# Module-level singleton used across all pipeline steps
audit = AuditLog()

