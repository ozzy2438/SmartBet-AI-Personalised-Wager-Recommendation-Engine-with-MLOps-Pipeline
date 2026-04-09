"""Schema-driven validation for generated raw data."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from smartbet_ai.common.config import load_data_schema
from smartbet_ai.common.paths import RAW_DATA_DIR


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _invalid_type_mask(series: pd.Series, expected_type: str) -> pd.Series:
    non_null = series.dropna()
    invalid_mask = pd.Series(False, index=series.index)

    if expected_type == "int":
        numeric = _coerce_numeric(non_null)
        invalid = numeric.isna() | ~np.isclose(numeric, np.round(numeric))
        invalid_mask.loc[non_null.index] = invalid
    elif expected_type == "float":
        numeric = _coerce_numeric(non_null)
        invalid_mask.loc[non_null.index] = numeric.isna()
    elif expected_type == "bool":
        normalized = non_null.astype(str).str.lower()
        invalid_mask.loc[non_null.index] = ~normalized.isin({"true", "false", "1", "0"})
    elif expected_type == "datetime":
        invalid_mask.loc[non_null.index] = pd.to_datetime(non_null, errors="coerce").isna()

    return invalid_mask


def validate_data(data_path: str | None = None) -> bool:
    """
    Validate raw CSVs against the JSON schema.
    Returns True when all checks pass and exits non-zero on failure.
    """
    schema = load_data_schema()
    base_path = RAW_DATA_DIR if data_path is None else pd.io.common.stringify_path(data_path)
    all_valid = True

    for table_name, columns in schema.items():
        filepath = RAW_DATA_DIR / f"{table_name}.csv" if data_path is None else pd.io.common.stringify_path(f"{base_path}/{table_name}.csv")
        print(f"\nValidating {table_name}...")

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"  Missing file: {filepath}")
            all_valid = False
            continue

        for column_name, rules in columns.items():
            if column_name not in df.columns:
                if rules.get("required", False):
                    print(f"  Missing required column: {column_name}")
                    all_valid = False
                continue

            series = df[column_name]

            if rules.get("required", False) and series.isna().any():
                print(f"  {column_name}: {int(series.isna().sum())} null values")
                all_valid = False

            invalid_type = _invalid_type_mask(series, rules["type"])
            if invalid_type.any():
                print(f"  {column_name}: {int(invalid_type.sum())} values failed type={rules['type']}")
                all_valid = False

            if "allowed" in rules:
                invalid_allowed = ~series.dropna().isin(rules["allowed"])
                if invalid_allowed.any():
                    print(f"  {column_name}: {int(invalid_allowed.sum())} values outside allowed set")
                    all_valid = False

            if rules["type"] in {"int", "float"}:
                numeric = _coerce_numeric(series)
                if "min" in rules:
                    below = numeric < rules["min"]
                    below = below.fillna(False)
                    if below.any():
                        print(f"  {column_name}: {int(below.sum())} values below minimum {rules['min']}")
                        all_valid = False
                if "max" in rules:
                    above = numeric > rules["max"]
                    above = above.fillna(False)
                    if above.any():
                        print(f"  {column_name}: {int(above.sum())} values above maximum {rules['max']}")
                        all_valid = False

        print(f"  {table_name}: {len(df)} rows checked")

    if all_valid:
        print("\nAll data validation checks passed.")
        return True

    print("\nData validation failed. Pipeline should not proceed.")
    sys.exit(1)


if __name__ == "__main__":
    validate_data()
