import pandas as pd
import numpy as np


def clean_tabular(df: pd.DataFrame, datetime_cols_to_infer: list[str] = None) -> pd.DataFrame:
    """
    Apply standard tabular cleaning procedures:
    - Deduplication
    - Numeric type coercion for generic text columns
    - Strip whitespace from strings
    - Replace Sentinel missing values (-999, "N/A") with np.nan
    - Optional automatic Date/Time parsing
    """
    df = df.copy()

    # 1. Duplicate rows
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Type coercion - numeric columns stored as strings
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Convert only if >90% of values parse successfully
        if coerced.notna().mean() > 0.90:
            df[col] = coerced

    # 3. Date/time standardisation
    # Automatically infer 'date' column or user provided ones
    dt_cols = datetime_cols_to_infer or []
    if "date" in df.columns and "date" not in dt_cols:
        dt_cols.append("date")

    for dt_col in dt_cols:
        if dt_col in df.columns:
            # Using format='mixed' replaces infer_datetime_format which is deprecated in modern pandas
            df[dt_col] = pd.to_datetime(df[dt_col], format='mixed', errors="coerce")
            df[f"{dt_col}_year"]  = df[dt_col].dt.year
            df[f"{dt_col}_month"] = df[dt_col].dt.month
            df[f"{dt_col}_day"]   = df[dt_col].dt.day
            df = df.drop(columns=[dt_col])   # drop raw date column

    # 4. Strip whitespace from all string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass # Mixed types occasionally fail

    # 5. Replace sentinel missing values
    df = df.replace([-999, -9999, 999, "-999", "-9999", "999", "N/A", "NA", "n/a", ""], np.nan)

    return df
