""" Data loading and Pandas-based financial calculations """
from pathlib import Path

import pandas as pd

from config import REQUIRED_CSV_COLUMNS

DATA_DIR = Path(__file__).parent.parent / "data"


def load_expenses(filepath=None) -> pd.DataFrame:
    """Load CSV expense data, parse dates and cast amounts to float."""
    if filepath is None:
        filepath = DATA_DIR / "sample_expenses.csv"
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


def get_spending_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-category totals, transaction counts, and averages."""
    summary = df.groupby("category")["amount"].agg(["sum", "count", "mean"])
    summary.columns = ["total", "transactions", "avg_per_transaction"]
    return summary.sort_values("total", ascending=False)


def get_monthly_summary(df: pd.DataFrame) -> pd.Series:
    """Return total spending per calendar month as a Period-indexed Series."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["amount"].sum()


def get_category_breakdown(df: pd.DataFrame):
    """Return (absolute amounts by category, percentage share by category)."""
    total = df["amount"].sum()
    breakdown = df.groupby("category")["amount"].sum()
    percentages = (breakdown / total * 100).round(2)
    return breakdown, percentages


def get_top_transactions(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the n largest individual transactions."""
    return df.nlargest(n, "amount")[["date", "amount", "category", "description"]]


def validate_dataframe(df: pd.DataFrame) -> tuple:
    """Return (True, 'OK') or (False, error_message) for uploaded expense DataFrames."""
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if df.empty:
        return False, "File contains no data rows."
    return True, "OK"
