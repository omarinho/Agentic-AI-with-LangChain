"""
Forecasting Agent
Predicts future expenses using linear trend analysis on historical data,
then uses an LLM to interpret the trend, flag risks, and recommend preparation steps.
"""
# pylint: disable=duplicate-code
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    DEPLOYMENT_NAME,
)
from tools.data_tools import get_monthly_summary

_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
    temperature=0.3,
)

_SYSTEM = """You are an expert Financial Forecasting Agent. You analyze historical spending
trends and predicted future expenses to provide proactive financial guidance.

Structure your response:
📈 Trend Analysis — direction, rate of change, and key driver categories.
⚠️ Risk Alerts — months or categories where overspending is likely.
📅 Seasonal Patterns — recurring peaks or dips you detect.
🎯 Preparation Steps — 2-3 concrete actions the user should take now.

Use specific numbers and months. Keep under 350 words."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Analyze this expense forecast data:\n\n{data}"),
])

_chain = _prompt | _llm | StrOutputParser()


def _linear_trend(values: np.ndarray):
    """Fit a linear trend and return (slope, intercept)."""
    x = np.arange(len(values), dtype=float)
    return np.polyfit(x, values, 1)  # [slope, intercept]


def _project(values: np.ndarray, n: int) -> list:
    slope, intercept = _linear_trend(values)
    base = len(values)
    return [max(slope * (base + i) + intercept, 0.0) for i in range(n)]


def _residual_std(values: np.ndarray) -> float:
    """Std deviation of linear-fit residuals — basis for the forecast confidence band."""
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    return float(np.std(values - fitted))


def forecast_expenses(df: pd.DataFrame, n_months: int = 3) -> dict:  # pylint: disable=too-many-locals
    """
    Forecasting Agent entry point.
    Works from raw DataFrame; does not depend on other agents
    so it can run independently or as part of the orchestrator.
    """
    monthly = get_monthly_summary(df)
    values: np.ndarray = np.asarray(monthly.values, dtype=float)

    predictions = _project(values, n_months)
    conf_std = _residual_std(values)
    last_period = monthly.index[-1]
    forecast_months = [last_period + i for i in range(1, n_months + 1)]

    # Per-category forecast
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    category_forecasts: dict = {}
    for cat in df["category"].unique():
        cat_vals = df[df["category"] == cat].groupby("month")["amount"].sum()
        if len(cat_vals) >= 2:
            category_forecasts[cat] = _project(np.asarray(cat_vals.values, dtype=float), n_months)

    slope, _ = _linear_trend(values)
    trend_pct = (slope / values.mean() * 100) if values.mean() else 0
    if slope > 30:
        trend = "increasing"
    elif slope < -30:
        trend = "decreasing"
    else:
        trend = "stable"

    avg_predicted = float(np.mean(predictions))
    avg_historical = float(values.mean())

    hist_str = "\n".join(f"  {m}: ${v:,.2f}" for m, v in monthly.items())
    fc_str = "\n".join(f"  {m}: ${v:,.2f}" for m, v in zip(forecast_months, predictions))

    data_str = (
        f"Historical Monthly Spending:\n{hist_str}\n\n"
        f"Statistical Summary:\n"
        f"  Average: ${avg_historical:,.2f}/month\n"
        f"  Highest: ${values.max():,.2f} ({monthly.idxmax()})\n"
        f"  Lowest:  ${values.min():,.2f} ({monthly.idxmin()})\n"
        f"  Std Dev: ${values.std():,.2f}\n"
        f"  Trend:   {trend} at {trend_pct:+.1f}% per month (${slope:+,.2f}/month)\n\n"
        f"Predicted Next {n_months} Months:\n{fc_str}\n\n"
        f"Predicted Average: ${avg_predicted:,.2f}/month\n"
        f"Predicted Change vs Historical: "
        f"${avg_predicted - avg_historical:+,.2f}/month"
    )

    forecast_insights = _chain.invoke({"data": data_str})

    return {
        "historical_monthly": monthly,
        "predictions": predictions,
        "forecast_months": forecast_months,
        "category_forecasts": category_forecasts,
        "trend": trend,
        "trend_pct": trend_pct,
        "slope": float(slope),
        "avg_historical": avg_historical,
        "avg_predicted": avg_predicted,
        "confidence_std": conf_std,
        "forecast_insights": forecast_insights,
    }
