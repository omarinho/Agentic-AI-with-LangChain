"""
Expense Tracker Agent
Monitors and categorizes spending habits using LangChain + Azure OpenAI.
Communicates results to other agents via its returned dict.
"""
# pylint: disable=duplicate-code
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
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
from tools.data_tools import (
    get_spending_summary,
    get_monthly_summary,
    get_category_breakdown,
    get_top_transactions,
)

_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
    temperature=0.2,
)

_SYSTEM = """You are an expert Expense Tracker Agent specializing in personal finance analytics.
Analyze the provided spending data and identify:
1. The top 3 spending categories and why they are significant.
2. Any unusual spikes or patterns in monthly spending.
3. Positive spending behaviors worth reinforcing.
4. A concise overall spending health assessment (Good / Needs Attention / Critical).
Be specific with numbers and percentages. Keep the response under 300 words."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Analyze this expense data:\n\n{data}"),
])

_chain = _prompt | _llm | StrOutputParser()


def _detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag per-category outliers using IQR upper fence (Q3 + 1.5 × IQR)."""
    flagged = []
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        if len(sub) < 3:
            continue
        q1 = sub["amount"].quantile(0.25)
        q3 = sub["amount"].quantile(0.75)
        upper = q3 + 1.5 * (q3 - q1)
        flagged.append(sub[sub["amount"] > upper])
    if flagged:
        return pd.concat(flagged).sort_values("amount", ascending=False)
    return pd.DataFrame(columns=df.columns)


def analyze_expenses(df: pd.DataFrame) -> dict:
    """
    Core function for the Expense Tracker Agent.
    Returns a rich dict consumed by Budget Planner, Advisor, and Forecasting agents.
    """
    summary = get_spending_summary(df)
    monthly = get_monthly_summary(df)
    breakdown, percentages = get_category_breakdown(df)
    top_txns = get_top_transactions(df, n=5)
    anomalies = _detect_anomalies(df)

    total_spent = df["amount"].sum()
    avg_monthly = float(monthly.mean())
    n_months = len(monthly)

    anomaly_str = (
        anomalies[["date", "amount", "category", "description"]].to_string(index=False)
        if not anomalies.empty else "None detected"
    )

    data_str = (
        f"Period Covered: {df['date'].min().date()} to {df['date'].max().date()}\n"
        f"Total Spent: ${total_spent:,.2f} over {n_months} months\n"
        f"Average Monthly Spending: ${avg_monthly:,.2f}\n"
        f"Total Transactions: {len(df)}\n\n"
        f"Spending by Category:\n{summary.to_string()}\n\n"
        f"Monthly Totals:\n{monthly.to_string()}\n\n"
        f"Category Share (%):\n{percentages.to_string()}\n\n"
        f"Top 5 Largest Transactions:\n{top_txns.to_string(index=False)}\n\n"
        f"Anomalous Transactions ({len(anomalies)} flagged by IQR method):\n{anomaly_str}"
    )

    ai_analysis = _chain.invoke({"data": data_str})

    return {
        "total_spent": total_spent,
        "avg_monthly": avg_monthly,
        "n_months": n_months,
        "summary": summary,
        "monthly": monthly,
        "breakdown": breakdown,
        "percentages": percentages,
        "top_transactions": top_txns,
        "anomalies": anomalies,
        "ai_analysis": ai_analysis,
    }


def categorize_transaction(description: str, amount: float) -> str:
    """Classify a single transaction into a spending category using AI."""
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a financial categorization assistant. "
            "Classify the transaction into EXACTLY ONE of these categories: "
            "Housing, Food & Dining, Transportation, Entertainment, Healthcare, "
            "Utilities, Shopping, Education, Savings, Others. "
            "Reply with ONLY the category name — no explanation."
        )),
        ("human", "Description: {description}\nAmount: ${amount}"),
    ])
    chain = classify_prompt | _llm | StrOutputParser()
    return chain.invoke({"description": description, "amount": amount}).strip()
