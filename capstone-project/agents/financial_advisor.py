"""
Financial Advisor Agent
Synthesizes outputs from Expense Tracker and Budget Planner agents to deliver
personalized, actionable financial advice and answer ad-hoc user questions.
Includes risk-based investment allocation and emergency fund calculations.
"""
# pylint: disable=duplicate-code
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    DEPLOYMENT_NAME,
    OVERSPENDING_THRESHOLD_PCT,
)

_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
    temperature=0.5,
)

# Deterministic investment allocation by risk tolerance
_INVEST_ALLOC = {
    "conservative": {
        "High-Yield Savings/CDs": 0.50,
        "Bond Index Funds": 0.40,
        "Stock Index Funds": 0.10,
    },
    "moderate": {
        "Stock Index Funds": 0.60,
        "Bond Index Funds": 0.30,
        "Cash Reserve": 0.10,
    },
    "aggressive": {
        "Growth ETFs": 0.70,
        "Individual Stocks": 0.20,
        "Alternatives": 0.10,
    },
}

_ADVICE_SYSTEM = """You are a world-class Personal Financial Advisor. You combine data-driven
analysis with empathetic, practical guidance. Your advice is specific, realistic, and
immediately actionable — not generic platitudes.

Structure your response:
🔴 Immediate Actions (this month) — 2-3 concrete steps with exact dollar targets.
🟡 Short-Term Strategy (3-6 months) — milestones with amounts and dates.
🟢 Long-Term Vision (6-12 months) — investment and wealth-building moves.
💡 Quick Win — one change that can save the most money fastest.

Cite specific numbers from the data. Keep under 450 words."""

_QA_SYSTEM = """You are a knowledgeable Personal Financial Advisor answering a specific
question from a user about their finances. Use the provided financial context.
Give a direct, specific answer backed by numbers from the context. Under 200 words."""

_advice_prompt = ChatPromptTemplate.from_messages([
    ("system", _ADVICE_SYSTEM),
    ("human", "Provide financial advice based on:\n\n{data}"),
])

_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM),
    ("human", "Financial Context:\n{context}\n\nUser Question: {question}"),
])

_advice_chain = _advice_prompt | _llm | StrOutputParser()
_qa_chain = _qa_prompt | _llm | StrOutputParser()


def get_financial_advice(  # pylint: disable=too-many-locals
    expense_analysis: dict,
    budget_plan: dict,
    financial_goals: str,
    risk_tolerance: str = "moderate",
) -> dict:
    """
    Financial Advisor Agent entry point.
    Consumes outputs from Expense Tracker and Budget Planner agents.
    Produces risk-aware investment recommendations and emergency fund targets.
    """
    monthly_income = budget_plan["monthly_income"]
    avg_monthly = expense_analysis["avg_monthly"]
    monthly_savings = monthly_income - avg_monthly
    savings_rate = budget_plan["savings_rate"]
    deltas = budget_plan["deltas"]

    # Dynamic threshold: 1% of income, minimum $10
    threshold = max(monthly_income * OVERSPENDING_THRESHOLD_PCT, 10.0)

    overspending = {
        cat: {
            "monthly_actual": budget_plan["monthly_actuals"].get(cat, 0),
            "recommended": budget_plan["recommended_allocations"].get(cat, 0),
            "excess": delta,
        }
        for cat, delta in deltas.items()
        if delta > threshold
    }

    over_str = "\n".join(
        f"  {cat}: spending ${v['monthly_actual']:,.2f}/mo vs "
        f"${v['recommended']:,.2f}/mo recommended (${v['excess']:,.2f} over)"
        for cat, v in sorted(overspending.items(), key=lambda x: -x[1]["excess"])
    ) or "  None — great job staying within budget!"

    # Deterministic investment allocation
    invest_alloc = _INVEST_ALLOC.get(risk_tolerance, _INVEST_ALLOC["moderate"])
    investable_monthly = max(monthly_savings * 0.70, 0)
    emergency_fund_target = avg_monthly * 6  # standard 6-month cushion

    invest_str = " | ".join(
        f"{name}: ${investable_monthly * pct:,.0f}/mo ({pct*100:.0f}%)"
        for name, pct in invest_alloc.items()
    )

    data_str = (
        f"Monthly Income: ${monthly_income:,.2f}\n"
        f"Average Monthly Spending: ${avg_monthly:,.2f}\n"
        f"Monthly Savings: ${monthly_savings:,.2f} ({savings_rate:.1f}% savings rate)\n"
        f"Risk Tolerance: {risk_tolerance}\n"
        f"Financial Goals: {financial_goals}\n\n"
        f"Overspending Categories (threshold ${threshold:,.0f}/mo):\n{over_str}\n\n"
        f"Investment Recommendation ({risk_tolerance} risk):\n  {invest_str}\n"
        f"  Total investable: ${investable_monthly:,.0f}/month (70% of savings)\n\n"
        f"Emergency Fund Target: ${emergency_fund_target:,.0f} (6 months of expenses)\n\n"
        f"Expense Tracker Insights:\n{expense_analysis['ai_analysis']}\n\n"
        f"Budget Planner Recommendations:\n{budget_plan['budget_plan_text'][:400]}"
    )

    advice_text = _advice_chain.invoke({"data": data_str})

    return {
        "advice": advice_text,
        "overspending_categories": overspending,
        "monthly_savings": monthly_savings,
        "savings_rate": savings_rate,
        "investment_allocation": invest_alloc,
        "investable_monthly": investable_monthly,
        "emergency_fund_target": emergency_fund_target,
    }


def answer_financial_question(question: str, financial_context: str) -> str:
    """Answer an ad-hoc user question using their financial context."""
    return _qa_chain.invoke({"context": financial_context, "question": question})
