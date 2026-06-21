"""
Budget Planner Agent
Generates a personalized monthly budget using the 50/30/20 rule as a baseline,
then adjusts for the user's income, actual spending, and stated goals.
Receives expense analysis from the Expense Tracker Agent via import.
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
    BUDGET_ALLOCATIONS,
    GOAL_SAVINGS_BOOST,
)

_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
    temperature=0.4,
)

_SYSTEM = """You are an expert Budget Planner Agent with deep knowledge of personal finance.
Using the 50/30/20 rule (50% needs, 30% wants, 20% savings) as a baseline, create a
realistic monthly budget tailored to the user's actual spending patterns and goals.

Structure your response as:
1. Budget Health Score (X/10) — one sentence rationale.
2. Recommended Monthly Allocations — table with category | current | recommended | delta.
3. Top 3 Adjustments to make immediately.
4. Goal-Specific Strategy — how the budget helps achieve the stated goals.

Be specific with dollar amounts. Keep under 400 words."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Create a personalized budget plan from:\n\n{data}"),
])

_chain = _prompt | _llm | StrOutputParser()


def _goal_aware_allocations(monthly_income: float, financial_goals: str) -> tuple:
    """
    Adjust 50/30/20 savings allocation upward when goal keywords are detected.
    Returns (allocations_dict, savings_boost_fraction).
    The savings boost is funded by proportionally reducing Food/Entertainment/Shopping.
    """
    alloc = dict(BUDGET_ALLOCATIONS)
    gl = financial_goals.lower()

    savings_lift = sum(v for k, v in GOAL_SAVINGS_BOOST.items() if k in gl)
    if savings_lift:
        alloc["Savings"] = min(alloc["Savings"] + savings_lift, 0.40)
        wants = ["Food & Dining", "Entertainment", "Shopping"]
        per = savings_lift / len(wants)
        for cat in wants:
            alloc[cat] = max(alloc[cat] - per, 0.02)

    return {cat: monthly_income * pct for cat, pct in alloc.items()}, savings_lift


def create_budget(monthly_income: float, expense_analysis: dict, financial_goals: str) -> dict:
    """
    Budget Planner Agent entry point.
    Accepts expense_analysis produced by the Expense Tracker Agent.
    Returns a budget dict consumed by the Financial Advisor Agent.
    """
    avg_monthly = expense_analysis["avg_monthly"]
    breakdown = expense_analysis["breakdown"]
    n_months = expense_analysis["n_months"]

    # Monthly average per category
    monthly_by_cat = (breakdown / n_months).to_dict()

    # Goal-aware recommended allocations
    recommended, savings_boost = _goal_aware_allocations(monthly_income, financial_goals)

    # Delta: positive = overspending, negative = under budget
    deltas = {
        cat: monthly_by_cat.get(cat, 0) - recommended.get(cat, 0)
        for cat in set(list(monthly_by_cat.keys()) + list(recommended.keys()))
    }

    savings_rate = ((monthly_income - avg_monthly) / monthly_income * 100) if monthly_income else 0

    boost_note = (
        f"  (boosted by {savings_boost*100:.0f}% due to goal keywords)" if savings_boost else ""
    )

    cat_table = "\n".join(
        f"  {cat}: Current ${monthly_by_cat.get(cat, 0):,.2f}/mo | "
        f"Recommended ${recommended.get(cat, 0):,.2f}/mo | "
        f"Delta ${deltas.get(cat, 0):+,.2f}"
        for cat in sorted(set(list(monthly_by_cat.keys()) + list(recommended.keys())))
    )

    data_str = (
        f"Monthly Income: ${monthly_income:,.2f}\n"
        f"Average Monthly Spending: ${avg_monthly:,.2f}\n"
        f"Current Savings Rate: {savings_rate:.1f}%\n"
        f"Financial Goals: {financial_goals}\n\n"
        f"50/30/20 Targets (goal-adjusted){boost_note}:\n"
        f"  Needs (50%): ${monthly_income * 0.50:,.2f}\n"
        f"  Wants (30%): ${monthly_income * 0.30:,.2f}\n"
        f"  Savings ({recommended.get('Savings', monthly_income*0.20)/monthly_income*100:.0f}%): "
        f"${recommended.get('Savings', monthly_income * 0.20):,.2f}\n\n"
        f"Category Breakdown (monthly avg vs recommended):\n{cat_table}"
    )

    budget_plan_text = _chain.invoke({"data": data_str})

    return {
        "budget_plan_text": budget_plan_text,
        "monthly_income": monthly_income,
        "needs_budget": monthly_income * 0.50,
        "wants_budget": monthly_income * 0.30,
        "savings_budget": recommended.get("Savings", monthly_income * 0.20),
        "recommended_allocations": recommended,
        "monthly_actuals": monthly_by_cat,
        "deltas": deltas,
        "savings_rate": savings_rate,
        "savings_boost_pct": savings_boost,
    }
