"""
Financial Orchestrator
Central coordinator that runs all four agents in sequence and synthesizes
their outputs into a unified executive report.

Agent pipeline:
  1. Expense Tracker  →  expense_analysis
  2. Budget Planner   →  budget_plan      (consumes expense_analysis)
  3. Financial Advisor→  advice           (consumes expense_analysis + budget_plan)
  4. Forecasting      →  forecast         (runs independently on raw DataFrame)
  5. Orchestrator     →  executive report (synthesizes all four)
"""
# pylint: disable=duplicate-code
import logging
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from agents.expense_tracker import analyze_expenses
from agents.budget_planner import create_budget
from agents.financial_advisor import get_financial_advice, answer_financial_question
from agents.forecasting_agent import forecast_expenses
from tools.data_tools import validate_dataframe
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    DEPLOYMENT_NAME,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s — %(message)s")

_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_API_KEY) if AZURE_OPENAI_API_KEY else None,
    temperature=0.4,
)

_REPORT_SYSTEM = """You are the Chief Financial Intelligence Officer delivering an executive
financial health report. Synthesize the four AI agent outputs into a clear, motivating
summary. Use emojis, bold key numbers, and bullet points.

Sections:
💼 Financial Snapshot — 4 key metrics in one glance.
✅ What You're Doing Well — 2 positives.
🚨 Top Priorities — 3 ranked action items with specific amounts.
🔮 Outlook — one-sentence forecast summary.

Max 300 words. Be direct, warm, and encouraging."""

_report_prompt = ChatPromptTemplate.from_messages([
    ("system", _REPORT_SYSTEM),
    ("human", "Generate the executive financial report from:\n\n{context}"),
])

_report_chain = _report_prompt | _llm | StrOutputParser()


class FinancialOrchestrator:
    """
    Stateful orchestrator — run `run_full_analysis()` once,
    then call `ask()` for follow-up questions without re-running all agents.
    """

    def __init__(self) -> None:
        self._expense_analysis: dict | None = None
        self._budget_plan: dict | None = None
        self._advice: dict | None = None
        self._forecast: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        monthly_income: float,
        financial_goals: str,
        risk_tolerance: str = "moderate",
    ) -> dict:
        """
        Run all four agents in sequence and return a combined results dict.
        Raises ValueError for bad input data; RuntimeError names the failing agent.
        """
        valid, msg = validate_dataframe(df)
        if not valid:
            raise ValueError(msg)

        try:
            _logger.info("Agent 1/4 — Expense Tracker")
            self._expense_analysis = analyze_expenses(df)
        except Exception as e:
            raise RuntimeError(f"Expense Tracker Agent failed: {e}") from e

        try:
            _logger.info("Agent 2/4 — Budget Planner")
            self._budget_plan = create_budget(
                monthly_income, self._expense_analysis, financial_goals
            )
        except Exception as e:
            raise RuntimeError(f"Budget Planner Agent failed: {e}") from e

        try:
            _logger.info("Agent 3/4 — Financial Advisor")
            self._advice = get_financial_advice(
                self._expense_analysis, self._budget_plan, financial_goals, risk_tolerance
            )
        except Exception as e:
            raise RuntimeError(f"Financial Advisor Agent failed: {e}") from e

        try:
            _logger.info("Agent 4/4 — Forecasting Agent")
            self._forecast = forecast_expenses(df)
        except Exception as e:
            raise RuntimeError(f"Forecasting Agent failed: {e}") from e

        try:
            _logger.info("Orchestrator — generating executive report")
            report = self._build_executive_report()
        except Exception as e:
            raise RuntimeError(f"Executive report generation failed: {e}") from e

        return {
            "expense_analysis": self._expense_analysis,
            "budget_plan": self._budget_plan,
            "advice": self._advice,
            "forecast": self._forecast,
            "executive_report": report,
        }

    def ask(self, question: str) -> str:
        """Answer a follow-up question using cached agent context."""
        if self._expense_analysis is None:
            return "Please run the full analysis first."
        assert self._budget_plan is not None
        assert self._advice is not None
        assert self._forecast is not None

        context = (
            f"Monthly Income: ${self._budget_plan['monthly_income']:,.2f}\n"
            f"Avg Monthly Spending: ${self._expense_analysis['avg_monthly']:,.2f}\n"
            f"Savings Rate: {self._advice['savings_rate']:.1f}%\n"
            f"Spending Trend: {self._forecast['trend']} "
            f"({self._forecast['trend_pct']:+.1f}%/month)\n"
            f"Top Spending Categories: "
            + ", ".join(self._expense_analysis["breakdown"].nlargest(3).index.tolist())
            + "\nOverspending Areas: "
            + (", ".join(self._advice["overspending_categories"].keys()) or "None")
            + f"\nEmergency Fund Target: ${self._advice['emergency_fund_target']:,.0f}"
        )
        return answer_financial_question(question, context)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_executive_report(self) -> str:
        assert self._expense_analysis is not None
        assert self._budget_plan is not None
        assert self._advice is not None
        assert self._forecast is not None
        context = (
            f"EXPENSE SNAPSHOT\n"
            f"  Total Spent: ${self._expense_analysis['total_spent']:,.2f} "
            f"over {self._expense_analysis['n_months']} months\n"
            f"  Avg Monthly: ${self._expense_analysis['avg_monthly']:,.2f}\n"
            f"  Monthly Income: ${self._budget_plan['monthly_income']:,.2f}\n"
            f"  Savings Rate: {self._advice['savings_rate']:.1f}%\n\n"
            f"BUDGET STATUS\n"
            f"  Target Savings: ${self._budget_plan['savings_budget']:,.2f}/month\n"
            f"  Actual Savings: ${self._advice['monthly_savings']:,.2f}/month\n"
            f"  Overspending Categories: "
            + (", ".join(self._advice["overspending_categories"].keys()) or "None")
            + f"\n\nFORECAST\n"
            f"  Trend: {self._forecast['trend']} "
            f"({self._forecast['trend_pct']:+.1f}%/month)\n"
            f"  Next Month Estimate: ${self._forecast['predictions'][0]:,.2f}\n\n"
            f"AGENT INSIGHTS\n"
            f"  Expense Tracker: {self._expense_analysis['ai_analysis'][:250]}\n"
            f"  Financial Advisor: {self._advice['advice'][:250]}\n"
            f"  Forecast Agent: {self._forecast['forecast_insights'][:200]}"
        )
        return _report_chain.invoke({"context": context})
