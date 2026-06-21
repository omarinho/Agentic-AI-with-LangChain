""" Matplotlib chart builders — return Figure objects for Streamlit's st.pyplot() """
from typing import Optional

import matplotlib
matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
# pylint: enable=wrong-import-position

_YLABEL_AMOUNT = "Amount ($)"

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def plot_spending_pie(df: pd.DataFrame) -> plt.Figure:
    """Pie chart of total spending broken down by category."""
    totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    _, _, autotexts = ax.pie(
        totals.to_numpy(),
        labels=totals.index.tolist(),
        autopct="%1.1f%%",
        startangle=140,
        colors=_PALETTE[: len(totals)],
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Spending by Category", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def plot_monthly_trend(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of total spending aggregated by calendar month."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(monthly)),
        monthly.to_numpy(),
        color=_PALETTE[0],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="$%.0f", fontsize=9, padding=3)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=30, ha="right")
    ax.set_title("Monthly Spending Trend", fontsize=14, fontweight="bold")
    ax.set_ylabel(_YLABEL_AMOUNT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_forecast(  # pylint: disable=too-many-locals
    historical_monthly: pd.Series,
    predictions: list,
    forecast_months: list,
    confidence_std: Optional[float] = None,
) -> plt.Figure:
    """Line chart of historical spending with a forward forecast and confidence band."""
    hist_vals: np.ndarray = np.asarray(historical_monthly.values, dtype=float)
    hist_labels = [str(m) for m in historical_monthly.index]
    fc_labels = [str(m) for m in forecast_months]
    all_labels = hist_labels + fc_labels

    hist_x = list(range(len(hist_vals)))
    fc_x = list(range(len(hist_vals) - 1, len(hist_vals) + len(predictions)))
    bridge = [hist_vals[-1]] + list(predictions)

    # Use statistical confidence band when available, fall back to ±10%
    if confidence_std and confidence_std > 0:
        half = confidence_std * 1.5
        band_label = "±1.5σ confidence band"
    else:
        half = max(bridge) * 0.10
        band_label = "±10% confidence band"

    lower = [max(v - half, 0) for v in bridge]
    upper = [v + half for v in bridge]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_x, hist_vals, "o-", color=_PALETTE[0], linewidth=2.5,
            label="Historical", zorder=3)
    ax.plot(fc_x, bridge, "o--", color=_PALETTE[1], linewidth=2.5,
            label="Forecast", zorder=3)
    ax.fill_between(fc_x, lower, upper, color=_PALETTE[1], alpha=0.15, label=band_label)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_title("Expense Forecast", fontsize=14, fontweight="bold")
    ax.set_ylabel(_YLABEL_AMOUNT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_budget_vs_actual(categories: list, budgets: list, actuals: list) -> plt.Figure:
    """Grouped bar chart comparing recommended budget vs actual spending per category."""
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, budgets, width, label="Budget", color=_PALETTE[2], alpha=0.85)
    ax.bar(x + width / 2, actuals, width, label="Actual", color=_PALETTE[3], alpha=0.85)

    # Highlight overspending bars in red
    for i, (bgt, act) in enumerate(zip(budgets, actuals)):
        if act > bgt:
            ax.bar(x[i] + width / 2, act, width, color="#E63946", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_title("Budget vs Actual Spending (Monthly)", fontsize=14, fontweight="bold")
    ax.set_ylabel(_YLABEL_AMOUNT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_architecture() -> plt.Figure:
    """System architecture flowchart showing the 5-stage multi-agent pipeline."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    def box(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        x, y, w, h, text, color, fontsize=9, text_color="white"
    ):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                zorder=4, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.8}, zorder=2)

    def label(x, y, text, color="#555555"):
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                color=color, style="italic", zorder=5)

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(7, 8.6, "Personal Finance AI — Multi-Agent System Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

    # ── INPUT layer ────────────────────────────────────────────────────────
    box(0.4, 7.0, 3.0, 0.8,
        "User Profile\n(Income · Goals · Risk)", "#6C757D", fontsize=8)
    box(4.2, 7.0, 3.0, 0.8,
        "Expense Data\n(CSV Upload / Sample)", "#6C757D", fontsize=8)

    # ── ORCHESTRATOR frame ─────────────────────────────────────────────────
    orch_rect = mpatches.FancyBboxPatch(
        (0.2, 1.2), 13.6, 5.4,
        boxstyle="round,pad=0.15",
        facecolor="#EAF2FB", edgecolor="#2980B9", linewidth=2, zorder=1
    )
    ax.add_patch(orch_rect)
    ax.text(7, 6.45, "ORCHESTRATOR  (orchestrator.py)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#2980B9")

    # ── AGENT boxes ────────────────────────────────────────────────────────
    box(0.5, 4.4, 2.8, 1.4,
        "Agent 1\nExpense Tracker\n(expense_tracker.py)", "#1A73E8", fontsize=8)
    box(3.8, 4.4, 2.8, 1.4,
        "Agent 2\nBudget Planner\n(budget_planner.py)", "#0F9D58", fontsize=8)
    box(7.1, 4.4, 2.8, 1.4,
        "Agent 3\nFinancial Advisor\n(financial_advisor.py)", "#E37400", fontsize=8)
    box(10.4, 4.4, 2.8, 1.4,
        "Agent 4\nForecasting Agent\n(forecasting_agent.py)", "#9B27AF", fontsize=8)

    # ── DATA-FLOW arrows between agents ────────────────────────────────────
    arrow(3.3, 5.1, 3.8, 5.1)
    label(3.55, 5.35, "expense_analysis")

    arrow(6.6, 5.1, 7.1, 5.1)
    label(6.85, 5.35, "budget_plan")

    # ── TOOLS boxes (below agents) ─────────────────────────────────────────
    box(1.5, 2.4, 5.0, 0.9,
        "tools/data_tools.py  ·  tools/visualization.py\n(Pandas · Matplotlib)",
        "#455A64", fontsize=8)
    box(7.5, 2.4, 4.5, 0.9,
        "LangChain + Azure OpenAI\n(AzureChatOpenAI · ChatPromptTemplate)",
        "#455A64", fontsize=8)

    # Arrows from agents to tools
    arrow(1.9, 4.4, 2.5, 3.3)
    arrow(5.2, 4.4, 4.5, 3.3)
    arrow(8.5, 4.4, 8.8, 3.3)
    arrow(11.8, 4.4, 10.5, 3.3)

    # ── Synthesized report ─────────────────────────────────────────────────
    box(3.5, 1.5, 7.0, 0.7,
        "Executive Report  (synthesized from all 4 agent outputs)",
        "#34495E", fontsize=8.5)
    arrow(7.0, 4.4, 7.0, 2.2)
    label(7.6, 3.35, "all outputs")

    # ── Arrows from inputs to orchestrator ────────────────────────────────
    arrow(1.9, 7.0, 3.5, 6.55)
    arrow(5.7, 7.0, 6.5, 6.55)

    # Forecasting arrow from raw data (independent path)
    arrow(7.2, 7.0, 11.8, 5.84)
    label(9.8, 6.6, "raw DataFrame (independent)", "#9B27AF")

    # ── OUTPUT layer ──────────────────────────────────────────────────────
    box(4.0, 0.2, 6.0, 0.9,
        "Streamlit UI — 6 Tabs + Live Chat\n(app.py)", "#C0392B", fontsize=9)
    arrow(7.0, 1.5, 7.0, 1.1)

    plt.tight_layout(pad=0.5)
    return fig
