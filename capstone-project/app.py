"""
Personal Finance AI Assistant — Streamlit UI
Multi-agent system: Expense Tracker | Budget Planner | Financial Advisor | Forecasting
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
import pandas as pd
import streamlit as st

from orchestrator import FinancialOrchestrator
from tools.data_tools import load_expenses, validate_dataframe
from tools.visualization import (
    plot_architecture,
    plot_budget_vs_actual,
    plot_forecast,
    plot_monthly_trend,
    plot_spending_pie,
)
# pylint: enable=wrong-import-position

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = FinancialOrchestrator()
if "results" not in st.session_state:
    st.session_state.results = None
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Finance AI Assistant")
    st.markdown("---")

    st.subheader("📂 Expense Data")
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload CSV"],
        horizontal=True,
    )

    if data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload expenses CSV",
            type=["csv"],
            help="Required columns: date, amount, category, description",
        )
        if uploaded:
            try:
                df_uploaded = pd.read_csv(uploaded)
                df_uploaded["date"] = pd.to_datetime(df_uploaded["date"])
                df_uploaded["amount"] = df_uploaded["amount"].astype(float)
                ok, msg = validate_dataframe(df_uploaded)
                if not ok:
                    st.warning(f"CSV issue: {msg}")
                else:
                    st.session_state.df = df_uploaded
                    st.success(f"Loaded {len(df_uploaded)} transactions.")
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Could not read file: {e}")
        if st.session_state.df is None:
            st.session_state.df = load_expenses()
            st.info("No valid file yet — using sample data.")
    else:
        st.session_state.df = load_expenses()
        st.info("Using built-in sample data (April–June 2025).")

    st.markdown("---")
    st.subheader("⚙️ Financial Profile")

    monthly_income = st.number_input(
        "Monthly Income ($)",
        min_value=500.0,
        max_value=50000.0,
        value=5000.0,
        step=100.0,
    )

    financial_goals = st.text_area(
        "Financial Goals",
        value="Build a 6-month emergency fund, save for a vacation, and reduce dining expenses.",
        height=90,
    )

    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=1,
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")

    if run_btn:
        if st.session_state.df is None:
            st.error("No expense data loaded.")
        else:
            with st.spinner("Running 4 AI agents… this may take ~30 seconds."):
                try:
                    results = st.session_state.orchestrator.run_full_analysis(
                        df=st.session_state.df,
                        monthly_income=monthly_income,
                        financial_goals=financial_goals,
                        risk_tolerance=risk_tolerance,
                    )
                    st.session_state.results = results
                    st.session_state.chat_history = []
                    st.success("Analysis complete!")
                except ValueError as e:
                    st.error(f"Data validation error: {e}")
                except RuntimeError as e:
                    st.error(str(e))
                except (OSError, AttributeError) as e:  # final safety net
                    st.error(f"Unexpected error: {e}")

    if st.session_state.results is None:
        st.caption("Set your profile and click Run Full Analysis to start.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Personal Finance AI Assistant")
st.caption("Powered by 4 specialized AI agents · LangChain · Azure OpenAI")

if st.session_state.results is None:
    st.info(
        "👈  Configure your financial profile in the sidebar "
        "and click **Run Full Analysis** to begin."
    )
    st.markdown(
        "### What this system does\n"
        "| Agent | Role |\n"
        "|---|---|\n"
        "| 🔍 **Expense Tracker** | Categorizes spending, detects anomalies |\n"
        "| 📊 **Budget Planner** | Goal-aware budget based on 50/30/20 |\n"
        "| 💡 **Financial Advisor** | Risk-based investment & emergency fund advice |\n"
        "| 🔮 **Forecasting Agent** | Predicts next 3 months with confidence intervals |"
    )
    st.stop()

# ── Unpack results ────────────────────────────────────────────────────────────
res = st.session_state.results
ea = res["expense_analysis"]
bp = res["budget_plan"]
adv = res["advice"]
fc = res["forecast"]
df = st.session_state.df

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spent", f"${ea['total_spent']:,.0f}", f"{ea['n_months']} months")
c2.metric("Avg Monthly", f"${ea['avg_monthly']:,.0f}")
c3.metric(
    "Savings Rate",
    f"{adv['savings_rate']:.1f}%",
    "Target 20%" if adv["savings_rate"] < 20 else "On track ✓",
    delta_color="inverse" if adv["savings_rate"] < 20 else "normal",
)
c4.metric(
    "Next Month Forecast",
    f"${fc['predictions'][0]:,.0f}",
    f"{fc['trend']} trend",
    delta_color="inverse" if fc["trend"] == "increasing" else "normal",
)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Report",
    "🔍 Expense Tracker",
    "📊 Budget Planner",
    "💡 Financial Advisor",
    "🔮 Forecast + Ask AI",
    "🏗️ Architecture",
])

# ─── Tab 1: Executive Report ──────────────────────────────────────────────────
with tab1:
    st.subheader("Executive Financial Report")
    st.markdown(res["executive_report"])

    st.download_button(
        label="⬇️ Download Report",
        data=res["executive_report"],
        file_name="financial_report.txt",
        mime="text/plain",
    )

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Spending by Category")
        st.pyplot(plot_spending_pie(df))
    with col_r:
        st.subheader("Monthly Spending Trend")
        st.pyplot(plot_monthly_trend(df))

# ─── Tab 2: Expense Tracker ───────────────────────────────────────────────────
with tab2:
    st.subheader("Expense Tracker Agent")
    st.info(
        "**Agent role:** Monitors spending habits, "
        "categorizes transactions, and flags anomalies."
    )

    st.markdown(ea["ai_analysis"])

    _FMT_USD = "${:,.2f}".format
    _COL_TOTAL = "Total ($)"
    _COL_AVG = "Avg/Transaction ($)"

    # Anomaly alert
    if not ea["anomalies"].empty:
        st.warning(f"⚠️ {len(ea['anomalies'])} anomalous transaction(s) detected (IQR method)")
        with st.expander("View Anomalous Transactions"):
            disp = ea["anomalies"][["date", "amount", "category", "description"]].copy()
            disp["amount"] = disp["amount"].map(_FMT_USD)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalous transactions detected.")

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Category Breakdown")
        summary_display = ea["summary"].copy()
        summary_display.index.name = "Category"
        summary_display.columns = [_COL_TOTAL, "Transactions", _COL_AVG]
        summary_display[_COL_TOTAL] = summary_display[_COL_TOTAL].map(_FMT_USD)
        summary_display[_COL_AVG] = summary_display[_COL_AVG].map(_FMT_USD)
        st.dataframe(summary_display, use_container_width=True)

    with col_b:
        st.subheader("Top 5 Transactions")
        top = ea["top_transactions"].copy()
        top["amount"] = top["amount"].map(_FMT_USD)
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Add & Categorize a New Transaction")
    col_x, col_y, col_z = st.columns([3, 1, 1])
    with col_x:
        new_desc = st.text_input("Description", placeholder="e.g. Starbucks coffee")
    with col_y:
        new_amt = st.number_input("Amount ($)", min_value=0.01, value=10.00)
    with col_z:
        st.write("")
        st.write("")
        categorize_btn = st.button("Categorize", use_container_width=True)

    if categorize_btn and new_desc:
        from agents.expense_tracker import categorize_transaction
        with st.spinner("Asking AI…"):
            cat = categorize_transaction(new_desc, new_amt)
        st.success(f"**{new_desc}** → Category: **{cat}**")

# ─── Tab 3: Budget Planner ────────────────────────────────────────────────────
with tab3:
    st.subheader("Budget Planner Agent")
    st.info("**Agent role:** Generates a personalized goal-aware budget using the 50/30/20 rule.")

    if bp.get("savings_boost_pct", 0) > 0:
        st.success(
            f"Savings target boosted by **{bp['savings_boost_pct']*100:.0f}%** "
            f"(${bp['savings_boost_pct'] * bp['monthly_income']:,.0f}/mo extra) "
            f"based on your financial goals."
        )

    # 50/30/20 progress bars
    col_n, col_w, col_s = st.columns(3)
    housing_transport_util = sum(
        bp["monthly_actuals"].get(c, 0)
        for c in ["Housing", "Transportation", "Utilities", "Healthcare"]
    )
    wants_spend = sum(
        bp["monthly_actuals"].get(c, 0)
        for c in ["Food & Dining", "Entertainment", "Shopping", "Others"]
    )

    with col_n:
        st.metric("Needs Budget (50%)", f"${bp['needs_budget']:,.0f}/mo")
        pct_needs = (
            min(housing_transport_util / bp["needs_budget"], 1.0)
            if bp["needs_budget"] else 0
        )
        st.progress(pct_needs, text=f"${housing_transport_util:,.0f} used")

    with col_w:
        st.metric("Wants Budget (30%)", f"${bp['wants_budget']:,.0f}/mo")
        pct_wants = min(wants_spend / bp["wants_budget"], 1.0) if bp["wants_budget"] else 0
        st.progress(pct_wants, text=f"${wants_spend:,.0f} used")

    with col_s:
        st.metric("Savings Target", f"${bp['savings_budget']:,.0f}/mo")
        pct_saving = (
            min(adv["monthly_savings"] / bp["savings_budget"], 1.0)
            if bp["savings_budget"] else 0
        )
        st.progress(max(pct_saving, 0), text=f"${adv['monthly_savings']:,.0f} saved")

    st.markdown("---")
    st.subheader("Budget vs Actual (Monthly Average)")
    shared_cats = list(bp["recommended_allocations"].keys())
    budgets_list = [bp["recommended_allocations"].get(c, 0) for c in shared_cats]
    actuals_list = [bp["monthly_actuals"].get(c, 0) for c in shared_cats]
    st.pyplot(plot_budget_vs_actual(shared_cats, budgets_list, actuals_list))

    st.markdown("---")
    st.subheader("Budget Planner Recommendations")
    st.markdown(bp["budget_plan_text"])

# ─── Tab 4: Financial Advisor ─────────────────────────────────────────────────
with tab4:
    st.subheader("Financial Advisor Agent")
    st.info(
        "**Agent role:** Delivers risk-based investment advice, emergency fund targets, "
        "and prioritized action steps based on all agent outputs."
    )

    if adv["overspending_categories"]:
        st.warning(
            "**Overspending detected in:** "
            + ", ".join(
                f"{cat} (${v['excess']:,.0f}/mo over)"
                for cat, v in adv["overspending_categories"].items()
            )
        )

    st.markdown(adv["advice"])

    st.markdown("---")
    st.subheader("Investment Allocation Recommendation")
    inv_cols = st.columns(len(adv["investment_allocation"]))
    for col, (name, pct) in zip(inv_cols, adv["investment_allocation"].items()):
        col.metric(name, f"${adv['investable_monthly'] * pct:,.0f}/mo", f"{pct*100:.0f}%")

    st.markdown("---")
    col_ef1, col_ef2, col_ef3 = st.columns(3)
    col_ef1.metric("Monthly Savings", f"${adv['monthly_savings']:,.0f}")
    col_ef2.metric("Annual Projection", f"${adv['monthly_savings'] * 12:,.0f}")
    col_ef3.metric(
        "Emergency Fund Target",
        f"${adv['emergency_fund_target']:,.0f}",
        "6 months of expenses",
    )

# ─── Tab 5: Forecast + Ask AI ─────────────────────────────────────────────────
with tab5:
    col_fc, col_qa = st.columns([3, 2])

    with col_fc:
        st.subheader("Forecasting Agent — Next 3 Months")
        st.info(
            "**Agent role:** Predicts future expenses using linear trend analysis "
            "with statistical confidence intervals."
        )

        fc_col1, fc_col2, fc_col3 = st.columns(3)
        for col, month, pred in zip(
            [fc_col1, fc_col2, fc_col3], fc["forecast_months"], fc["predictions"]
        ):
            delta = pred - fc["avg_historical"]
            col.metric(str(month), f"${pred:,.0f}", f"{delta:+,.0f} vs avg")

        st.pyplot(
            plot_forecast(
                fc["historical_monthly"],
                fc["predictions"],
                fc["forecast_months"],
                confidence_std=fc.get("confidence_std"),
            )
        )

        st.markdown("#### Forecast Insights")
        st.markdown(fc["forecast_insights"])

        if fc["category_forecasts"]:
            with st.expander("Per-Category Forecasts"):
                fc_df = pd.DataFrame(
                    {
                        cat: [f"${v:,.0f}" for v in vals]
                        for cat, vals in fc["category_forecasts"].items()
                    },
                    index=[str(m) for m in fc["forecast_months"]],
                )
                st.dataframe(fc_df, use_container_width=True)

    with col_qa:
        st.subheader("Ask Your Financial AI")
        st.caption("Ask anything about your finances — the advisor uses your live data.")

        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

        user_q = st.chat_input("e.g. How can I save an extra $300 next month?")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = st.session_state.orchestrator.ask(user_q)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ─── Tab 6: Architecture ──────────────────────────────────────────────────────
with tab6:
    st.subheader("System Architecture — Multi-Agent Pipeline")
    st.caption(
        "High-level flowchart showing how the four AI agents collaborate "
        "through the central orchestrator."
    )
    st.pyplot(plot_architecture())

    st.markdown("""
    ### Agent Interaction Protocol

    | Stage | Agent | Receives | Produces |
    |---|---|---|---|
    | 1 | **Expense Tracker** | Raw DataFrame (CSV) | `expense_analysis` dict — totals, summaries, anomalies, AI insights |
    | 2 | **Budget Planner** | `expense_analysis` | `budget_plan` dict — goal-aware allocations, deltas, 50/30/20 targets |
    | 3 | **Financial Advisor** | `expense_analysis` + `budget_plan` | `advice` dict — investment allocation, emergency fund, prioritized steps |
    | 4 | **Forecasting Agent** | Raw DataFrame (independent) | `forecast` dict — linear predictions, confidence std, category forecasts |
    | 5 | **Orchestrator** | All four outputs | Executive Report (LLM synthesis) |

    ### Key Design Decisions
    - **Sequential pipeline with explicit dependencies**: Agents 2 and 3 build on Agent 1's output, ensuring each layer enriches the context.
    - **Agent 4 is independent**: Forecasting runs directly on raw data so it can be called standalone without the full pipeline.
    - **Stateful orchestrator**: `FinancialOrchestrator` caches agent outputs, enabling the live Q&A chat without re-running expensive LLM calls.
    - **Error isolation**: Each agent call is wrapped in a try/except so a single API failure names exactly which agent failed.
    - **Goal-aware budget**: The Budget Planner reads financial goal keywords and dynamically adjusts savings allocation (e.g., "emergency fund" → +5% savings).
    """)
