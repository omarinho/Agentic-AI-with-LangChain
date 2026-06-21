# Personal Finance AI Assistant

> A multi-agent AI system that analyzes your expenses, builds a goal-aware budget, delivers personalized investment advice, and forecasts your spending — all powered by **LangChain** and **Azure OpenAI**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Using the App](#using-the-app)
- [CSV Format](#csv-format)
- [Code Quality](#code-quality)
- [Tech Stack](#tech-stack)

---

## Overview

This capstone project implements a **4-agent orchestration pipeline** for personal finance intelligence. Each agent is a specialized LangChain chain backed by `gpt-4.1-mini` on Azure OpenAI. A central orchestrator coordinates the agents, caches their outputs for follow-up Q&A, and synthesizes an executive financial report.

```
User Input → Orchestrator → 4 AI Agents → Charts + Report → Streamlit UI
```

Live demo output (sample data, April–June 2025):

| Metric | Value |
|---|---|
| Total Spent | $9,508 over 3 months |
| Avg Monthly | $3,169 |
| Savings Rate | 36.6% — above target |
| Next Month Forecast | $3,619 (increasing trend) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                        │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐                   │
│  │ Agent 1      │──▶│ Agent 2      │                   │
│  │ Expense      │   │ Budget       │                   │
│  │ Tracker      │   │ Planner      │──▶ Agent 3        │
│  └──────────────┘   └──────────────┘    Financial      │
│                                          Advisor        │
│  ┌──────────────┐                                       │
│  │ Agent 4      │  (independent — runs on raw data)    │
│  │ Forecasting  │                                       │
│  └──────────────┘                                       │
│                                                         │
│        ▼ All outputs synthesized into:                  │
│        Executive Financial Report                       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Streamlit UI (6 tabs)
```

| Agent | Input | Output |
|---|---|---|
| **Expense Tracker** | Raw expense DataFrame | Spending summary, anomaly detection, AI narrative |
| **Budget Planner** | Expense analysis + income + goals | Goal-aware 50/30/20 allocations, deltas, AI plan |
| **Financial Advisor** | Expense + budget outputs | Investment allocation, emergency fund target, AI advice |
| **Forecasting Agent** | Raw DataFrame (independent) | 3-month linear forecast, confidence intervals, AI insights |
| **Orchestrator** | All four agent outputs | Executive report + stateful Q&A |

---

## Features

### Agent Intelligence
- **IQR anomaly detection** — flags per-category spending outliers automatically
- **Goal-aware budgeting** — detects keywords ("emergency fund", "retirement", "invest") and dynamically boosts the savings allocation
- **Risk-based investment allocation** — conservative / moderate / aggressive portfolios with monthly dollar amounts
- **Statistical forecasting** — linear regression with residual-std confidence bands (not hardcoded ±10%)
- **6-month emergency fund target** — calculated from actual average monthly spending

### Streamlit UI (6 tabs)
| Tab | Contents |
|---|---|
| Executive Report | 4 KPI metrics, AI-generated financial snapshot, priorities, outlook |
| Expense Tracker | Spending pie chart, monthly trend bar chart, anomaly table |
| Budget Planner | Budget vs actual grouped bar chart, goal-aware allocation breakdown |
| Financial Advisor | Investment allocation metrics, emergency fund target, AI advice |
| Forecast + Ask AI | Forecast line chart with confidence band, 3-month projections, live Q&A chat |
| Architecture | System flowchart of the multi-agent pipeline |

### Code Quality
- **Pylint 10.00/10** across all source files
- **Mypy** — 0 type errors (11 files)
- **SonarQube** — 0 issues
- Input validation, per-agent error handling, structured logging throughout

---

## Project Structure

```
capstone-project/
├── app.py                        # Streamlit UI — 6 tabs, sidebar, session state
├── config.py                     # Azure credentials, budget constants, goal keywords
├── orchestrator.py               # Central coordinator + stateful Q&A
├── requirements.txt
│
├── agents/
│   ├── expense_tracker.py        # Agent 1 — spending analysis + anomaly detection
│   ├── budget_planner.py         # Agent 2 — goal-aware 50/30/20 budget
│   ├── financial_advisor.py      # Agent 3 — investment allocation + emergency fund
│   └── forecasting_agent.py      # Agent 4 — linear forecast + confidence intervals
│
├── tools/
│   ├── data_tools.py             # CSV loading, monthly aggregations, validation
│   └── visualization.py          # Matplotlib chart builders (pie, bar, forecast, arch)
│
└── data/
    └── sample_expenses.csv       # 53 transactions, Apr–Jun 2025, 9 categories
```

---

## Prerequisites

- Python 3.11+
- An **Azure OpenAI** resource with a `gpt-4.1-mini` deployment
- API version: `2024-12-01-preview`

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Agentic-AI-with-LangChain

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r capstone-project/requirements.txt
```

---

## Configuration

Create a `.env` file in the **`capstone-project/`** directory (never committed — covered by `.gitignore`):

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | API key from your Azure OpenAI resource |
| `AZURE_OPENAI_ENDPOINT` | Endpoint URL (include trailing `/`) |

The deployment name (`gpt-4.1-mini`) and API version (`2024-12-01-preview`) are set in `config.py`.

---

## Running the App

```bash
streamlit run capstone-project/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the App

1. **Choose data source** — use the built-in sample CSV (April–June 2025) or upload your own
2. **Set your financial profile** in the sidebar:
   - Monthly income
   - Financial goals (free text — goal keywords trigger smart budget adjustments)
   - Risk tolerance (conservative / moderate / aggressive)
3. **Click "Run Full Analysis"** — all 4 agents execute in sequence (~15–30 s)
4. **Explore the 6 tabs** — charts, AI narratives, investment breakdown, forecast
5. **Ask follow-up questions** in the Forecast + Ask AI tab chat

---

## CSV Format

Upload your own expense data with these required columns:

| Column | Type | Example |
|---|---|---|
| `date` | YYYY-MM-DD | `2025-04-03` |
| `amount` | numeric | `1500.00` |
| `category` | string | `Housing` |
| `description` | string | `Monthly rent` |

Suggested categories: `Housing`, `Food & Dining`, `Transportation`, `Entertainment`, `Healthcare`, `Utilities`, `Shopping`, `Education`, `Savings`, `Others`

---

## Code Quality

```bash
# Pylint (10.00/10)
python -m pylint capstone-project/ --max-line-length=100

# Mypy (0 issues)
python -m mypy capstone-project --ignore-missing-imports

# Run the app
streamlit run capstone-project/app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI — `gpt-4.1-mini` (API version `2024-12-01-preview`) |
| Agent framework | LangChain (`ChatPromptTemplate` → `AzureChatOpenAI` → `StrOutputParser`) |
| Data processing | Pandas, NumPy |
| Visualizations | Matplotlib (Agg backend — Streamlit compatible) |
| UI | Streamlit |
| Type safety | Pydantic v2, Mypy |
| Code quality | Pylint, SonarQube |
