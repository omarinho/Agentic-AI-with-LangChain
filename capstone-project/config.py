""" Central configuration for the Personal Finance AI Assistant """
import os

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-mini"

EXPENSE_CATEGORIES = [
    "Housing",
    "Food & Dining",
    "Transportation",
    "Entertainment",
    "Healthcare",
    "Utilities",
    "Shopping",
    "Education",
    "Savings",
    "Others",
]

# 50/30/20 budget rule allocations (as fraction of income)
BUDGET_ALLOCATIONS = {
    "Housing": 0.25,
    "Food & Dining": 0.12,
    "Transportation": 0.10,
    "Utilities": 0.05,
    "Healthcare": 0.05,
    "Entertainment": 0.08,
    "Shopping": 0.10,
    "Education": 0.05,
    "Savings": 0.20,
}

# Overspending threshold: 1% of monthly income (scales with earner, min $10)
OVERSPENDING_THRESHOLD_PCT = 0.01

# Required columns for uploaded expense CSVs
REQUIRED_CSV_COLUMNS = {"date", "amount", "category", "description"}

# Goal keywords that boost the savings allocation (fractional)
GOAL_SAVINGS_BOOST = {
    "emergency fund": 0.05,
    "retirement": 0.05,
    "invest": 0.03,
}

# Goal keywords that allow slight wants increase (not used to cut savings)
GOAL_WANTS_ADJUSTMENT = {
    "vacation": 0.03,
    "travel": 0.03,
}
