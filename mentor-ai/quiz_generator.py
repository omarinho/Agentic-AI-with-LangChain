""" Quiz Generator Agent — generates 5 MCQ questions per module and evaluates answers """
import os
import json
import re
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
)
DEPLOYMENT_NAME = "gpt-4.1-mini"


def generate_quiz(skill: str, module: dict) -> list[dict]:
    """Return 5 MCQ questions for the given module as a list of dicts."""
    prompt = (
        f"You are a quiz generator. Generate exactly 5 multiple-choice questions "
        f"to test understanding of the '{module['title']}' module in the '{skill}' skill. "
        f"Return ONLY a JSON array with no markdown fences. Each element must have: "
        f'"question" (string), "options" (object with keys A, B, C, D), "answer" (A/B/C/D).'
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a quiz generator agent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip optional Markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def evaluate_quiz(questions: list[dict], user_answers: dict) -> dict:
    """
    Evaluate user answers against correct answers and ask the LLM for feedback.
    user_answers: {0: 'A', 1: 'C', ...} keyed by question index.
    Returns {"score": int, "total": int, "feedback": str, "details": list[dict]}.
    """
    details = []
    score = 0
    for i, q in enumerate(questions):
        selected = user_answers.get(i, "")
        correct = q["answer"]
        is_correct = selected.upper() == correct.upper()
        if is_correct:
            score += 1
        details.append({
            "question": q["question"],
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
        })

    summary = "\n".join(
        f"Q{i+1}: {d['question']} | Selected: {d['selected']} | "
        f"Correct: {d['correct']} | {'✅' if d['is_correct'] else '❌'}"
        for i, d in enumerate(details)
    )
    prompt = (
        f"A learner scored {score}/{len(questions)} on a module quiz.\n"
        f"Here are the results:\n{summary}\n"
        f"Provide concise feedback (3-4 sentences) highlighting strengths and areas to improve."
    )
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a quiz evaluation agent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
    )
    feedback = (response.choices[0].message.content or "").strip()
    return {"score": score, "total": len(questions), "feedback": feedback, "details": details}
