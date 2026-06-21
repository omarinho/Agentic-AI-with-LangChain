""" Question and Feedback generator agent """
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
)
DEPLOYMENT_NAME = "gpt-4.1-mini"

def generate_question(skill_name, module_name):
    """`Generate Question` """
    prompt = (f"Generate a followup question that must be asked after a person "
              f"complete the learning of the {module_name} module of {skill_name} skill")
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are a follow-up agent"},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

def generate_feedback(question, answer):
    """ Generate Feedback """
    prompt = (f"You need to provide feedback to learner's "
              f"answer - {answer} to the follow-up question - {question}")
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are a follow-up agent"},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content
