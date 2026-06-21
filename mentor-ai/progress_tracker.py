""" Progress Tracker """
import os
from openai import AzureOpenAI
import streamlit as st

TOTAL_MODULES = 5
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
)
DEPLOYMENT_NAME = "gpt-4.1-mini"

@st.dialog("Track Progress")
def track_progress(learning_history, module_finished):
    """ Track progress """
    completed_count = len(module_finished)
    progress = completed_count / TOTAL_MODULES
    prompt = (f"You are learning progress tracking agent, analyse "
              f"the learner's learning history - {learning_history} "
              f"and generate its progress in four line.")

    if not learning_history:
        return "📌 You have not completed any module yet. Start learning to track progress!"
    with st.spinner("Generating Progress"):
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a progress tracking agent"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )
    if response:
        st.progress(progress)
        st.write(f"{len(module_finished)} module completed" )
    st.write(response.choices[0].message.content)
    return None
