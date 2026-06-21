""" Course generator """
import os
import re
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=str(os.getenv("AZURE_OPENAI_ENDPOINT")),
)

DEPLOYMENT_NAME = "gpt-4.1-mini"


def generate_course(skill, level, num_modules=5):
    """ Generate a course """
    print("Sending a test completion job")
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an AI course generator."},
            {
                "role": "user",
                "content": f"Generate a course outline for {skill} at {level} level "
                           f"with exactly {num_modules} modules. List each module "
                           f"with a title and a brief description containing "
                           f"reference link to learn.",
            },
        ],
        max_tokens=1000,
    )
    generated_response = response.choices[0].message.content
    module_titles = re.findall(r"Module\s\d+:.*", generated_response)
    # Extract Module Descriptions
    module_descriptions = re.split(r"Module\s\d+:.*", generated_response)[1:]

    # Cleaning descriptions
    clean_descriptions = [desc.strip() for desc in module_descriptions]

    # Ensure both lists are equal length
    if len(module_titles) != len(clean_descriptions):
        print("⚠ Warning: Titles and Descriptions count mismatch!")

    # Combine and return modules as structured data
    course_modules = [
        {
            "title": title,
            "description": desc
        } for title, desc in zip(module_titles, clean_descriptions)
    ]

    return course_modules
