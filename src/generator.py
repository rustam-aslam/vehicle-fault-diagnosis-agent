from openai import OpenAI

from retriever import retrieve_documents
from config import OPENAI_API_KEY


client = OpenAI(
    api_key=OPENAI_API_KEY
)


def generate_diagnosis(question):

    results = retrieve_documents(
        question,
        top_k=2
    )

    context = ""

    for match in results["matches"]:

        context += (
            f"Source: {match['metadata']['source']}\n"
            f"{match['metadata']['text']}\n\n"
        )

    prompt = f"""
You are a vehicle fault diagnosis assistant.

Use the technical information provided below
to help diagnose the vehicle problem.

Do not invent technical information that is
not supported by the provided documentation.

Technical Information:
{context}

User Question:
{question}

Provide:
1. Possible causes
2. Recommended diagnostic checks
3. A brief explanation of your reasoning
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful vehicle "
                    "diagnosis assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    question = (
        "My transmission is slipping "
        "when I accelerate. What could "
        "be causing this?"
    )

    diagnosis = generate_diagnosis(
        question
    )

    print("\nDiagnosis:\n")
    print(diagnosis)