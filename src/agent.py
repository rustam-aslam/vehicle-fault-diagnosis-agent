from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from src.retriever import retrieve_documents
from src.config import OPENAI_API_KEY


client = OpenAI(
    api_key=OPENAI_API_KEY
)


class DiagnosisState(TypedDict):
    question: str
    context: str
    diagnosis: str
    documents: list
    relevant: bool
    retrieval_attempts: int


def retrieve_node(state: DiagnosisState):

    results = retrieve_documents(
        state["question"],
        top_k=2
    )

    context = ""

    for match in results:
        context += (
            f"Source: {match['metadata']['source']}\n"
            f"{match['metadata']['text']}\n\n"
        )

    return {
        "context": context,
        "documents": results,
        "retrieval_attempts": state["retrieval_attempts"] + 1
    }


def generate_node(state: DiagnosisState):

    prompt = f"""
You are a vehicle fault diagnosis assistant.

Use only the technical information provided
below to help answer the user's question.

Do not invent technical information that is
not supported by the provided documentation.

Technical Information:
{state["context"]}

User Question:
{state["question"]}

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

    return {
        "diagnosis": response.choices[0].message.content
    }


def check_relevance(state):

    matches = state["documents"]

    if not matches:
        print("No documents retrieved.")
        return {"relevant": False}

    best_score = matches[0]["score"]

    print(f"Best similarity score: {best_score:.3f}")

    if best_score >= 0.60:
        print("Retrieval is relevant.")
        return {"relevant": True}

    if state["retrieval_attempts"] >= 2:
        print("Maximum retrieval attempts reached.")
        return {"relevant": True}

    print("Retrieval is not relevant. Retrying...")
    return {"relevant": False}


workflow = StateGraph(DiagnosisState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_node(
    "check_relevance",
    check_relevance
)

workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "check_relevance"
)

workflow.add_conditional_edges(
    "check_relevance",
    lambda state: "generate" if state["relevant"] else "retrieve"
)

workflow.add_edge(
    "generate",
    END
)

app = workflow.compile()


if __name__ == "__main__":

    result = app.invoke({
        "question": (
            "My transmission is slipping when I accelerate. "
            "What could be causing this?"
        ),
        "context": "",
        "diagnosis": "",
        "documents": [],
        "relevant": False,
        "retrieval_attempts": 0
    })

    print("\nDiagnosis:\n")
    print(result["diagnosis"])