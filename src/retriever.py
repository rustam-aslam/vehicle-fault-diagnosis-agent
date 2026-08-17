from openai import OpenAI
from pinecone import Pinecone

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)


def create_query_embedding(query):

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return response.data[0].embedding


def retrieve_documents(query, top_k=2):
    
    query_embedding = create_query_embedding(
        query
    )

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = results["matches"]

    return matches


if __name__ == "__main__":

    question = (
        "My transmission is slipping "
        "when I accelerate. What could "
        "be causing this?"
    )

    results = retrieve_documents(question)

    print("\nRetrieved documents:\n")

    for match in results["matches"]:

        print(
            f"Score: {match['score']}"
        )

        print(
            f"Source: "
            f"{match['metadata']['source']}"
        )

        print(
            match["metadata"]["text"]
        )

        print("-" * 50)