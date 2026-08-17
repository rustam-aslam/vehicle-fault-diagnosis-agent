from pathlib import Path

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


def create_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def load_documents():

    documents = []

    for file_path in Path("data").glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "text": text,
            "source": file_path.name
        })

    return documents


def main():

    documents = load_documents()

    vectors = []

    for i, document in enumerate(documents):

        print(
            f"Creating embedding for "
            f"{document['source']}..."
        )

        embedding = create_embedding(
            document["text"]
        )

        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {
                "source": document["source"],
                "text": document["text"]
            }
        })

    index.upsert(
        vectors=vectors
    )

    print(
        f"Uploaded {len(vectors)} documents to Pinecone."
    )


if __name__ == "__main__":
    main()