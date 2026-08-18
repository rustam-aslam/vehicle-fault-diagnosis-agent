# Vehicle Fault Diagnosis Agent

## Overview

The Vehicle Fault Diagnosis Agent is an AI-powered diagnostic application designed to help identify potential vehicle faults from user-described symptoms.

The project combines a **Retrieval-Augmented Generation (RAG)** pipeline with a **LangGraph-based workflow** to ground AI-generated diagnoses in a collection of vehicle fault documentation. Instead of relying solely on the language model's pretrained knowledge, the system first searches a vector database for relevant technical information and then provides that information to the model as context when generating a response.

For example, a user can enter:

> "My transmission is slipping when I accelerate. What could be causing this?"

The system converts the question into an embedding and searches a **Pinecone vector database** containing technical vehicle fault documentation. The most relevant documents are retrieved and evaluated for relevance. If the retrieved information does not meet the required similarity threshold, the LangGraph workflow performs another retrieval attempt.

Once relevant information is available, the retrieved documentation is passed to an **OpenAI language model**, which generates a structured diagnostic response containing possible causes, recommended diagnostic checks, and an explanation of the reasoning.

The application is exposed through a **FastAPI** backend and provides a user-friendly **Streamlit** interface for interacting with the system. The entire application can also be containerized using **Docker**.

The project demonstrates an end-to-end AI application architecture rather than simply making a direct LLM API call. It includes document ingestion, vector search, retrieval evaluation, conditional agent workflow, LLM generation, API development, frontend development, and containerization.

## Architecture

The system follows the architecture below:

```text
                         User
                           │
                           ▼
                    Streamlit UI
                           │
                           │ HTTP POST
                           ▼
                      FastAPI API
                           │
                           ▼
                     LangGraph
                    Agent Workflow
                           │
                           ▼
                  Generate Query Embedding
                           │
                           ▼
                     Pinecone Search
                           │
                           ▼
                  Retrieved Documents
                           │
                           ▼
                  Relevance Evaluation
                     /            \
                    /              \
              Relevant          Not Relevant
                 │                    │
                 │                    ▼
                 │              Retry Retrieval
                 │                    │
                 └────────────────────┘
                           │
                           ▼
                   Retrieved Context
                           │
                           ▼
                      OpenAI LLM
                           │
                           ▼
                   Generated Diagnosis
                           │
                           ▼
                      FastAPI
                           │
                           ▼
                    Streamlit UI
```

### 1. User Interface

The user interacts with the application through a **Streamlit** web interface.

The user enters a description of a vehicle problem, such as a transmission symptom. Streamlit sends the question to the FastAPI backend through an HTTP POST request.

The UI is intentionally separated from the backend so that the application has a clear frontend/backend architecture.

### 2. FastAPI Backend

**FastAPI** provides the REST API layer between the frontend and the AI system.

The `/diagnose` endpoint accepts a vehicle fault description:

```text
POST /diagnose
```

The request is validated using **Pydantic**, and the question is passed into the LangGraph workflow.

FastAPI then returns the generated diagnosis to the Streamlit application.

This separation allows the AI workflow to operate independently from the user interface and makes the system easier to integrate with other applications in the future.

### 3. LangGraph Workflow

**LangGraph** is used to control the diagnostic workflow.

The workflow maintains a shared state containing information such as:

* User question
* Retrieved context
* Retrieved documents
* Retrieval relevance
* Number of retrieval attempts
* Final diagnosis

The primary workflow consists of three nodes:

```text
Retrieve
   ↓
Check Relevance
   ↓
Generate
```

However, the workflow also contains a conditional path.

After documents are retrieved, their Pinecone similarity scores are evaluated. If the retrieved information meets the relevance threshold, the workflow proceeds to generation.

If the retrieval result is below the threshold, the workflow returns to the retrieval node and performs another retrieval attempt.

This makes the system more robust than a simple linear RAG pipeline.

### 4. Document Retrieval

The retrieval system uses **Pinecone** as the vector database.

The project contains technical documentation covering different vehicle fault categories, including:

```text
engine_faults.txt
sensor_faults.txt
transmission_faults.txt
```

The documents are converted into embeddings and stored in Pinecone.

When a user submits a question, the question is also converted into an embedding. Pinecone compares the query vector against the stored document vectors using cosine similarity and returns the most relevant results.

The retrieval process therefore looks like:

```text
User Question
      ↓
OpenAI Embedding
      ↓
1536-dimensional Vector
      ↓
Pinecone Similarity Search
      ↓
Top-K Documents
```

The retrieved documents are then passed into the LangGraph workflow.

### 5. Conditional Retrieval

A key feature of the architecture is the relevance check after retrieval.

The system examines the similarity score of the highest-ranked result.

If the score is above the configured threshold:

```text
Retrieval → Relevant → Generate
```

If the score is below the threshold:

```text
Retrieval → Not Relevant → Retrieve Again
```

The workflow also tracks the number of retrieval attempts and prevents the system from continuously retrying.

This demonstrates how LangGraph can be used to build decision-based AI pipelines rather than simply chaining together fixed operations.

### 6. LLM Generation

After relevant documentation has been retrieved, the context is provided to an **OpenAI language model**.

The generation prompt instructs the model to:

1. Use the retrieved technical information.
2. Avoid inventing unsupported technical information.
3. Identify possible causes.
4. Recommend diagnostic checks.
5. Explain the reasoning behind the diagnosis.

The overall generation process is:

```text
User Question
      +
Retrieved Technical Context
      ↓
OpenAI
      ↓
Diagnostic Response
```

This approach uses the language model primarily for reasoning and synthesis while relying on the retrieved documentation to provide the technical knowledge.

* Git
* GitHub
