from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import app


api = FastAPI(
    title="Vehicle Fault Diagnosis API",
    description="AI-powered vehicle fault diagnosis using RAG",
    version="1.0"
)


class DiagnosisRequest(BaseModel):
    question: str


class DiagnosisResponse(BaseModel):
    diagnosis: str


@api.get("/")
def health_check():

    return {
        "status": "Vehicle Fault Diagnosis API is running"
    }


@api.post(
    "/diagnose",
    response_model=DiagnosisResponse
)
def diagnose(request: DiagnosisRequest):

    result = app.invoke({
        "question": request.question,
        "context": "",
        "diagnosis": "",
        "documents": [],
        "relevant": False,
        "retrieval_attempts": 0
    })

    return {
        "diagnosis": result["diagnosis"]
    }