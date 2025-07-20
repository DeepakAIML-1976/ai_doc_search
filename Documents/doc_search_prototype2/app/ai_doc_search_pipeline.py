from fastapi import FastAPI, UploadFile, File
from app.doc_ingestor import ingest_documents, search_documents
from app.feedback_logger import log_feedback
from app.learner import retrain_model_from_feedback
import os

app = FastAPI()

@app.post("/upload_docs/")
async def upload_docs(file: UploadFile = File(...)):
    file_path = f"data/documents/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    ingest_documents()
    return {"status": "Document uploaded and indexed"}

@app.get("/search/")
async def search(query: str):
    results = search_documents(query)
    return {"query": query, "results": results}

@app.post("/feedback/")
async def feedback(query: str, selected_result: str, relevance: bool):
    log_feedback(query, selected_result, relevance)
    return {"status": "Feedback logged"}

@app.post("/retrain/")
def retrain():
    retrain_model_from_feedback()
    return {"status": "Model updated with feedback"}
