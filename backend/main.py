from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model import TransformerEvaluatorService

app = FastAPI(
    title="Confidence-Based Token Evaluator API",
    description="Backend API for Confidence-Based Token Evaluator in Transformer Models"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Transformer Service lazily or on startup
service = None

@app.on_event("startup")
def startup_event():
    global service
    try:
        service = TransformerEvaluatorService("distilgpt2")
        print("[FastAPI] TransformerEvaluatorService successfully initialized!")
    except Exception as e:
        print(f"[FastAPI Error] Failed to initialize model: {e}")

class EvaluateRequest(BaseModel):
    text: str
    threshold: float = 0.50

class RecalculateRequest(BaseModel):
    token_analysis: List[dict]
    threshold: float = 0.50

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": service is not None}

@app.post("/api/evaluate")
def evaluate_text(req: EvaluateRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Transformer Model service is loading...")
    
    result = service.evaluate_text(req.text, req.threshold)
    return result

@app.post("/api/recalculate")
def recalculate_metrics(req: RecalculateRequest):
    threshold = req.threshold
    token_analysis = req.token_analysis
    
    retained_tokens = []
    filtered_tokens = []
    updated_analysis = []
    
    for item in token_analysis:
        score = item.get("confidence", 0.0)
        is_retained = score >= threshold
        status = "Retained" if is_retained else "Filtered"
        
        updated_item = {
            "id": item.get("id"),
            "token": item.get("token"),
            "confidence": score,
            "status": status,
            "retained": is_retained
        }
        updated_analysis.append(updated_item)
        
        if is_retained:
            retained_tokens.append(item.get("token"))
        else:
            filtered_tokens.append(item.get("token"))
            
    # Guarantee at least 1 token retained
    if not retained_tokens and updated_analysis:
        highest_item = max(updated_analysis, key=lambda x: x["confidence"])
        highest_item["status"] = "Retained"
        highest_item["retained"] = True
        retained_tokens.append(highest_item["token"])
        if highest_item["token"] in filtered_tokens:
            filtered_tokens.remove(highest_item["token"])

    original_tokens = len(updated_analysis)
    selected_tokens = len(retained_tokens)
    tokens_saved = original_tokens - selected_tokens
    saving_percentage = round((tokens_saved / original_tokens) * 100, 2) if original_tokens > 0 else 0.0

    return {
        "token_analysis": updated_analysis,
        "original_tokens": original_tokens,
        "selected_tokens": selected_tokens,
        "tokens_saved": tokens_saved,
        "saving_percentage": saving_percentage,
        "retained_tokens": retained_tokens,
        "filtered_tokens": filtered_tokens,
        "confidence_threshold": threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
