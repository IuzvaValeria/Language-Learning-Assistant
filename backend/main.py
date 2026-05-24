from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.llm_service import load_model, generate_response
from backend.prompts import AVAILABLE_MODES

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    mode: str
    text: str

class AskResponse(BaseModel):
    response: str
    mode: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Japanese tutor backend is running"}

@app.get("/modes")
def get_modes():
    return {"available_modes": list(AVAILABLE_MODES.keys())}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.mode not in AVAILABLE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: {request.mode}. Available: {list(AVAILABLE_MODES.keys())}"
        )
    try:
        result = generate_response(request.mode, request.text)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    return AskResponse(response=result, mode=request.mode)