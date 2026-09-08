from fastapi import APIRouter, HTTPException

from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse
from app.services.analyzer import analyze_code


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        return analyze_code(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
