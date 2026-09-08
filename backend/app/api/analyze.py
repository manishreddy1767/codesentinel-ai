from fastapi import APIRouter, HTTPException

from app.schemas.request import AnalyzeRequest
from app.services.analyzer import analyze_code


router = APIRouter()


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Analyze source code and generate
    its Code Property Graph.
    """

    try:

        result = analyze_code(
            code=request.code,
            language=request.language,
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )