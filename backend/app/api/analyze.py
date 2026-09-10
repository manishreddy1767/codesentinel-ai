from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse
from app.services.analyzer import analyze_code


router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
def analyze(request: AnalyzeRequest):
    """
    Analyze source code for vulnerabilities.

    Returns a compact analysis response suitable
    for frontend consumption.
    """

    try:

        result = analyze_code(
            code=request.code,
            language=request.language,
        )

        return {
            "analysis_id": str(uuid4()),

            "filename": request.filename,

            "language": result["language"],

            "security_risk": result["security_risk"],

            "vulnerability_summary":
                result["vulnerability_summary"],

            "vulnerabilities":
                result["vulnerabilities"],

            "cpg_metadata":
                result["cpg"]["metadata"],
        }

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
