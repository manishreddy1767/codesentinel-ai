from uuid import uuid4

from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse
from app.services.language_service import normalize_language
from app.services.parser_service import parse_source_code


def analyze_code(request: AnalyzeRequest) -> AnalyzeResponse:
    language = normalize_language(request.language)

    parsed_code = parse_source_code(
        code=request.code,
        language=language,
    )

    return AnalyzeResponse(
        analysis_id=str(uuid4()),
        filename=request.filename,
        language=language,
        security_score=100,
        vulnerabilities=[],
    )