from pydantic import BaseModel, Field


class Vulnerability(BaseModel):
    type: str
    severity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    line: int | None = None
    description: str | None = None


class AnalyzeResponse(BaseModel):
    analysis_id: str
    filename: str | None
    language: str
    security_score: int = Field(..., ge=0, le=100)
    vulnerabilities: list[Vulnerability]
