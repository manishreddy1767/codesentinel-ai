from pydantic import BaseModel, Field


class CPGNode(BaseModel):
    id: int
    type: str
    graph_type: str
    start_line: int | None = None
    end_line: int | None = None


class Vulnerability(BaseModel):
    type: str
    severity: str
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    title: str | None = None
    description: str | None = None
    recommendation: str | None = None
    cwe: str | None = None
    line: int | None = None
    code: str | None = None
    function: str | None = None
    sink: str | None = None
    cpg_nodes: list[CPGNode] = []


class SecurityRisk(BaseModel):
    security_score: int = Field(
        ...,
        ge=0,
        le=100,
    )
    risk_level: str
    severity_counts: dict[str, int]


class AnalyzeResponse(BaseModel):
    analysis_id: str
    filename: str | None = None
    language: str

    security_risk: SecurityRisk

    vulnerability_summary: dict

    vulnerabilities: list[Vulnerability]

    cpg_metadata: dict
