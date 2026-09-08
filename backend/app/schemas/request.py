from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    code: str = Field(
        ...,
        min_length=1,
        description="Source code to analyze",
    )

    language: str = Field(
        ...,
        description="Programming language of the source code",
    )

    filename: str | None = Field(
        default=None,
        description="Optional source filename",
    )