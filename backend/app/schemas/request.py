from pydantic import BaseModel, Field, field_validator


SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "java",
    "c",
    "cpp",
}


class AnalyzeRequest(BaseModel):

    code: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Source code to analyze (maximum 1 MB)",
    )

    language: str = Field(
        ...,
        description="Programming language of the source code",
    )

    filename: str | None = Field(
        default=None,
        max_length=255,
        description="Optional source filename",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str):

        language = value.strip().lower()

        aliases = {
            "js": "javascript",
            "py": "python",
            "c++": "cpp",
        }

        language = aliases.get(
            language,
            language,
        )

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                "Unsupported language. Supported languages are: "
                "python, javascript, java, c, cpp."
            )

        return language
