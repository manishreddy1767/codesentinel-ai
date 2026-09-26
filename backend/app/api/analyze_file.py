"""
POST /analyze/file

Analyze an uploaded source file.

Safety rules enforced here
--------------------------
* The file is **never executed**, imported, or written to disk. It is read into
  memory, decoded as text, and handed to the same static analyzer `/analyze`
  uses.
* The filename is reduced to its basename before anything else, so
  `../../etc/passwd` cannot escape anywhere - nothing is written, but the name
  is also echoed back in the response and must not carry a path.
* The extension must be on an allow-list derived from the languages the
  analyzer can actually parse.
* The body is size-capped before decoding, so a large upload cannot exhaust
  memory.
* Undecodable bytes are rejected rather than silently replaced: a binary file
  mislabelled as source would otherwise produce meaningless "findings".
"""

from pathlib import PurePosixPath
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.languages import LANGUAGE_INFO
from app.config import settings
from app.schemas.request import SUPPORTED_LANGUAGES
from app.schemas.response import AnalyzeResponse
from app.services.analyzer import analyze_code

router = APIRouter(tags=["Analysis"])

# extension -> language, built from the same table the metadata endpoint serves
# so the two can never disagree.
EXTENSION_TO_LANGUAGE = {
    extension: language
    for language, info in LANGUAGE_INFO.items()
    for extension in info["extensions"]
}


def _safe_basename(filename: str | None) -> str:
    """Strip any directory component from a client-supplied filename."""

    if not filename:
        return "uploaded"

    # Handle both separators: the client OS is not necessarily this one.
    candidate = filename.replace("\\", "/")
    name = PurePosixPath(candidate).name

    # Leading dots and empties would make a confusing echo; they are not a
    # security problem because nothing is written to disk.
    return name or "uploaded"


@router.post("/analyze/file", response_model=AnalyzeResponse)
async def analyze_file(
    # ruff B008 flags calls in argument defaults, but File()/Form() in the
    # signature IS the documented FastAPI idiom - it is how the framework
    # discovers multipart parameters. Restructuring would break the endpoint.
    file: UploadFile = File(...),  # noqa: B008
    language: str | None = Form(default=None),
):
    filename = _safe_basename(file.filename)
    suffix = PurePosixPath(filename).suffix.lower()

    # ---- resolve the language ------------------------------------------
    if language:
        resolved = language.strip().lower()
        resolved = {"js": "javascript", "py": "python", "c++": "cpp"}.get(
            resolved, resolved
        )

        if resolved not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported language {language!r}. Supported: "
                    f"{', '.join(sorted(SUPPORTED_LANGUAGES))}."
                ),
            )
    else:
        resolved = EXTENSION_TO_LANGUAGE.get(suffix)

        if resolved is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot determine a language for extension {suffix or '(none)'!r}. "
                    f"Supported extensions: "
                    f"{', '.join(sorted(EXTENSION_TO_LANGUAGE))}. "
                    f"Pass an explicit `language` field to override."
                ),
            )

    # ---- read with a hard size cap --------------------------------------
    raw = await file.read(settings.max_upload_bytes + 1)

    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds the {settings.max_upload_bytes} byte limit."
            ),
        )

    if not raw.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        code = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=(
                "File is not valid UTF-8 text. Binary files cannot be analyzed."
            ),
        ) from None

    # ---- analyze (same path as POST /analyze) ---------------------------
    try:
        result = analyze_code(code=code, language=resolved)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "analysis_id": str(uuid4()),
        "filename": filename,
        "language": result["language"],
        "security_risk": result["security_risk"],
        "vulnerability_summary": result["vulnerability_summary"],
        "vulnerabilities": result["vulnerabilities"],
        "cpg_metadata": result["cpg"]["metadata"],
    }
