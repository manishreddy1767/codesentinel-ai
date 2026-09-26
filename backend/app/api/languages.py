"""
GET /supported-languages

Reports the languages the analyzer can actually parse, so the frontend
populates its language selector from the backend rather than from a hardcoded
list that can drift out of sync with `SUPPORTED_LANGUAGES`.

Each entry reports whether a tree-sitter grammar is importable *right now*.
A language whose grammar package is missing is returned with
`parser_available: false` rather than being hidden, because silently omitting
it would make a real installation problem look like an intentional limitation.
"""

from fastapi import APIRouter

from app.schemas.request import SUPPORTED_LANGUAGES

router = APIRouter(tags=["Metadata"])

# Display metadata. Kept next to the endpoint because it is presentation
# detail, not analysis behaviour.
LANGUAGE_INFO = {
    "c": {"label": "C", "extensions": [".c", ".h"]},
    "cpp": {"label": "C++", "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hh"]},
    "python": {"label": "Python", "extensions": [".py"]},
    "javascript": {"label": "JavaScript", "extensions": [".js", ".mjs", ".cjs"]},
    "java": {"label": "Java", "extensions": [".java"]},
}

# Aliases accepted by the request validator, surfaced so the UI can accept the
# same spellings a user might type.
ALIASES = {"js": "javascript", "py": "python", "c++": "cpp"}


def _parser_available(language: str) -> bool:
    """True if a parser can actually be built for this language."""

    try:
        from app.services.parser_service import get_parser

        return get_parser(language) is not None
    except Exception:  # noqa: BLE001
        # A missing grammar package raises ImportError; a bad language name
        # raises ValueError. Either way the honest answer is "not available".
        return False


@router.get("/supported-languages")
async def supported_languages():
    languages = []

    for name in sorted(SUPPORTED_LANGUAGES):
        info = LANGUAGE_INFO.get(name, {})
        languages.append(
            {
                "id": name,
                "label": info.get("label", name),
                "extensions": info.get("extensions", []),
                "parser_available": _parser_available(name),
            }
        )

    return {
        "languages": languages,
        "aliases": ALIASES,
        "count": len(languages),
    }
