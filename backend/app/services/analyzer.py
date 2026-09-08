from app.services.language_service import detect_language
from app.services.parser_service import get_parser
from app.services.cpg_service import build_cpg


def analyze_code(code: str, language: str | None = None):
    """
    Main code analysis pipeline.

    Steps:
    1. Detect language if not provided
    2. Get Tree-sitter parser
    3. Parse source code
    4. Build the Code Property Graph (CPG)
    """

    if not code or not code.strip():
        raise ValueError("Code cannot be empty.")

    # Detect language automatically if needed
    if language is None:
        language = detect_language(code)

    # Get parser
    parser = get_parser(language)

    if parser is None:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    # Parse source code
    tree = parser.parse(
        bytes(code, "utf-8")
    )

    # Build Code Property Graph
    cpg = build_cpg(
        tree.root_node
    )

    return {
        "language": language,
        "cpg": cpg,
    }