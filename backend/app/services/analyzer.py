from app.services.language_service import detect_language
from app.services.parser_service import get_parser
from app.services.cpg_service import build_cpg
from app.services.vulnerability_service import detect_vulnerabilities


def analyze_code(code: str, language: str | None = None):
    """
    Main CodeSentinel-AI analysis pipeline.

    Steps:
    1. Validate code
    2. Detect language if not provided
    3. Get Tree-sitter parser
    4. Parse source code
    5. Build the Code Property Graph (CPG)
    6. Detect potential vulnerabilities
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

    # Detect vulnerabilities
    vulnerabilities = detect_vulnerabilities(
        code
    )

    # Calculate vulnerability summary
    severity_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for vulnerability in vulnerabilities:
        severity = vulnerability["severity"]

        if severity in severity_counts:
            severity_counts[severity] += 1

    return {
        "language": language,
        "vulnerabilities": vulnerabilities,
        "vulnerability_summary": {
            "total": len(vulnerabilities),
            "severity_counts": severity_counts,
        },
        "cpg": cpg,
    }
