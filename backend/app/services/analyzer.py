from app.services.language_service import detect_language
from app.services.parser_service import get_parser
from app.services.cpg_service import build_cpg
from app.services.vulnerability_service import (
    detect_vulnerabilities,
    get_vulnerability_summary,
)
from app.services.taint_service import detect_taint_flows
from app.services.cpg_vulnerability_service import (
    link_vulnerabilities_to_cpg,
)


def analyze_code(code: str, language: str | None = None):
    """
    Main CodeSentinel analysis pipeline.

    Steps:
    1. Detect language if not provided
    2. Parse source code using Tree-sitter
    3. Build the Code Property Graph (CPG)
    4. Detect rule-based vulnerabilities
    5. Detect source-to-sink taint flows
    6. Combine vulnerabilities
    7. Link vulnerabilities to relevant CPG nodes
    8. Generate vulnerability summary
    """

    if not code or not code.strip():
        raise ValueError("Code cannot be empty.")

    # Detect language automatically if needed
    if language is None:
        language = detect_language(code)

    # Get Tree-sitter parser
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

    # Detect rule-based vulnerabilities
    rule_vulnerabilities = detect_vulnerabilities(
        code
    )

    # Detect source-to-sink taint flows
    taint_vulnerabilities = detect_taint_flows(
        code
    )

    # Combine all vulnerabilities
    vulnerabilities = (
        rule_vulnerabilities +
        taint_vulnerabilities
    )

    # Link vulnerabilities to relevant CPG nodes
    vulnerabilities = link_vulnerabilities_to_cpg(
        vulnerabilities,
        cpg
    )

    # Generate vulnerability summary
    vulnerability_summary = get_vulnerability_summary(
        vulnerabilities
    )

    return {
        "language": language,
        "vulnerabilities": vulnerabilities,
        "vulnerability_summary": vulnerability_summary,
        "cpg": cpg,
    }
