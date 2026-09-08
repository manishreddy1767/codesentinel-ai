from app.services.language_service import detect_language
from app.services.parser_service import get_parser
from app.services.cpg_service import build_cpg
from app.services.vulnerability_service import detect_vulnerabilities
from app.services.cpg_vulnerability_service import link_vulnerabilities_to_cpg
from app.services.taint_service import detect_taint_flows
from app.services.risk_service import calculate_security_risk
from app.services.confidence_service import assign_confidence
from app.services.deduplication_service import deduplicate_vulnerabilities


def analyze_code(code: str, language: str | None = None):
    """
    Main CodeSentinel AI analysis pipeline.

    Steps:
    1. Detect language
    2. Parse source code
    3. Build Code Property Graph
    4. Detect rule-based vulnerabilities
    5. Detect taint-flow vulnerabilities
    6. Remove duplicate findings
    7. Assign confidence scores
    8. Link vulnerabilities to CPG nodes
    9. Calculate security risk
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

    # Rule-based vulnerability detection
    vulnerabilities = detect_vulnerabilities(
        code,
        language,
    )

    # Taint-flow vulnerability detection
    taint_vulnerabilities = detect_taint_flows(
        code,
        language,
    )

    vulnerabilities.extend(
        taint_vulnerabilities
    )

    # Remove duplicate findings
    vulnerabilities = deduplicate_vulnerabilities(
        vulnerabilities
    )

    # Assign confidence scores
    vulnerabilities = assign_confidence(
        vulnerabilities
    )

    # Link vulnerabilities to relevant CPG nodes
    vulnerabilities = link_vulnerabilities_to_cpg(
        vulnerabilities,
        cpg,
    )

    # Calculate overall security risk
    security_risk = calculate_security_risk(
        vulnerabilities
    )

    vulnerability_summary = {
        "total": len(vulnerabilities),
        "severity_counts": security_risk[
            "severity_counts"
        ],
    }

    return {
        "language": language,
        "cpg": cpg,
        "vulnerabilities": vulnerabilities,
        "vulnerability_summary": vulnerability_summary,
        "security_risk": security_risk,
    }
