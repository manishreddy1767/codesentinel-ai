from app.services.language_service import detect_language
from app.services.parser_service import get_parser
from app.services.cpg_service import build_cpg
from app.services.vulnerability_service import detect_vulnerabilities
from app.services.cpg_vulnerability_service import link_vulnerabilities_to_cpg
from app.services.taint_service import detect_taint_flows
from app.services.risk_service import calculate_security_risk


def analyze_code(code: str, language: str | None = None):
    """
    Main CodeSentinel AI analysis pipeline.

    Steps:
    1. Detect programming language
    2. Parse source code using Tree-sitter
    3. Build Code Property Graph
    4. Detect rule-based vulnerabilities
    5. Detect taint flows
    6. Link vulnerabilities to CPG nodes
    7. Calculate security risk score
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
    vulnerabilities = detect_vulnerabilities(code)

    # Detect taint-flow vulnerabilities
    taint_vulnerabilities = detect_taint_flows(code)

    # Combine all vulnerabilities
    vulnerabilities.extend(
        taint_vulnerabilities
    )

    # Link vulnerabilities to relevant CPG nodes
    vulnerabilities = link_vulnerabilities_to_cpg(
        vulnerabilities,
        cpg,
    )

    # Existing vulnerability summary
    vulnerability_summary = {
        "total": len(vulnerabilities),
        "severity_counts": {
            "CRITICAL": sum(
                1 for v in vulnerabilities
                if v.get("severity") == "CRITICAL"
            ),
            "HIGH": sum(
                1 for v in vulnerabilities
                if v.get("severity") == "HIGH"
            ),
            "MEDIUM": sum(
                1 for v in vulnerabilities
                if v.get("severity") == "MEDIUM"
            ),
            "LOW": sum(
                1 for v in vulnerabilities
                if v.get("severity") == "LOW"
            ),
        },
    }

    # Calculate overall security risk
    security_risk = calculate_security_risk(
        vulnerabilities
    )

    return {
        "language": language,
        "cpg": cpg,
        "vulnerabilities": vulnerabilities,
        "vulnerability_summary": vulnerability_summary,
        "security_risk": security_risk,
    }
