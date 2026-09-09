from app.services.analyzer import analyze_code


def test_analyze_python_eval():
    code = """
user_input = input()
eval(user_input)
"""

    result = analyze_code(
        code,
        "python",
    )

    assert result["language"] == "python"

    assert any(
        vulnerability["cwe"] == "CWE-95"
        for vulnerability in result["vulnerabilities"]
    )


def test_analyze_python_sql_injection():
    code = """
user_id = input()
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
"""

    result = analyze_code(
        code,
        "python",
    )

    assert result["vulnerability_summary"]["total"] >= 2

    assert any(
        vulnerability["type"] == "SQL_INJECTION"
        and vulnerability["cwe"] == "CWE-89"
        for vulnerability in result["vulnerabilities"]
    )

    assert any(
        vulnerability["type"] == "TAINT_FLOW"
        and vulnerability["cwe"] == "CWE-89"
        for vulnerability in result["vulnerabilities"]
    )


def test_analyzer_calculates_security_risk():
    code = """
user_input = input()
eval(user_input)
"""

    result = analyze_code(
        code,
        "python",
    )

    security_risk = result["security_risk"]

    assert "security_score" in security_risk
    assert "risk_level" in security_risk
    assert "severity_counts" in security_risk


def test_analyzer_builds_cpg():
    code = """
user_input = input()
eval(user_input)
"""

    result = analyze_code(
        code,
        "python",
    )

    assert "nodes" in result["cpg"]
    assert len(result["cpg"]["nodes"]) > 0


def test_analyzer_links_vulnerabilities_to_cpg():
    code = """
user_id = input()
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
"""

    result = analyze_code(
        code,
        "python",
    )

    sql_vulnerabilities = [
        vulnerability
        for vulnerability in result["vulnerabilities"]
        if vulnerability["cwe"] == "CWE-89"
    ]

    assert len(sql_vulnerabilities) > 0

    assert any(
        len(
            vulnerability.get(
                "cpg_nodes",
                [],
            )
        ) > 0
        for vulnerability in sql_vulnerabilities
    )
