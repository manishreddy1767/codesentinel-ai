def calculate_security_risk(vulnerabilities: list) -> dict:
    """
    Calculate an overall security score and risk level.

    The score starts at 100 and decreases based on the
    severity of detected vulnerabilities.
    """

    severity_penalties = {
        "CRITICAL": 30,
        "HIGH": 15,
        "MEDIUM": 7,
        "LOW": 3,
    }

    severity_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    score = 100

    for vulnerability in vulnerabilities:
        severity = vulnerability.get("severity", "LOW").upper()

        if severity in severity_counts:
            severity_counts[severity] += 1
            score -= severity_penalties[severity]

    # Keep score within 0-100
    score = max(0, min(100, score))

    # Determine overall risk level
    if score >= 90:
        risk_level = "LOW"
    elif score >= 70:
        risk_level = "MEDIUM"
    elif score >= 40:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "security_score": score,
        "risk_level": risk_level,
        "severity_counts": severity_counts,
    }
