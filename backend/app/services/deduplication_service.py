def deduplicate_vulnerabilities(vulnerabilities: list) -> list:
    """
    Remove duplicate vulnerability findings.

    A finding is considered a duplicate when it has the same:
    - vulnerability type
    - CWE
    - source line
    - function or sink
    """

    unique_vulnerabilities = []
    seen = set()

    for vulnerability in vulnerabilities:

        vulnerability_type = vulnerability.get("type", "")
        cwe = vulnerability.get("cwe", "")
        line = vulnerability.get("line")

        function = vulnerability.get(
            "function",
            vulnerability.get("sink", ""),
        )

        key = (
            vulnerability_type,
            cwe,
            line,
            function,
        )

        if key in seen:
            continue

        seen.add(key)

        unique_vulnerabilities.append(
            vulnerability
        )

    return unique_vulnerabilities
