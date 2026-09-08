import re


# Functions that can introduce untrusted data
TAINT_SOURCES = {
    "gets",
    "scanf",
    "fscanf",
    "cin",
    "fgets",
}


# Functions where tainted data can become dangerous
TAINT_SINKS = {
    "system": {
        "severity": "CRITICAL",
        "cwe": "CWE-78",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches system(), which may allow "
            "command injection."
        ),
        "recommendation": (
            "Validate and sanitize input and avoid passing "
            "untrusted data to system()."
        ),
    },
    "popen": {
        "severity": "CRITICAL",
        "cwe": "CWE-78",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches popen(), which may allow "
            "command injection."
        ),
        "recommendation": (
            "Avoid passing untrusted input to shell commands."
        ),
    },
}


def _get_line_number(code: str, position: int) -> int:
    """Return the 1-based line number for a character position."""
    return code[:position].count("\n") + 1


def find_tainted_variables(code: str):
    """
    Identify variables that receive data from known taint sources.

    Example:
        gets(buffer);

    Result:
        {"buffer"}
    """

    tainted_variables = set()

    # gets(buffer)
    pattern = r"\bgets\s*\(\s*([A-Za-z_]\w*)\s*\)"

    for match in re.finditer(pattern, code):
        tainted_variables.add(match.group(1))

    # scanf("%s", buffer)
    pattern = (
        r'\bscanf\s*\(\s*"[^"]*"\s*,\s*&?\s*'
        r'([A-Za-z_]\w*)'
    )

    for match in re.finditer(pattern, code):
        tainted_variables.add(match.group(1))

    # fgets(buffer, size, stdin)
    pattern = (
        r"\bfgets\s*\(\s*([A-Za-z_]\w*)\s*,"
    )

    for match in re.finditer(pattern, code):
        tainted_variables.add(match.group(1))

    return tainted_variables


def detect_taint_flows(code: str):
    """
    Detect flows where tainted variables reach dangerous sinks.

    Example:

        gets(buffer);
        system(buffer);

    Result:
        A taint-flow vulnerability.
    """

    tainted_variables = find_tainted_variables(code)

    vulnerabilities = []

    for variable in tainted_variables:

        for sink, details in TAINT_SINKS.items():

            pattern = (
                rf"\b{sink}\s*\(\s*"
                rf"{re.escape(variable)}\s*\)"
            )

            for match in re.finditer(pattern, code):

                vulnerabilities.append({
                    "type": "TAINT_FLOW",
                    "source_variable": variable,
                    "sink": sink,
                    "severity": details["severity"],
                    "title": details["title"],
                    "description": details["description"],
                    "recommendation": details["recommendation"],
                    "cwe": details["cwe"],
                    "line": _get_line_number(
                        code,
                        match.start()
                    ),
                    "code": match.group(0),
                })

    return vulnerabilities
