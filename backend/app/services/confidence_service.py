def assign_confidence(vulnerabilities: list) -> list:
    """
    Assign confidence scores to detected vulnerabilities.

    Confidence represents how certain the detection is
    based on the detection rule and vulnerability type.
    """

    function_confidence = {
        "gets": 0.99,
        "strcpy": 0.95,
        "strcat": 0.95,
        "sprintf": 0.95,
        "scanf": 0.90,
        "system": 0.95,
        "rand": 0.90,
        "md5": 0.98,
        "sha1": 0.98,
    }

    type_confidence = {
        "FORMAT_STRING": 0.90,
        "HARDCODED_SECRET": 0.85,
        "WEAK_CRYPTO": 0.95,
        "TAINT_FLOW": 0.90,
        "RULE_BASED": 0.85,
    }

    for vulnerability in vulnerabilities:

        function_name = vulnerability.get("function")
        vulnerability_type = vulnerability.get("type")

        # Prefer specific function confidence
        if function_name in function_confidence:
            confidence = function_confidence[function_name]

        # Otherwise use vulnerability type confidence
        else:
            confidence = type_confidence.get(
                vulnerability_type,
                0.75,
            )

        vulnerability["confidence"] = confidence

    return vulnerabilities
