import re


# =========================================================
# COMMON SANITIZERS
# =========================================================

COMMON_SANITIZERS = {
    "sanitize",
    "validate",
    "escape",
    "html_escape",
    "shell_escape",
}


# =========================================================
# C / C++ TAINT SOURCES
# =========================================================

CPP_TAINT_SOURCES = {
    "gets",
    "scanf",
    "fgets",
    "read",
    "recv",
}


# =========================================================
# PYTHON TAINT SOURCES
# =========================================================

PYTHON_TAINT_SOURCE_PATTERNS = [
    (
        r'\b([A-Za-z_]\w*)\s*=\s*input\s*\(',
        1,
    ),
    (
        r'\b([A-Za-z_]\w*)\s*=\s*request\.args\.',
        1,
    ),
    (
        r'\b([A-Za-z_]\w*)\s*=\s*request\.form\.',
        1,
    ),
    (
        r'\b([A-Za-z_]\w*)\s*=\s*request\.json',
        1,
    ),
    (
        r'\b([A-Za-z_]\w*)\s*=\s*os\.environ',
        1,
    ),
]


# =========================================================
# JAVASCRIPT TAINT SOURCES
# =========================================================

JAVASCRIPT_TAINT_SOURCE_PATTERNS = [
    # const userInput = prompt()
    (
        r'\b(?:const|let|var)\s+([A-Za-z_$]\w*)\s*='
        r'\s*prompt\s*\(',
        1,
    ),

    # const value = req.query.xxx
    (
        r'\b(?:const|let|var)\s+([A-Za-z_$]\w*)\s*='
        r'\s*req\.query',
        1,
    ),

    # const value = req.body.xxx
    (
        r'\b(?:const|let|var)\s+([A-Za-z_$]\w*)\s*='
        r'\s*req\.body',
        1,
    ),

    # const value = req.params.xxx
    (
        r'\b(?:const|let|var)\s+([A-Za-z_$]\w*)\s*='
        r'\s*req\.params',
        1,
    ),

    # process.argv
    (
        r'\b(?:const|let|var)\s+([A-Za-z_$]\w*)\s*='
        r'\s*process\.argv',
        1,
    ),
]


# =========================================================
# JAVA TAINT SOURCES
# =========================================================

JAVA_TAINT_SOURCE_PATTERNS = [
    # String command = request.getParameter("cmd");
    (
        r'\b(?:String|Object|var)\s+([A-Za-z_]\w*)\s*='
        r'\s*request\.getParameter\s*\(',
        1,
    ),

    # String command = request.getHeader("...");
    (
        r'\b(?:String|Object|var)\s+([A-Za-z_]\w*)\s*='
        r'\s*request\.getHeader\s*\(',
        1,
    ),
]


# =========================================================
# C / C++ TAINT SINKS
# =========================================================

CPP_TAINT_SINKS = {

    "system": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches system(), which may allow "
            "command injection."
        ),
        "recommendation": (
            "Validate and sanitize input and avoid passing "
            "untrusted data to system()."
        ),
        "cwe": "CWE-78",
    },

    "strcpy": {
        "severity": "HIGH",
        "title": "Potential buffer overflow",
        "description": (
            "Tainted input reaches strcpy(), which may cause "
            "a buffer overflow."
        ),
        "recommendation": (
            "Use bounded string-copy operations and validate "
            "input length."
        ),
        "cwe": "CWE-120",
    },

    "strcat": {
        "severity": "HIGH",
        "title": "Potential buffer overflow",
        "description": (
            "Tainted input reaches strcat(), which may cause "
            "a buffer overflow."
        ),
        "recommendation": (
            "Use bounded string concatenation and validate "
            "buffer sizes."
        ),
        "cwe": "CWE-120",
    },

    "printf": {
        "severity": "HIGH",
        "title": "Potential format string vulnerability",
        "description": (
            "Tainted input is used directly as a printf() "
            "format string."
        ),
        "recommendation": (
            'Use a fixed format string, for example '
            'printf("%s", input).'
        ),
        "cwe": "CWE-134",
    },

    "sprintf": {
        "severity": "HIGH",
        "title": "Potential format string or buffer overflow",
        "description": (
            "Tainted input reaches sprintf(), which may allow "
            "format string attacks or buffer overflows."
        ),
        "recommendation": (
            "Use snprintf() with explicit buffer limits and "
            "fixed format strings."
        ),
        "cwe": "CWE-120",
    },
}


# =========================================================
# PYTHON TAINT SINKS
# =========================================================

PYTHON_TAINT_SINKS = {

    "os.system": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches os.system(), which may "
            "allow command injection."
        ),
        "recommendation": (
            "Avoid os.system() with untrusted input. Use "
            "subprocess with fixed arguments and validation."
        ),
        "cwe": "CWE-78",
    },

    "subprocess.run": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches subprocess.run(), which "
            "may allow command injection depending on usage."
        ),
        "recommendation": (
            "Avoid shell=True and pass validated arguments "
            "as a list."
        ),
        "cwe": "CWE-78",
    },

    "subprocess.call": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches subprocess.call(), which "
            "may allow command injection."
        ),
        "recommendation": (
            "Validate input and avoid shell=True."
        ),
        "cwe": "CWE-78",
    },

    "subprocess.Popen": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches subprocess.Popen(), which "
            "may allow command injection."
        ),
        "recommendation": (
            "Use validated arguments and avoid shell=True."
        ),
        "cwe": "CWE-78",
    },

    "eval": {
        "severity": "CRITICAL",
        "title": "Potential code injection",
        "description": (
            "Tainted input reaches eval(), which may allow "
            "arbitrary code execution."
        ),
        "recommendation": (
            "Avoid eval() on untrusted input."
        ),
        "cwe": "CWE-95",
    },

    "exec": {
        "severity": "CRITICAL",
        "title": "Potential code injection",
        "description": (
            "Tainted input reaches exec(), which may allow "
            "arbitrary code execution."
        ),
        "recommendation": (
            "Avoid exec() on untrusted input."
        ),
        "cwe": "CWE-95",
    },

    "execute": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches a database execute() call, "
            "which may allow SQL injection."
        ),
        "recommendation": (
            "Use parameterized queries instead of constructing "
            "SQL statements with untrusted input."
        ),
        "cwe": "CWE-89",
    },

    "executemany": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches a database executemany() call, "
            "which may allow SQL injection."
        ),
        "recommendation": (
            "Use parameterized queries and avoid dynamically "
            "constructing SQL statements."
        ),
        "cwe": "CWE-89",
    },
}


# =========================================================
# JAVASCRIPT TAINT SINKS
# =========================================================

JAVASCRIPT_TAINT_SINKS = {

    "child_process.exec": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches child_process.exec(), which "
            "may allow command injection."
        ),
        "recommendation": (
            "Avoid constructing shell commands from untrusted input "
            "and validate all command arguments."
        ),
        "cwe": "CWE-78",
    },

    "child_process.execSync": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches child_process.execSync(), which "
            "may allow command injection."
        ),
        "recommendation": (
            "Avoid constructing shell commands from untrusted input "
            "and validate all command arguments."
        ),
        "cwe": "CWE-78",
    },

    "eval": {
        "severity": "CRITICAL",
        "title": "Potential code injection",
        "description": (
            "Tainted input reaches eval(), which may allow "
            "arbitrary code execution."
        ),
        "recommendation": (
            "Avoid eval() on untrusted input."
        ),
        "cwe": "CWE-95",
    },

    "Function": {
        "severity": "CRITICAL",
        "title": "Potential code injection",
        "description": (
            "Tainted input reaches the Function() constructor, which "
            "may allow arbitrary code execution."
        ),
        "recommendation": (
            "Avoid dynamically constructing executable code."
        ),
        "cwe": "CWE-95",
    },


    "query": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches a database query() call, "
            "which may allow SQL injection."
        ),
        "recommendation": (
            "Use parameterized queries instead of dynamically "
            "constructing SQL statements."
        ),
        "cwe": "CWE-89",
    },

    "execute": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches a database execute() call, "
            "which may allow SQL injection."
        ),
        "recommendation": (
            "Use parameterized queries and avoid concatenating "
            "untrusted input into SQL statements."
        ),
        "cwe": "CWE-89",
    },

}


# =========================================================
# JAVA TAINT SINKS
# =========================================================

JAVA_TAINT_SINKS = {

    "Runtime.getRuntime().exec": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches Runtime.exec(), which may allow "
            "command injection."
        ),
        "recommendation": (
            "Avoid executing commands constructed from untrusted input "
            "and validate all command arguments."
        ),
        "cwe": "CWE-78",
    },

    "ProcessBuilder": {
        "severity": "CRITICAL",
        "title": "Potential command injection",
        "description": (
            "Tainted input reaches ProcessBuilder, which may allow "
            "command injection."
        ),
        "recommendation": (
            "Validate command arguments and avoid constructing "
            "commands from untrusted input."
        ),
        "cwe": "CWE-78",
    },

    "executeQuery": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches executeQuery(), which may allow "
            "SQL injection."
        ),
        "recommendation": (
            "Use PreparedStatement with parameterized queries instead "
            "of dynamically constructing SQL statements."
        ),
        "cwe": "CWE-89",
    },

    "executeUpdate": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches executeUpdate(), which may allow "
            "SQL injection."
        ),
        "recommendation": (
            "Use PreparedStatement with parameterized queries instead "
            "of dynamically constructing SQL statements."
        ),
        "cwe": "CWE-89",
    },

    "execute": {
        "severity": "CRITICAL",
        "title": "Potential SQL Injection",
        "description": (
            "Tainted input reaches execute(), which may allow "
            "SQL injection."
        ),
        "recommendation": (
            "Use PreparedStatement with parameterized queries instead "
            "of dynamically constructing SQL statements."
        ),
        "cwe": "CWE-89",
    },
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_identifiers(text: str) -> set:
    """Extract identifier-like tokens."""

    return set(
        re.findall(
            r'\b[A-Za-z_$]\w*\b',
            text,
        )
    )


def get_sanitizers(language: str | None = None) -> set:
    """Return known sanitizer functions."""

    return COMMON_SANITIZERS


def get_taint_sinks(language: str | None = None) -> dict:
    """Return language-specific taint sinks."""

    if language == "python":
        return PYTHON_TAINT_SINKS

    if language == "javascript":
        return JAVASCRIPT_TAINT_SINKS

    if language == "java":
        return JAVA_TAINT_SINKS

    return CPP_TAINT_SINKS


# =========================================================
# BACKWARD-COMPATIBLE HELPERS
# =========================================================

def find_tainted_variables(code: str) -> set:
    """
    Find tainted variables using C/C++, Python,
    and JavaScript sources.
    """

    tainted = set()

    # -------------------------
    # C/C++ sources
    # -------------------------

    for match in re.finditer(
        r'\bgets\s*\(\s*([A-Za-z_]\w*)\s*\)',
        code,
    ):
        tainted.add(match.group(1))

    for match in re.finditer(
        r'\bscanf\s*\(\s*[^,]+,\s*&?\s*([A-Za-z_]\w*)',
        code,
    ):
        tainted.add(match.group(1))

    for match in re.finditer(
        r'\bfgets\s*\(\s*([A-Za-z_]\w*)',
        code,
    ):
        tainted.add(match.group(1))

    for match in re.finditer(
        r'\bread\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)',
        code,
    ):
        tainted.add(match.group(1))

    for match in re.finditer(
        r'\brecv\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)',
        code,
    ):
        tainted.add(match.group(1))

    # -------------------------
    # Python sources
    # -------------------------

    for pattern, group in PYTHON_TAINT_SOURCE_PATTERNS:

        for match in re.finditer(pattern, code):
            tainted.add(match.group(group))

    if re.search(r'\bsys\.argv\b', code):
        tainted.add("argv")

    elif re.search(r'\bargv\b', code):
        tainted.add("argv")

    # -------------------------
    # JavaScript sources
    # -------------------------

    for pattern, group in JAVASCRIPT_TAINT_SOURCE_PATTERNS:

        for match in re.finditer(pattern, code):
            tainted.add(match.group(group))

    return tainted


def find_sanitized_variables(code: str) -> set:
    """
    Return variables passed to known sanitizer functions.
    """

    sanitized = set()

    for sanitizer in COMMON_SANITIZERS:

        pattern = (
            rf'\b{re.escape(sanitizer)}'
            r'\s*\(\s*'
            r'([A-Za-z_$]\w*)'
            r'\s*\)'
        )

        for match in re.finditer(pattern, code):
            sanitized.add(match.group(1))

    return sanitized


def propagate_taint(
    code: str,
    tainted: set,
    sanitized: set | None = None,
) -> set:
    """
    Backward-compatible multi-pass taint propagation helper.
    """

    if sanitized is None:
        sanitized = set()

    lines = code.splitlines()

    changed = True

    while changed:

        changed = False

        for line in lines:

            match = re.search(
                r'\b(?:const|let|var)?\s*'
                r'([A-Za-z_$]\w*)\s*'
                r'(?:\[[^\]]*\])?\s*=\s*'
                r'([^;#\n]+)',
                line,
            )

            if not match:
                continue

            target = match.group(1)
            expression = match.group(2)

            identifiers = extract_identifiers(
                expression
            )

            active_tainted = tainted - sanitized

            if identifiers & active_tainted:

                if target not in tainted:

                    tainted.add(target)
                    sanitized.discard(target)
                    changed = True

    return tainted


# =========================================================
# LINE-AWARE MULTI-LANGUAGE TAINT DETECTION
# =========================================================

def detect_taint_flows(
    code: str,
    language: str | None = None,
) -> list:
    """
    Detect taint flows while respecting source-code order.

    Supports:
    - C/C++
    - Python
    - JavaScript
    """

    vulnerabilities = []

    tainted = set()
    sanitized = set()

    if language is None:

        if (
            "console.log" in code
            or "const " in code
            or "let " in code
            or "function " in code
            or "=>" in code
        ):
            language = "javascript"

        elif (
            re.search(r'\bdef\s+\w+\s*\(', code)
            or re.search(r'\bimport\s+\w+', code)
            or re.search(r'\binput\s*\(', code)
        ):
            language = "python"

        else:
            language = "cpp"

    sinks = get_taint_sinks(language)
    sanitizers = get_sanitizers(language)

    lines = code.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        stripped_line = line.strip()

        if not stripped_line:
            continue

        # -------------------------------------------------
        # PYTHON TAINT SOURCES
        # -------------------------------------------------

        if language == "python":

            for pattern, group in PYTHON_TAINT_SOURCE_PATTERNS:

                match = re.search(
                    pattern,
                    stripped_line,
                )

                if match:

                    variable = match.group(group)

                    tainted.add(variable)
                    sanitized.discard(variable)

        # -------------------------------------------------
        # JAVASCRIPT TAINT SOURCES
        # -------------------------------------------------

        elif language == "javascript":

            for pattern, group in JAVASCRIPT_TAINT_SOURCE_PATTERNS:

                match = re.search(
                    pattern,
                    stripped_line,
                )

                if match:

                    variable = match.group(group)

                    tainted.add(variable)
                    sanitized.discard(variable)

        # -------------------------------------------------
        # JAVA TAINT SOURCES
        # -------------------------------------------------

        elif language == "java":

            # Method arguments are potentially user-controlled.
            if re.search(
                r'\bString\[\]\s+args\b',
                stripped_line,
            ):
                tainted.add("args")
                sanitized.discard("args")

            # String command = request.getParameter(...)
            for pattern, group in JAVA_TAINT_SOURCE_PATTERNS:

                match = re.search(
                    pattern,
                    stripped_line,
                )

                if match:

                    variable = match.group(group)

                    tainted.add(variable)
                    sanitized.discard(variable)

        # -------------------------------------------------
        # C / C++ TAINT SOURCES
        # -------------------------------------------------

        else:

            match = re.search(
                r'\bgets\s*\(\s*([A-Za-z_]\w*)\s*\)',
                stripped_line,
            )

            if match:

                variable = match.group(1)

                tainted.add(variable)
                sanitized.discard(variable)

            match = re.search(
                r'\bscanf\s*\(\s*[^,]+,\s*&?\s*([A-Za-z_]\w*)',
                stripped_line,
            )

            if match:

                variable = match.group(1)

                tainted.add(variable)
                sanitized.discard(variable)

            match = re.search(
                r'\bfgets\s*\(\s*([A-Za-z_]\w*)',
                stripped_line,
            )

            if match:

                variable = match.group(1)

                tainted.add(variable)
                sanitized.discard(variable)

            match = re.search(
                r'\bread\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)',
                stripped_line,
            )

            if match:

                variable = match.group(1)

                tainted.add(variable)
                sanitized.discard(variable)

            match = re.search(
                r'\brecv\s*\(\s*[^,]+,\s*([A-Za-z_]\w*)',
                stripped_line,
            )

            if match:

                variable = match.group(1)

                tainted.add(variable)
                sanitized.discard(variable)

        # -------------------------------------------------
        # TAINT PROPAGATION
        # -------------------------------------------------

        if language == "javascript":

            assignment = re.search(
                r'\b(?:const|let|var)?\s*'
                r'([A-Za-z_$]\w*)\s*=\s*(.+)',
                stripped_line,
            )

        elif language == "java":

            assignment = re.search(
                r'\b(?:String|Object|var|int|long|double|float|boolean)?\s*'
                r'([A-Za-z_]\w*)\s*=\s*(.+)',
                stripped_line,
            )

        else:

            assignment = re.search(
                r'\b([A-Za-z_]\w*)\s*=\s*(.+)',
                stripped_line,
            )

        if assignment:

            target = assignment.group(1)
            expression = assignment.group(2)

            identifiers = extract_identifiers(
                expression
            )

            active_tainted = tainted - sanitized

            if identifiers & active_tainted:

                tainted.add(target)
                sanitized.discard(target)

        # -------------------------------------------------
        # SANITIZATION
        # -------------------------------------------------

        for sanitizer in sanitizers:

            sanitizer_match = re.search(
                rf'\b{re.escape(sanitizer)}'
                r'\s*\(\s*'
                r'([A-Za-z_$]\w*)'
                r'\s*\)',
                stripped_line,
            )

            if sanitizer_match:

                variable = sanitizer_match.group(1)

                if variable in tainted:

                    sanitized.add(variable)

        # -------------------------------------------------
        # DANGEROUS SINKS
        # -------------------------------------------------

        active_tainted = tainted - sanitized

        for sink, details in sinks.items():

            sink_match = re.search(
                rf'\b{re.escape(sink)}\s*\((.*?)\)',
                stripped_line,
            )

            if not sink_match:
                continue

            arguments = sink_match.group(1)

            tainted_variable = None

            for variable in active_tainted:

                if re.search(
                    rf'\b{re.escape(variable)}\b',
                    arguments,
                ):

                    tainted_variable = variable
                    break

            if not tainted_variable:
                continue

            vulnerabilities.append(
                {
                    "type": "TAINT_FLOW",
                    "source_variable": tainted_variable,
                    "sink": sink,
                    "severity": details["severity"],
                    "title": details["title"],
                    "description": details["description"],
                    "recommendation": details[
                        "recommendation"
                    ],
                    "cwe": details["cwe"],
                    "line": line_number,
                    "code": stripped_line,
                }
            )

    return vulnerabilities
