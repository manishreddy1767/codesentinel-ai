import re


def validate_source_code(code: str) -> bool:
    """
    Performs basic validation to ensure source code
    is not empty or whitespace-only.
    """
    return bool(code and code.strip())


def extract_functions(code: str) -> list[str]:
    """
    Performs basic function-name extraction for C/C++ code.

    This is a temporary parser foundation and will later
    be replaced or extended with proper AST parsing.
    """

    pattern = re.compile(
        r"""
        [\w\s\*]+
        \s+
        (\w+)
        \s*
        \(
        [^)]*
        \)
        \s*
        \{
        """,
        re.VERBOSE,
    )

    return pattern.findall(code)


def parse_source_code(code: str, language: str) -> dict:
    """
    Parses source code into a structured representation.

    Future versions will include:
    - AST
    - CFG
    - DFG
    - Code Property Graph
    """

    if not validate_source_code(code):
        raise ValueError("Source code cannot be empty.")

    functions = extract_functions(code)

    return {
        "language": language,
        "line_count": len(code.splitlines()),
        "function_count": len(functions),
        "functions": functions,
    }

