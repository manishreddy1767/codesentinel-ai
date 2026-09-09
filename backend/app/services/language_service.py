def detect_language(code: str) -> str:
    """
    Detect the programming language from source code.

    Supports:
    - C
    - C++
    - Python
    - JavaScript
    """

    code = code.strip()

    if not code:
        return "unknown"

    # ---------------------------------
    # JavaScript
    # ---------------------------------

    if (
        "console.log" in code
        or "function " in code
        or "const " in code
        or "let " in code
        or "var " in code
        or "=>" in code
        or "require(" in code
    ):
        return "javascript"

    # ---------------------------------
    # Python
    # ---------------------------------

    if (
        "def " in code
        or "import " in code
        or "from " in code
        or "print(" in code
        or "elif " in code
        or "__name__" in code
    ):
        return "python"

    # ---------------------------------
    # C++
    # ---------------------------------

    if (
        "std::" in code
        or "using namespace" in code
        or "#include <iostream>" in code
        or "cout <<" in code
    ):
        return "cpp"

    # ---------------------------------
    # C
    # ---------------------------------

    if (
        "#include" in code
        or "int main" in code
        or "printf(" in code
    ):
        return "c"

    return "unknown"
