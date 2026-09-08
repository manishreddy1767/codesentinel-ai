def detect_language(code: str) -> str:
    """
    Detect the programming language from source code.

    Currently supports basic detection for C++,
    Python, Java, and JavaScript.
    """

    code = code.strip()

    if not code:
        return "unknown"

    # Python
    if (
        "def " in code
        or "import " in code
        or "print(" in code
        or "elif " in code
    ):
        return "python"

    # Java
    if (
        "public class " in code
        or "private class " in code
        or "System.out.println" in code
    ):
        return "java"

    # JavaScript
    if (
        "console.log" in code
        or "function " in code
        or "const " in code
        or "let " in code
    ):
        return "javascript"

    # C / C++
    if (
        "#include" in code
        or "std::" in code
        or "int main" in code
        or "using namespace" in code
    ):
        return "cpp"

    return "unknown"
