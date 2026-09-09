from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_java


def get_parser(language: str) -> Parser:
    """
    Returns a Tree-sitter parser for the requested language.
    """

    parser = Parser()

    if language == "c":
        parser.language = Language(tree_sitter_c.language())

    elif language == "cpp":
        parser.language = Language(tree_sitter_cpp.language())

    elif language == "python":
        parser.language = Language(tree_sitter_python.language())

    elif language == "javascript":
        parser.language = Language(tree_sitter_javascript.language())

    elif language == "java":
        parser.language = Language(tree_sitter_java.language())

    else:
        raise ValueError(
            f"Unsupported parser language: {language}"
        )

    return parser


def validate_source_code(code: str) -> bool:
    """
    Ensures that the source code is not empty.
    """

    return bool(code and code.strip())


def get_node_count(node) -> int:
    """
    Recursively counts all nodes in an AST.
    """

    count = 1

    for child in node.children:
        count += get_node_count(child)

    return count


def extract_functions(root_node) -> list[str]:
    """
    Extracts function names from the AST.
    """

    functions = []

    function_node_types = {
        "function_definition",
        "function_declaration",
        "method_declaration",
    }

    def traverse(node):

        if node.type in function_node_types:

            declarator = node.child_by_field_name(
                "declarator"
            )

            if declarator:
                function_name = find_identifier(
                    declarator
                )

                if function_name:
                    functions.append(function_name)

            else:
                name_node = node.child_by_field_name(
                    "name"
                )

                if name_node:
                    functions.append(
                        name_node.text.decode("utf-8")
                    )

        for child in node.children:
            traverse(child)

    traverse(root_node)

    return functions


def find_identifier(node) -> str | None:
    """
    Finds an identifier inside an AST node.
    """

    if node.type == "identifier":
        return node.text.decode("utf-8")

    for child in node.children:

        result = find_identifier(child)

        if result:
            return result

    return None


def parse_source_code(
    code: str,
    language: str,
) -> dict:
    """
    Parses source code using Tree-sitter
    and returns structured AST information.
    """

    if not validate_source_code(code):
        raise ValueError(
            "Source code cannot be empty."
        )

    parser = get_parser(language)

    tree = parser.parse(
        bytes(code, "utf-8")
    )

    root_node = tree.root_node

    functions = extract_functions(
        root_node
    )

    return {
        "language": language,
        "line_count": len(
            code.splitlines()
        ),
        "function_count": len(
            functions
        ),
        "functions": functions,
        "ast_node_count": get_node_count(
            root_node
        ),
        "has_syntax_error": root_node.has_error,
    }
