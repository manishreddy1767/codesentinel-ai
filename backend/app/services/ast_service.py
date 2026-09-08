def build_ast_graph(root_node):
    """
    Builds an Abstract Syntax Tree (AST) graph from a
    Tree-sitter root node.
    """

    nodes = []
    edges = []

    node_id = 0

    def traverse(node, parent_id=None):
        nonlocal node_id

        current_id = node_id
        node_id += 1

        node_text = node.text.decode(
            "utf-8",
            errors="ignore",
        )

        nodes.append(
            {
                "id": current_id,
                "type": node.type,
                "parent_id": parent_id,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "text": node_text,
            }
        )

        if parent_id is not None:
            edges.append(
                {
                    "source": parent_id,
                    "target": current_id,
                    "type": "AST",
                }
            )

        for child in node.children:
            traverse(
                child,
                current_id,
            )

        return current_id

    traverse(root_node)

    return {
        "nodes": nodes,
        "edges": edges,
    }