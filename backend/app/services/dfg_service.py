def build_dfg(root_node):
    """
    Builds a basic Data Flow Graph (DFG).

    Tracks:

    - Variable definitions
    - Variable uses
    - DEF -> USE relationships

    Current implementation supports:

    - Variable declarations
    - Assignments
    - Update expressions
    - Return statements
    """

    nodes = []
    edges = []

    node_id = 0

    # Stores the latest definition node
    # for each variable.
    latest_definition = {}

    def create_node(
        variable,
        node_type,
        ast_node,
    ):
        """
        Creates a DFG node.
        """

        nonlocal node_id

        current_id = node_id
        node_id += 1

        node_text = ast_node.text.decode(
            "utf-8",
            errors="ignore",
        )

        nodes.append(
            {
                "id": current_id,
                "type": node_type,
                "variable": variable,
                "start_line": (
                    ast_node.start_point[0] + 1
                ),
                "end_line": (
                    ast_node.end_point[0] + 1
                ),
                "start_byte": ast_node.start_byte,
                "end_byte": ast_node.end_byte,
                "text": node_text,
            }
        )

        return current_id

    def add_edge(
        source,
        target,
        edge_type="DEF_USE",
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    def extract_identifiers(node):
        """
        Returns identifier nodes inside
        an AST node.
        """

        identifiers = []

        def traverse(current):

            if current.type == "identifier":

                identifiers.append(
                    current
                )

            for child in current.named_children:

                traverse(child)

        traverse(node)

        return identifiers

    def create_use(
        identifier_node,
    ):
        """
        Creates a USE node and connects it
        to the latest definition.
        """

        variable = (
            identifier_node.text.decode(
                "utf-8",
                errors="ignore",
            )
        )

        use_id = create_node(
            variable,
            "USE",
            identifier_node,
        )

        if variable in latest_definition:

            add_edge(
                latest_definition[
                    variable
                ],
                use_id,
            )

        return use_id

    def create_definition(
        identifier_node,
    ):
        """
        Creates a DEF node and stores it
        as the latest definition.
        """

        variable = (
            identifier_node.text.decode(
                "utf-8",
                errors="ignore",
            )
        )

        def_id = create_node(
            variable,
            "DEF",
            identifier_node,
        )

        latest_definition[
            variable
        ] = def_id

        return def_id

    def process_declaration(node):
        """
        Handles variable declarations.

        Example:

        int x = y + 1;
        """

        declarator = None

        for child in node.named_children:

            if child.type in {
                "init_declarator",
                "identifier",
            }:

                declarator = child
                break

        if declarator is None:

            return

        if declarator.type == "identifier":

            create_definition(
                declarator
            )

            return

        variable_node = None

        value_node = None

        for child in declarator.named_children:

            if child.type == "identifier":

                variable_node = child

            else:

                value_node = child

        # Process uses first
        #
        # int y = x + 1;
        #
        # x should be a USE before y
        # becomes a definition.

        if value_node is not None:

            identifiers = (
                extract_identifiers(
                    value_node
                )
            )

            for identifier in identifiers:

                create_use(
                    identifier
                )

        if variable_node is not None:

            create_definition(
                variable_node
            )

    def process_assignment(node):
        """
        Handles assignment expressions.

        Example:

        x = y + 1;
        """

        left = (
            node.child_by_field_name(
                "left"
            )
        )

        right = (
            node.child_by_field_name(
                "right"
            )
        )

        if left is None:

            return

        # ----------------------------------
        # Process RHS uses
        # ----------------------------------

        if right is not None:

            identifiers = (
                extract_identifiers(
                    right
                )
            )

            for identifier in identifiers:

                create_use(
                    identifier
                )

        # ----------------------------------
        # Process LHS definition
        # ----------------------------------

        if left.type == "identifier":

            create_definition(
                left
            )

    def process_update_expression(node):
        """
        Handles:

        x++
        x--
        ++x
        --x

        These are both a USE and a DEF.
        """

        identifiers = (
            extract_identifiers(node)
        )

        for identifier in identifiers:

            create_use(
                identifier
            )

            create_definition(
                identifier
            )

    def process_return_statement(node):
        """
        Handles:

        return x;
        """

        identifiers = (
            extract_identifiers(node)
        )

        for identifier in identifiers:

            create_use(
                identifier
            )

    def traverse(node):
        """
        Traverses AST and builds DFG.
        """

        if node.type == "declaration":

            process_declaration(
                node
            )

            return

        if node.type == "assignment_expression":

            process_assignment(
                node
            )

            return

        if node.type == "update_expression":

            process_update_expression(
                node
            )

            return

        if node.type == "return_statement":

            process_return_statement(
                node
            )

            return

        for child in node.named_children:

            traverse(child)

    traverse(root_node)

    return {
        "nodes": nodes,
        "edges": edges,
    }