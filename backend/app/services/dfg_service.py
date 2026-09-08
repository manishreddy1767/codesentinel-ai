def build_dfg(root_node):
    """
    Builds a basic Data Flow Graph (DFG) from a Tree-sitter AST.

    Current support:
    - Variable declarations
    - Variable definitions
    - Variable assignments
    - Variable usages
    - Def-use relationships
    - Assignment data dependencies

    The graph contains:
    - Nodes representing variable definitions and usages
    - Edges representing data-flow dependencies

    This implementation is designed as the foundation for
    combining AST + CFG + DFG into a unified Code Property Graph.
    """

    nodes = []
    edges = []

    node_id = 0

    # Stores the latest definition of each variable.
    #
    # Example:
    #
    # int x = 10;
    #
    # latest_definition["x"] = definition node ID
    #
    latest_definition = {}

    def create_node(
        node_type,
        variable,
        ast_node,
    ):
        """
        Creates a DFG node.
        """

        nonlocal node_id

        current_id = node_id
        node_id += 1

        start_line = ast_node.start_point[0] + 1
        end_line = ast_node.end_point[0] + 1

        node_text = ast_node.text.decode(
            "utf-8",
            errors="ignore",
        )

        nodes.append(
            {
                "id": current_id,
                "type": node_type,
                "variable": variable,
                "start_line": start_line,
                "end_line": end_line,
                "text": node_text,
            }
        )

        return current_id

    def add_edge(
        source,
        target,
        edge_type="DFG",
    ):
        """
        Adds a data-flow edge.
        """

        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    def get_identifier_nodes(node):
        """
        Recursively collects identifier nodes.

        Example:

        x = a + b

        Returns identifiers:

        x
        a
        b
        """

        identifiers = []

        def traverse(current_node):

            if current_node.type == "identifier":

                identifiers.append(
                    current_node
                )

            for child in current_node.named_children:

                traverse(child)

        traverse(node)

        return identifiers

    def process_variable_use(
        identifier_node,
    ):
        """
        Creates a USE node for a variable.

        If a previous definition exists,
        creates a DEF_USE edge.
        """

        variable_name = identifier_node.text.decode(
            "utf-8",
            errors="ignore",
        )

        use_id = create_node(
            node_type="USE",
            variable=variable_name,
            ast_node=identifier_node,
        )

        if variable_name in latest_definition:

            add_edge(
                latest_definition[variable_name],
                use_id,
                "DEF_USE",
            )

        return use_id

    def process_declaration(
        declaration_node,
    ):
        """
        Processes variable declarations.

        Example:

        int x = 10;

        Creates:

        DEF(x)
        """

        for declarator in declaration_node.named_children:

            if declarator.type != "init_declarator":

                continue

            variable_node = declarator.child_by_field_name(
                "declarator"
            )

            value_node = declarator.child_by_field_name(
                "value"
            )

            if variable_node is None:

                continue

            variable_name = variable_node.text.decode(
                "utf-8",
                errors="ignore",
            )

            # Process variables used in the initializer.
            #
            # Example:
            #
            # int y = x + 1;
            #
            # x is a USE.
            #
            if value_node is not None:

                identifiers = get_identifier_nodes(
                    value_node
                )

                for identifier in identifiers:

                    process_variable_use(
                        identifier
                    )

            # Create the variable definition.
            definition_id = create_node(
                node_type="DEF",
                variable=variable_name,
                ast_node=variable_node,
            )

            latest_definition[
                variable_name
            ] = definition_id

    def process_assignment(
        assignment_node,
    ):
        """
        Processes assignments.

        Example:

        x = y + 1;

        Data flow:

        previous DEF(y)
              |
              v
             USE(y)

        and:

        DEF(x)
        """

        left_node = assignment_node.child_by_field_name(
            "left"
        )

        right_node = assignment_node.child_by_field_name(
            "right"
        )

        if left_node is None:

            return

        # Process variables used on the right side.
        #
        # Example:
        #
        # x = y + z;
        #
        # y and z are USE nodes.
        #
        if right_node is not None:

            identifiers = get_identifier_nodes(
                right_node
            )

            for identifier in identifiers:

                process_variable_use(
                    identifier
                )

        # Create a new definition for the variable
        # on the left side.
        #
        # Example:
        #
        # x = y;
        #
        # Creates a new DEF(x).
        #
        if left_node.type == "identifier":

            variable_name = left_node.text.decode(
                "utf-8",
                errors="ignore",
            )

            definition_id = create_node(
                node_type="DEF",
                variable=variable_name,
                ast_node=left_node,
            )

            latest_definition[
                variable_name
            ] = definition_id

    def process_update_expression(
        update_node,
    ):
        """
        Processes increment and decrement expressions.

        Examples:

        x++
        x--
        ++x
        --x

        These both USE and redefine the variable.
        """

        identifiers = get_identifier_nodes(
            update_node
        )

        for identifier in identifiers:

            variable_name = identifier.text.decode(
                "utf-8",
                errors="ignore",
            )

            # Previous value is used.
            process_variable_use(
                identifier
            )

            # Then a new definition is created.
            definition_id = create_node(
                node_type="DEF",
                variable=variable_name,
                ast_node=identifier,
            )

            latest_definition[
                variable_name
            ] = definition_id

    def process_return_statement(
        return_node,
    ):
        """
        Processes variables used in return statements.

        Example:

        return x;

        Creates:

        DEF(x) -> USE(x)
        """

        identifiers = get_identifier_nodes(
            return_node
        )

        for identifier in identifiers:

            process_variable_use(
                identifier
            )

    def process_node(node):
        """
        Recursively processes AST nodes.

        Handles important statement types and
        recursively traverses their children.
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

        if node.type in {
            "update_expression",
        }:

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

            process_node(
                child
            )

    process_node(root_node)

    return {
        "nodes": nodes,
        "edges": edges,
    }