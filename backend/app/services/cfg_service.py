def build_cfg(root_node):
    """
    Builds a statement-level Control Flow Graph (CFG)
    from a Tree-sitter AST.

    Current support:
    - Sequential statements
    - Return statements
    - if statements
    - if/else statements
    - while loops
    - for loops
    - break statements
    - continue statements
    - Function entry and exit nodes
    """

    nodes = []
    edges = []

    node_id = 0

    def create_node(
        ast_node=None,
        node_type=None,
        text="",
    ):
        """
        Creates a CFG node and returns its ID.
        """

        nonlocal node_id

        current_id = node_id
        node_id += 1

        if ast_node is not None:

            start_line = ast_node.start_point[0] + 1
            end_line = ast_node.end_point[0] + 1

            start_byte = ast_node.start_byte
            end_byte = ast_node.end_byte

            node_text = ast_node.text.decode(
                "utf-8",
                errors="ignore",
            )

        else:

            start_line = None
            end_line = None

            start_byte = None
            end_byte = None

            node_text = text

        nodes.append(
            {
                "id": current_id,
                "type": node_type or (
                    ast_node.type
                    if ast_node is not None
                    else None
                ),
                "start_line": start_line,
                "end_line": end_line,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "text": node_text,
            }
        )

        return current_id

    def add_edge(
        source,
        target,
        edge_type="CFG",
    ):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    def get_statements(block_node):
        """
        Returns statement-like children from
        a compound block.
        """

        statement_types = {
            "declaration",
            "expression_statement",
            "return_statement",
            "if_statement",
            "while_statement",
            "for_statement",
            "switch_statement",
            "do_statement",
            "break_statement",
            "continue_statement",
        }

        return [
            child
            for child in block_node.named_children
            if child.type in statement_types
        ]

    def process_statements(
        statements,
        previous_nodes,
        function_exit,
        loop_context=None,
    ):
        """
        Processes statements and returns nodes where
        normal control flow exits.

        loop_context contains:

        {
            "continue_target": node_id,
            "break_target": node_id,
        }
        """

        current_exits = previous_nodes

        for statement in statements:

            if not current_exits:
                break

            if statement.type == "if_statement":

                current_exits = process_if_statement(
                    statement,
                    current_exits,
                    function_exit,
                    loop_context,
                )

            elif statement.type == "while_statement":

                current_exits = process_while_statement(
                    statement,
                    current_exits,
                    function_exit,
                )

            elif statement.type == "for_statement":

                current_exits = process_for_statement(
                    statement,
                    current_exits,
                    function_exit,
                )

            else:

                if statement.type == "break_statement":

                    statement_id = create_node(
                        ast_node=statement,
                        node_type="BREAK",
                    )

                    for previous_id in current_exits:
                        add_edge(
                            previous_id,
                            statement_id,
                        )

                    if (
                        loop_context is not None
                        and loop_context.get(
                            "break_target"
                        ) is not None
                    ):

                        add_edge(
                            statement_id,
                            loop_context[
                                "break_target"
                            ],
                            "CFG_BREAK",
                        )

                    current_exits = []

                    continue

                if statement.type == "continue_statement":

                    statement_id = create_node(
                        ast_node=statement,
                        node_type="CONTINUE",
                    )

                    for previous_id in current_exits:
                        add_edge(
                            previous_id,
                            statement_id,
                        )

                    if (
                        loop_context is not None
                        and loop_context.get(
                            "continue_target"
                        ) is not None
                    ):

                        add_edge(
                            statement_id,
                            loop_context[
                                "continue_target"
                            ],
                            "CFG_CONTINUE",
                        )

                    current_exits = []

                    continue

                statement_id = create_node(
                    ast_node=statement
                )

                for previous_id in current_exits:

                    add_edge(
                        previous_id,
                        statement_id,
                    )

                if statement.type == "return_statement":

                    add_edge(
                        statement_id,
                        function_exit,
                        "CFG_RETURN",
                    )

                    current_exits = []

                else:

                    current_exits = [
                        statement_id
                    ]

        return current_exits

    def process_if_statement(
        if_node,
        previous_nodes,
        function_exit,
        loop_context=None,
    ):
        """
        Builds CFG branching for if and if/else.
        """

        condition = if_node.child_by_field_name(
            "condition"
        )

        consequence = if_node.child_by_field_name(
            "consequence"
        )

        alternative = if_node.child_by_field_name(
            "alternative"
        )

        condition_id = create_node(
            ast_node=condition,
            node_type="IF_CONDITION",
        )

        for previous_id in previous_nodes:

            add_edge(
                previous_id,
                condition_id,
            )

        merge_id = create_node(
            node_type="IF_MERGE",
            text="",
        )

        # ----------------------------------------
        # TRUE BRANCH
        # ----------------------------------------

        true_exits = []

        if consequence is not None:

            if (
                consequence.type
                == "compound_statement"
            ):

                consequence_statements = (
                    get_statements(
                        consequence
                    )
                )

                if consequence_statements:

                    first_statement = (
                        consequence_statements[0]
                    )

                    first_node_id = None

                    # Process statements
                    true_exits = (
                        process_statements(
                            consequence_statements,
                            [condition_id],
                            function_exit,
                            loop_context,
                        )
                    )

                    # Change the first outgoing edge
                    # from condition to CFG_TRUE
                    for edge in edges:

                        if (
                            edge["source"]
                            == condition_id
                            and edge["type"]
                            == "CFG"
                        ):

                            edge["type"] = (
                                "CFG_TRUE"
                            )

                            break

                else:

                    true_exits = [
                        condition_id
                    ]

            else:

                consequence_id = create_node(
                    ast_node=consequence
                )

                add_edge(
                    condition_id,
                    consequence_id,
                    "CFG_TRUE",
                )

                true_exits = [
                    consequence_id
                ]

        else:

            true_exits = [
                condition_id
            ]

        # ----------------------------------------
        # TRUE -> MERGE
        # ----------------------------------------

        for node in true_exits:

            add_edge(
                node,
                merge_id,
                "CFG",
            )

        # ----------------------------------------
        # FALSE BRANCH
        # ----------------------------------------

        false_exits = []

        if alternative is not None:

            if (
                alternative.type
                == "else_clause"
            ):

                alternative_body = None

                for child in (
                    alternative.named_children
                ):

                    if (
                        child.type
                        == "compound_statement"
                    ):

                        alternative_body = child
                        break

                    if (
                        child.type
                        == "if_statement"
                    ):

                        alternative_body = child
                        break

                if (
                    alternative_body is not None
                    and alternative_body.type
                    == "compound_statement"
                ):

                    alternative_statements = (
                        get_statements(
                            alternative_body
                        )
                    )

                    if alternative_statements:

                        false_exits = (
                            process_statements(
                                alternative_statements,
                                [condition_id],
                                function_exit,
                                loop_context,
                            )
                        )

                        for edge in edges:

                            if (
                                edge["source"]
                                == condition_id
                                and edge["type"]
                                == "CFG"
                            ):

                                edge["type"] = (
                                    "CFG_FALSE"
                                )

                                break

                    else:

                        false_exits = [
                            condition_id
                        ]

                elif (
                    alternative_body is not None
                    and alternative_body.type
                    == "if_statement"
                ):

                    false_exits = (
                        process_if_statement(
                            alternative_body,
                            [condition_id],
                            function_exit,
                            loop_context,
                        )
                    )

                    for edge in edges:

                        if (
                            edge["source"]
                            == condition_id
                            and edge["type"]
                            == "CFG"
                        ):

                            edge["type"] = (
                                "CFG_FALSE"
                            )

                            break

            else:

                alternative_id = create_node(
                    ast_node=alternative
                )

                add_edge(
                    condition_id,
                    alternative_id,
                    "CFG_FALSE",
                )

                false_exits = [
                    alternative_id
                ]

        else:

            add_edge(
                condition_id,
                merge_id,
                "CFG_FALSE",
            )

        # ----------------------------------------
        # FALSE -> MERGE
        # ----------------------------------------

        for node in false_exits:

            add_edge(
                node,
                merge_id,
                "CFG",
            )

        return [
            merge_id
        ]

    def process_while_statement(
        while_node,
        previous_nodes,
        function_exit,
    ):
        """
        Builds CFG for while loops.
        """

        condition = while_node.child_by_field_name(
            "condition"
        )

        body = while_node.child_by_field_name(
            "body"
        )

        condition_id = create_node(
            ast_node=condition,
            node_type="WHILE_CONDITION",
        )

        loop_exit_id = create_node(
            node_type="WHILE_EXIT",
            text="",
        )

        for previous_id in previous_nodes:

            add_edge(
                previous_id,
                condition_id,
            )

        loop_context = {
            "continue_target": condition_id,
            "break_target": loop_exit_id,
        }

        body_exits = []

        if body is not None:

            if (
                body.type
                == "compound_statement"
            ):

                body_exits = process_statements(
                    get_statements(body),
                    [condition_id],
                    function_exit,
                    loop_context,
                )

                # Mark first body edge TRUE
                for edge in edges:

                    if (
                        edge["source"]
                        == condition_id
                        and edge["type"]
                        == "CFG"
                    ):

                        edge["type"] = (
                            "CFG_TRUE"
                        )

                        break

            else:

                body_id = create_node(
                    ast_node=body
                )

                add_edge(
                    condition_id,
                    body_id,
                    "CFG_TRUE",
                )

                body_exits = [
                    body_id
                ]

        for node in body_exits:

            add_edge(
                node,
                condition_id,
                "CFG_LOOP_BACK",
            )

        add_edge(
            condition_id,
            loop_exit_id,
            "CFG_FALSE",
        )

        return [
            loop_exit_id
        ]

    def process_for_statement(
        for_node,
        previous_nodes,
        function_exit,
    ):
        """
        Builds CFG for for loops.
        """

        initializer = (
            for_node.child_by_field_name(
                "initializer"
            )
        )

        condition = (
            for_node.child_by_field_name(
                "condition"
            )
        )

        update = (
            for_node.child_by_field_name(
                "update"
            )
        )

        body = (
            for_node.child_by_field_name(
                "body"
            )
        )

        loop_exit_id = create_node(
            node_type="FOR_EXIT",
            text="",
        )

        # ----------------------------------------
        # INITIALIZER
        # ----------------------------------------

        if initializer is not None:

            initializer_id = create_node(
                ast_node=initializer,
                node_type="FOR_INITIALIZER",
            )

            for previous_id in previous_nodes:

                add_edge(
                    previous_id,
                    initializer_id,
                )

            current_previous = [
                initializer_id
            ]

        else:

            current_previous = previous_nodes

        # ----------------------------------------
        # CONDITION
        # ----------------------------------------

        if condition is not None:

            condition_id = create_node(
                ast_node=condition,
                node_type="FOR_CONDITION",
            )

        else:

            condition_id = create_node(
                ast_node=for_node,
                node_type="FOR_CONDITION",
            )

        for previous_id in current_previous:

            add_edge(
                previous_id,
                condition_id,
            )

        # ----------------------------------------
        # UPDATE
        # ----------------------------------------

        if update is not None:

            update_id = create_node(
                ast_node=update,
                node_type="FOR_UPDATE",
            )

            continue_target = update_id

        else:

            update_id = None

            continue_target = condition_id

        loop_context = {
            "continue_target": continue_target,
            "break_target": loop_exit_id,
        }

        # ----------------------------------------
        # BODY
        # ----------------------------------------

        body_exits = []

        if body is not None:

            if (
                body.type
                == "compound_statement"
            ):

                body_exits = process_statements(
                    get_statements(body),
                    [condition_id],
                    function_exit,
                    loop_context,
                )

                for edge in edges:

                    if (
                        edge["source"]
                        == condition_id
                        and edge["type"]
                        == "CFG"
                    ):

                        edge["type"] = (
                            "CFG_TRUE"
                        )

                        break

            else:

                body_id = create_node(
                    ast_node=body
                )

                add_edge(
                    condition_id,
                    body_id,
                    "CFG_TRUE",
                )

                body_exits = [
                    body_id
                ]

        # ----------------------------------------
        # BODY -> UPDATE
        # ----------------------------------------

        if update_id is not None:

            for node in body_exits:

                add_edge(
                    node,
                    update_id,
                    "CFG",
                )

            add_edge(
                update_id,
                condition_id,
                "CFG_LOOP_BACK",
            )

        else:

            for node in body_exits:

                add_edge(
                    node,
                    condition_id,
                    "CFG_LOOP_BACK",
                )

        # ----------------------------------------
        # FALSE -> EXIT
        # ----------------------------------------

        add_edge(
            condition_id,
            loop_exit_id,
            "CFG_FALSE",
        )

        return [
            loop_exit_id
        ]

    def process_function(function_node):
        """
        Builds CFG for one function.
        """

        entry_id = create_node(
            ast_node=function_node,
            node_type="FUNCTION_ENTRY",
        )

        exit_id = create_node(
            node_type="FUNCTION_EXIT",
            text="",
        )

        body = function_node.child_by_field_name(
            "body"
        )

        if body is None:

            add_edge(
                entry_id,
                exit_id,
            )

            return

        final_exits = process_statements(
            get_statements(body),
            [entry_id],
            exit_id,
        )

        for node in final_exits:

            add_edge(
                node,
                exit_id,
            )

    def traverse(node):
        """
        Finds and processes functions.
        """

        if node.type == "function_definition":

            process_function(node)

        for child in node.named_children:

            traverse(child)

    traverse(root_node)

    return {
        "nodes": nodes,
        "edges": edges,
    }