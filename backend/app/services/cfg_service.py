def build_cfg(root_node):
    """
    Builds a statement-level Control Flow Graph (CFG)
    from a Tree-sitter AST.

    Supported:
    - Sequential statements
    - Function entry and exit
    - return
    - if
    - if/else
    - nested if statements
    - while loops
    - for loops
    - do-while loops
    - break
    - continue
    """

    nodes = []
    edges = []
    node_id = 0

    # ==================================================
    # NODE AND EDGE HELPERS
    # ==================================================

    def create_node(ast_node=None, node_type=None, text=""):
        """
        Creates a CFG node and returns its ID.
        """

        nonlocal node_id

        current_id = node_id
        node_id += 1

        if ast_node is not None:

            start_line = ast_node.start_point[0] + 1
            end_line = ast_node.end_point[0] + 1

            node_text = ast_node.text.decode(
                "utf-8",
                errors="ignore",
            )

            final_type = node_type or ast_node.type

        else:

            start_line = None
            end_line = None
            node_text = text
            final_type = node_type

        nodes.append(
            {
                "id": current_id,
                "type": final_type,
                "start_line": start_line,
                "end_line": end_line,
                "text": node_text,
            }
        )

        return current_id


    def add_edge(source, target, edge_type="CFG"):
        """
        Adds an edge to the CFG.
        """

        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )


    # ==================================================
    # AST HELPERS
    # ==================================================

    def get_statements(block_node):
        """
        Returns CFG-relevant statements from a compound block.
        """

        if block_node is None:
            return []

        statement_types = {
            "declaration",
            "expression_statement",
            "return_statement",
            "if_statement",
            "while_statement",
            "for_statement",
            "do_statement",
            "break_statement",
            "continue_statement",
            "switch_statement",
        }

        return [
            child
            for child in block_node.named_children
            if child.type in statement_types
        ]


    # ==================================================
    # PROCESS STATEMENT LIST
    # ==================================================

    def process_statements(
        statements,
        incoming_nodes,
        function_exit,
        loop_condition=None,
        loop_exit=None,
    ):
        """
        Processes a sequence of statements.

        incoming_nodes:
            CFG nodes that connect into the first statement.

        loop_condition:
            Target used by continue statements.

        loop_exit:
            Target used by break statements.

        Returns:
            Nodes representing normal exits from the block.
        """

        current_exits = incoming_nodes

        for statement in statements:

            # No normal path remains.
            # Example:
            #
            # return;
            # next_statement;
            #
            if not current_exits:
                break

            # ----------------------------------------------
            # IF
            # ----------------------------------------------

            if statement.type == "if_statement":

                current_exits = process_if_statement(
                    statement,
                    current_exits,
                    function_exit,
                    loop_condition,
                    loop_exit,
                )

            # ----------------------------------------------
            # WHILE
            # ----------------------------------------------

            elif statement.type == "while_statement":

                current_exits = process_while_statement(
                    statement,
                    current_exits,
                    function_exit,
                )

            # ----------------------------------------------
            # FOR
            # ----------------------------------------------

            elif statement.type == "for_statement":

                current_exits = process_for_statement(
                    statement,
                    current_exits,
                    function_exit,
                )

            # ----------------------------------------------
            # DO WHILE
            # ----------------------------------------------

            elif statement.type == "do_statement":

                current_exits = process_do_statement(
                    statement,
                    current_exits,
                    function_exit,
                )

            # ----------------------------------------------
            # BREAK
            # ----------------------------------------------

            elif statement.type == "break_statement":

                break_id = create_node(
                    ast_node=statement,
                    node_type="BREAK",
                )

                for previous_id in current_exits:

                    add_edge(
                        previous_id,
                        break_id,
                    )

                if loop_exit is not None:

                    add_edge(
                        break_id,
                        loop_exit,
                        "CFG_BREAK",
                    )

                # break terminates normal flow
                current_exits = []

            # ----------------------------------------------
            # CONTINUE
            # ----------------------------------------------

            elif statement.type == "continue_statement":

                continue_id = create_node(
                    ast_node=statement,
                    node_type="CONTINUE",
                )

                for previous_id in current_exits:

                    add_edge(
                        previous_id,
                        continue_id,
                    )

                if loop_condition is not None:

                    add_edge(
                        continue_id,
                        loop_condition,
                        "CFG_CONTINUE",
                    )

                # continue terminates normal flow
                current_exits = []

            # ----------------------------------------------
            # RETURN
            # ----------------------------------------------

            elif statement.type == "return_statement":

                return_id = create_node(
                    ast_node=statement
                )

                for previous_id in current_exits:

                    add_edge(
                        previous_id,
                        return_id,
                    )

                add_edge(
                    return_id,
                    function_exit,
                    "CFG_RETURN",
                )

                # return terminates normal flow
                current_exits = []

            # ----------------------------------------------
            # NORMAL STATEMENT
            # ----------------------------------------------

            else:

                statement_id = create_node(
                    ast_node=statement
                )

                for previous_id in current_exits:

                    add_edge(
                        previous_id,
                        statement_id,
                    )

                current_exits = [
                    statement_id
                ]

        return current_exits


    # ==================================================
    # IF / ELSE
    # ==================================================

    def process_if_statement(
        if_node,
        incoming_nodes,
        function_exit,
        loop_condition=None,
        loop_exit=None,
    ):
        """
        Builds CFG for:

        if (condition)
        {
            ...
        }

        if (condition)
        {
            ...
        }
        else
        {
            ...
        }
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

        # ----------------------------------------------
        # CONDITION
        # ----------------------------------------------

        condition_id = create_node(
            ast_node=condition,
            node_type="IF_CONDITION",
        )

        for previous_id in incoming_nodes:

            add_edge(
                previous_id,
                condition_id,
            )

        # ----------------------------------------------
        # MERGE
        # ----------------------------------------------

        merge_id = create_node(
            node_type="IF_MERGE",
            text="",
        )

        # ----------------------------------------------
        # TRUE BRANCH
        # ----------------------------------------------

        true_exits = []

        if consequence is not None:

            if consequence.type == "compound_statement":

                statements = get_statements(
                    consequence
                )

                if statements:

                    # Recursively process statements
                    #
                    # This is the important part that fixes
                    # nested if/break/continue handling.

                    true_exits = process_statements(
                        statements,
                        [condition_id],
                        function_exit,
                        loop_condition,
                        loop_exit,
                    )

                    # Mark first edge as TRUE

                    if true_exits is not None:

                        # Replace the first CFG edge
                        # from condition to true branch
                        # with CFG_TRUE.

                        for edge in edges:

                            if (
                                edge["source"] == condition_id
                                and edge["type"] == "CFG"
                            ):

                                edge["type"] = "CFG_TRUE"
                                break

                else:

                    add_edge(
                        condition_id,
                        merge_id,
                        "CFG_TRUE",
                    )

            else:

                consequence_exits = process_statements(
                    [consequence],
                    [condition_id],
                    function_exit,
                    loop_condition,
                    loop_exit,
                )

                true_exits = consequence_exits

                # Convert condition edge to TRUE

                for edge in edges:

                    if (
                        edge["source"] == condition_id
                        and edge["type"] == "CFG"
                    ):

                        edge["type"] = "CFG_TRUE"
                        break

        else:

            add_edge(
                condition_id,
                merge_id,
                "CFG_TRUE",
            )

        # Connect normal TRUE exits to merge

        for node in true_exits:

            add_edge(
                node,
                merge_id,
                "CFG",
            )

        # ----------------------------------------------
        # FALSE BRANCH
        # ----------------------------------------------

        false_exits = []

        if alternative is not None:

            # C/C++ Tree-sitter generally stores
            # else as an else_clause.

            if alternative.type == "else_clause":

                else_body = None

                for child in alternative.named_children:

                    if child.type in {
                        "compound_statement",
                        "if_statement",
                        "expression_statement",
                        "return_statement",
                        "while_statement",
                        "for_statement",
                        "do_statement",
                        "break_statement",
                        "continue_statement",
                    }:

                        else_body = child
                        break

                if else_body is not None:

                    # ----------------------------------
                    # ELSE BLOCK
                    # ----------------------------------

                    if else_body.type == "compound_statement":

                        statements = get_statements(
                            else_body
                        )

                        if statements:

                            false_exits = process_statements(
                                statements,
                                [condition_id],
                                function_exit,
                                loop_condition,
                                loop_exit,
                            )

                            # Mark first connection
                            # as FALSE

                            for edge in edges:

                                if (
                                    edge["source"] == condition_id
                                    and edge["type"] == "CFG"
                                ):

                                    edge["type"] = "CFG_FALSE"
                                    break

                        else:

                            add_edge(
                                condition_id,
                                merge_id,
                                "CFG_FALSE",
                            )

                    # ----------------------------------
                    # ELSE IF
                    # ----------------------------------

                    elif else_body.type == "if_statement":

                        false_exits = process_if_statement(
                            else_body,
                            [condition_id],
                            function_exit,
                            loop_condition,
                            loop_exit,
                        )

                        # Mark edge to nested condition
                        # as FALSE

                        for edge in edges:

                            if (
                                edge["source"] == condition_id
                                and edge["type"] == "CFG"
                            ):

                                edge["type"] = "CFG_FALSE"
                                break

                    # ----------------------------------
                    # ELSE SINGLE STATEMENT
                    # ----------------------------------

                    else:

                        false_exits = process_statements(
                            [else_body],
                            [condition_id],
                            function_exit,
                            loop_condition,
                            loop_exit,
                        )

                        for edge in edges:

                            if (
                                edge["source"] == condition_id
                                and edge["type"] == "CFG"
                            ):

                                edge["type"] = "CFG_FALSE"
                                break

                else:

                    add_edge(
                        condition_id,
                        merge_id,
                        "CFG_FALSE",
                    )

            else:

                false_exits = process_statements(
                    [alternative],
                    [condition_id],
                    function_exit,
                    loop_condition,
                    loop_exit,
                )

                for edge in edges:

                    if (
                        edge["source"] == condition_id
                        and edge["type"] == "CFG"
                    ):

                        edge["type"] = "CFG_FALSE"
                        break

        else:

            # No ELSE.
            #
            # False path directly reaches merge.

            add_edge(
                condition_id,
                merge_id,
                "CFG_FALSE",
            )

        # Connect normal FALSE exits to merge

        for node in false_exits:

            add_edge(
                node,
                merge_id,
                "CFG",
            )

        # ----------------------------------------------
        # DETERMINE NORMAL EXITS
        # ----------------------------------------------

        # If both branches terminate
        # (for example return/break/continue),
        # there may be no valid path through merge.

        has_true_path = len(true_exits) > 0
        has_false_path = (
            alternative is None
            or len(false_exits) > 0
        )

        if has_true_path or has_false_path:

            return [
                merge_id
            ]

        return []


    # ==================================================
    # WHILE LOOP
    # ==================================================

    def process_while_statement(
        while_node,
        incoming_nodes,
        function_exit,
    ):
        """
        Builds CFG for:

        while (condition)
        {
            body
        }
        """

        condition = while_node.child_by_field_name(
            "condition"
        )

        body = while_node.child_by_field_name(
            "body"
        )

        # ----------------------------------------------
        # CONDITION
        # ----------------------------------------------

        condition_id = create_node(
            ast_node=condition,
            node_type="WHILE_CONDITION",
        )

        for previous_id in incoming_nodes:

            add_edge(
                previous_id,
                condition_id,
            )

        # ----------------------------------------------
        # EXIT
        # ----------------------------------------------

        exit_id = create_node(
            node_type="WHILE_EXIT",
            text="",
        )

        # ----------------------------------------------
        # BODY
        # ----------------------------------------------

        body_exits = []

        if body is not None:

            if body.type == "compound_statement":

                statements = get_statements(
                    body
                )

                if statements:

                    body_exits = process_statements(
                        statements,
                        [condition_id],
                        function_exit,
                        condition_id,
                        exit_id,
                    )

                    # Convert first condition edge
                    # into TRUE branch.

                    for edge in edges:

                        if (
                            edge["source"] == condition_id
                            and edge["type"] == "CFG"
                        ):

                            edge["type"] = "CFG_TRUE"
                            break

                else:

                    add_edge(
                        condition_id,
                        condition_id,
                        "CFG_TRUE",
                    )

            else:

                body_exits = process_statements(
                    [body],
                    [condition_id],
                    function_exit,
                    condition_id,
                    exit_id,
                )

                for edge in edges:

                    if (
                        edge["source"] == condition_id
                        and edge["type"] == "CFG"
                    ):

                        edge["type"] = "CFG_TRUE"
                        break

        # ----------------------------------------------
        # LOOP BACK
        # ----------------------------------------------

        for node in body_exits:

            add_edge(
                node,
                condition_id,
                "CFG_LOOP_BACK",
            )

        # ----------------------------------------------
        # FALSE PATH
        # ----------------------------------------------

        add_edge(
            condition_id,
            exit_id,
            "CFG_FALSE",
        )

        return [
            exit_id
        ]


    # ==================================================
    # FOR LOOP
    # ==================================================

    def process_for_statement(
        for_node,
        incoming_nodes,
        function_exit,
    ):
        """
        Builds CFG for a for loop.
        """

        body = for_node.child_by_field_name(
            "body"
        )

        condition = for_node.child_by_field_name(
            "condition"
        )

        update = for_node.child_by_field_name(
            "update"
        )

        # ----------------------------------------------
        # CONDITION
        # ----------------------------------------------

        condition_id = create_node(
            ast_node=condition
            if condition is not None
            else for_node,
            node_type="FOR_CONDITION",
        )

        for previous_id in incoming_nodes:

            add_edge(
                previous_id,
                condition_id,
            )

        # ----------------------------------------------
        # EXIT
        # ----------------------------------------------

        exit_id = create_node(
            node_type="FOR_EXIT",
            text="",
        )

        # ----------------------------------------------
        # UPDATE NODE
        # ----------------------------------------------

        update_id = None

        if update is not None:

            update_id = create_node(
                ast_node=update,
                node_type="FOR_UPDATE",
            )

        # Continue should go to UPDATE
        # if update exists.

        continue_target = (
            update_id
            if update_id is not None
            else condition_id
        )

        # ----------------------------------------------
        # BODY
        # ----------------------------------------------

        body_exits = []

        if body is not None:

            if body.type == "compound_statement":

                statements = get_statements(
                    body
                )

                if statements:

                    body_exits = process_statements(
                        statements,
                        [condition_id],
                        function_exit,
                        continue_target,
                        exit_id,
                    )

                    for edge in edges:

                        if (
                            edge["source"] == condition_id
                            and edge["type"] == "CFG"
                        ):

                            edge["type"] = "CFG_TRUE"
                            break

            else:

                body_exits = process_statements(
                    [body],
                    [condition_id],
                    function_exit,
                    continue_target,
                    exit_id,
                )

                for edge in edges:

                    if (
                        edge["source"] == condition_id
                        and edge["type"] == "CFG"
                    ):

                        edge["type"] = "CFG_TRUE"
                        break

        # ----------------------------------------------
        # UPDATE AND LOOP BACK
        # ----------------------------------------------

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

        # ----------------------------------------------
        # FALSE PATH
        # ----------------------------------------------

        add_edge(
            condition_id,
            exit_id,
            "CFG_FALSE",
        )

        return [
            exit_id
        ]


    # ==================================================
    # DO-WHILE LOOP
    # ==================================================

    def process_do_statement(
        do_node,
        incoming_nodes,
        function_exit,
    ):
        """
        Builds CFG for:

        do
        {
            body
        }
        while (condition);
        """

        body = do_node.child_by_field_name(
            "body"
        )

        condition = do_node.child_by_field_name(
            "condition"
        )

        # ----------------------------------------------
        # EXIT
        # ----------------------------------------------

        exit_id = create_node(
            node_type="DO_WHILE_EXIT",
            text="",
        )

        # ----------------------------------------------
        # CONDITION
        # ----------------------------------------------

        condition_id = create_node(
            ast_node=condition
            if condition is not None
            else do_node,
            node_type="DO_WHILE_CONDITION",
        )

        # ----------------------------------------------
        # BODY
        # ----------------------------------------------

        body_exits = []

        first_body_nodes = []

        if body is not None:

            if body.type == "compound_statement":

                statements = get_statements(
                    body
                )

                if statements:

                    body_exits = process_statements(
                        statements,
                        incoming_nodes,
                        function_exit,
                        condition_id,
                        exit_id,
                    )

                    # Find first node connected
                    # from incoming nodes.

                    for edge in edges:

                        if (
                            edge["source"] in incoming_nodes
                            and edge["type"] == "CFG"
                        ):

                            first_body_nodes.append(
                                edge["target"]
                            )

                            break

            else:

                body_exits = process_statements(
                    [body],
                    incoming_nodes,
                    function_exit,
                    condition_id,
                    exit_id,
                )

                for edge in edges:

                    if (
                        edge["source"] in incoming_nodes
                        and edge["type"] == "CFG"
                    ):

                        first_body_nodes.append(
                            edge["target"]
                        )

                        break

        # ----------------------------------------------
        # BODY → CONDITION
        # ----------------------------------------------

        for node in body_exits:

            add_edge(
                node,
                condition_id,
                "CFG",
            )

        # ----------------------------------------------
        # TRUE → BODY
        # ----------------------------------------------

        for first_body_id in first_body_nodes:

            add_edge(
                condition_id,
                first_body_id,
                "CFG_TRUE",
            )

        # ----------------------------------------------
        # FALSE → EXIT
        # ----------------------------------------------

        add_edge(
            condition_id,
            exit_id,
            "CFG_FALSE",
        )

        return [
            exit_id
        ]


    # ==================================================
    # FUNCTION PROCESSING
    # ==================================================

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

        statements = get_statements(
            body
        )

        if not statements:

            add_edge(
                entry_id,
                exit_id,
            )

            return

        final_exits = process_statements(
            statements,
            [entry_id],
            exit_id,
        )

        for node in final_exits:

            add_edge(
                node,
                exit_id,
                "CFG",
            )


    # ==================================================
    # AST TRAVERSAL
    # ==================================================

    def traverse(node):
        """
        Finds function definitions.
        """

        if node.type == "function_definition":

            process_function(node)

            return

        for child in node.named_children:

            traverse(child)


    traverse(root_node)

    return {
        "nodes": nodes,
        "edges": edges,
    }