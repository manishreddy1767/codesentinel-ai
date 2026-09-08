from app.services.ast_service import build_ast_graph
from app.services.cfg_service import build_cfg
from app.services.dfg_service import build_dfg


def build_cpg(root_node):
    """
    Builds a Code Property Graph (CPG).

    Combines:

    - AST
    - CFG
    - DFG

    Cross-graph edges are created using precise
    Tree-sitter source byte ranges.
    """

    # ----------------------------------------------
    # Build graphs
    # ----------------------------------------------

    ast_graph = build_ast_graph(
        root_node
    )

    cfg_graph = build_cfg(
        root_node
    )

    dfg_graph = build_dfg(
        root_node
    )

    ast_nodes = ast_graph["nodes"]
    ast_edges = ast_graph["edges"]

    cfg_nodes = cfg_graph["nodes"]
    cfg_edges = cfg_graph["edges"]

    dfg_nodes = dfg_graph["nodes"]
    dfg_edges = dfg_graph["edges"]

    # ----------------------------------------------
    # Final graph
    # ----------------------------------------------

    nodes = []
    edges = []

    # ==============================================
    # ADD AST
    # ==============================================

    for node in ast_nodes:

        combined_node = node.copy()

        combined_node[
            "graph_type"
        ] = "AST"

        nodes.append(
            combined_node
        )

    for edge in ast_edges:

        combined_edge = edge.copy()

        combined_edge[
            "graph_type"
        ] = "AST"

        edges.append(
            combined_edge
        )

    # ==============================================
    # ADD CFG
    # ==============================================

    cfg_offset = len(
        ast_nodes
    )

    for node in cfg_nodes:

        combined_node = node.copy()

        combined_node["id"] = (
            node["id"]
            + cfg_offset
        )

        combined_node[
            "graph_type"
        ] = "CFG"

        nodes.append(
            combined_node
        )

    for edge in cfg_edges:

        combined_edge = edge.copy()

        combined_edge["source"] = (
            edge["source"]
            + cfg_offset
        )

        combined_edge["target"] = (
            edge["target"]
            + cfg_offset
        )

        combined_edge[
            "graph_type"
        ] = "CFG"

        edges.append(
            combined_edge
        )

    # ==============================================
    # ADD DFG
    # ==============================================

    dfg_offset = (
        len(ast_nodes)
        + len(cfg_nodes)
    )

    for node in dfg_nodes:

        combined_node = node.copy()

        combined_node["id"] = (
            node["id"]
            + dfg_offset
        )

        combined_node[
            "graph_type"
        ] = "DFG"

        nodes.append(
            combined_node
        )

    for edge in dfg_edges:

        combined_edge = edge.copy()

        combined_edge["source"] = (
            edge["source"]
            + dfg_offset
        )

        combined_edge["target"] = (
            edge["target"]
            + dfg_offset
        )

        combined_edge[
            "graph_type"
        ] = "DFG"

        edges.append(
            combined_edge
        )

    # ==============================================
    # AST -> CFG CONNECTIONS
    # ==============================================

    ast_cfg_edges = []

    for cfg_node in cfg_nodes:

        cfg_start = cfg_node.get(
            "start_byte"
        )

        cfg_end = cfg_node.get(
            "end_byte"
        )

        # Skip synthetic nodes
        #
        # FUNCTION_EXIT
        # IF_MERGE
        # WHILE_EXIT
        # FOR_EXIT

        if (
            cfg_start is None
            or cfg_end is None
        ):

            continue

        best_ast_node = None

        # ------------------------------------------
        # Find exact source span match
        # ------------------------------------------

        for ast_node in ast_nodes:

            if (
                ast_node.get(
                    "start_byte"
                )
                == cfg_start
                and ast_node.get(
                    "end_byte"
                )
                == cfg_end
            ):

                best_ast_node = ast_node

                break

        # ------------------------------------------
        # If no exact match, find smallest AST node
        # that contains the CFG source range.
        # ------------------------------------------

        if best_ast_node is None:

            containing_nodes = []

            for ast_node in ast_nodes:

                ast_start = ast_node.get(
                    "start_byte"
                )

                ast_end = ast_node.get(
                    "end_byte"
                )

                if (
                    ast_start is None
                    or ast_end is None
                ):

                    continue

                if (
                    ast_start <= cfg_start
                    and ast_end >= cfg_end
                ):

                    containing_nodes.append(
                        ast_node
                    )

            if containing_nodes:

                best_ast_node = min(
                    containing_nodes,
                    key=lambda node: (
                        node["end_byte"]
                        - node["start_byte"]
                    ),
                )

        # ------------------------------------------
        # Create exactly ONE AST_CFG edge
        # ------------------------------------------

        if best_ast_node is not None:

            ast_cfg_edges.append(
                {
                    "source": best_ast_node[
                        "id"
                    ],
                    "target": (
                        cfg_node["id"]
                        + cfg_offset
                    ),
                    "type": "AST_CFG",
                    "graph_type": "CPG",
                }
            )

    edges.extend(
        ast_cfg_edges
    )

    # ==============================================
    # AST -> DFG CONNECTIONS
    # ==============================================

    ast_dfg_edges = []

    for dfg_node in dfg_nodes:

        dfg_start = dfg_node.get(
            "start_byte"
        )

        dfg_end = dfg_node.get(
            "end_byte"
        )

        variable = dfg_node.get(
            "variable",
            ""
        )

        if (
            dfg_start is None
            or dfg_end is None
        ):

            continue

        best_ast_node = None

        # ------------------------------------------
        # Exact identifier span match
        # ------------------------------------------

        for ast_node in ast_nodes:

            if (
                ast_node.get("type")
                != "identifier"
            ):

                continue

            if (
                ast_node.get("text")
                != variable
            ):

                continue

            if (
                ast_node.get(
                    "start_byte"
                )
                == dfg_start
                and ast_node.get(
                    "end_byte"
                )
                == dfg_end
            ):

                best_ast_node = ast_node

                break

        # ------------------------------------------
        # Create exactly ONE AST_DFG edge
        # ------------------------------------------

        if best_ast_node is not None:

            ast_dfg_edges.append(
                {
                    "source": best_ast_node[
                        "id"
                    ],
                    "target": (
                        dfg_node["id"]
                        + dfg_offset
                    ),
                    "type": "AST_DFG",
                    "graph_type": "CPG",
                }
            )

    edges.extend(
        ast_dfg_edges
    )

    # ==============================================
    # METADATA
    # ==============================================

    metadata = {
        "ast_nodes": len(
            ast_nodes
        ),

        "ast_edges": len(
            ast_edges
        ),

        "cfg_nodes": len(
            cfg_nodes
        ),

        "cfg_edges": len(
            cfg_edges
        ),

        "dfg_nodes": len(
            dfg_nodes
        ),

        "dfg_edges": len(
            dfg_edges
        ),

        "ast_cfg_edges": len(
            ast_cfg_edges
        ),

        "ast_dfg_edges": len(
            ast_dfg_edges
        ),

        "total_nodes": len(
            nodes
        ),

        "total_edges": len(
            edges
        ),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }