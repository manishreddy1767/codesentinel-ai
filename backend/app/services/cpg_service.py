from app.services.ast_service import build_ast_graph
from app.services.cfg_service import build_cfg
from app.services.dfg_service import build_dfg


def build_cpg(root_node):
    """
    Builds a unified Code Property Graph (CPG).

    The CPG combines:

    - Abstract Syntax Tree (AST)
    - Control Flow Graph (CFG)
    - Data Flow Graph (DFG)

    The returned graph contains all nodes and edges
    from the three program representations.

    Since AST, CFG, and DFG generate their own node IDs,
    this service assigns globally unique IDs to every node
    before combining them into a single graph.
    """

    ast_graph = build_ast_graph(root_node)
    cfg_graph = build_cfg(root_node)
    dfg_graph = build_dfg(root_node)

    nodes = []
    edges = []

    next_node_id = 0

    def add_graph(graph, graph_type):
        """
        Adds one graph to the unified CPG.

        Parameters:
        - graph: Graph containing nodes and edges
        - graph_type: AST, CFG, or DFG
        """

        nonlocal next_node_id

        node_id_mapping = {}

        # Add nodes with globally unique IDs.
        for node in graph["nodes"]:

            old_id = node["id"]
            new_id = next_node_id

            next_node_id += 1

            node_id_mapping[old_id] = new_id

            cpg_node = node.copy()

            cpg_node["id"] = new_id
            cpg_node["graph_type"] = graph_type

            nodes.append(cpg_node)

        # Add edges using the new global IDs.
        for edge in graph["edges"]:

            source = edge["source"]
            target = edge["target"]

            if (
                source in node_id_mapping
                and target in node_id_mapping
            ):

                cpg_edge = edge.copy()

                cpg_edge["source"] = node_id_mapping[
                    source
                ]

                cpg_edge["target"] = node_id_mapping[
                    target
                ]

                cpg_edge["graph_type"] = graph_type

                edges.append(cpg_edge)

    # Add all program graphs.
    add_graph(
        ast_graph,
        "AST",
    )

    add_graph(
        cfg_graph,
        "CFG",
    )

    add_graph(
        dfg_graph,
        "DFG",
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "ast_nodes": len(
                ast_graph["nodes"]
            ),
            "ast_edges": len(
                ast_graph["edges"]
            ),
            "cfg_nodes": len(
                cfg_graph["nodes"]
            ),
            "cfg_edges": len(
                cfg_graph["edges"]
            ),
            "dfg_nodes": len(
                dfg_graph["nodes"]
            ),
            "dfg_edges": len(
                dfg_graph["edges"]
            ),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }