from os import terminal_size
import networkx as nx
import matplotlib.pyplot as plt


def get_networkx(dnf_model, clause_offset=0, label_or_post="", use_weak=False):
    G = nx.DiGraph()
    terminal = "out"
    if dnf_model.target_name is not None:
        terminal = dnf_model.target_name
    # Add nodes for input variables and their negations
    for s in dnf_model.feature_names:
        G.add_node(s)
    if use_weak:
        dnf = dnf_model.learned_dnf_weak
    else:
        dnf = dnf_model.learned_dnf
    G.add_node(terminal)
    if len(dnf) == 0:
        return G
    # Add nodes for AND gates (clauses)
    for i in range(len(dnf)):
        G.add_node(f"AND_{clause_offset + i}")
    # Add node for the final OR gate
    or_label = "OR" + label_or_post
    G.add_node(or_label)
    # ======================================
    # Add edges from literals to AND gates and from AND gates to the OR gate
    for clause_index, clause in enumerate(dnf):
        and_gate_node = f"AND_{clause_offset + clause_index}"
        for literal_index in np.where(clause)[0]:
            if literal_index < dnf_model.n_feature:
                # Positive literal
                literal_node = dnf_model.feature_names[literal_index]
                edge_label = "+"
            else:
                # Negative literal
                literal_node = dnf_model.feature_names[
                    literal_index - dnf_model.n_feature
                ]
                edge_label = "-"
            G.add_edge(literal_node, and_gate_node, label=edge_label)
        # Add edge from the AND gate to the OR gate
        G.add_edge(and_gate_node, or_label)
    G.add_edge(or_label, terminal)
    # ======================================
    return G


def draw_and_or(G):
    # Use spring_layout with a good k value from previous step
    pos = nx.spring_layout(G, k=0.7)

    input_nodes = [
        n for n in G.nodes() if n.startswith("x")
    ]  # Only include positive literal nodes
    and_nodes = [n for n in G.nodes() if n.startswith("AND_")]
    or_node = ["OR"]

    # Draw the nodes with different styles and set the label for the legend
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_size=70,
        node_color="skyblue",
        label="Literals",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=and_nodes,
        node_size=100,
        node_color="lightgreen",
        node_shape="s",
        label="AND Gates",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=or_node,
        node_size=150,
        node_color="salmon",
        node_shape="o",
        label="OR Gate",
    )

    # Separate edges based on their label
    positive_literal_edges = [
        (u, v) for u, v, d in G.edges(data=True) if "label" in d and d["label"] == "+"
    ]
    negative_literal_edges = [
        (u, v) for u, v, d in G.edges(data=True) if "label" in d and d["label"] == "-"
    ]
    and_to_or_edges = [(u, v) for u, v, d in G.edges(data=True) if "label" not in d]

    # Draw the edges with different colors
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=positive_literal_edges,
        edge_color="gray",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=negative_literal_edges,
        edge_color="red",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=and_to_or_edges,
        edge_color="black",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )

    # Draw the labels for nodes and edges
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple')

    # Set the title of the plot
    plt.title("AND-OR Graph of Learned DNF Formula")

    # Turn off the axis
    plt.axis("off")

    # Add a legend
    # Create custom legend handles
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="gray", label="Positive Literal Edge"),
        Patch(color="red", label="Negated Literal Edge"),
        # Patch(color='black', label='AND to OR Edge'),
        plt.scatter([], [], color="skyblue", label="Literals", s=50),
        plt.scatter([], [], color="lightgreen", marker="s", label="AND Gates", s=50),
        plt.scatter([], [], color="salmon", marker="o", label="OR Gate", s=50),
        # Patch(color='purple', label='Edge Label (+/-)')
    ]

    plt.legend(handles=legend_handles, scatterpoints=1)

    # Display the plot
    plt.show()


def get_combined_graph(bn_model, use_weak=False):
    Gs = []
    clause_offset = 0
    for k, dnf_model in bn_model.learned_dnf_cls_.items():
        # get_networkx needs the MatDNFClassifier instance to access learned_dnf and n_feature
        # We can pass the MatDNFClassifier instance stored in bn_model.learned_dnf_cls_
        G = get_networkx(
            dnf_model,
            clause_offset=clause_offset,
            label_or_post="_" + str(k),
            use_weak=use_weak,
        )
        Gs.append(G)
        clause_offset += len(dnf_model.learned_dnf)

    # Combine the graphs using disjoint_union
    combined_G = nx.compose_all(Gs)

    # Generate positions for the nodes in the combined graph
    # Trying a different layout algorithm
    # pos = nx.spectral_layout(combined_G)
    # pos = nx.spring_layout(combined_G, k=0.7)
    pos = nx.kamada_kawai_layout(combined_G)
    return combined_G, pos


def draw_combined_and_or(G, pos):
    plt.figure(figsize=(24, 16))  # Adjust figure size for potentially larger graph
    input_nodes = [
        n for n in G.nodes() if not (n.startswith("AND") or n.startswith("OR"))
    ]
    and_nodes = [n for n in G.nodes() if n.startswith("AND")]
    or_nodes = [
        n for n in G.nodes() if n.startswith("OR")
    ]  # Renamed 'or_node' to 'or_nodes'

    # Draw the nodes with different styles
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_size=70,
        node_color="skyblue",
        label="Literals",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=and_nodes,
        node_size=100,
        node_color="lightgreen",
        node_shape="s",
        label="AND Gates",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=or_nodes,
        node_size=150,
        node_color="salmon",
        node_shape="o",
        label="OR Gate",
    )

    # Separate edges based on their label (now stored in edge attributes)
    positive_literal_edges = [
        (u, v) for u, v, d in G.edges(data=True) if "label" in d and d["label"] == "+"
    ]
    negative_literal_edges = [
        (u, v) for u, v, d in G.edges(data=True) if "label" in d and d["label"] == "-"
    ]
    and_to_or_edges = [(u, v) for u, v, d in G.edges(data=True) if "label" not in d]

    # Draw the edges with different colors
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=positive_literal_edges,
        edge_color="blue",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=negative_literal_edges,
        edge_color="red",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=and_to_or_edges,
        edge_color="black",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
    )

    # Draw the labels for nodes and edges
    # Use original node names for labels
    node_labels = {n: n[1] if isinstance(n, tuple) else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight="bold")
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='purple', font_size=8)

    # Set the title of the plot
    plt.title("Combined AND-OR Graph of Learned DNF Formulas")

    # Turn off the axis
    # plt.axis("off")

    # Add a legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color="gray", label="Positive Literal Edge"),
        Patch(color="red", label="Negated Literal Edge"),
        Patch(color="black", label="AND to OR Edge"),
        plt.scatter([], [], color="skyblue", label="Literals", s=50),
        plt.scatter([], [], color="lightgreen", marker="s", label="AND Gates", s=50),
        plt.scatter([], [], color="salmon", marker="o", label="OR Gate", s=50),
        Patch(color="purple", label="Edge Label (+/-)"),
    ]

    plt.legend(handles=legend_handles, scatterpoints=1)

    # Display the plot
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt


def get_variable_dependency_graph(bn_model):
    G = nx.DiGraph()

    # Add nodes for input variables
    nodes = []
    for _, dnf_classifier in bn_model.learned_dnf_cls_.items():
        nodes.extend(dnf_classifier.feature_names)

    for n in set(nodes):
        G.add_node(n)

    # Add edges based on which variables appear in the learned DNF of other variables
    for _, dnf_classifier in bn_model.learned_dnf_cls_.items():
        learned_dnf = dnf_classifier.learned_dnf
        target_variable_node = dnf_classifier.target_name
        m = dnf_classifier.get_dependent_vars()
        # print(dnf_classifier.feature_names)
        # print(m)
        vars = np.array(dnf_classifier.feature_names)[m]
        for v in vars:
            G.add_edge(v, target_variable_node)

    return G


def draw_variable_dependency_graph(G):
    pos = nx.spring_layout(G, k=0.5)  # Adjust k for spacing

    nx.draw_networkx_nodes(G, pos, node_size=70, node_color="skyblue")
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), arrows=True, arrowstyle="-|>", arrowsize=10
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Variable Dependency Graph (based on DNF)")
    plt.axis("off")
    plt.show()
