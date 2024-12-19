import networkx as nx
import matplotlib.pyplot as plt
from generate_graphs import load_graph
import random

def guests_with_degree_bounded(graph, num_edges, lower_bound, upper_bound):

    """
    # Get all edges and their weights
    edges = list(graph.edges(data=True))
    weights = [edge[2].get('weight', 1)* 10 for edge in edges]  # Default weight is 1 if not present

    # Sample edges with weights
    sampled_edges = random.choices(edges, weights=weights, k=min(num_edges, len(edges)))
    """
    # TODO rewrite with bounds to get the guests with bounds
    sampled_edges = []
    # Create a new graph with the sampled edges
    sampled_graph = nx.Graph()
    sampled_graph.add_edges_from([(u, v, data) for u, v, data in sampled_edges])

    return sampled_graph
    

def hosts_with_degree_bounded(graph, num_edges, lower_bound, upper_bound):

    # Step 1: Filter hosts based on in-degree bounds
    eligible_hosts = [
        node for node in graph.nodes()
        if node.endswith("_h") and lower_bound <= graph.in_degree(node) <= upper_bound
    ]

    print(len(eligible_hosts))

    # Step 2: Get all edges connected to the eligible hosts
    candidate_edges = [
        edge for edge in graph.edges()
        if edge[1] in eligible_hosts  # Only edges pointing to eligible hosts
    ]

    limited_edges = candidate_edges[:num_edges]


    # Step 4: Create a new graph with the sampled edges
    sampled_graph = nx.DiGraph()  # Assuming the input graph is directed
    sampled_graph.add_edges_from(limited_edges)

    return sampled_graph


def both_biased_edge_sampling(graph, num_edges, host_lower_bound, host_upper_bound, guest_lower_bound, guest_upper_bound):
    pass
"""
    # Step 1: Filter hosts  and guests based on degree bounds
    eligible_hosts = [
        node for node in graph.nodes()
        if node.endswith("_h") and host_lower_bound <= graph.in_degree(node) <= host_upper_bound
    ]

    eligible_guests = [
        node for node in graph.nodes()
        if node.endswith("_g") and guest_lower_bound <= graph.out_degree(node) <= guest_upper_bound
    ]

    print(len(eligible_hosts))
    print(len(eligible_guests))

    # Step 2: Get all edges connected to the eligible hosts
    candidate_edges = []
    for edge in graph.edges():
        print(edge)
        if edge[0] in eligible_guests and edge[1] in eligible_hosts:
            candidate_edges.append(edge)
        if len(candidate_edges) == num_edges:
            break
    
    limited_edges = candidate_edges
    #limited_edges = candidate_edges[:num_edges]


    # Step 4: Create a new graph with the sampled edges
    sampled_graph = nx.DiGraph()  # Assuming the input graph is directed
    sampled_graph.add_edges_from(limited_edges)

    return sampled_graph
"""


def visualize_bipartite_graph(graph, num_edges, bias):


    # Perform biased sampling
    if bias == "guest":
        sampled_graph = guest_biased_edge_sampling(graph, num_edges)
    elif bias == "both":
        sampled_graph = both_biased_edge_sampling(graph, num_edges, 7, 10, 2, 4)
    else:
        sampled_graph = host_biased_edge_sampling(graph, num_edges, 7, 10)
    
    # Separate the two sets (guests and hosts)
    guests = [node for node in sampled_graph.nodes() if node.endswith("_g")]
    hosts = [node for node in sampled_graph.nodes() if node.endswith("_h")]
    hosts_sorted = sorted(hosts, key=lambda h: sampled_graph.in_degree(h))

      # Scale vertical positions for better spacing
    max_y = max(len(guests), len(hosts_sorted))
    pos_guests = {guest: (0, i * (max_y / len(guests))) for i, guest in enumerate(guests)}
    pos_hosts = {host: (1, i * (max_y / len(hosts_sorted))) for i, host in enumerate(hosts_sorted)}


    # Combine positions
    pos = {**pos_guests, **pos_hosts}
    
    # Draw the graph
    plt.figure(figsize=(6, 12))
    nx.draw(
        sampled_graph,
        pos,
        with_labels=True,
        node_size=100,
        node_color=["skyblue" if node.endswith("_g") else "red" for node in sampled_graph.nodes()],
        edge_color="gray",
        font_size=7,
    )
    
    # Add titles for hosts and guests
    plt.text(-0.2, len(guests) / 2, "Guests", fontsize=12, color="skyblue", ha="center", va="center", rotation=90)
    plt.text(1.2, len(hosts_sorted) / 2, "Hosts", fontsize=12, color="red", ha="center", va="center", rotation=90)
    
    # Add a title
    plt.title(f"Bipartite Graph Visualization (Top {num_edges} Edges)", fontsize=14)
    
    save_path = f"data/{city}/graph_visualisation_{bias}_bias.png"
    # Save and show the plot
    plt.savefig(save_path, format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    cities= ["london","seattle","san-diego","san-francisco"]
    cities= ["seattle","san-diego","san-francisco"]
    cities= ["san-francisco"]
    for city in cities:
        guest_host = load_graph(city, "guest_host")
        visualize_bipartite_graph(guest_host, 100, "host")
