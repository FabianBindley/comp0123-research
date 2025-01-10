import networkx as nx
import matplotlib.pyplot as plt
from generate_graphs import load_graph
import random



def edge_sampling_bounded(graph, num_edges, host_lower_bound, host_upper_bound, guest_lower_bound, guest_upper_bound):
    

    # Filter hosts  and guests based on degree bounds
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

    # Get all edges connected to the eligible hosts
    candidate_edges = []
    for edge in graph.edges():
        if edge[0] in eligible_guests and edge[1] in eligible_hosts:
            candidate_edges.append(edge)
        if len(candidate_edges) == num_edges:
            break
    
    # Sort edges by host ID to group edges by hosts
    candidate_edges = sorted(candidate_edges, key=lambda edge: edge[1])

    # Limit the number of edges to num_edges
    limited_edges = candidate_edges[:num_edges]
    #random.shuffle(limited_edges)


    # Create a new graph with the sampled edges
    sampled_graph = nx.DiGraph()  # Assuming the input graph is directed
    sampled_graph.add_edges_from(limited_edges)
    print("Generated Graph")
    return sampled_graph



def visualize_bipartite_graph(graph, num_edges):


    # Perform sampling based on degree
    sampled_graph = edge_sampling_bounded(graph, num_edges, 10, 300, 1, 25)

    
    # Separate the two sets (guests and hosts)
    hosts = [node for node in sampled_graph.nodes() if node.endswith("_h")]
    hosts_sorted = sorted(hosts, key=lambda h: sampled_graph.in_degree(h))


    guests = []
    for node in sampled_graph.nodes:
        if node.endswith("_g"):
            hosts = list(sampled_graph.successors(node))
            positions = list(map(lambda host: hosts_sorted.index(host), hosts))
            print(hosts)
            positions.sort(reverse=True)
            print(positions)
            guests.append((node, positions))

    guests_sorted = sorted(guests, key=lambda guest: guest[1], reverse=False)
    print(guests_sorted)
    guests = list(map(lambda guest: guest[0], guests_sorted))
    #guests = [node for node in sampled_graph.nodes() if node.endswith("_g")]


      # Scale vertical positions for better spacing
    max_y = max(len(guests), len(hosts_sorted))
    pos_guests = {guest: (0, i * (max_y / len(guests))) for i, guest in enumerate(guests)}
    pos_hosts = {host: (1, i * (max_y / len(hosts_sorted))) for i, host in enumerate(hosts_sorted)}


    # Combine positions
    pos = {**pos_guests, **pos_hosts}
    
    # Draw the graph
    plt.figure(figsize=(8, 14))
    plt.subplots_adjust(top=0.85)  # Adjust space for the title

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
    
    plt.suptitle(f"Bipartite Graph Visualization ({num_edges} Stays)", fontsize=16, y=0.97)


    plt.text(-0.5, max_y / 2, "Guests", fontsize=14, color="skyblue", ha="center", va="center", rotation=90)
    plt.text(1.5, max_y / 2, "Hosts", fontsize=14, color="red", ha="center", va="center", rotation=90)
    plt.text(0.01, max_y + 0.2, "Guest Nodes", fontsize=12, color="black", ha="center")
    plt.text(0.99, max_y + 0.2, "Host Nodes", fontsize=12, color="black", ha="center")

    
    save_path = f"data/{city}/graph_visualisation.png"
    # Save and show the plot
    plt.savefig(save_path, format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    cities= ["seattle","san-diego","san-francisco"]

    for city in cities:
        guest_host = load_graph(city, "guest_host", None)
        visualize_bipartite_graph(guest_host, 80)
