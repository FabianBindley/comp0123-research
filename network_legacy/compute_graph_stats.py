from network.generate_graphs import load_graph
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random



def gini(values):
    if not values or sum(values) == 0:
        return 0.0  # Handle empty list or all-zero values

    # Sort the values in ascending order
    values = sorted(values)
    n = len(values)
    cumulative_values = [sum(values[:i + 1]) for i in range(n)]

    # Numerator and denominator
    numerator = sum((i + 1) * value for i, value in enumerate(cumulative_values))
    denominator = sum(values) * n

    # Gini coefficient
    return (2 * numerator / denominator) - (n + 1) / n




def compute_directed_stats(graph):
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # Count edges where weight > 1
    weighted_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('weight', 0) > 1)
    print(f"Edges with weight > 1: {weighted_edges}")


    # Separate in-degree and out-degree analysis
    host_in_degrees = [graph.in_degree(n) for n in graph.nodes() if graph.in_degree(n) > 0]
    guest_out_degrees = [graph.out_degree(n) for n in graph.nodes() if graph.out_degree(n) > 0]

    # Compute statistics for hosts
    avg_host_in_degree = sum(host_in_degrees) / len(host_in_degrees) if host_in_degrees else 0
    max_host_in_degree = max(host_in_degrees) if host_in_degrees else 0

    # Compute statistics for guests
    avg_guest_out_degree = sum(guest_out_degrees) / len(guest_out_degrees) if guest_out_degrees else 0
    max_guest_out_degree = max(guest_out_degrees) if guest_out_degrees else 0

    # Print the results
    print(f"Number of Hosts: {len(host_in_degrees)}")
    print(f"Average Host In-Degree: {avg_host_in_degree}")
    print(f"Max Host In-Degree: {max_host_in_degree}")

    print(f"Number of Guests: {len(guest_out_degrees)}")
    print(f"Average Guest Out-Degree: {avg_guest_out_degree}")
    print(f"Max Guest Out-Degree: {max_guest_out_degree}")


    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    print(f"Average Edge Weight: {sum(edge_weights) / len(edge_weights)}")
    print(f"Max Edge Weight: {max(edge_weights)}")

    #assortativity = nx.degree_assortativity_coefficient(graph)
    #print(f"Degree Assortativity Coefficient: {assortativity}")

    in_degree_gini = gini(host_in_degrees)
    #out_degree_gini = gini(out_degrees)

    print(f"Host In-Degree Gini Coefficient: {in_degree_gini}")
    #print(f"Out-Degree Gini Coefficient: {out_degree_gini}")


def compute_degree_distribution(graph):
    # Get the degree of each node
    degrees = [graph.degree(n) for n in graph.nodes()]
    
    # Count the frequency of each degree
    degree_counts = {}
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    
    # Sort the degree frequencies
    sorted_degrees = sorted(degree_counts.items())
    x, y = zip(*sorted_degrees)
    
    return x, y

def plot_degree_distribution(graph, city, type):
    # Compute the degree distribution
    x, y = compute_degree_distribution(graph)
    
    # Plot the degree distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="", markersize=5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"{city} {type}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"data/{city}/{type}_degree_distribution.png", format="png", dpi=300)


def compute_undirected_stats(graph):
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # Count edges where weight > 1
    weighted_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('weight', 0) > 1)
    print(f"Edges with weight > 1: {weighted_edges}")

    degrees = [graph.degree(n) for n in graph.nodes()]
    print(f"Average Degree: {sum(degrees) / len(degrees)}")
    print(f"Max Degree: {max(degrees)}")


    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    print(f"Average Edge Weight: {sum(edge_weights) / len(edge_weights)}")
    print(f"Max Edge Weight: {max(edge_weights)}")

    assortativity = nx.degree_assortativity_coefficient(graph)
    print(f"Degree Assortativity Coefficient: {assortativity}")

    clustering = nx.clustering(graph)
    print(f"Average Clustering Coefficient: {sum(clustering.values()) / len(clustering)}")

    plot_degree_distribution(graph, city, type)

    #rich_club_coefficient = weighted_rich_club_coefficient(graph)
    #print(f"Weighted Rich Club Coefficient: {rich_club_coefficient}")

    #degree_gini = gini(degrees)

    #print(f"Degree Gini Coefficient: {degree_gini}")

    #node_weights = {node: sum(data['weight'] for _, _, data in graph.edges(node, data=True)) for node in graph.nodes()}
    #top_hosts = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    #print(f"Top 5 Nodes by Total Edge Weight: {top_hosts}")


def compute_rich_club(graph):
    rich_club = nx.rich_club_coefficient(graph, normalized=False)
    print(f"Rich Club Coefficient: {rich_club}")

def weighted_rich_club_coefficient(G, normalized=True):
    """
    Compute the weighted rich-club coefficient for a weighted graph G.

    Parameters:
    G : NetworkX graph
        A graph with edge weights.
    normalized : bool
        If True, normalize the coefficient by comparing it to a randomized version of the graph.

    Returns:
    dict
        Weighted rich-club coefficient for each degree k.
    """
    # Compute the weighted degrees
    weighted_degrees = dict(G.degree(weight='weight'))
    
    # Sort nodes by weighted degree
    nodes_by_degree = sorted(weighted_degrees, key=weighted_degrees.get, reverse=True)

    # Initialize result dictionary
    rich_club_coeff = {}

    for k in nodes_by_degree:
        # Filter nodes with degree > k
        high_degree_nodes = [n for n, deg in weighted_degrees.items() if deg > k]
        
        subgraph = G.edge_subgraph((u, v) for u, v, data in G.edges(data=True) if u in high_degree_nodes and v in high_degree_nodes)


        # Compute total weight of edges in the subgraph
        total_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
        
        # Maximum possible weight (complete graph)
        n = len(high_degree_nodes)
        max_weight = n * (n - 1) / 2  # Complete graph

        # Compute coefficient
        if max_weight > 0:
            coeff = total_weight / max_weight
        else:
            coeff = 0

        rich_club_coeff[k] = coeff

    # Normalize if needed
    if normalized:
        randomized_graph = nx.configuration_model([d for _, d in weighted_degrees.items()])
        rich_club_random = weighted_rich_club_coefficient(randomized_graph, normalized=False)
        rich_club_coeff = {k: rc / rich_club_random.get(k, 1) for k, rc in rich_club_coeff.items()}

    return rich_club_coeff




def visualize_graph_matplotlib(graph, city, type):
    # Draw the graph
    filename = f"data/{city}/{type}.png"
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)  # Use spring layout for a natural positioning of nodes

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="skyblue")

    # Draw edges
    #nx.draw_networkx_edges(graph, pos, alpha=0.5)

    plt.title(f"{type} degree distribution")
    plt.axis("off")
    # Save the figure
    plt.savefig(filename, format="png", dpi=300)
    print(f"Graph figure saved as {filename}")



def sample_and_visualize(graph, city, sample_size, type):
    # Randomly sample nodes
    print(f"Generating chart for {city} {type}")
    sampled_nodes = random.sample(list(graph.nodes()), min(sample_size, graph.number_of_nodes()))
    
    # Create a subgraph with the sampled nodes
    sampled_subgraph = graph.subgraph(sampled_nodes)
    
    # Visualize the sampled graph using Plotly
    visualize_graph_plotly(sampled_subgraph, city, type)

def visualize_graph_plotly(graph, city, type):
    # Compute the positions using spring layout
    pos = nx.spring_layout(graph, seed=42)

    # Extract edge positions
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Separate edges
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Extract node positions and degrees
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]
    fixed_node_size = 5

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=fixed_node_size,
            color="blue",
            line_width=2
        ),
        text=[f"Node {node}<br>Degree: {graph.degree(node)}" for node in graph.nodes()],
        hoverinfo="text"
    )

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"{type} Graph",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                    ))
    
    # Save to HTML file
    fig.write_html(f"data/{city}/{type}.html")
    print(f"Graph visualization saved as data/{city}/{type}.html")



if __name__ == "__main__":
    city = "seattle"
    type = "host_host"
    graph = load_graph(city, type)
    if type == "guest_host":
        compute_directed_stats(graph)
    else:
        compute_undirected_stats(graph)
    #sample_and_visualize(graph, city, 15000, type)
    #compute_undirected_stats(graph)
    """
    if nx.is_connected(graph):
        print("computing rich club")
        rich_club = weighted_rich_club(graph)
        print(rich_club)
    else:
        print("The graph is disconnected. Rich club coefficient may not be meaningful.")
    """

