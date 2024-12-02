from generate_graphs import load_graph
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random



def gini(values):
    values = sorted(values)
    n = len(values)
    cumulative_values = [sum(values[:i+1]) for i in range(n)]
    numerator = sum((i+1) * value for i, value in enumerate(cumulative_values))
    denominator = sum(values) * n
    return (2 * numerator / denominator) - (n + 1) / n


def compute_directed_stats(graph):
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # Count edges where weight > 1
    weighted_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get('weight', 0) > 1)
    print(f"Edges with weight > 1: {weighted_edges}")

    in_degrees = [graph.in_degree(n) for n in graph.nodes()]
    print(f"Average In-Degree: {sum(in_degrees) / len(in_degrees)}")
    print(f"Max In-Degree: {max(in_degrees)}")

    out_degrees = [graph.out_degree(n) for n in graph.nodes()]
    print(f"Average Out-Degree: {sum(out_degrees) / len(out_degrees)}")
    print(f"Max Out-Degree: {max(out_degrees)}")

    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    print(f"Average Edge Weight: {sum(edge_weights) / len(edge_weights)}")
    print(f"Max Edge Weight: {max(edge_weights)}")

    #assortativity = nx.degree_assortativity_coefficient(graph)
    #print(f"Degree Assortativity Coefficient: {assortativity}")

    #in_degree_gini = gini(in_degrees)
    #out_degree_gini = gini(out_degrees)

    #print(f"In-Degree Gini Coefficient: {in_degree_gini}")
    #print(f"Out-Degree Gini Coefficient: {out_degree_gini}")

    #node_weights = {node: sum(data['weight'] for _, _, data in graph.edges(node, data=True)) for node in graph.nodes()}
    #top_hosts = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    #print(f"Top 5 Nodes by Total Edge Weight: {top_hosts}")

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

def plot_degree_distribution(graph, title="Degree Distribution"):
    # Compute the degree distribution
    x, y = compute_degree_distribution(graph)
    
    # Plot the degree distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="", markersize=5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("data/london/host_host_degree_distribution.png", format="png", dpi=300)


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

    plot_degree_distribution(graph, title="Unweighted Degree Distribution")

    #degree_gini = gini(degrees)

    #print(f"Degree Gini Coefficient: {degree_gini}")

    #node_weights = {node: sum(data['weight'] for _, _, data in graph.edges(node, data=True)) for node in graph.nodes()}
    #top_hosts = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    #print(f"Top 5 Nodes by Total Edge Weight: {top_hosts}")


def compute_rich_club(graph):
    rich_club = nx.rich_club_coefficient(graph, normalized=False)
    print(f"Rich Club Coefficient: {rich_club}")



def visualize_graph_matplotlib(graph, title="Host-Host Graph"):
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)  # Use spring layout for a natural positioning of nodes

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="skyblue")

    # Draw edges
    #nx.draw_networkx_edges(graph, pos, alpha=0.5)

    plt.title(title)
    plt.axis("off")
    # Save the figure
    plt.savefig("data/london/host_host.png", format="png", dpi=300)
    print(f"Graph figure saved as data/london/host_host")



def sample_and_visualize(graph, sample_size, title="Sampled Host-Host Graph", output_file="data/london/host_host.html"):
    # Randomly sample nodes
    sampled_nodes = random.sample(list(graph.nodes()), min(sample_size, graph.number_of_nodes()))
    
    # Create a subgraph with the sampled nodes
    sampled_subgraph = graph.subgraph(sampled_nodes)
    
    # Visualize the sampled graph using Plotly
    visualize_graph_plotly(sampled_subgraph, title, output_file)

def visualize_graph_plotly(graph, title="Host-Host Graph", output_file="data/london/host_host.html"):
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
                        title=title,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                    ))
    
    # Save to HTML file
    fig.write_html(output_file)
    print(f"Graph visualization saved as {output_file}")



if __name__ == "__main__":
    graph = load_graph("london", "host_host")
    #compute_directed_stats(graph)
    compute_undirected_stats(graph)
    #sample_and_visualize(graph, 15000)
    #compute_undirected_stats(graph)
    """
    if nx.is_connected(graph):
        print("computing rich club")
        rich_club = weighted_rich_club(graph)
        print(rich_club)
    else:
        print("The graph is disconnected. Rich club coefficient may not be meaningful.")
    """

