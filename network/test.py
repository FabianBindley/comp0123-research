import networkx as nx

def compute_bipartite_clustering(graph):
    """
    Computes the bipartite clustering coefficient for a bipartite graph.
    
    Parameters:
        graph (nx.Graph): A bipartite graph (nodes are split into two disjoint sets).
    
    Returns:
        float: The bipartite clustering coefficient.
    """
    # Ensure the graph is bipartite
    if not nx.is_bipartite(graph):
        raise ValueError("The provided graph is not bipartite.")
    
    # Initialize variables for counting squares and triples
    square_count = 0
    triple_count = 0
    
    for node in graph.nodes():
        # Neighbors of the current node
        neighbors = set(graph.neighbors(node))
        
        # For each pair of neighbors, check if they share a common neighbor
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 != neighbor2:
                    # Check if neighbor1 and neighbor2 share a common neighbor (a square is formed)
                    common_neighbors = set(graph.neighbors(neighbor1)) & set(graph.neighbors(neighbor2))
                    square_count += len(common_neighbors)  # Each common neighbor adds a square
        
        # Each neighbor pair counts as a "triple" for bipartite clustering
        triple_count += len(neighbors) * (len(neighbors) - 1)
    
    # Avoid division by zero
    if triple_count == 0:
        return 0.0
    
    print(f"square count: {square_count}")
    # Bipartite clustering coefficient is the ratio of squares to triples
    clustering_coefficient = square_count / triple_count
    return clustering_coefficient

# Create a bipartite graph
B = nx.Graph()
# Add nodes to two sets
B.add_nodes_from(["g1", "g2"], bipartite=0)  # Guests
B.add_nodes_from(["h1", "h2"], bipartite=1)  # Hosts
# Add edges (guest stays with host)
B.add_edges_from([("g1", "h1"), ("g1", "h2"), ("g2", "h1"), ("g2", "h2")])  # 4-cycle

# Compute the bipartite clustering coefficient
clustering_coefficient = compute_bipartite_clustering(B)
print(f"Bipartite Clustering Coefficient: {clustering_coefficient}")
