from generate_graphs import load_graph
import pandas as pd
import networkx as nx
import pickle
import os

def select_superguests(city, graph, num_superguests, percentile, min_degree):

    # Need to select some superguests according to some criteria
    # * Degree over a certain amount
    # * Top X Degree, 99th Percentile
    # * HITS Hubs

    guests = []

    for n in graph.nodes():
        if guest_host.nodes[n]['node_type'] == "guest":
            total_stays= sum(graph[n][neighbor]['weight'] for neighbor in graph.successors(n))
            out_degree =  graph.out_degree(n)
            guests.append((n, out_degree, total_stays))

    guests.sort(key=lambda guest: guest[1], reverse=True)
    
    if num_superguests:
        return guests[:num_superguests]
    if percentile:
        divisor = 100 / (100-percentile)
        return guests[:int(len(guests)//divisor)]

    if min_degree:
        return list(filter(lambda x: x[1] >= min_degree, guests))

def generate_superguest_network(city, superguests, graph):
    superguest_ids = set([guest[0] for guest in superguests])

    guest_host = nx.DiGraph()
    for superguest_id in superguest_ids:

        if not guest_host.has_node(superguest_id):
            guest_host.add_node(superguest_id, **graph.nodes[superguest_id])

        edges = graph.out_edges(superguest_id, data=True) 
        for edge in edges:
            source, target, attributes = edge
            if not guest_host.has_node(target):
                guest_host.add_node(target, **graph.nodes[target])

            guest_host.add_edge(source, target, **attributes)
    

    num_guests_new = len([node for node in guest_host.nodes() if guest_host.nodes[node]['node_type']=="guest"])
    num_hosts_new = len([node for node in guest_host.nodes() if guest_host.nodes[node]['node_type']=="host"])

    num_guests_old = len([node for node in graph.nodes() if graph.nodes[node]['node_type']=="guest"])
    num_hosts_old = len([node for node in graph.nodes() if graph.nodes[node]['node_type']=="host"])

    print(f"Generated superguest network for {city} with {num_guests_new}/{num_guests_old} guests, {num_hosts_new}/{num_hosts_old} hosts and {guest_host.number_of_edges()}/{graph.number_of_edges()} edges.")
    # Save the graph output
    os.makedirs(f"data/{city}/superguest/", exist_ok=True)  # Create the directory if it doesn't exist
    pickle.dump(guest_host, open(f"data/{city}/superguest/guest_host.pickle", "wb"))

if __name__ == "__main__":
    cities= ["san-francisco","san-diego","seattle"]

    for city in cities:
        network_type = None
        guest_host = load_graph(city, "guest_host", network_type)
        print(city)
        #superguests = select_superguests(city, guest_host, num_superguests=200, percentile=None, min_degree=None)
        #superguests = select_superguests(city, guest_host, num_superguests=None, percentile=None, min_degree=20)

        superguests = select_superguests(city, guest_host, num_superguests=None, percentile=99.9, min_degree=None)
        superguest_network = generate_superguest_network(city, superguests, guest_host)
