import pandas as pd
import networkx as nx
import pickle


def generate_guest_host(city):
    listings = pd.read_csv(f"data/{city}/listings.csv", usecols=['id', 'host_id', 'host_total_listings_count'])
    listings.rename(columns={'id': 'listing_id'}, inplace=True)

    reviews = pd.read_csv(f"data/{city}/reviews.csv", usecols=['listing_id', 'id', 'reviewer_id'])
    reviews.rename(columns={'id': 'review_id', 'reviewer_id':'guest_id'}, inplace=True)

    # Merge the listings and reviews DataFrames
    stays = pd.merge(reviews, listings, on='listing_id', how='inner')

    # Limit to 100 stays
    #stays = stays[:1000000]

    # Iterate over the stays and generate the Guest - Host graph
    # Nodes have id: guest_id, host_id 
    guest_host = nx.DiGraph()

    for _, stay in stays.iterrows():
        guest = int(stay["guest_id"])
        host = int(stay["host_id"])
        if guest_host.has_edge(guest, host):
            guest_host[guest][host]['weight'] += 1
        else:
            guest_host.add_edge(guest, host, weight=1)

    
    # Save the graph output
    pickle.dump(guest_host, open(f"data/{city}/guest_host.pickle", "wb"))


def generate_host_host(city):
    
    host_host = nx.Graph()
    guest_host = load_graph(city, "guest_host")

    for guest in guest_host.nodes():

        if guest_host.out_degree(guest) >= 2:
            hosts = [neighbour for neighbour in guest_host.successors(guest)]

            # Add edges between all pairs of hosts
            for i in range(len(hosts)):
                for j in range(i + 1, len(hosts)):
                    host1, host2 = hosts[i], hosts[j]
                    if host_host.has_edge(host1, host2):
                        host_host[host1][host2]['weight'] += 1
                    else:
                        host_host.add_edge(host1, host2, weight=1)
    
    largest_cc = max(nx.connected_components(host_host), key=len)
    host_host_largest = host_host.subgraph(largest_cc).copy()

    # Save the graph output
    pickle.dump(host_host_largest, open(f"data/{city}/host_host.pickle", "wb"))


def print_graph(graph):
    # Print the edges with their weights
    print("Edges with weights:")

    for u, v, data in graph.edges(data=True):
        print(f"{u} -> {v}, weight: {data['weight']}")

def load_graph(city, type):
    return pickle.load(open(f"data/{city}/{type}.pickle", "rb"))


if __name__ == "__main__":
    city = "seattle"
    generate_guest_host(city)
    guest_host = load_graph(city, "guest_host")
    print_graph(guest_host)

    generate_host_host(city)
    host_host = load_graph(city, "host_host")
    print("saved host-host graph")
    #print_graph(host_host)