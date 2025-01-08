import pandas as pd
import networkx as nx
import pickle


def generate_guest_host(city):
    listings = pd.read_csv(f"data/{city}/listings.csv", usecols=['id', 'host_id', 'host_total_listings_count', 'host_is_superhost'])
    # Convert host_is_superhost to boolean
    listings['host_is_superhost'] = listings['host_is_superhost'] == "t"

    listings.rename(columns={'id': 'listing_id'}, inplace=True)

    reviews = pd.read_csv(f"data/{city}/reviews.csv", usecols=['listing_id', 'id', 'reviewer_id'])
    reviews.rename(columns={'id': 'review_id', 'reviewer_id':'guest_id'}, inplace=True)

    # Merge the listings and reviews DataFrames
    stays = pd.merge(reviews, listings, on='listing_id', how='inner')

    # Limit to 100 stays
    #stays = stays[:100]
    #stays = stays.sample(frac=1)
    #stays = stays[:20]

    # Iterate over the stays and generate the Guest - Host graph
    # Nodes have id: guest_id, host_id 
    guest_host = nx.DiGraph()
    count = 0
    for _, stay in stays.iterrows():
        guest = f"{str(int(stay['guest_id']))}_g"
        host  = f"{str(int(stay['host_id']))}_h"
        if guest_host.has_edge(guest, host):
            guest_host[guest][host]['weight'] += 1
        else:
            guest_host.add_edge(guest, host, weight=1)
        
        if 'host_is_superhost' not in guest_host.nodes[host]:
            guest_host.nodes[host]['host_is_superhost'] = stay['host_is_superhost']
            guest_host.nodes[host]['host_total_listings_count'] = stay['host_total_listings_count']

        guest_host.nodes[guest]['node_type'] = "guest"
        guest_host.nodes[host]['node_type'] = "host"

    # Save the graph output
    pickle.dump(guest_host, open(f"data/{city}/guest_host.pickle", "wb"))


def print_graph(graph):
    # Print the edges with their weights
    #print("Edges with weights:")
    
    #for u, v, data in graph.edges(data=True):
       #print(f"{u} -> {v}, weight: {data['weight']}")

    counter = 0
    for node in graph.nodes():
        if node[-1] == "h":
            if graph.nodes[node]['host_is_superhost'] is True:
                counter += 1

    print(f"Superhosts: {counter}")
    

def load_graph(city, type, network_type):
    if not network_type:
        return pickle.load(open(f"data/{city}/{type}.pickle", "rb"))
    return pickle.load(open(f"data/{city}/{network_type}/{type}.pickle", "rb"))

def export_graphml(city, type, graph):
    print("Exporting graph in graphml format")
    nx.write_graphml(guest_host, f"data/{city}/{type}.graphml")


if __name__ == "__main__":
    cities= ["london","seattle","san-diego","san-francisco"]
    cities= ["san-diego","san-francisco"]
    #cities= ["seattle","san-diego","san-francisco"]
    #cities= ["san-francisco"]
    for city in cities:
        print(city)
        generate_guest_host(city)
        guest_host = load_graph(city, "guest_host")
        print_graph(guest_host)
        export_graphml(city, "guest_host", guest_host)
