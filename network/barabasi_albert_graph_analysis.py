import pandas as pd
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import random


def host_degrees_on_date(graph, host_degrees_on_date, date, alpha, the_host):

    if date in host_degrees_on_date:
        return host_degrees_on_date[date]
    
    hosts = [host for host in graph.nodes() if graph.nodes[host]['node_type'] == "host"]

    hosts_available_on_date =  [host for host in hosts if graph.nodes[host]['host_since'] < date]


    k_hosts_on_date = [graph.in_degree(host)+alpha for host in hosts_available_on_date]

    host_degrees_on_date[date] = k_hosts_on_date
    return k_hosts_on_date



def generate_random_temporal_ba_guest_host_network(start_num_hosts, start_num_stays, stays, listings):

    initial_network = nx.DiGraph()

    initial_hosts = list()  # Use a set to track unique host IDs
    index = 0
    host_index = 0

    # Set host_since for all of the hosts to be the same as the first
    start_host_since = listings.iloc[0]['host_since']

    while len(initial_hosts) < start_num_hosts and index < len(listings):
        listing = listings.iloc[index]
        host_id = listing['host_id']
        
        # Check if host_id is not already in the initial_hosts set
        if host_id not in initial_hosts:
            initial_hosts.append(host_id)  # Add to the unique host set
            initial_network.add_node(f"{host_id}_h", node_type="host", host_since=start_host_since)
            host_index += 1
        index += 1 

    # Add remaining hosts to the graph (without edges)
    
    for _, listing in listings.iterrows():
        host_id = listing['host_id']
        if host_id not in initial_hosts:
            initial_network.add_node(f"{host_id}_h", node_type="host", host_since=listing['host_since'])

    # Add the first start_num_stays from stays, with random hosts but guaranteeing each host has at least 1 guest
    host_index = 0
    for index, stay in stays.iterrows():
        if len(initial_network.edges()) >= start_num_stays:
            break  
        
        guest_id = f"{stay['guest_id']}_g"

        if host_index < start_num_hosts:
            # Create the guest node and edge to the selected host
            if not(initial_network.has_node(guest_id)):
                initial_network.add_node(guest_id, node_type="guest")

            initial_network.add_edge(guest_id, f"{initial_hosts[host_index]}_h")  
            host_index += 1
        else:
            if not(initial_network.has_node(guest_id)):
                initial_network.add_node(guest_id, node_type="guest")

            # Pick a random host from the initial hosts
            random_host = random.choice(list(initial_hosts))
            host_id = f"{random_host}_h"
        
            initial_network.add_edge(guest_id, host_id)  # Use random_host here, not host_index

    return initial_network


def host_degrees_on_date(graph, degrees_on_date, date, alpha):

    if date in degrees_on_date:
        return degrees_on_date[date]
    
    hosts = [host for host in graph.nodes() if graph.nodes[host]['node_type'] == "host"]

    hosts_available_on_date =  [host for host in hosts if graph.nodes[host]['host_since'] < date]


    k_hosts_on_date = [(host,graph.in_degree(host)+alpha) for host in hosts_available_on_date]

    degrees_on_date[date] = k_hosts_on_date
    return k_hosts_on_date
    

def add_guests_temporal_ba_network(graph, stays, degrees_on_date, alpha):

    stay_probabilities = []
    count = 0
    for _, stay in stays.iterrows():
        guest_id = f"{stay['guest_id']}_g"
        stay_date = stay['date']

        if not graph.has_node(guest_id):
            graph.add_node(guest_id, node_type="guest")

        k_hosts_on_date = host_degrees_on_date(graph, degrees_on_date, stay_date, alpha)
        #print(k_hosts_on_date)

        # Compute the total attractiveness and probabilities of all hosts
        total_attractiveness = sum([k+alpha for _, k in k_hosts_on_date])

        hosts, probabilities = zip(*[
            (host, k / total_attractiveness) for host, k in k_hosts_on_date
        ])

        # Randomly select a host based on normalized probabilities
        selected_host_id = random.choices(hosts, weights=probabilities, k=1)[0]

        # Add edge between guest and selected host
        graph.add_edge(guest_id, selected_host_id)

        # Record probabilities for analysis
        P_deg = (graph.in_degree(selected_host_id) + alpha) / total_attractiveness
        P_rand = 1 / len(hosts)  # Probability under random selection
        stay_probabilities.append([P_deg, P_rand, stay_date])


        if count % 10000 == 0:
            print(count)
        count += 1

    return graph, stay_probabilities




def generate_guest_host_barabasi_albert(city, alpha, load_degrees_by_date):
    listings = pd.read_csv(f"data/{city}/listings.csv", usecols=['id', 'host_id', 'calculated_host_listings_count', 'host_is_superhost', 'host_since'])
    # Convert host_is_superhost to boolean
    listings['host_is_superhost'] = listings['host_is_superhost'] == "t"

    listings.rename(columns={'id': 'listing_id', 'calculated_host_listings_count': 'host_listings_count'}, inplace=True)

    reviews = pd.read_csv(f"data/{city}/reviews.csv", usecols=['listing_id', 'id', 'reviewer_id', 'date'])
    reviews.rename(columns={'id': 'review_id', 'reviewer_id':'guest_id'}, inplace=True)

    # Merge the listings and reviews DataFrames
    stays = pd.merge(reviews, listings, on='listing_id', how='inner')

    # Sort by date
    stays = stays.sample(frac=1)
    stays.sort_values(by="date", inplace=True)
    #stays=stays[:50000]
    listings.sort_values(by="host_since", inplace=True)

    # Create the barabasi_albert graph, with all the hosts, and set the date of the first start_num_stays stays to be the host_since 
    # of the latest of the start_num_hosts to join, so that the guests may randomly chose to join any of them
    start_num_hosts = 20
    start_num_stays = 40

    temporal_ba_guest_host_network = generate_random_temporal_ba_guest_host_network(start_num_hosts, start_num_stays, stays, listings)
    #visualize_bipartite_graph(temporal_ba_guest_host_network, 50)

    # Now we need to add the guests to the BA network according to the stay. We also drop all columns referencing specific hosts
    stays_reduced = stays[['guest_id', 'date']][20:]

    if load_degrees_by_date:
        try:
            with open(f"data/{city}/barabasi_albert/guest_host_barabasi_albert_degrees.pickle", "rb") as f:
                degrees_on_date = pickle.load(f)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
    else:
        degrees_on_date = {}

    temporal_ba_guest_host_network, stay_probabilities = add_guests_temporal_ba_network(
        temporal_ba_guest_host_network, 
        stays_reduced, 
        alpha=alpha,
        degrees_on_date=degrees_on_date
    )

     # Convert probabilities to DataFrame
    stay_probabilities_df = pd.DataFrame(stay_probabilities, columns=["P_deg", "P_rand", "date"])
    
    output_dir = f"data/{city}/barabasi_albert"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the graph output
    pickle.dump(temporal_ba_guest_host_network, open(f"data/{city}/barabasi_albert/guest_host_barabasi_albert.pickle", "wb"))
    pickle.dump(degrees_on_date, open(f"data/{city}/barabasi_albert/guest_host_barabasi_albert_degrees.pickle", "wb"))
    pickle.dump(stay_probabilities_df, open(f"data/{city}/barabasi_albert/guest_host_barabasi_albert_stay_probabilities.pickle", "wb"))
    
    return temporal_ba_guest_host_network, stay_probabilities_df


def edge_sampling_bounded(graph, num_edges, host_lower_bound, host_upper_bound, guest_lower_bound, guest_upper_bound):
    

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

    candidate_edges = []
    for edge in graph.edges():
        if edge[0] in eligible_guests and edge[1] in eligible_hosts:
            candidate_edges.append(edge)
        if len(candidate_edges) == num_edges:
            break
    
    limited_edges = candidate_edges

    sampled_graph = nx.DiGraph() 
    sampled_graph.add_edges_from(limited_edges)
    print("Generated Graph")
    return sampled_graph


def visualize_bipartite_graph(graph, num_edges):


    # Perform sampling based on degree
    sampled_graph = edge_sampling_bounded(graph, num_edges, 1, 100, 1, 100)

    
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
    
    save_path = f"data/{city}/graph_visualisation.png"
    # Save and show the plot
    plt.savefig(save_path, format="png", dpi=300)
    plt.show()


def analyse_stay_probabilities(city, stay_probabilities):

    # Plot histograms for P_deg and P_rand with logarithmic y-axis
    plt.figure(figsize=(12, 6))
    plt.hist(stay_probabilities["P_deg"], bins=100, alpha=0.7, label="P_deg (Degree-Based)", color="blue")
    plt.hist(stay_probabilities["P_rand"], bins=100, alpha=0.7, label="P_rand (Random)", color="orange")
    plt.xlabel("Probability")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f"BA Model Host Selection Probability Distributions in {city}")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)  # Major and minor grid lines
    plt.savefig(f"data/{city}/barabasi_albert/probability_distributions_log.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/barabasi_albert/probability_distributions_log.png")

    # Plot the probabilities for each stay against time, to see how the
    stay_probabilities["date"] = pd.to_datetime(stay_probabilities["date"])
    barabasi_albert_means = stay_probabilities.groupby("date")[["P_deg", "P_rand"]].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(barabasi_albert_means.index, barabasi_albert_means["P_deg"], label="P_deg (Degree-Based)", color="blue")
    plt.plot(barabasi_albert_means.index, barabasi_albert_means["P_rand"], label="P_rand (Random)", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Average Probability")
    plt.title(f"BA Model Temporal Trends in Stay Probabilities - {city}")
    plt.legend()
    plt.grid()
    plt.savefig(f"data/{city}/barabasi_albert/trends_probability_distributions.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/barabasi_albert/trends_probability_distributions.png")

    # Calculate barabasi_albert means for probabilities less than a value X
    X = 0.01
    low_probabilities = stay_probabilities[(stay_probabilities["P_deg"] < X) & (stay_probabilities["P_rand"] < X)]
    low_prob_means = low_probabilities.groupby("date")[["P_deg", "P_rand"]].mean()

    # Plot BA temporal trends for probabilities below X
    plt.figure(figsize=(12, 6))
    plt.plot(low_prob_means.index, low_prob_means["P_deg"], label=f"P_deg < {X}", color="blue")
    plt.plot(low_prob_means.index, low_prob_means["P_rand"], label=f"P_rand < {X}", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Average Probability")
    plt.title(f"BA Model Temporal Trends in Stay Probabilities (P < {X}) - {city}")
    plt.legend()
    plt.grid()
    plt.savefig(f"data/{city}/barabasi_albert/trends_low_probabilities.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/barabasi_albert/trends_low_probabilities.png")


    # Ensure 'date' is datetime and compute timestamps
    stay_probabilities["date"] = pd.to_datetime(stay_probabilities["date"])
    timestamps = stay_probabilities["date"].apply(lambda x: x.timestamp())

    # Filter out the point with the maximum P_rand
    stay_probabilities_filtered = stay_probabilities[stay_probabilities["P_rand"] != stay_probabilities["P_rand"].max()]

    # Normalize timestamps for color mapping
    norm = plt.Normalize(vmin=timestamps.min(), vmax=timestamps.max())  
    colors = stay_probabilities_filtered["date"].apply(lambda x: x.timestamp()) 

    # Plot scatterplot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        stay_probabilities_filtered["P_rand"],
        stay_probabilities_filtered["P_deg"],
        c=colors, 
        cmap="plasma",  
        norm=norm,  
        alpha=0.6,
        label="Stays",
    )

    # Add diagonal line (x = y)
    max_value = min(stay_probabilities_filtered["P_deg"].max(), stay_probabilities_filtered["P_rand"].max())
    plt.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1, label="x = y")


    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca(), orientation="vertical", pad=0.02)
    cbar.set_label("Stay Date", rotation=270, labelpad=15)
    tick_locs = np.linspace(timestamps.min(), timestamps.max(), num=6)
    tick_labels = pd.to_datetime(tick_locs, unit="s").strftime("%Y-%m")
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)

    # Labels, title, and grid
    plt.xlabel("P_rand (Random Probability)")
    plt.ylabel("P_deg (Degree-Based Probability)")
    plt.title(f"BA Model P_deg vs P_rand in {city}")
    plt.legend(loc="center right") 
    plt.grid()

    # Save and show plot
    plt.savefig(f"data/{city}/barabasi_albert/pdeg_vs_prand_color_by_date.png", format="png", dpi=300)
    #plt.show()
    print("Saved: data/{city}/barabasi_albert/pdeg_vs_prand_color_by_date.png")



if __name__ == "__main__":

    cities= ["san-francisco"]
    cities= ["san-diego", "seattle", "san-francisco"]
    compute_stay_probabilities = True
    for city in cities:
        if compute_stay_probabilities:
            _, stay_probabilities = generate_guest_host_barabasi_albert(city, alpha=1, load_degrees_by_date=False)
        else:
            stay_probabilities = pickle.load(open(f"data/{city}/barabasi_albert/guest_host_barabasi_albert_stay_probabilities.pickle", "rb"))

    analyse_stay_probabilities(city, stay_probabilities)