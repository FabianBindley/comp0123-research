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

def host_degrees_on_date(graph, host_degrees_on_date, date, alpha):

    if date in host_degrees_on_date:
        return host_degrees_on_date[date]
    
    hosts = [host for host in graph.nodes() if graph.nodes[host]['node_type'] == "host"]

    hosts_available_on_date =  [host for host in hosts if graph.nodes[host]['host_since'] < date]


    k_hosts_on_date = [graph.in_degree(host)+alpha for host in hosts_available_on_date]

    host_degrees_on_date[date] = k_hosts_on_date
    return k_hosts_on_date


def generate_guest_host_temporal(city, alpha, load_degrees_by_date):
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

    # Generate the temporal graph
    guest_host_temporal = nx.DiGraph()

    # Add all the hosts and their details to the temporal graph
    for _, listing in listings.iterrows():
        host  = f"{str(int(listing['host_id']))}_h"
    
        if not(guest_host_temporal.has_node(host)):
            if not(isinstance(listing['host_since'], str)):
                continue
            guest_host_temporal.add_node(host, node_type="host", host_is_superhost=listing['host_is_superhost'], 
                                         host_listings_count=listing['host_listings_count'], host_since=datetime.strptime(listing['host_since'], "%Y-%m-%d") )

    if load_degrees_by_date:
        try:
            with open(f"data/{city}/temporal/guest_host_temporal_degrees.pickle", "rb") as f:
                degrees_on_date = pickle.load(f)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
    else:
        degrees_on_date = {}

    stay_probabilities = []
    counter = 0
    # Iterate over all reviews representing stays
    for _, stay in stays.iterrows():

        if counter % 50000 == 0:
            print(f"Stay: {counter}")
        # Get the guest and host IDs
        guest = f"{str(int(stay['guest_id']))}_g"
        host  = f"{str(int(stay['host_id']))}_h"

        stay_date = datetime.strptime(stay['date'], "%Y-%m-%d")

        # Get the host degree and the total degree of all hosts at the time
        k_host = guest_host_temporal.in_degree(host)
        k_hosts_on_date= host_degrees_on_date(guest_host_temporal, degrees_on_date, stay_date, alpha, host)
        total_attractiveness = sum(k_hosts_on_date)

        # Compute the probability P_deg(i) that a guest stays with the given host according to host degree
        P_deg = (k_host + alpha) / total_attractiveness

        if P_deg > 1:
            print(f"{k_host + alpha} {sum(k_hosts_on_date)} {P_deg}")

        #print(P_deg)
        # Compute the probability P_rand(i) that a guest stays with the given host according to random chance
        P_rand = 1 / len(k_hosts_on_date)

        stay_probabilities.append([P_deg, P_rand, stay_date])

        # All hosts have a base attractiveness of alpha, to which the degree is added \cite{Structure of Growing Networks with Preferential Linking}

        if guest_host_temporal.has_edge(guest, host):
            guest_host_temporal[guest][host]['weight'] += 1
        else:
            guest_host_temporal.add_edge(guest, host, weight=1)

        guest_host_temporal.nodes[guest]['node_type'] = "guest"

        counter += 1

     # Convert probabilities to DataFrame
    stay_probabilities_df = pd.DataFrame(stay_probabilities, columns=["P_deg", "P_rand", "date"])

    output_dir = f"data/{city}/temporal"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the graph output
    pickle.dump(guest_host_temporal, open(f"data/{city}/temporal/guest_host_temporal.pickle", "wb"))
    pickle.dump(degrees_on_date, open(f"data/{city}/temporal/guest_host_temporal_degrees.pickle", "wb"))
    pickle.dump(stay_probabilities_df, open(f"data/{city}/temporal/guest_host_temporal_stay_probabilities.pickle", "wb"))

    return guest_host_temporal, stay_probabilities_df



def analyse_stay_probabilities(city, stay_probabilities):

    # Plot histograms for P_deg and P_rand with logarithmic y-axis
    plt.figure(figsize=(12, 6))
    plt.hist(stay_probabilities["P_deg"], bins=100, alpha=0.7, label="P_deg (Degree-Based)", color="blue")
    plt.hist(stay_probabilities["P_rand"], bins=100, alpha=0.7, label="P_rand (Random)", color="orange")
    plt.xlabel("Probability")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f"Host Selection Probability Distributions in {city}")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)  # Major and minor grid lines
    plt.savefig(f"data/{city}/temporal/probability_distributions_log.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/temporal/probability_distributions_log.png")

    # Plot the probabilities for each stay against time, to see how the
    stay_probabilities["date"] = pd.to_datetime(stay_probabilities["date"])
    temporal_means = stay_probabilities.groupby("date")[["P_deg", "P_rand"]].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(temporal_means.index, temporal_means["P_deg"], label="P_deg (Degree-Based)", color="blue")
    plt.plot(temporal_means.index, temporal_means["P_rand"], label="P_rand (Random)", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Average Probability")
    plt.title(f"Temporal Trends in Stay Probabilities - {city}")
    plt.legend()
    plt.grid()
    plt.savefig(f"data/{city}/temporal/trends_probability_distributions.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/temporal/trends_probability_distributions.png")

    # Calculate temporal means for probabilities less than a value X
    X = 0.01
    low_probabilities = stay_probabilities[(stay_probabilities["P_deg"] < X) & (stay_probabilities["P_rand"] < X)]
    low_prob_means = low_probabilities.groupby("date")[["P_deg", "P_rand"]].mean()

    # Plot temporal trends for probabilities below X
    plt.figure(figsize=(12, 6))
    plt.plot(low_prob_means.index, low_prob_means["P_deg"], label=f"P_deg < {X}", color="blue")
    plt.plot(low_prob_means.index, low_prob_means["P_rand"], label=f"P_rand < {X}", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Average Probability")
    plt.title(f"Temporal Trends in Stay Probabilities (P < {X}) - {city}")
    plt.legend()
    plt.grid()
    plt.savefig(f"data/{city}/temporal/trends_low_probabilities.png", format="png", dpi=300)
    #plt.show()
    plt.close()
    print("Saved: data/{city}/temporal/trends_low_probabilities.png")


    # Ensure 'date' is datetime and compute timestamps
    stay_probabilities["date"] = pd.to_datetime(stay_probabilities["date"])
    timestamps = stay_probabilities["date"].apply(lambda x: x.timestamp())

    # Filter out the point with the maximum P_rand
    stay_probabilities_filtered = stay_probabilities[stay_probabilities["P_rand"] != stay_probabilities["P_rand"].max()]

    # Normalize timestamps for color mapping
    norm = plt.Normalize(vmin=timestamps.min(), vmax=timestamps.max())  # Explicitly set vmin and vmax
    colors = stay_probabilities_filtered["date"].apply(lambda x: x.timestamp())  # Use filtered timestamps directly for colormapping

    # Plot scatterplot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        stay_probabilities_filtered["P_rand"],
        stay_probabilities_filtered["P_deg"],
        c=colors,  # Pass filtered timestamps directly as c
        cmap="plasma",  # Explicitly specify the colormap
        norm=norm,      # Match normalization to c
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
    plt.title(f"P_deg vs P_rand in {city}")
    plt.legend(loc="center right") 
    plt.grid()

    # Save and show plot
    plt.savefig(f"data/{city}/temporal/pdeg_vs_prand_color_by_date.png", format="png", dpi=300)
    #plt.show()
    print("Saved: data/{city}/temporal/pdeg_vs_prand_color_by_date.png")


def print_graph(graph):
    # Print the edges with their weights
    #print("Edges with weights:")
    
    #for u, v, data in graph.edges(data=True):
       #print(f"{u} -> {v}, weight: {data['weight']}")

    counter = 0
    max_listings_count = 0
    max_id = -1
    for node in graph.nodes():
        if node[-1] == "h":
            if graph.nodes[node]['host_is_superhost'] is True:
                counter += 1
            #print(graph.nodes[node]['host_since'])
            if graph.nodes[node]['host_listings_count'] > max_listings_count:
                #print( graph.nodes[node]['host_listings_count'])
                #print(node)
                max_listings_count = graph.nodes[node]['host_listings_count']
                max_id = node
            #print(graph.nodes[node]['host_listings_count'])

    print(f"Superhosts: {counter}")
    print(f"max listings count: {max_listings_count}")
    print(max_id)
    print(len(graph.nodes()))
    



if __name__ == "__main__":
    cities= ["london","seattle","san-diego","san-francisco"]
    #cities= ["seattle","san-diego","san-francisco"]
    #cities= ["seattle","san-diego","san-francisco"]
    #cities= ["seattle","san-diego"]
    cities= ["san-francisco"]
    #cities= ["london"]
    compute_stay_probabilities = False
    print(datetime.now())
    for city in cities:
        print(city)
        if compute_stay_probabilities:
            _, stay_probabilities = generate_guest_host_temporal(city, alpha=0.1, load_degrees_by_date=False)
        else:
            stay_probabilities = pickle.load(open(f"data/{city}/temporal/guest_host_temporal_stay_probabilities.pickle", "rb"))
        
        analyse_stay_probabilities(city, stay_probabilities)
        print(datetime.now())





        #print_graph(guest_host_temporal)
    