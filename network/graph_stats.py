from generate_graphs import load_graph
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import numpy as np
from scipy.stats import linregress
import pandas as pd


def compute_and_plot_degree_distribution(graph, node_type, city):
    # Get the degree of each node
    nodes = list(filter(lambda node: node[-1] == node_type, graph.nodes()))
    degrees = [graph.degree(n) for n in nodes]

    # Count the frequency of each degree
    degree_counts = {}
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    
    # Sort the degree frequencies and store as probabilities
    sorted_degrees = sorted(degree_counts.items())
    x, y = zip(*sorted_degrees)

    # Normalize frequencies to probabilities
    total_frequency = sum(y)
    y = [freq / total_frequency for freq in y]  # Convert to probabilities

    plot_degree_distribution(x, y, node_type, city)
    power_law_exponent, power_law_r2 = compute_power_law_coefficient_ccdf(graph, node_type, city)
    return round(power_law_exponent,4), round(power_law_r2,4)

def plot_degree_distribution(x, y, node_type, city):
    
    type_label = "guest" if node_type == "g" else "host"
    dist_type = "in" if node_type == "h" else "out"

    # Plot the degree distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label="Degree Probabilities", marker="o")
    # Set log-log scale but keep normal labels
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")  # Normal axis label
    plt.ylabel("Degree Probability P(k)")  # Normal axis label
    plt.title(f"{city} | {type_label.capitalize()} {dist_type.capitalize()}-Degree Distribution")

    # Grid only for major ticks
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")

    # Save the plot
    plt.savefig(f"data/{city}/{type_label.lower()}_{dist_type}_degree_distribution.png", format="png", dpi=300)
    plt.close()


def compute_ccdf(graph, node_type):

    # Get the degree of each node
    nodes = list(filter(lambda node: node[-1] == node_type, graph.nodes()))
    degrees = [graph.degree(n) for n in nodes]
    
    # Count the frequency of each degree
    degree_counts = {}
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    
    # Total number of nodes
    total_nodes = len(graph.nodes())
    
    # Compute CCDF
    ccdf_dict = {}
    for degree, count in degree_counts.items():
        # Probability of degree >= k
        ccdf_dict[degree] = sum(freq for d, freq in degree_counts.items() if d >= degree) / total_nodes
    
    # Sort the degrees
    sorted_degrees = sorted(ccdf_dict.items())
    degrees, ccdf = zip(*sorted_degrees)
    
    return degrees, ccdf



def plot_ccdf(x, y, log_x, log_y, slope, intercept, r_squared, node_type, city):

    type_label = "guest" if node_type == "g" else "host"
    dist_type = "in" if node_type == "h" else "out"

    # Convert fitted line back to the original scale
    x_fit = np.linspace(min(x), max(x), 100)  # Generate smooth degree values
    y_fit = np.exp(intercept) * x_fit ** slope  # Back to original scale

    # Compute the power-law exponent
    exponent = 1-slope  # Exponent is the negative slope

    # Plot the CCDF
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Empirical CCDF', alpha=0.7)

    # Overlay the fitted line and add exponent and R² to the legend
    plt.plot(x_fit, y_fit, 'r-', 
             label=f'Fitted Line\nα: {exponent:.2f}\nR²: {r_squared:.4f}')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (k)')
    plt.ylabel('P(K >= k) (CCDF)')
    plt.title(f'{city} | {type_label.capitalize()} {dist_type.capitalize()}-Degree CCDF')
    plt.legend(loc='best')  # Add legend with location
    plt.grid(visible=True, which="major", linestyle="-", linewidth=0.5)
    plt.minorticks_off()  # Remove minor ticks

    # Save the plot
    plt.savefig(f"data/{city}/{type_label.lower()}_{dist_type}_ccdf_fit.png", format="png", dpi=300)
    plt.close()


def compute_power_law_coefficient_ccdf(graph, node_type, city):
    
    degrees, ccdf = compute_ccdf(graph, node_type)

    # Step 2: Convert to numpy arrays
    x = np.array(degrees)
    y = np.array(ccdf)

    # Step 3: Remove zero or near-zero values (log(0) is undefined)
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Step 4: Log-log transformation
    log_x = np.log(x_filtered)
    log_y = np.log(y_filtered)

    # Step 5: Focus on the linear portion (truncate small noisy degrees)
    # Step 5: Focus on the linear portion (truncate small noisy degrees)
    min_degree = 1
    max_log_k = 4  # Upper bound for log(k)

    # Create a mask for the range [log(min_degree), max_log_k]
    valid_mask = (log_x >= np.log(min_degree))

    # Apply the mask to filter the data
    log_x_trimmed = log_x[valid_mask]
    log_y_trimmed = log_y[valid_mask]

    valid_mask = (log_x >= np.log(min_degree)) & (log_x <= max_log_k)
    log_x_polyfit_trimmed = log_x[valid_mask]
    log_y_polyfit_trimmed = log_y[valid_mask]


    # Step 6: Perform linear regression on log-log data
    slope, intercept, r_value, _, _ = linregress(log_x_polyfit_trimmed, log_y_polyfit_trimmed)

    # Compute the power-law exponent
    power_law_exponent = 1-slope  # Negative slope corresponds to exponent

    # Compute R² for the goodness of fit
    r_squared = r_value ** 2

    plot_ccdf(x, y, log_x_trimmed, log_y_trimmed, slope, intercept, r_squared, node_type, city)

    print(f"Power-Law Exponent: {power_law_exponent:.4f}")
    print(f"R² Value: {r_squared:.4f}")

    return power_law_exponent, r_squared
    
    
def compute_guest_host_stats(city, graph, df):
    print("=================================")
    print(city)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    # Count edges where weight > 1
    edges_weight_greater_1 = sum(1 for _, _, data in graph.edges(data=True) if data.get('weight', 0) > 1)
    print(f"Edges with weight > 1: {edges_weight_greater_1}")
    


    # Separate in-degree and out-degree analysis
    host_in_degrees = [graph.in_degree(n) for n in graph.nodes() if graph.in_degree(n) > 0]
    guest_out_degrees = [graph.out_degree(n) for n in graph.nodes() if graph.out_degree(n) > 0]

    num_hosts = len(host_in_degrees)
    num_guests = len(guest_out_degrees)

    # Compute statistics for hosts
    mean_host_in_degree = round(sum(host_in_degrees) / len(host_in_degrees) if host_in_degrees else 0 , 4)
    med_host_in_degree = round(sorted(host_in_degrees)[len(host_in_degrees)//2], 4)
    max_host_in_degree = max(host_in_degrees) if host_in_degrees else 0

    # Compute statistics for guests
    mean_guest_out_degree = round(sum(guest_out_degrees) / len(guest_out_degrees) if guest_out_degrees else 0, 4)
    med_guest_out_degree = round(sorted(guest_out_degrees)[len(guest_out_degrees)//2], 4)
    max_guest_out_degree = max(guest_out_degrees) if guest_out_degrees else 0

    # Print the results
    print(f"Number of Hosts: {num_hosts}")
    print(f"Mean Host In-Degree: {mean_host_in_degree}")
    print(f"Median Host In-Degree: {med_host_in_degree}")
    print(f"Max Host In-Degree: {max_host_in_degree}")

    print(f"Number of Guests: {num_guests}")
    print(f"Average Guest Out-Degree: {mean_guest_out_degree}")
    print(f"Median Guest Out-Degree: {med_guest_out_degree}")
    print(f"Max Guest Out-Degree: {max_guest_out_degree}")


    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]
    mean_edge_weight = round(sum(edge_weights) / len(edge_weights),2)
    print(f"Average Edge Weight: {mean_edge_weight}")
    print(f"Max Edge Weight: {max(edge_weights)}")

    print("In Degree Distribution")
    guest_power_law_exponent, guest_power_law_r2 = compute_and_plot_degree_distribution(graph, "g", city)

    print("Out Degree Distribution")
    host_power_law_exponent, host_power_law_r2 = compute_and_plot_degree_distribution(graph, "h", city)

    superhosts = sum(1 for node in graph.nodes if node[-1] == "h" and graph.nodes[node]['host_is_superhost'] is True)
    superhosts_percent_hosts = superhosts/num_hosts * 100

    print(f"Superhosts: {superhosts} % of hosts: {superhosts_percent_hosts}")

    df.loc[len(df)] = [city,graph.number_of_nodes(), num_guests, num_hosts, graph.number_of_edges(), mean_host_in_degree, med_host_in_degree, max_host_in_degree, mean_guest_out_degree, med_guest_out_degree, max_guest_out_degree, mean_edge_weight, edges_weight_greater_1, guest_power_law_exponent, guest_power_law_r2, host_power_law_exponent, host_power_law_r2, superhosts, superhosts_percent_hosts ]

if __name__ == "__main__":
    df = pd.DataFrame(columns=['city','num_nodes','guests','hosts','edges','mean_host_in_degree','median_host_in_degree','max_host_in_degree','mean_guest_out_degree','median_guest_out_degree','max_guest_out_degree','mean_edge_weight','edges_weight_greater_1','guest_power_law_exponent','guest_power_law_r2','host_power_law_exponent','host_power_law_r2', 'superhosts', 'superhosts_percent_hosts'])
    cities= ["san-francisco","san-diego","seattle","london",]
    #cities= ["san-francisco"]
    for city in cities:
        guest_host = load_graph(city, "guest_host")
        compute_guest_host_stats(city, guest_host, df)


    df.to_csv("data/stats_output.csv", encoding='utf-8', index=False)