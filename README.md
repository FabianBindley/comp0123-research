# comp0123-research
Research Project investigating Airbnb guest-host networks

# Instructions
1) Download and extract the relevant city detailed listings.csv.gz and reviews.csv.gz files from https://insideairbnb.com/get-the-data/ in the data/{city-name} directory eg: data/san-francisco/listings.csv - data/san-francisco/reviews.csv.

> Please note, if you would prefer not to do download the original data, pregenerated guest-host graph object files (guest_host.pickle) can be found in each city's directory, however you will not be able to perform temporal or BA analysis with these.

2) (Optional) Create a Python virtual environment
3) Install the required libraries - shown in requirements.txt by running:
```
pip3 install -r network/requirements.txt 
```
4) If you have downloaded and extracted the listings and reviews data, run the network/generate_graphs.py script to generate the guest-host graphs for the selected cities.
5) Analyse the graphs with network/graph_stats.py and visualise a small portion with network/visualise_graphs.py
Please Note:

You may need to change the run configurations to select which cities you are generating graphs for and analysing. This can be done by adding or removing cities from the cities list in each file:
```
   cities = ["seattle","san-diego","san-francisco"]
```
By default, we assume that you wish to investigate the guest-host graphs for the 3 cities above.

6) To investigate the temporal graphs, barabasi albert, and superguest - please generate them with their respective analysis scripts.
7) Barabasi-albert and superguest graphs can have their stats computed in graph_stats.py, by setting their network_type to "barabasi_albert" or "superguest"
