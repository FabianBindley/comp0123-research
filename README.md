# comp0123-research
Research Project investigating Airbnb guest-host networks

# Instructions
1) Download and extract the relevant city detailed 	listings.csv.gz and reviews.csv.gz files from https://insideairbnb.com/get-the-data/ in the data/{city-name} directory eg: data/san-francisco/listings.csv - data/san-francisco/reviews.csv
2) Create a Python virtual environment and install the required libraries - shown in requirements.txt by running:
```
pip3 install -r network/requirements.txt 
```
3) Run the network/generate_graphs.py script to generate the guest-host graphs for the selected city
4) Analyse the graphs with network/graph_stats.py and visualise a small portion with network/visualise_graphs.py
Please Note:

You may need to change the run configurations to select which cities you are analysing. This can be done by adding or removing cities from the cities list in each file:
```
   cities= ["seattle","san-diego","san-francisco"]
```
By default, we assume that you have generated guest-host graphs for the 3 cities above.
