from flask import Flask, jsonify
from flask_cors import CORS
import networkx as nx
import random


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


# Create a graph with 10 nodes
G = nx.Graph()


# Create a random weighted graph with a specified number of nodes
def generate_random_graph(num_nodes):
    G = nx.Graph()

    # Add nodes with labels and groups (without "type")
    for node_id in range(num_nodes):
        label = f"Node {node_id}"
        group = random.choice(["Group 1", "Group 2", "Group 3"])
        
        G.add_node(node_id, label=label, group=group)

    # Randomly add weighted edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Generate a random number to determine whether to add an edge
            if random.random() < 0.1:  # Adjust the probability as needed
                weight = random.uniform(0.1, 1.0)
                G.add_edge(i, j, weight=weight)

    # Ensure connectivity
    while not nx.is_connected(G):
        # Find disconnected nodes
        disconnected_nodes = [node for node in G.nodes if not nx.has_path(G, source=node, target=list(G.nodes)[0])]
        
        # Randomly select a disconnected node and connect it
        selected_node = random.choice(disconnected_nodes)
        target_node = random.choice(list(G.nodes))
        weight = random.uniform(0.1, 1.0)
        G.add_edge(selected_node, target_node, weight=weight)           

    return G

# Generate a random graph with 10 nodes
RandG = generate_random_graph(50)

# Prepare the graph data in the specified format (without "type")
RandData = {
    "directed": False,
    "graph": {},
    "multigraph": False,
    "nodes": [
        {
            "id": node,
            "label": RandG.nodes[node]["label"],
            "group": RandG.nodes[node]["group"] if "group" in RandG.nodes[node] else None,
        }
        for node in RandG.nodes
    ],
    "links": [
        {
            "source": source,
            "target": target,
            "weight": RandG[source][target]['weight']
        }
        for source, target in RandG.edges
    ],
}



# Add nodes
num_nodes = 10
for i in range(num_nodes):
    RandG.add_node(i)

# Add random weighted edges
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        # Generate a random weight between 0.1 and 1.0
        weight = random.uniform(0.1, 1.0)
        RandG.add_edge(i, j, weight=weight)


# Add nodes representing departments
departments = [
    {"id": "CS", "label": "Computer Science"},
    {"id": "EE", "label": "Electrical Engineering"},
    {"id": "ME", "label": "Mechanical Engineering"},
    {"id": "CE", "label": "Civil Engineering"},
    {"id": "BA", "label": "Business Administration"},
    {"id": "LA", "label": "Liberal Arts"},
    # Add more departments here
]

# Add nodes representing devices within each department
devices = [
    {"id": "Router", "label": "Router"},
    {"id": "AccessPoint", "label": "Access Point"},
    {"id": "PC", "label": "PC"},
    {"id": "Printer", "label": "Printer"},
    {"id": "Server", "label": "Server"},
    {"id": "Laptop", "label": "Laptop"},
    # Add more devices for each department
]

# Add main network admin center
network_admin_center = {"id": "AdminCenter", "label": "Network Admin Center"}

# Add main routers
main_routers = [
    {"id": "MainRouter1", "label": "Main Router 1"},
    {"id": "MainRouter2", "label": "Main Router 2"},
]

# Add departments as nodes
for department in departments:
    G.add_node(department["id"], label=department["label"], type="Department", group=department["id"])

# Add devices as nodes within departments and connect them
for department in departments:
    for device in devices:
        device_id = f"{department['id']}-{device['id']}"
        G.add_node(device_id, label=device["label"], type=device["id"], group=department["id"])
        G.add_edge(department["id"], device_id)  # Connect device to department

# Add main network admin center
G.add_node(network_admin_center["id"], label=network_admin_center["label"], type="AdminCenter")

# Add main routers and connect them to departments
for router in main_routers:
    G.add_node(router["id"], label=router["label"], type="MainRouter")
    for department in departments:
        G.add_edge(department["id"], router["id"])  # Connect department to main router

# Connect main routers to network admin center
for router in main_routers:
    G.add_edge(router["id"], network_admin_center["id"])

# Export the graph data as JSON
data = {
    "directed": False,
    "graph": {},
    "multigraph": False,
    "nodes": [
        {
            "id": node,
            "label": G.nodes[node]["label"],
            "type": G.nodes[node]["type"],
            "group": G.nodes[node]["group"] if "group" in G.nodes[node] else None,
        }
        for node in G.nodes
    ],
    "links": [{"source": source, "target": target} for source, target in G.edges],
}

@app.route('/get_graph', methods=['GET'])
def get_graph():
    return jsonify(data)



@app.route('/get_rand_graph', methods=['GET'])
def get_rand_graph():
    
    # Return the JSON data as a response
    return jsonify(RandData)


@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    clustering_coefficient = nx.clustering(G)
    diameter = nx.diameter(G)

    metrics = {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'clustering_coefficient': clustering_coefficient,
        'diameter': diameter
    }
    return (metrics)

@app.route('/get_rand_metrics', methods=['GET'])
def get_rand_metrics():
    degree_centrality = nx.degree_centrality(RandG)
    betweenness_centrality = nx.betweenness_centrality(RandG)
    closeness_centrality = nx.closeness_centrality(RandG)
    clustering_coefficient = nx.clustering(RandG)
    diameter = nx.diameter(RandG)

    metrics = {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'clustering_coefficient': clustering_coefficient,
        'diameter': diameter
    }
    return (metrics)

if __name__ == '__main__':
    app.run(debug=True)