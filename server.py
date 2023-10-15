from flask import Flask, jsonify, request
from flask_cors import CORS
import networkx as nx
import random
import json
import heapq
import math


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# # Create a random weighted graph with a specified number of nodes
# def generate_random_graph(num_nodes):
#     G = nx.Graph()

#     # Add nodes with labels and groups (without "type")
#     for node_id in range(num_nodes):
#         label = f"Node {node_id}"
#         group = random.choice(["Group 1", "Group 2", "Group 3"])
        
#         G.add_node(node_id, label=label, group=group)

#     # Randomly add weighted edges
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             # Generate a random number to determine whether to add an edge
#             if random.random() < 0.1:  # Adjust the probability as needed
#                 weight = random.uniform(0.1, 1.0)
#                 G.add_edge(i, j, weight=weight)

#     # Ensure connectivity
#     while not nx.is_connected(G):
#         # Find disconnected nodes
#         disconnected_nodes = [node for node in G.nodes if not nx.has_path(G, source=node, target=list(G.nodes)[0])]
        
#         # Randomly select a disconnected node and connect it
#         selected_node = random.choice(disconnected_nodes)
#         target_node = random.choice(list(G.nodes))
#         weight = random.uniform(0.1, 1.0)
#         G.add_edge(selected_node, target_node, weight=weight)           

#     return G

# Create a random connected graph with Watts-Strogatz properties
# def generate_random_graph(num_nodes, k=5, p=0.1):
#     G = nx.random_internet_as_graph(num_nodes,seed = 4)

#     # Assign labels and groups to nodes
#     for node_id, node_data in G.nodes(data=True):
#         node_data['label'] = f"Node {node_id}"
#         node_data['group'] = random.choice(["Group 1", "Group 2", "Group 3"])

#     # Assign random weights to edges
#     for edge in G.edges():
#         G[edge[0]][edge[1]]['weight'] = random.uniform(1, 20)

#     return G


def generate_random_graph(num_nodes, k=5, p=0.1):
    G = nx.connected_watts_strogatz_graph(num_nodes, k, p)

    # Assign labels and groups to nodes
    for node_id, node_data in G.nodes(data=True):
        node_data['label'] = f"Node {node_id}"
        # node_data['group'] = random.choice(["Group 1", "Group 2", "Group 3"])

    # Assign random weights to edges
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = random.randint(1, 20)

    return G
# Generate a random graph with 10 nodes

RandG = None







@app.route('/get_rand_graph', methods=['GET'])
def get_rand_graph():
    global RandG
    RandG = generate_random_graph(150)
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
    # print(json.dumps( RandG.edges, indent= 2))
    list(RandG.edges)
    print(nx.shortest_path(RandG,0, 17))
    # Return the JSON data as a response
    shortest_path = a_star(RandG, 0, 29)
    print("Shortest Path A*:", shortest_path)
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

@app.route('/find_shortest_path', methods=['POST'])
def find_shortest_path():

    #Not making this global can result in incorrect data when finding the shrotest path
    global RandG 
   
    # Parse the request data
    data = request.get_json()
    source = data['source']
    target = data['target']
    algorithm = data['algorithm']
  # Default to Dijkstra's algorithm

    # Ensure that source and target nodes exist in the RandG graph
    if source not in RandG.nodes or target not in RandG.nodes:
        return jsonify({'error': 'Source or target node not found in the graph'})

    # Find the shortest path based on the selected algorithm
    if algorithm == 'dijkstra':
        path = nx.shortest_path(RandG, source, target)
        print("Dijkstra Algo executed...")
    elif algorithm == 'dijkstraW':
        path = nx.shortest_path(RandG, source, target, weight='weight')
        print("Dijkstra (WEIGHTED) Algo executed...")
    elif algorithm == 'bellman-ford':
        path = nx.shortest_path(RandG, source, target, method="bellman-ford")
        print("Bellman-Ford Algo executed...")
    elif algorithm == 'bellman-fordW':
        path = nx.shortest_path(RandG, source, target, weight='weight',method="bellman-ford")
        print("Bellman-Ford (WEIGHTED) Algo executed...")
    elif algorithm == 'astar':
        path = a_star(RandG, source, target)
        print("A* Algo executed...")
    elif algorithm == 'astarW':
        path = a_starW(RandG, source, target)
        print("A* (WEIGHTED) Algo executed...")
      
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400
    print(path)
    list(RandG.edges)
    return jsonify({'shortest_path': path})

# Function to calculate Euclidean distance between two nodes
def euclidean_distance(graph, node1, node2):
    x1, y1 = graph.nodes[node1]['pos']
    x2, y2 = graph.nodes[node2]['pos']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


#Custom A* Algorithm
def a_star(graph, start_node, target_node):
    # Initialize the open list (priority queue) with the start node and its cost
    open_list = [(0, start_node)]
    # Initialize dictionaries to track costs and predecessors
    g_costs = {node: float('inf') for node in graph.nodes}
    g_costs[start_node] = 0
    predecessors = {}

    while open_list:
        # Get the node with the lowest cost from the open list
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == target_node:
            # Reconstruct the path
            path = [current_node]
            while current_node in predecessors:
                current_node = predecessors[current_node]
                path.insert(0, current_node)
            return path

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            # Access the weight of the edge directly from the graph
            # edge_weight = graph[current_node][neighbor].get('weight', 1)

            # Calculate the tentative g cost from the start node to this neighbor
            tentative_g_cost = g_costs[current_node] + 1

            if tentative_g_cost < g_costs[neighbor]:
                # This path to the neighbor is better, so record it
                predecessors[neighbor] = current_node
                g_costs[neighbor] = tentative_g_cost
                # Calculate the heuristic (in this case, we use Euclidean distance as the heuristic)
                h = nx.shortest_path_length(graph, neighbor, target_node, weight=1)
                # Add the neighbor to the open list with the total estimated cost
                f = tentative_g_cost + h
                heapq.heappush(open_list, (f, neighbor))

    # If no path is found, return None
    return None

def a_starW(graph, start_node, target_node):
    # Initialize the open list (priority queue) with the start node and its cost
    open_list = [(0, start_node)]
    # Initialize dictionaries to track costs and predecessors
    g_costs = {node: float('inf') for node in graph.nodes}
    g_costs[start_node] = 0
    predecessors = {}

    while open_list:
        # Get the node with the lowest cost from the open list
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == target_node:
            # Reconstruct the path
            path = [current_node]
            while current_node in predecessors:
                current_node = predecessors[current_node]
                path.insert(0, current_node)
            return path

        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            # Access the weight of the edge directly from the graph
            edge_weight = graph[current_node][neighbor].get('weight', 1)

            # Calculate the tentative g cost from the start node to this neighbor
            tentative_g_cost = g_costs[current_node] + edge_weight

            if tentative_g_cost < g_costs[neighbor]:
                # This path to the neighbor is better, so record it
                predecessors[neighbor] = current_node
                g_costs[neighbor] = tentative_g_cost
                # Calculate the heuristic
                h = nx.shortest_path_length(graph, neighbor, target_node, weight='weight')
                # Add the neighbor to the open list with the total estimated cost
                f = tentative_g_cost + h
                heapq.heappush(open_list, (f, neighbor))

    # If no path is found, return None
    return None




# # Add nodes
# num_nodes = 10
# for i in range(num_nodes):
#     RandG.add_node(i)

# # Add random weighted edges
# for i in range(num_nodes):
#     for j in range(i + 1, num_nodes):
#         # Generate a random weight between 0.1 and 1.0
#         weight = random.uniform(0.1, 1.0)
#         RandG.add_edge(i, j, weight=weight)


# # Add nodes representing departments
# departments = [
#     {"id": "CS", "label": "Computer Science"},
#     {"id": "EE", "label": "Electrical Engineering"},
#     {"id": "ME", "label": "Mechanical Engineering"},
#     {"id": "CE", "label": "Civil Engineering"},
#     {"id": "BA", "label": "Business Administration"},
#     {"id": "LA", "label": "Liberal Arts"},
#     # Add more departments here
# ]

# # Add nodes representing devices within each department
# devices = [
#     {"id": "Router", "label": "Router"},
#     {"id": "AccessPoint", "label": "Access Point"},
#     {"id": "PC", "label": "PC"},
#     {"id": "Printer", "label": "Printer"},
#     {"id": "Server", "label": "Server"},
#     {"id": "Laptop", "label": "Laptop"},
#     # Add more devices for each department
# ]

# # Add main network admin center
# network_admin_center = {"id": "AdminCenter", "label": "Network Admin Center"}

# # Add main routers
# main_routers = [
#     {"id": "MainRouter1", "label": "Main Router 1"},
#     {"id": "MainRouter2", "label": "Main Router 2"},
# ]

# # Add departments as nodes
# for department in departments:
#     G.add_node(department["id"], label=department["label"], type="Department", group=department["id"])

# # Add devices as nodes within departments and connect them
# for department in departments:
#     for device in devices:
#         device_id = f"{department['id']}-{device['id']}"
#         G.add_node(device_id, label=device["label"], type=device["id"], group=department["id"])
#         G.add_edge(department["id"], device_id)  # Connect device to department

# # Add main network admin center
# G.add_node(network_admin_center["id"], label=network_admin_center["label"], type="AdminCenter")

# # Add main routers and connect them to departments
# for router in main_routers:
#     G.add_node(router["id"], label=router["label"], type="MainRouter")
#     for department in departments:
#         G.add_edge(department["id"], router["id"])  # Connect department to main router

# # Connect main routers to network admin center
# for router in main_routers:
#     G.add_edge(router["id"], network_admin_center["id"])

# # Export the graph data as JSON
# data = {
#     "directed": False,
#     "graph": {},
#     "multigraph": False,
#     "nodes": [
#         {
#             "id": node,
#             "label": G.nodes[node]["label"],
#             "type": G.nodes[node]["type"],
#             "group": G.nodes[node]["group"] if "group" in G.nodes[node] else None,
#         }
#         for node in G.nodes
#     ],
#     "links": [{"source": source, "target": target} for source, target in G.edges],
# }

# @app.route('/get_graph', methods=['GET'])
# def get_graph():
#     return jsonify(data)
if __name__ == '__main__':
    app.run(debug=True)