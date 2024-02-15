from flask import Flask, jsonify, request
from flask_cors import CORS
import networkx as nx
import random
import json
import heapq
import math


app = Flask(__name__)
CORS(app, origins=["*"]) #Put your localhost 


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
