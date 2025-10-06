import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edges_df = pd.read_csv("graph.csv")
heuristic_df = pd.read_csv("heuristic.csv")

graph = {}
for _, row in edges_df.iterrows():
    src, dst = row['from'], row['to']
    if src in graph:
        graph[src].append(dst)
    else:
        graph[src] = [dst]

for node in heuristic_df['node']:
    if node not in graph:
        graph[node] = []

heuristic = dict(zip(heuristic_df['node'], heuristic_df['heuristic']))

start_node = 'A'
goal_node = 'G'

current_node = start_node
path = [current_node]

while current_node != goal_node:
    neighbors = graph[current_node]
   
    if not neighbors:
        print("Dead end at:", current_node)
        break
   
    next_node = min(neighbors, key=lambda node: heuristic[node])
   
    if heuristic[next_node] >= heuristic[current_node]:
        print("Stuck at local minimum at:", current_node)
        break

    current_node = next_node
    path.append(current_node)

G = nx.DiGraph()

# Add edges to the graph
for _, row in edges_df.iterrows():
    G.add_edge(row['from'], row['to'])

pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_weight='bold')

heuristic_labels = {node: f"h={heuristic[node]}" for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=heuristic_labels, font_color='red', verticalalignment='bottom')

path_edges = list(zip(path, path[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=3)
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='limegreen')

if path[-1] == goal_node:
    print("Goal reached!")
else:
    print("Failed to reach the goal.")

print("Path taken:", path)

plt.title("Hill Climbing Path Visualization")
plt.show()