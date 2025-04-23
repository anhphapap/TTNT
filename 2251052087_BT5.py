import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

G = nx.Graph()
num_nodes_per_layer = 10
positions = {}
node_weights = []

for i in range(num_nodes_per_layer):
    angle = 2 * np.pi * i / num_nodes_per_layer
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle)
    node = f"L1_{i}"
    positions[node] = (x, y)
    G.add_node(node)
    weight = 0.2 - (i / num_nodes_per_layer) * (0.2 - 0.025)
    node_weights.append(weight)

x_vals = [positions[f"L1_{i}"][0] for i in range(num_nodes_per_layer)]
y_vals = [positions[f"L1_{i}"][1] for i in range(num_nodes_per_layer)]
x_center = sum(x_vals) / num_nodes_per_layer
y_center = sum(y_vals) / num_nodes_per_layer

extra_node = "L1_extra"
positions[extra_node] = (x_center, y_center)
G.add_node(extra_node)
node_weights.append(0.2)
G.add_edge("L1_5", extra_node)

center_node_L2 = "L2_center"
G.add_node(center_node_L2)
positions[center_node_L2] = (0, -5)
node_weights.append(0.2)

for i in range(num_nodes_per_layer):
    angle = 2 * np.pi * i / num_nodes_per_layer
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle) - 5
    node = f"L2_{i}"
    positions[node] = (x, y)
    G.add_node(node)
    weight = 0.2 - (i / num_nodes_per_layer) * (0.2 - 0.025)
    node_weights.append(weight)
    G.add_edge(center_node_L2, node)

for i in range(num_nodes_per_layer):
    G.add_edge(f"L1_{i}", f"L2_{i}")

for i in range(num_nodes_per_layer):
    if i != 5:
        G.add_edge(f"L1_{i}", f"L1_{(i + 1) % num_nodes_per_layer}")

G.add_edge(extra_node, center_node_L2)

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_colormap",["black", "saddlebrown", "red", "orange", "yellow"]
)

plt.figure(figsize=(6, 6))
nodes = nx.draw_networkx_nodes(G, positions, node_color=node_weights, cmap=custom_cmap, node_size=300)
for u, v in G.edges():
    if (u.startswith("L1_") and v.startswith("L1_")) or (u == extra_node) and v.startswith("L1_"):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='black', width=7)
    elif u.startswith("L1_") and v.startswith("L2_"):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='gray', alpha=0.3, width=10)
    elif (u == extra_node and v == center_node_L2) or (v == extra_node and u == center_node_L2):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='gray', alpha=0.3, width=7)
    else:
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='black', width=7)

plt.colorbar(nodes, shrink=0.7, label="weight")

plt.text(-4.5, 1.3, 'layer $i = 1$', fontsize=12, rotation=90, va="center")
plt.text(-4.5, -5.7, 'layer $i = 2$', fontsize=12, rotation=90, va="center")

plt.axis('off')
plt.tight_layout()
plt.show()

