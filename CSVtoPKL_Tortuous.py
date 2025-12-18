"""
Create FULL vascular graph pickle:
- igraph graph
- node attributes
- edge attributes
- full geometry (coords + indices)
"""

import pandas as pd
import igraph as ig
import numpy as np
import pickle
import os

# ============================
# Parameters
# ============================

graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_full.pkl"

print("=== START CSV → PKL ===")

# ============================
# Load CSVs
# ============================

vertices_df = pd.read_csv(folder + "vertices.csv", header=None)
coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None)
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None)

edges_df = pd.read_csv(folder + "edges.csv", header=None)
length_df = pd.read_csv(folder + "length.csv", header=None)
radii_df = pd.read_csv(folder + "radii_edge.csv", header=None)
vein_df = pd.read_csv(folder + "vein.csv", header=None)
artery_df = pd.read_csv(folder + "artery.csv", header=None)

radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None)
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None)
distance_to_surface_df = pd.read_csv(folder + "distance_to_surface.csv", header=None)

geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)

print("CSVs loaded")

# ============================
# Create graph
# ============================

G = ig.Graph()
G.add_vertices(len(vertices_df))

# ----------------------------
# Vertex attributes
# ----------------------------

G.vs["id"] = vertices_df[0].astype(int).tolist()

G.vs["coords"] = list(zip(
    coordinates_df[0], coordinates_df[1], coordinates_df[2]
))

G.vs["coords_image"] = list(zip(
    coordinates_images_df[0],
    coordinates_images_df[1],
    coordinates_images_df[2]
))

G.vs["annotation"] = annotation_vertex_df[0].astype(int).tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()

# ----------------------------
# Edges
# ----------------------------

edges = []
edge_nkind = []
edge_radius = []
edge_length = []

for i, row in edges_df.iterrows():
    s = int(row[0])
    t = int(row[1])

    nkind = 4
    if artery_df[0][i] == 1:
        nkind = 2
    elif vein_df[0][i] == 1:
        nkind = 3

    edges.append((s, t))
    edge_nkind.append(nkind)
    edge_radius.append(radii_df[0][i])
    edge_length.append(length_df[0][i])

G.add_edges(edges)

G.es["nkind"] = edge_nkind
G.es["radius"] = np.array(edge_radius, dtype=np.float32).tolist()
G.es["diameter"] = (2 * np.array(edge_radius, dtype=np.float32)).tolist()
G.es["length"] = np.array(edge_length, dtype=np.float32).tolist()

# ----------------------------
# Geometry (MEMORY SAFE)
# ----------------------------

x = edge_geometry_df[0].to_numpy(dtype=np.float32)
y = edge_geometry_df[1].to_numpy(dtype=np.float32)
z = edge_geometry_df[2].to_numpy(dtype=np.float32)

G.es["geom_start"] = geom_index_df[0].astype(np.int64).tolist()
G.es["geom_end"]   = geom_index_df[1].astype(np.int64).tolist()

# ----------------------------
# Calculate tortuosity
# ----------------------------
coords_v = np.array(G.vs["coords"])

length_tortuous = []
straight_dist = []

for e in range(G.ecount()):
    # Lenght of the tortuous should be computed using the edge geometry coordinates
    s = G.es[e]["geom_start"]
    e_ = G.es[e]["geom_end"]

    L = 0.0
    for i in range(s, e_ - 1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        dz = z[i+1] - z[i]
        L += np.sqrt(dx*dx + dy*dy + dz*dz)

    length_tortuous.append(L)

    # Straight distance between endpoints
    v0, v1 = G.es[e].tuple
    d = np.linalg.norm(coords_v[v0] - coords_v[v1])
    straight_dist.append(d)

length_tortuous = np.array(length_tortuous) #current length should be the sum of all the tortuous points.
straight_dist = np.array(straight_dist)

tortuosity = length_tortuous / straight_dist
tortuosity[straight_dist == 0] = np.nan

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()


# ----------------------------
# Save EVERYTHING in ONE pkl
# ----------------------------

data = {
    "graph": G,
    "coords": {
        "x": x,
        "y": y,
        "z": z
    }
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved FULL graph to:", out_path)
print("=== DONE ===")
