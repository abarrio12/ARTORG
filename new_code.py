"""
Code written by Sofia to read the csv files extracted from Renier
and transform them into an igraph pickle file.

UPDATED VERSION:
- Correct geometry handling (start/end indices)
- Geometry coordinates saved separately (memory safe)
- Suitable for very large graphs
"""

import pandas as pd
import igraph as ig
import pickle
import numpy as np
import os

# ============================
# Parameters
# ============================

graph_number = 18

folder = (
    "/home/admin/Ana/MicroBrain/CSV/"
)

out_folder = (
    "/home/admin/Ana/MicroBrain/output"
)

os.makedirs(out_folder, exist_ok=True)

print("=== START ===")

# ============================
# Load CSV files
# ============================

vertices_df = pd.read_csv(folder + "vertices.csv", header=None)

coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None)
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None)

edges_df = pd.read_csv(folder + "edges.csv", header=None)
length_df = pd.read_csv(folder + "length.csv", header=None)
radii_df = pd.read_csv(folder + "radii_edge.csv", header=None)
radii_atlas_df = pd.read_csv(folder + "radii_atlas.csv", header=None)

vein_df = pd.read_csv(folder + "vein.csv", header=None)
artery_df = pd.read_csv(folder + "artery.csv", header=None)

radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None)
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None)
distance_to_surface_df = pd.read_csv(folder + "distance_to_surface.csv", header=None)

# Geometry (IMPORTANT)
geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)

print("CSV files loaded")

# ============================
# Quick sanity info
# ============================

print("Mean radii check:",
      np.mean([np.mean(radii_vertex_df[0]), np.mean(radii_df[0])]))

print("Vertices:", len(vertices_df))
print("Edges:", len(edges_df))
print("Geometry points:", len(edge_geometry_df))

# ============================
# 1) Create empty graph
# ============================

G = ig.Graph()
G.add_vertices(len(vertices_df))

# ============================
# Vertex attributes
# ============================

G.vs["id"] = vertices_df[0].astype(int).tolist()

coordinates_df.columns = ["x", "y", "z"]
coordinates_images_df.columns = ["x", "y", "z"]

G.vs["coords"] = list(
    zip(coordinates_df["x"], coordinates_df["y"], coordinates_df["z"])
)
G.vs["coords_image"] = list(
    zip(coordinates_images_df["x"],
        coordinates_images_df["y"],
        coordinates_images_df["z"])
)

G.vs["annotation"] = annotation_vertex_df[0].astype(int).tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()

# ============================
# 2) Edges
# ============================

edges = []
edge_nkind = []   # 2=artery, 3=vein, 4=capillary
edge_radius = []
edge_length = []

for i, row in edges_df.iterrows():
    source = int(row[0])
    target = int(row[1])

    nkind = 4
    if artery_df[0][i] == 1:
        nkind = 2
    elif vein_df[0][i] == 1:
        nkind = 3

    edges.append((source, target))
    edge_nkind.append(nkind)
    edge_radius.append(radii_df[0][i])
    edge_length.append(length_df[0][i])

G.add_edges(edges)

G.es["nkind"] = edge_nkind
G.es["radius"] = np.array(edge_radius, dtype=np.float32).tolist()
G.es["diameter"] = (2 * np.array(edge_radius, dtype=np.float32)).tolist()
G.es["length"] = np.array(edge_length, dtype=np.float32).tolist()

# ============================
# 3) Geometry INDICES ONLY (CORRECT + MEMORY SAFE)
# ============================

# geom_index_df columns:
# col 0 = start index
# col 1 = end index   (slice on edge_geometry_coordinates)

G.es["geom_start"] = geom_index_df[0].astype(np.int32).tolist()
G.es["geom_end"]   = geom_index_df[1].astype(np.int32).tolist()

# Sanity check
last_end = int(geom_index_df.iloc[-1, 1])
assert last_end <= len(edge_geometry_df), "Geometry indices out of bounds"

# ============================
# 4) Save geometry coordinates separately
# ============================

coords_path = os.path.join(out_folder, "edge_geometry_coords.npz")

np.savez_compressed(
    coords_path,
    x=edge_geometry_df[0].astype(np.float32).values,
    y=edge_geometry_df[1].astype(np.float32).values,
    z=edge_geometry_df[2].astype(np.float32).values,
)

print("Geometry coordinates saved to:", coords_path)

# ============================
# 5) Save igraph pickle
# ============================

graph_path = os.path.join(
    out_folder,
    f"graph_{graph_number}_igraph.pkl"
)

with open(graph_path, "wb") as f:
    pickle.dump(G, f)

print("Graph saved to:", graph_path)

# ============================
# DONE
# ============================

print("=== DONE ===")
