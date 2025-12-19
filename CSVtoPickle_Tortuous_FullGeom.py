"""
Code written by Sofia to read the csv files extracted from Renier and transform them into a pickles file
"""
import pandas as pd
import igraph as ig
import pickle
import numpy as np
import os

# import sys
# graph_number = sys.argv[1]

graph_number = 18

# Load CSV files


folder = "/home/admin/Ana/MicroBrain/CSV/"


print("=== START CSV → PKL (FULL GEOM) ===")

vertices_df = pd.read_csv(folder + "vertices.csv", header=None)  # Nodes
coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None)  # Coordinates nodes atlas verison
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None)  # Coordinates nodes from Image

edges_df = pd.read_csv(folder + "edges.csv", header=None)        # Edges
length_df = pd.read_csv(folder + "length.csv", header=None)      # Extra edge attributes
radii_df = pd.read_csv(folder + "radii_edge.csv", header=None)   # More edge attributes
radii_atlas_df = pd.read_csv(folder + "radii_atlas.csv", header=None)  # More edge attributes
vein_df = pd.read_csv(folder + "vein.csv", header=None)          # Label for vein edges
artery_df = pd.read_csv(folder + "artery.csv", header=None)      # Label for artery edges
radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None) # More vertex attributes
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None) # Annotation from ABA
distance_to_surface_df = pd.read_csv(folder + "distance_to_surface.csv", header=None) # distance to Surface

# Geometry CSVs
geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)  # geometry indices per edge
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)  # geometry coordinates
print("despues DE LEER CSVs")


print(np.mean([np.mean(radii_vertex_df[0]), np.mean(radii_df)]))

# ============================
# 1) Create empty graph
# ============================
G = ig.Graph()
G.add_vertices(len(vertices_df))

# original IDs
G.vs["id"] = vertices_df[0].tolist()

# node coordinates
if coordinates_df.shape[1] >= 3:
    G.vs["coords"] = list(zip(coordinates_df[0], coordinates_df[1], coordinates_df[2]))
    G.vs["coords_image"] = list(zip(coordinates_images_df[0], coordinates_images_df[1], coordinates_images_df[2]))

G.vs["annotation"] = annotation_vertex_df[0].tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].tolist()
G.vs["radii"] = radii_vertex_df[0].tolist()

# ============================
# 2) Edges
# ============================
edges = []
edge_nkind = []  # 2=artery, 3=vein, 4=capillary
radius = []
lengths = []

for i, row in edges_df.iterrows():
    source = int(row[0])
    target = int(row[1])

    nkind = 4  # default = capilar
    if artery_df[0][i] == 1:
        nkind = 2
    elif vein_df[0][i] == 1:
        nkind = 3

    edges.append((source, target))
    edge_nkind.append(nkind)
    lengths.append(length_df[0][i])
    radius.append(radii_df[0][i])

G.add_edges(edges)

G.es["connectivity"] = edges
G.es["nkind"] = edge_nkind
G.es["radius"] = radius
G.es["diameter"] = [r * 2 for r in radius]
G.es["length"] = lengths

# ============================
# 3) Geometry (POINTS per edge) — MEMORY SAFE VERSION
# ============================

edge_geometry_df.columns = ["x", "y", "z"]

# Use numpy arrays  (more compact than lists of tuples) 
x = edge_geometry_df["x"].to_numpy(dtype=np.float32)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32)

starts = geom_index_df[0].to_numpy(dtype=np.int64)
ends   = geom_index_df[1].to_numpy(dtype=np.int64)

# Check
assert int(ends[-1]) <= len(edge_geometry_df), "Last end index exceeds points length"

edge_geometries = []
edge_geometries_append = edge_geometries.append

for s, e in zip(starts, ends):
    # Save as Nx3 array float32 (compact)
    geom = np.column_stack((x[s:e], y[s:e], z[s:e])).astype(np.float32, copy=False)
    edge_geometries_append(geom)

G.es["geometry"] = edge_geometries

# ============================
# 3b) Compute tortuous length and tortuosity
# ============================

coords_v = np.array(G.vs["coords"])

length_tortuous = []
straight_dist = []

for edge_id, geom in enumerate(G.es["geometry"]):
    # --- tortuous length from geometry ---
    L = 0.0
    for i in range(len(geom) - 1):
        L += np.linalg.norm(geom[i+1] - geom[i])
    length_tortuous.append(L)

    # --- straight-line distance between endpoints ---
    v0, v1 = G.es[edge_id].tuple
    straight_dist.append(
        np.linalg.norm(coords_v[v0] - coords_v[v1])
    )

length_tortuous = np.array(length_tortuous)
straight_dist   = np.array(straight_dist)

tortuosity = length_tortuous / straight_dist
tortuosity[straight_dist == 0] = np.nan

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"]      = tortuosity.tolist()




# ============================
# 4) Comprobaciones de tipos
# ============================

def check_column_types(series, col_name):
    types_found = set(type(x) for x in series if pd.notnull(x))
    if len(types_found) > 1:
        print(f"Column '{col_name}' has multiple types: {types_found}")
    else:
        print(f"Column '{col_name}' all values type: {types_found.pop()}")

check_column_types(radii_vertex_df[0], "radii_vertex_df")
check_column_types(annotation_vertex_df[0], "annotation_vertex_df")
check_column_types(distance_to_surface_df[0], "distance_to_surface_df")
check_column_types(artery_df[0], "artery_df")
check_column_types(vein_df[0], "vein_df")
check_column_types(length_df[0], "length_df")
check_column_types(radii_df[0], "radii_df")
check_column_types(vertices_df[0], "vertices_df")

# ============================
# 5) Guardar en pickle
# ============================

out_path = r"/home/admin/Ana/MicroBrain" + str(graph_number) + f"/add_coord_{graph_number}_igraph.pkl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


print('Graph saved in pickle format at:', out_path)
print("=== DONE ===")
