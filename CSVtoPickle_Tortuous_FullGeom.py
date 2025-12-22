"""
Code written by Sofia to read the csv files extracted from Renier
and transform them into a pickles file (FULL GEOMETRY + CHECKS)
"""

import pandas as pd
import igraph as ig
import pickle
import numpy as np
import os

graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"

print("=== START CSV → PKL (FULL GEOM) ===")

# ============================
# Load CSVs
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

geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)

print("despues DE LEER CSVs")
print(np.mean([np.mean(radii_vertex_df[0]), np.mean(radii_df)]))

# ============================
# 1) Create graph
# ============================

G = ig.Graph()
G.add_vertices(len(vertices_df))

G.vs["id"] = vertices_df[0].tolist()

# --- node coordinates ---
G.vs["coords"] = list(zip(
    coordinates_df[0], coordinates_df[1], coordinates_df[2]
))
G.vs["coords_image"] = list(zip(
    coordinates_images_df[0], coordinates_images_df[1], coordinates_images_df[2]
))

G.vs["annotation"] = annotation_vertex_df[0].tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].tolist()
G.vs["radii"] = radii_vertex_df[0].tolist()

# ============================
# CHECK 1: duplicated node coordinates
# ============================

coords = np.array(G.vs["coords"])
n_nodes = len(coords)
n_unique = len(np.unique(coords, axis=0))

print("CHECK nodes:")
print("  N nodos:", n_nodes)
print("  N coords únicas:", n_unique)
print("  N duplicados:", n_nodes - n_unique)

# ============================
# 2) Edges
# ============================

edges = []
edge_nkind = []
radius = []
lengths = []

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
    lengths.append(length_df[0][i])
    radius.append(radii_df[0][i])

# ============================
# CHECK 2: edge index sanity
# ============================

edges_arr = np.array(edges)

print("CHECK edges indices:")
print("  Min source:", edges_arr[:, 0].min())
print("  Max source:", edges_arr[:, 0].max())
print("  Min target:", edges_arr[:, 1].min())
print("  Max target:", edges_arr[:, 1].max())
print("  N vertices:", G.vcount())

assert edges_arr[:, 0].max() < G.vcount()
assert edges_arr[:, 1].max() < G.vcount()

G.add_edges(edges)

# ============================
# CHECK 3: self-loops
# ============================

self_loops = [e.index for e in G.es if e.tuple[0] == e.tuple[1]]
print("CHECK self-loops:")
print("  N self-loops:", len(self_loops))
if len(self_loops) > 0:
    print("  Example self-loop:", self_loops[0], G.es[self_loops[0]].tuple)

# ============================
# Edge attributes
# ============================

G.es["connectivity"] = edges
G.es["nkind"] = edge_nkind
G.es["radius"] = radius
G.es["diameter"] = [r * 2 for r in radius]
G.es["length"] = lengths

# ============================
# 3) Geometry (memory-safe)
# ============================

edge_geometry_df.columns = ["x", "y", "z"]

x = edge_geometry_df["x"].to_numpy(dtype=np.float32)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32)

starts = geom_index_df[0].to_numpy(dtype=np.int64)
ends   = geom_index_df[1].to_numpy(dtype=np.int64)

assert int(ends[-1]) <= len(edge_geometry_df)

edge_geometries = []
append_geom = edge_geometries.append

for s, e in zip(starts, ends):
    geom = np.column_stack((x[s:e], y[s:e], z[s:e]))
    append_geom(geom)

G.es["geometry"] = edge_geometries

# ============================
# 3b) Tortuous length & tortuosity (ROBUST)
# ============================

length_tortuous = np.zeros(G.ecount(), dtype=np.float32)
straight_dist   = np.zeros(G.ecount(), dtype=np.float32)

for edge_id, geom in enumerate(G.es["geometry"]):
    if len(geom) < 2:
        length_tortuous[edge_id] = 0.0
        straight_dist[edge_id] = 0.0
        continue

    # tortuous length
    length_tortuous[edge_id] = np.sum(
        np.linalg.norm(np.diff(geom, axis=0), axis=1)
    )

    # straight distance FROM GEOMETRY (not nodes!)
    straight_dist[edge_id] = np.linalg.norm(geom[-1] - geom[0])

tortuosity = np.full(G.ecount(), np.nan, dtype=np.float32)
mask = straight_dist > 0
tortuosity[mask] = length_tortuous[mask] / straight_dist[mask]

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()

print("CHECK tortuosity:")
print("  N edges:", G.ecount())
print("  N straight_dist == 0:", np.sum(~mask))

# ============================
# 4) Column type checks
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
# 5) Save pickle
# ============================

out_path = f"/home/admin/Ana/MicroBrain/output{graph_number}/{graph_number}_igraph_fullGeom.pkl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Graph saved at:", out_path)
print("=== DONE ===")
