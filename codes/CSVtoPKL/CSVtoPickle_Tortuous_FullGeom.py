"""
BUILD FULLGEOM graph from CSVs
- Each edge stores its full geometry in G.es["points"] (Gaia-compatible).
- Also computes length_tortuous + tortuosity from the points.
"""

import os
import pickle
import numpy as np
import pandas as pd
import igraph as ig


# -----------------------------
# Params
# -----------------------------
graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"
out_path = f"/home/admin/Ana/MicroBrain/output{graph_number}/{graph_number}_igraph_FULLGEOM_SUB.pkl"

# If you're testing, you can cap edges to avoid killing the process:
# set to None for full (WARNING: huge)
MAX_EDGES = 200000  # e.g. 200000  (for a test)

print("=== START CSV → PKL (FULLGEOM) ===")


# -----------------------------
# Load CSVs (use dtypes to reduce RAM)
# -----------------------------
vertices_df = pd.read_csv(folder + "vertices.csv", header=None, dtype=np.int64)

coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None, dtype=np.float32)
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None, dtype=np.float32)

edges_df  = pd.read_csv(folder + "edges.csv",  header=None, dtype=np.int64)
length_df = pd.read_csv(folder + "length.csv", header=None, dtype=np.float32)
radii_df  = pd.read_csv(folder + "radii_edge.csv", header=None, dtype=np.float32)

vein_df   = pd.read_csv(folder + "vein.csv",   header=None, dtype=np.int8)
artery_df = pd.read_csv(folder + "artery.csv", header=None, dtype=np.int8)

radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None, dtype=np.float32)
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None, dtype=np.int32)
distance_to_surface_df = pd.read_csv(folder + "distance_to_surface.csv", header=None, dtype=np.float32)

geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None, dtype=np.int64)
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None, dtype=np.float32)

edge_geometry_radii_df = pd.read_csv(folder + "edge_geometry_radii.csv", header=None, dtype=np.float32)


print("END reading CSVs")


# -----------------------------
# Create graph
# -----------------------------
n_vertices = len(vertices_df)
G = ig.Graph()
G.add_vertices(n_vertices)

# Vertex attributes
G.vs["id"] = vertices_df[0].astype(np.int64).tolist()

coords = np.column_stack([
    coordinates_df[0].to_numpy(np.float32, copy=False),
    coordinates_df[1].to_numpy(np.float32, copy=False),
    coordinates_df[2].to_numpy(np.float32, copy=False),
])
# store as tuples (stable + Gaia-friendly)
G.vs["coords"] = [tuple(row) for row in coords]

coords_img = np.column_stack([
    coordinates_images_df[0].to_numpy(np.float32, copy=False),
    coordinates_images_df[1].to_numpy(np.float32, copy=False),
    coordinates_images_df[2].to_numpy(np.float32, copy=False),
])
G.vs["coords_image"] = [tuple(row) for row in coords_img]

G.vs["annotation"] = annotation_vertex_df[0].astype(np.int32).tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()


# -----------------------------
# Checks: node duplicates
# -----------------------------
n_unique = len({c for c in G.vs["coords"]})
print("CHECK nodes:")
print("  N nodos:", n_vertices)
print("  UNIQUE N coords:", n_unique)
print("  N duplicated:", n_vertices - n_unique)


# -----------------------------
# Build edges (TEST cap)
# -----------------------------
if MAX_EDGES is not None:
    edges_df = edges_df.iloc[:MAX_EDGES].reset_index(drop=True)
    length_df = length_df.iloc[:MAX_EDGES].reset_index(drop=True)
    radii_df = radii_df.iloc[:MAX_EDGES].reset_index(drop=True)
    vein_df = vein_df.iloc[:MAX_EDGES].reset_index(drop=True)
    artery_df = artery_df.iloc[:MAX_EDGES].reset_index(drop=True)
    geom_index_df = geom_index_df.iloc[:MAX_EDGES].reset_index(drop=True)

n_edges = len(edges_df)

edges = list(zip(edges_df[0].astype(np.int64), edges_df[1].astype(np.int64)))

# Check edge indices
edges_arr = np.asarray(edges, dtype=np.int64)
print("CHECK edges indices:")
print("  Min source:", edges_arr[:, 0].min())
print("  Max source:", edges_arr[:, 0].max())
print("  Min target:", edges_arr[:, 1].min())
print("  Max target:", edges_arr[:, 1].max())
print("  N vertices:", G.vcount())

assert edges_arr[:, 0].max() < G.vcount()
assert edges_arr[:, 1].max() < G.vcount()

G.add_edges(edges)

# Edge attributes
nkind = np.full(n_edges, 4, dtype=np.int8)  # capillary default
nkind[artery_df[0].to_numpy(np.int8) == 1] = 2
nkind[vein_df[0].to_numpy(np.int8) == 1] = 3

radius_edge = radii_df[0].to_numpy(np.float32, copy=False)
lengths_edge = length_df[0].to_numpy(np.float32, copy=False)

G.es["nkind"] = nkind.tolist()
G.es["radius"] = radius_edge.tolist()
G.es["diameter"] = (2.0 * radius_edge).tolist()
G.es["length"] = lengths_edge.tolist()


# -----------------------------
# Geometry FULL (points per edge)
# -----------------------------
edge_geometry_df.columns = ["x", "y", "z"]

x = edge_geometry_df["x"].to_numpy(dtype=np.float32, copy=False)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32, copy=False)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32, copy=False)

starts = geom_index_df[0].to_numpy(dtype=np.int64, copy=False)
ends   = geom_index_df[1].to_numpy(dtype=np.int64, copy=False)

r_global = edge_geometry_radii_df[0].to_numpy(dtype=np.float32, copy=False)

assert int(ends[-1]) <= len(edge_geometry_df), "Last end index exceeds points length"
assert int(ends[-1]) <= len(r_global), "Last end index exceeds radii length"

# Build points list (Gaia expects e['points'])
points_list = []
diameters_list = []
lengths2_list = []
lengths_list = []

append_points = points_list.append
append_diams = diameters_list.append
append_l2 = lengths2_list.append
append_l = lengths_list.append

for s, e in zip(starts, ends):
    # Nx3 float32 array
    pts = np.column_stack((x[s:e], y[s:e], z[s:e])).astype(np.float32, copy=False)

    # N radii -> N diameters
    r_pts = r_global[s:e].astype(np.float32, copy=False)
    diams = (2.0 * r_pts).astype(np.float32, copy=False)

    # lengths2: (N-1) segment lengths
    if pts.shape[0] >= 2:
        l2 = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)  # (N-1,)
        l = np.empty(pts.shape[0], dtype=np.float32)  # (N,)
        l[:-1] = l2
        l[-1]  = l2[-1] # point 0 --> seg 0-1, point 1 --> seg 1-2. If not [-1], point 0 --> "fictional", point 1 --> seg 0-1 

    else:
        l2 = np.zeros(0, dtype=np.float32)
        l = np.zeros(pts.shape[0], dtype=np.float32)

    append_points(pts)
    append_diams(diams)
    append_l2(l2)
    append_l(l)

# Complies with Gaias cut the graph code attributes names
G.es["points"] = points_list
G.es["diameters"] = diameters_list
G.es["lengths2"] = lengths2_list
G.es["lengths"] = lengths_list

# -----------------------------
# Tortuosity (from points)
# -----------------------------
length_tortuous = np.zeros(n_edges, dtype=np.float32)
straight_dist   = np.zeros(n_edges, dtype=np.float32)

for i, pts in enumerate(points_list):
    if pts.shape[0] < 2:
        length_tortuous[i] = 0.0
        straight_dist[i] = 0.0
        continue

    diffs = np.diff(pts, axis=0)
    length_tortuous[i] = np.sum(np.linalg.norm(diffs, axis=1))
    straight_dist[i] = np.linalg.norm(pts[-1] - pts[0])

tortuosity = np.full(n_edges, np.nan, dtype=np.float32)
mask = straight_dist > 0
tortuosity[mask] = length_tortuous[mask] / straight_dist[mask]

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()

print("CHECK tortuosity:")
print("  N edges:", n_edges)
print("  N straight_dist == 0:", int(np.sum(~mask)))
if np.any(mask):
    print("  Max tortuosity (finite):", float(np.nanmax(tortuosity)))


# CHECKS

# Data structure
k = 0
print(points_list[k].shape, diameters_list[k].shape, lengths_list[k].shape, lengths2_list[k].shape)
# Esperado: (N,3) (N,) (N,) (N-1,)

# Edge info (Should be 0 or really close)
bad = 0
for e in range(G.ecount()):
    pts = G.es[e]["points"]
    d   = G.es[e]["diameters"]
    l   = G.es[e]["lengths"]
    l2  = G.es[e]["lengths2"]
    n = pts.shape[0]
    if len(d)!=n or len(l)!=n or len(l2)!=max(n-1,0):
        bad += 1

print("Edges with inconsistent per-point arrays:", bad)

# Leghths (should be very similar)
e = 0
print(np.sum(G.es[e]["lengths2"]), G.es[e]["length_tortuous"])


# -----------------------------
# Save
# -----------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Graph saved at:", out_path)
print("=== DONE ===")
