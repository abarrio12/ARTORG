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
# store as tuples (stable + Gaia-friendly)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
G.vs["coords"] = [np.asarray(row) for row in coords]

coords_img = np.column_stack([
    coordinates_images_df[0].to_numpy(np.float32, copy=False),
    coordinates_images_df[1].to_numpy(np.float32, copy=False),
    coordinates_images_df[2].to_numpy(np.float32, copy=False),
])
G.vs["coords_image"] = [np.asarray(row) for row in coords_img]

G.vs["annotation"] = annotation_vertex_df[0].astype(np.int32).tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()


# -----------------------------
# Checks: node duplicates
# -----------------------------
n_unique = len({tuple(c) for c in G.vs["coords"]})
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

edges = list(zip(edges_df[0].astype(np.int64), edges_df[1].astype(np.int64))) # Nx2 

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
length_list = []


# arrays of source/target nodes of polyline + coords (array of coordinates)
edges_sources = edges_arr[:,0]
edges_targets = edges_arr[:,1]

# arrays to compute tortuosity
length_tortuous_arr= np.zeros(n_edges, dtype=np.float32)
tortuosity = np.ones(n_edges, dtype=np.float32) # Initialize in 1 (min tortuosity)



for i, (s, e, src, tg) in enumerate(zip(starts, ends, edges_sources, edges_targets)):
    # Nx3 float32 array
    pts = np.column_stack((x[s:e], y[s:e], z[s:e])).astype(np.float32, copy=False)

    coords_source = coords[src] #coords source node
    p0 = np.asarray(pts[0], dtype=np.float32)
    p_last = np.asarray(pts[-1], dtype=np.float32)
    
    
    # N radii -> N diameters
    r_pts = r_global[s:e].astype(np.float32, copy=False)
    diams = (2.0 * r_pts).astype(np.float32, copy=False)


    # Usamos np.allclose para evitar errores de precisión decimal
    if not np.allclose(coords_source, p0, atol=1e-6):
        # Si no coincide con el primero, verificamos si coincide con el último
        if np.allclose(coords_source, p_last, atol=1e-6):
            # Invertimos el orden de los puntos para que el inicio sea el source
            pts = pts[::-1].copy()
            diams = diams[::-1].copy()


    # pts: (N,3)
    # lengths2: (N-1) segment lengths
    # lenght: (N)
    if pts.shape[0] >= 2:
        #lengths2 (per segment)
        lengths2 = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)  # (N-1)
        # length (per node)
        length = np.empty(pts.shape[0], dtype=np.float32)  # (N,)
        length[:-1] = lengths2
        length[-1]  = lengths2[-1] # point 0 --> seg 0-1, point 1 --> seg 1-2. If not [-1], point 0 --> "fictional", point 1 --> seg 0-1 

        # Calculate tortuosity
        length_tortuous = np.sum(lengths2)
        straight_dist = np.linalg.norm(pts[-1] - pts[0])
        tortu = length_tortuous / straight_dist if straight_dist > 0 else 1


    else:
        length = np.zeros(pts.shape[0], dtype=np.float32)
        lengths2 = np.zeros(0, dtype=np.float32)
        length_tortuous = 0.0
        tortu = 1
        
    # because each edge has != num of points/diams, we can't use normal np, should use a list of arrays. Using directly a list with .append is better.    
    points_list.append(pts)
    diameters_list.append(diams)
    lengths2_list.append(lengths2) 
    length_list.append(length) # list of vector value global length of edge 

    # because it is a scalar per edge, we can use normal np
    tortuosity[i] = tortu
    length_tortuous_arr[i] = length_tortuous

# --- CHECK OF LENGHTS START ---
# GTtoCSV uses euclidian distance in straight line between A and B, not intermediate points
# This check helps us see if the length from CSV (no points) and the tortuous length (using points) is the same

# 'lengths_edge' loaded from length.csv at the start
diff_vs_csv = np.abs(length_tortuous_arr - lengths_edge)
print("\n=== validation tortuous/csv legth ===")
print(f"Mean Diff: {np.mean(diff_vs_csv):.4f} um")
print(f"Max Diff: {np.max(diff_vs_csv):.4f} um")

# If big diff, CSV Si la diferencia es grande, es porque el CSV era la distancia recta
if np.mean(diff_vs_csv) > 0.1:
    print("CONFIRMADO: El CSV original no coincide con la suma de segmentos.")
    # Check extra contra la distancia recta
    # (Podrías guardar dist_recta en un array para comparar aquí también)

#  --- CHECK OF LENGHTS END ---



# --- GRAPH ASSIGNATION ---

# Complies with Gaias cut the graph code attributes names
G.es["points"] = points_list
G.es["diameters"] = diameters_list
G.es["lengths2"] = lengths2_list
G.es["length"] = length_list #vector

G.es["length_tortuous"] = length_tortuous_arr.tolist()
G.es["tortuosity"] = tortuosity.tolist()



# CHECKS

# Data structure
k = 0
print(points_list[k].shape, diameters_list[k].shape, length_list[k].shape, lengths2_list[k].shape)
# Esperado: (N,3) (N,) (N,) (N-1,)

# Edge info (Should be 0 or really close)
bad = 0
for e in range(G.ecount()):
    pts = G.es[e]["points"]
    d   = G.es[e]["diameters"]
    l_vec   = G.es[e]["length"]
    lengths2  = G.es[e]["lengths2"]
    n = pts.shape[0]
    if len(d)!=n or len(l_vec)!=n or len(lengths2)!=max(n-1,0):
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

def points_space_report(G, n_edges=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = G.ecount()
    idxs = rng.choice(m, size=min(n_edges, m), replace=False)    
    dA = []
    dI = []    
    for ei in idxs:
        e = G.es[ei]
        pts = np.asarray(e["points"], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue        
        p0 = pts[0]
        p1 = pts[-1]
        src, tgt = e.source, e.target        
        A0 = np.asarray(G.vs[src]["coords"], dtype=np.float64)
        A1 = np.asarray(G.vs[tgt]["coords"], dtype=np.float64)
        I0 = np.asarray(G.vs[src]["coords_image"], dtype=np.float64)
        I1 = np.asarray(G.vs[tgt]["coords_image"], dtype=np.float64)        # distancia del extremo de la polilínea al extremo de nodo (min sobre p0/p1 y src/tgt)
        dA.append(min(np.linalg.norm(p0-A0), np.linalg.norm(p0-A1),
                      np.linalg.norm(p1-A0), np.linalg.norm(p1-A1)))
        dI.append(min(np.linalg.norm(p0-I0), np.linalg.norm(p0-I1),
                      np.linalg.norm(p1-I0), np.linalg.norm(p1-I1)))    
    dA = np.array(dA)
    dI = np.array(dI)   
    print("Samples:", len(dA))
    print("Median dist to ATLAS (coords):", float(np.median(dA)))
    print("Median dist to IMAGE (coords_image):", float(np.median(dI)))
    print("Percent where ATLAS closer:", float(np.mean(dA < dI) * 100), "%")
    print("Percent where IMAGE closer:", float(np.mean(dI < dA) * 100), "%")

points_space_report(G, n_edges=5000)

X = np.asarray(G.vs["coords_image"], dtype=np.float64)  # imagen
Y = np.asarray(G.vs["coords"], dtype=np.float64)        # atlas
Xh = np.c_[X, np.ones(len(X))]                          # (N,4)
M, residuals, rank, s = np.linalg.lstsq(Xh, Y, rcond=None)
Yhat = Xh @ M
err = np.linalg.norm(Yhat - Y, axis=1)
print("rank:", rank)
print("median err:", float(np.median(err)))
print("max err:", float(err.max()))
print("residuals:", residuals[:10] if hasattr(residuals, "__len__") else residuals)
print("M (image -> atlas):\n", M)




