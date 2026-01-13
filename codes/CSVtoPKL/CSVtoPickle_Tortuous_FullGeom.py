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
import gc


# -----------------------------
# Params
# -----------------------------
graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"
out_path = f"/home/admin/Ana/MicroBrain/output{graph_number}/{graph_number}_igraph_FULLGEOM_SUB.pkl"

# If you're testing, you can cap edges to avoid killing the process:
# set to None for full (WARNING: huge)
MAX_EDGES = 200000  # e.g. 200000  (for a test)

# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------
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

# -------------------------------------------------------------------------


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

# Check coords vs coords img 
coords = np.asarray(G.vs["coords"], float)
coords_img = np.asarray(G.vs["coords_image"], float)

print("coords max:", coords.max(axis=0))
print("coords_image max:", coords_img.max(axis=0))
print("ratio img/coords:", coords_img.max(axis=0) / coords.max(axis=0))

'''
# Checks node duplicates
n_unique = len({tuple(c) for c in G.vs["coords"]})
print("CHECK nodes:")
print("  N nodos:", n_vertices)
print("  UNIQUE N coords:", n_unique)
print("  N duplicated:", n_vertices - n_unique)
'''


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

G.es["length_csv"] = lengths_edge.tolist()

'''
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
length_points_seg_list = [] # assignment to each position its  segment: 0 -> seg0

# arrays of source/target nodes of polyline + coords (array of coordinates)
edges_sources = edges_arr[:,0]
edges_targets = edges_arr[:,1]

# arrays to compute tortuosity
length_arr= np.zeros(n_edges, dtype=np.float32)
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
    # lengths2: (N-1) segment lengths --> a-a1-a2-a3-b
    # lenght_tortuous: (N) --> a-b
    # we use l_seg to go from N-1 to N
    
    if pts.shape[0] >= 2:
        #lengths2 (per segment)
        lengths2 = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)  # (N-1)
        # length (per node)
        length_points_seg = np.empty(pts.shape[0], dtype=np.float32)  # (N,)
        length_points_seg[:-1] = lengths2
        length_points_seg[-1]  = lengths2[-1] # point 0 --> seg 0-1, point 1 --> seg 1-2. If not [-1], point 0 --> "fictional", point 1 --> seg 0-1 

        # Calculate tortuosity
        length = np.sum(lengths2) # scalar (N)
        straight_dist = np.linalg.norm(pts[-1] - pts[0])
        tortu = length / straight_dist if straight_dist > 0 else 1


    else:
        length_points_seg = np.zeros(pts.shape[0], dtype=np.float32)
        lengths2 = np.zeros(0, dtype=np.float32)
        length = 0.0
        tortu = 1.0
        
    # because each edge has != num of points/diams, we can't use normal np, should use a list of arrays. Using directly a list with .append is better.    
    points_list.append(pts)
    diameters_list.append(diams)
    lengths2_list.append(lengths2) 
    length_points_seg_list.append(length_points_seg)

    # because it is a scalar per edge, we can use normal np
    tortuosity[i] = tortu
    length_arr[i] = length #global length 

# --- CHECK OF LENGHTS START ---
# GTtoCSV uses euclidian distance in straight line between A and B, not intermediate points
# This check helps us see if the length from CSV (no points) and the tortuous length (using points) is the same

# 'lengths_edge' loaded from length.csv at the start
diff_vs_csv = np.abs(length_arr - lengths_edge)
print("\n=== validation tortuous/csv legth ===")
print(f"Mean Diff: {np.mean(diff_vs_csv):.4f} um")
print(f"Max Diff: {np.max(diff_vs_csv):.4f} um")

# If big diff, CSV Si la diferencia es grande, es porque el CSV era la distancia recta
if np.mean(diff_vs_csv) > 0.1:
    print("CONFIRMADO: El CSV original no coincide con la suma de segmentos.")
    # Check extra contra la distancia recta
    # (Podrías guardar dist_recta en un array para comparar aquí también)


# --- GRAPH ASSIGNATION ---

G.es["points"]      = points_list      # Nx3
G.es["diameters"]   = diameters_list    # N
G.es["lengths2"]    = lengths2_list      # (N-1)
G.es["lengths"]     = length_points_seg_list        # (N) per-point (Gaia)
G.es["length"]      = length_arr.tolist()   # scalar tortuous 
#G.es["length_csv"]  = lengths_edge.tolist()          # scalar from CSV
#G.es["tortuosity"]  = tortuosity.tolist()



# CHECKS

# Data structure
k = 0
print(points_list[k].shape, diameters_list[k].shape, length_points_seg_list[k].shape, lengths2_list[k].shape)


# Edge info (Should be 0 or really close)
bad = 0
for e in range(G.ecount()):
    pts = G.es[e]["points"]
    d   = G.es[e]["diameters"]
    l_edge  = G.es[e]["length"] #scalar
    l_pt = G.es[e]["lengths"]

    lengths2  = G.es[e]["lengths2"]
    n = pts.shape[0]
    if len(d) != n or len(l_pt) != n or len(lengths2) != max(n-1, 0):
        bad += 1


print("Edges with inconsistent per-point arrays:", bad)

# Legths (should be very similar) = sum(lengths2)
e = 0
print(np.sum(G.es[e]["lengths2"]), G.es[e]["length"])


points_space_report(G, n_edges=5000)
'''

# -----------------------------
# Geometry FULL (points per edge)  -- OPTION 2: BYTES+SHAPE (OOM SAFE)
# -----------------------------
edge_geometry_df.columns = ["x", "y", "z"]

x = edge_geometry_df["x"].to_numpy(dtype=np.float32, copy=False)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32, copy=False)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32, copy=False)

starts = geom_index_df[0].to_numpy(dtype=np.int64, copy=False)
ends   = geom_index_df[1].to_numpy(dtype=np.int64, copy=False)
r_global = edge_geometry_radii_df[0].to_numpy(dtype=np.float32, copy=False)

assert int(ends[-1]) <= len(edge_geometry_df), "Last end index exceeds points length"
assert int(ends[-1]) <= len(r_global), "Last end index exceeds radii length"# arrays of source/target nodes of polyline + coords (array of coordinates)

edges_sources = edges_arr[:, 0]
edges_targets = edges_arr[:, 1]# scalars per edge

length_arr  = np.zeros(n_edges, dtype=np.float32)
tortuosity  = np.ones(n_edges, dtype=np.float32)# IMPORTANT: store coords as array for fast access (your code uses coords[src])

sx, sy, sz = 1.625, 1.625, 2.5
scale = np.array([sx, sy, sz], dtype=np.float64)

# Here, "coords" must be (n_vertices, 3) float32
# If you currently have coords as list-of-arrays, make this once:
coords_nodes = np.asarray(G.vs["coords"], dtype=np.float32)
for i, (s, e, src, tg) in enumerate(zip(starts, ends, edges_sources, edges_targets)):
    if i % 50000 == 0 and i > 0:
        print(f"  Processing edge {i}/{n_edges}")    # Nx3 float32 array (vox)
    pts = np.stack((x[s:e], y[s:e], z[s:e]), axis=1).astype(np.float32, copy=False)    
    coords_source = coords_nodes[src]    
    p0 = pts[0]
    p_last = pts[-1]    # N radii -> N diameters
    r_pts = r_global[s:e].astype(np.float32, copy=False)
    diams = (2.0 * r_pts).astype(np.float32, copy=False)    # Orient so first point matches source (if possible)
    if not np.allclose(coords_source, p0, atol=1e-6):
        if np.allclose(coords_source, p_last, atol=1e-6):
            pts = pts[::-1].copy()
            diams = diams[::-1].copy()    # Compute lengths (still in vox here; if you want µm multiply pts*scale first)
    # If you want tortuosity/length in µm (recommended):
    pts_um = pts.astype(np.float64, copy=False) * scale    
    if pts_um.shape[0] >= 2:
        lengths2 = np.linalg.norm(np.diff(pts_um, axis=0), axis=1).astype(np.float32)  # (N-1)        
        length_points_seg = np.empty(pts_um.shape[0], dtype=np.float32)  # (N,)
        length_points_seg[:-1] = lengths2
        length_points_seg[-1]  = lengths2[-1]        
        length = float(lengths2.sum())
        straight_dist = float(np.linalg.norm(pts_um[-1] - pts_um[0]))
        tortu = (length / straight_dist) if straight_dist > 0 else 1.0
    else:
        lengths2 = np.zeros(0, dtype=np.float32)
        length_points_seg = np.zeros(pts_um.shape[0], dtype=np.float32)
        length = 0.0
        tortu = 1.0    # -----------------------------
    # STORE COMPACTLY PER EDGE (bytes + lengths)
    # -----------------------------
    # store points in µm (float32) to reduce size and to be consistent downstream
    pts_store = (pts_um.astype(np.float32, copy=False))    
    eobj = G.es[i]    # points
    eobj["points_bytes"] = pts_store.tobytes()
    eobj["points_n"] = int(pts_store.shape[0])  # number of points    # diameters
    di_store = diams.astype(np.float32, copy=False)
    eobj["diameters_bytes"] = di_store.tobytes()
    eobj["diameters_n"] = int(di_store.shape[0])    # lengths2
    l2_store = lengths2.astype(np.float32, copy=False)
    eobj["lengths2_bytes"] = l2_store.tobytes()
    eobj["lengths2_n"] = int(l2_store.shape[0])    # lengths (per-point)
    l_store = length_points_seg.astype(np.float32, copy=False)
    eobj["lengths_bytes"] = l_store.tobytes()
    eobj["lengths_n"] = int(l_store.shape[0])    # flags/scalars
    eobj["points_um"] = True
    tortuosity[i] = tortu
    length_arr[i] = length# assign scalar arrays
G.es["length"] = length_arr.tolist()
G.es["tortuosity"] = tortuosity.tolist()
print("Geometry assignment complete (bytes per edge).")

# -------------------------------------------------
# Optional: remove old list-based attrs if they exist
# -------------------------------------------------
for old in ("points", "diameters", "lengths", "lengths2"):
    if old in G.es.attributes():
        try:
            del G.es[old]
            print(f"Removed old edge attribute: {old}")
        except Exception as ex:
            print(f"Could not remove {old}: {ex}")

gc.collect()

# -------------------------------------------------
# Quick sanity check: reconstruct a random edge
# -------------------------------------------------
def edge_points_from_bytes(e):
    n = int(e["points_n"])
    return np.frombuffer(e["points_bytes"], dtype=np.float32).reshape(n, 3)

def edge_diams_from_bytes(e):
    n = int(e["diameters_n"])
    return np.frombuffer(e["diameters_bytes"], dtype=np.float32, count=n)

k = 0  # or np.random.randint(G.ecount())
e0 = G.es[k]
pts0 = edge_points_from_bytes(e0)
d0   = edge_diams_from_bytes(e0)
print("CHECK edge 0 reconstructed shapes:",
      "points", pts0.shape, "diams", d0.shape)
assert pts0.shape[0] == d0.shape[0], "Mismatch points vs diameters length!"# -------------------------------------------------
# Save
# -------------------------------------------------
print("\n=== Saving graph ===")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph saved at: {out_path}")
print("=== DONE ===")
print(f"Final graph: {G.vcount()} vertices, {G.ecount()} edges")









''''
# -----------------------------
# Save
# -----------------------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Graph saved at:", out_path)
print("=== DONE ===")





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

'''