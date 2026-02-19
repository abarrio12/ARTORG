"""
BUILD FULLGEOM graph from CSVs - MEMORY OPTIMIZED
- Each edge stores its full geometry in G.es["points"]. 
- Also computes length_tortuous + tortuosity from the points.
- Optimized to avoid RAM duplication

Take into account that this code store the information in the edges/vertices. 
If you are using a big graph, most likely it will crash. That is the reason
building outside arrays with the geometry and other interesting data was 
implemented (outgeom)

Author: Ana Barrio
Date: January 2026
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
MAX_EDGES = None  # None = all edges

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
        I1 = np.asarray(G.vs[tgt]["coords_image"], dtype=np.float64)
        
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



    def get_points_um(G, ei, scale):
        s = G.es[ei]["p_start"]
        t = G.es[ei]["p_end"]
        pts = G["P"][s:t]
        return pts * scale

# -------------------------------------------------------------------------

print("=== START CSV → PKL (FULLGEOM) - MEMORY OPTIMIZED ===")

# -----------------------------
# Load CSVs (use dtypes to reduce RAM)
# -----------------------------
print("\n=== Loading CSVs ===")
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
print("\n=== Creating graph structure ===")
n_vertices = len(vertices_df)
G = ig.Graph()
G.add_vertices(n_vertices)

# Vertex attributes
G.vs["id"] = vertices_df[0].astype(np.int64).tolist()
del vertices_df
gc.collect()

coords = np.column_stack([
    coordinates_df[0].to_numpy(np.float32, copy=False),
    coordinates_df[1].to_numpy(np.float32, copy=False),
    coordinates_df[2].to_numpy(np.float32, copy=False),
])

G.vs["coords"] = [np.asarray(row) for row in coords]
del coordinates_df, coords
gc.collect()

coords_img = np.column_stack([
    coordinates_images_df[0].to_numpy(np.float32, copy=False),
    coordinates_images_df[1].to_numpy(np.float32, copy=False),
    coordinates_images_df[2].to_numpy(np.float32, copy=False),
])

G.vs["coords_image"] = [np.asarray(row) for row in coords_img]
del coordinates_images_df, coords_img
gc.collect()

G.vs["annotation"] = annotation_vertex_df[0].astype(np.int32).tolist()
del annotation_vertex_df
gc.collect()

G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
del distance_to_surface_df
gc.collect()

G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()
del radii_vertex_df
gc.collect()

# -----------------------------
# Checks: coords spaces
# -----------------------------
coords_check = np.asarray(G.vs["coords"], float)
coords_img_check = np.asarray(G.vs["coords_image"], float)

print("coords max:", coords_check.max(axis=0))
print("coords_image max:", coords_img_check.max(axis=0))
print("ratio img/coords:", coords_img_check.max(axis=0) / coords_check.max(axis=0))

del coords_check, coords_img_check
gc.collect()

# -----------------------------
# Build edges (TEST cap)
# -----------------------------
print("\n=== Building edges ===")
if MAX_EDGES is not None:
    edges_df = edges_df.iloc[:MAX_EDGES].reset_index(drop=True)
    length_df = length_df.iloc[:MAX_EDGES].reset_index(drop=True)
    radii_df = radii_df.iloc[:MAX_EDGES].reset_index(drop=True)
    vein_df = vein_df.iloc[:MAX_EDGES].reset_index(drop=True)
    artery_df = artery_df.iloc[:MAX_EDGES].reset_index(drop=True)
    geom_index_df = geom_index_df.iloc[:MAX_EDGES].reset_index(drop=True)

n_edges = len(edges_df)
print(f"Processing {n_edges} edges")

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
del edges
gc.collect()

# Edge attributes (scalars - efficient)
nkind = np.full(n_edges, 4, dtype=np.int8)
nkind[artery_df[0].to_numpy(np.int8) == 1] = 2
nkind[vein_df[0].to_numpy(np.int8) == 1] = 3

radius_edge = radii_df[0].to_numpy(np.float32, copy=False)
lengths_edge = length_df[0].to_numpy(np.float32, copy=False)

G.es["nkind"] = nkind.tolist()
G.es["radius"] = radius_edge.tolist()
G.es["diameter"] = (2.0 * radius_edge).tolist()
G.es["length_csv"] = lengths_edge.tolist()

# Free memory from edge CSVs
del radii_df, vein_df, artery_df, length_df, edges_df, nkind, radius_edge
gc.collect()

# -----------------------------
# Geometry FULL (points per edge) - OPTIMIZED
# -----------------------------
print("\n=== Building geometry (memory-optimized: edge by edge) ===")

edge_geometry_df.columns = ["x", "y", "z"]

x = edge_geometry_df["x"].to_numpy(dtype=np.float32, copy=False)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32, copy=False)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32, copy=False)

starts = geom_index_df[0].to_numpy(dtype=np.int64, copy=False)
ends   = geom_index_df[1].to_numpy(dtype=np.int64, copy=False)

r_global = edge_geometry_radii_df[0].to_numpy(dtype=np.float32, copy=False)

assert int(ends[-1]) <= len(edge_geometry_df), "Last end index exceeds points length"
assert int(ends[-1]) <= len(r_global), "Last end index exceeds radii length"

# Get node coords once (avoid repeated lookups)
coords_nodes = np.asarray(G.vs["coords"], dtype=np.float32)

# arrays of source/target nodes
edges_sources = edges_arr[:, 0]
edges_targets = edges_arr[:, 1]

# Scalar arrays (these are efficient)
length_arr = np.zeros(n_edges, dtype=np.float32)
tortuosity = np.ones(n_edges, dtype=np.float32)

# Process edges one by one (NO intermediate lists)
for i, (s, e, src, tg) in enumerate(zip(starts, ends, edges_sources, edges_targets)):
    if i % 50000 == 0:
        print(f"  Processing edge {i}/{n_edges}")
    
    # Get geometry for this edge
    pts = np.column_stack((x[s:e], y[s:e], z[s:e])).astype(np.float32, copy=False)

    coords_source = coords_nodes[src]
    
    p0 = pts[0]
    p_last = pts[-1]
    
    # Get radii/diameters
    r_pts = r_global[s:e].astype(np.float32, copy=False)
    diams = (2.0 * r_pts).astype(np.float32, copy=False)

    # Check orientation (flip if needed)
    if not np.allclose(coords_source, p0, atol=1e-6):
        if np.allclose(coords_source, p_last, atol=1e-6):
            pts = pts[::-1].copy()
            diams = diams[::-1].copy()

    # Compute lengths
    if pts.shape[0] >= 2:
        lengths2 = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
        
        length_points_seg = np.empty(pts.shape[0], dtype=np.float32)
        length_points_seg[:-1] = lengths2
        length_points_seg[-1] = lengths2[-1]

        length = np.sum(lengths2)
        straight_dist = np.linalg.norm(pts[-1] - pts[0])
        tortu = length / straight_dist if straight_dist > 0 else 1.0
    else:
        length_points_seg = np.zeros(pts.shape[0], dtype=np.float32)
        lengths2 = np.zeros(0, dtype=np.float32)
        length = 0.0
        tortu = 1.0
    
    # ASSIGN DIRECTLY TO EDGE (no intermediate lists)
    G.es[i]["points"] = pts.tolist()
    G.es[i]["diameters"] = diams.tolist()
    G.es[i]["lengths2"] = lengths2.tolist()
    G.es[i]["lengths"] = length_points_seg.tolist()
    
    # Store scalars in arrays
    length_arr[i] = length
    tortuosity[i] = tortu

# Assign scalar arrays
G.es["length"] = length_arr.tolist()
G.es["tortuosity"] = tortuosity.tolist()

# Free geometry memory
del edge_geometry_df, x, y, z, r_global, starts, ends, geom_index_df, edge_geometry_radii_df
del edges_arr, edges_sources, edges_targets, coords_nodes, length_arr, tortuosity, lengths_edge
gc.collect()

print("Geometry assignment complete!")

# -----------------------------
# CHECKS
# -----------------------------
print("\n=== Running checks ===")

# Data structure check
k = 0
pts_0 = np.asarray(G.es[k]["points"])
diams_0 = np.asarray(G.es[k]["diameters"])
lengths_0 = np.asarray(G.es[k]["lengths"])
lengths2_0 = np.asarray(G.es[k]["lengths2"])
print(f"Edge 0 shapes: points={pts_0.shape}, diameters={diams_0.shape}, "
      f"lengths={lengths_0.shape}, lengths2={lengths2_0.shape}")

# Edge consistency check
bad = 0
for e in range(G.ecount()):
    pts = np.asarray(G.es[e]["points"])
    d   = G.es[e]["diameters"]
    l_pt = G.es[e]["lengths"]
    lengths2  = G.es[e]["lengths2"]
    
    n = pts.shape[0]
    if len(d) != n or len(l_pt) != n or len(lengths2) != max(n-1, 0):
        bad += 1

print("Edges with inconsistent per-point arrays:", bad)

# Length check
e = 0
print(f"Edge 0 length check: sum(lengths2)={np.sum(G.es[e]['lengths2']):.4f}, "
      f"length={G.es[e]['length']:.4f}")

# Space report
points_space_report(G, n_edges=min(5000, G.ecount()))

# -----------------------------
# Save
# -----------------------------
print("\n=== Saving graph ===")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Option 1: Regular pickle
with open(out_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Graph saved at: {out_path}")

# Option 2: Compressed (uncomment if you want smaller file size)
# import gzip
# with gzip.open(out_path + ".gz", "wb", compresslevel=9) as f:
#     pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
# print(f"Graph saved at: {out_path}.gz")

print("=== DONE ===")
print(f"Final graph: {G.vcount()} vertices, {G.ecount()} edges")
