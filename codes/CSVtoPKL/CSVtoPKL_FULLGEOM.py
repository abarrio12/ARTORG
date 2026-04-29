"""
Code written by Sofia to read the csv files extracted from Renier and transform them into a pickles file
- Extended to include full tortuous geometry per edge
"""

import pandas as pd
import igraph as ig
import pickle
import numpy as np
import os

graph_number = 18
folder = "/storage/homefs/ab25c720/MicroBrain/ParisGraph/CSV/"
out_path = f"/storage/homefs/ab25c720/MicroBrain/halfbrain_{graph_number}_igraph.pkl"

# Load CSV files
vertices_df = pd.read_csv(folder + "vertices.csv", header=None)  # Nodes
coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None)  # Coordinates nodes atlas version
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None)  # Coordinates nodes from Image

edges_df = pd.read_csv(folder + "edges.csv", header=None)        # Edges
length_df = pd.read_csv(folder + "length.csv", header=None)     # Extra edge attributes
radii_df = pd.read_csv(folder + "radii_edge.csv", header=None)      # More edge attributes
vein_df = pd.read_csv(folder + "vein.csv", header=None)    # Label for vein edges
artery_df = pd.read_csv(folder + "artery.csv", header=None)         # Label for artery edges
radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None)         # More edge attributes
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None)         # Annotation from ABA
distance_to_surface_df = pd.read_csv(folder + "distance_to_surface.csv", header=None) #distance to Surface

# Geometry CSVs
geom_index_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)
edge_geometry_radii_df = pd.read_csv(folder + "edge_geometry_radii.csv", header=None)

print("CSVs loaded")

# Create an empty graph
G = ig.Graph()
G.add_vertices(len(vertices_df))

# Assign vertex attributes
G.vs["id"] = vertices_df[0].tolist()
G.vs["coords"] = list(zip(coordinates_df[0], coordinates_df[1], coordinates_df[2]))
G.vs["coords_image"] = list(zip(coordinates_images_df[0], coordinates_images_df[1], coordinates_images_df[2]))
G.vs["annotation"] = annotation_vertex_df[0].tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].tolist()
G.vs["radii"] = radii_vertex_df[0].tolist()

# Add edges and attributes
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

G.add_edges(edges)
G.es["nkind"] = edge_nkind
G.es["radius"] = radius
G.es["diameter"] = [r * 2 for r in radius]
G.es["length_csv"] = lengths

print("Basic graph created")

# -----------------------------
# Add tortuous geometry
# -----------------------------
print("Adding geometry...")

edge_geometry_df.columns = ["x", "y", "z"]
x = edge_geometry_df["x"].to_numpy(dtype=np.float32)
y = edge_geometry_df["y"].to_numpy(dtype=np.float32)
z = edge_geometry_df["z"].to_numpy(dtype=np.float32)

starts = geom_index_df[0].to_numpy(dtype=np.int64)
ends = geom_index_df[1].to_numpy(dtype=np.int64)
r_global = edge_geometry_radii_df[0].to_numpy(dtype=np.float32)

coords_nodes = np.asarray(G.vs["coords_image"], dtype=np.float32)

points_list = []
diameters_list = []
lengths2_list = []
lengths_list = []
length_arr = []
tortuosity_arr = []

for i in range(len(edges)):
    s, e = starts[i], ends[i]
    pts = np.column_stack((x[s:e], y[s:e], z[s:e])).astype(np.float32)
    
    coords_source = coords_nodes[G.es[i].source]
    p0 = pts[0] if pts.shape[0] > 0 else np.zeros(3)
    p_last = pts[-1] if pts.shape[0] > 0 else np.zeros(3)
    
    r_pts = r_global[s:e].astype(np.float32)
    diams = (2.0 * r_pts).astype(np.float32)
    
    # Flip if needed (ensure start point aligns with source node)
    if pts.shape[0] >= 2 and not np.allclose(coords_source, p0, atol=1e-6):
        if np.allclose(coords_source, p_last, atol=1e-6):
            # Reverse both geometry and diameters to align with source
            pts = pts[::-1]
            diams = diams[::-1]
            r_pts = r_pts[::-1]  # Also reverse radii for consistency
    
    # Compute lengths
    if pts.shape[0] >= 2:
        lengths2 = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
        length_points_seg = np.zeros(pts.shape[0], dtype=np.float32)
        length_points_seg[:-1] = lengths2
        length_points_seg[-1] = lengths2[-1] if lengths2.size > 0 else 0.0
        length = np.sum(lengths2)
        straight_dist = np.linalg.norm(pts[-1] - pts[0])
        tortu = length / straight_dist if straight_dist > 0 else 1.0
    else:
        lengths2 = np.zeros(0, dtype=np.float32)
        length_points_seg = np.zeros(pts.shape[0] if pts.shape[0] > 0 else 1, dtype=np.float32)
        length = 0.0
        tortu = 1.0
    
    points_list.append(pts.tolist())
    diameters_list.append(diams.tolist())
    lengths2_list.append(lengths2.tolist())
    lengths_list.append(length_points_seg.tolist())
    length_arr.append(length)
    tortuosity_arr.append(tortu)

G.es["points"] = points_list
G.es["diameters"] = diameters_list
G.es["lengths2"] = lengths2_list
G.es["lengths"] = lengths_list
G.es["length"] = length_arr
G.es["tortuosity"] = tortuosity_arr

print("Geometry added")

# -----------------------------
# Convert to Gaia/MVN1 format (keep only final attributes)
# For compatibility with Pkl2vtp_MVN_GAIA.py
# -----------------------------
# ====================================================================
# ATTRIBUTE DEFINITIONS FOR GAIA FORMAT
# ====================================================================
# 
# VERTEX ATTRIBUTES:
#   - coords (v):        (x, y, z) coordinates of node in voxels
#   - degree (v):        number of edges connected to this node
#   - index (v):         unique node index (0 to nV-1)
#   - diameter (v):      vessel diameter at node (2 * radii), in voxels
#   - pBC (v):           pressure boundary condition (None = no BC, -1000 in VTP)
#   - pressure (v):      pressure value at node (None by default)
#   - annotation (v):    brain region annotation (ABA or atlas-based)
#
# EDGE ATTRIBUTES:
#   - connectivity (e):  tuple (source, target) of node indices
#   - nkind (e):         vessel type: 2=arteriole, 3=venule, 4=capillary
#   - points (e):        list of (x,y,z) tuples for tortuous geometry polyline
#   - diameters (e):     list of diameters, one per point in the geometry
#   - lengths2 (e):      list of segment lengths between consecutive points
#   - lengths (e):       list of (segment) lengths per point (for interpolation)
#   - length (e):        total edge length (sum of lengths2)
#   - diameter (e):      maximum diameter of the edge (max of diameters in Paris data) -> 
# https://github.com/ClearAnatomics/ClearMap/blob/71444a5c7456901f15e8d0ceb06fab72b74161df/ClearMap/Scripts/TubeMap.py#L433
# ====================================================================

print("Converting to Gaia format...")

# ===== VERTEX ATTRIBUTES =====
# Calculate degree
G.vs["degree"] = G.degree()

# Use coords_image as coords (mvn name format, do not confuse with coords of atlas)
G.vs["coords"] = G.vs["coords_image"]

# Convert radii to diameter
G.vs["diameter"] = [r * 2 for r in G.vs["radii"]]

# Add index
G.vs["index"] = list(range(G.vcount()))

# Set pBC (boundary condition) - default to None (will be -1000 in VTP export)
# If available in data, use it; otherwise set to None
G.vs["pBC"] = [None] * G.vcount()

# Set pressure - default to None
# If available in data, use it; otherwise set to None
G.vs["pressure"] = [None] * G.vcount()


# Keep annotation if present
if "annotation" not in G.vs.attributes():
    G.vs["annotation"] = [None] * G.vcount()

# ===== EDGE ATTRIBUTES =====
# Add connectivity (tuple of source, target)
G.es["connectivity"] = [tuple(map(int, e.tuple)) for e in G.es]

# Ensure length is sum of lengths2 if not already set
for i, (l2, l) in enumerate(zip(lengths2_list, length_arr)):
    if len(l2) > 0:
        G.es[i]["length"] = float(np.sum(l2))
    else:
        G.es[i]["length"] = l

# Add diameter  (max diameter tortuous nodes)
diam_edge = []
for diams in diameters_list:
    if len(diams) > 0:
        diam_edge.append(float(np.max(diams)))
    else:
        diam_edge.append(float("nan"))
G.es["diameter"] = diam_edge

# Metadata
G["unit"] = "vox" # coords/length unit
G["coord_space"] = "image"
G["diameter_unit"] = "vox"
G["resolution_image_um_per_voxel"] = [1.625, 1.625, 2.5]
G["resolution_atlas_um_per_voxel"] = [25.0, 25.0, 25.0]

# ===== KEEP ONLY FINAL ATTRIBUTES =====
keep_v = {"coords", "degree", 
          "index", "diameter", "annotation"}
keep_e = {"connectivity", "diameter", "diameters", "length", "lengths",
          "lengths2", "nkind", "points"}

# Remove extra vertex attributes
for attr in list(G.vs.attributes()):
    if attr not in keep_v:
        del G.vs[attr]

# Remove extra edge attributes
for attr in list(G.es.attributes()):
    if attr not in keep_e:
        del G.es[attr]

print("Converted to Gaia format")

# Quick checks
print(f"Graph: {G.vcount()} vertices, {G.ecount()} edges")
print("V attrs:", sorted(list(G.vs.attributes())))
print("E attrs:", sorted(list(G.es.attributes())))
print("Units:", G["unit"], G["diameter_unit"])

# Sanity check: verify required attrs exist
required_v = {"coords", "degree", "index", "diameter"}
required_e = {"connectivity", "nkind", "diameter", "diameters", "length", "lengths2", "points"}
missing_v = required_v - set(G.vs.attributes())
missing_e = required_e - set(G.es.attributes())
if missing_v:
    print(f"WARNING: Missing vertex attributes: {missing_v}")
if missing_e:
    print(f"WARNING: Missing edge attributes: {missing_e}")

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(G, f)
print(f'Graph saved in {out_path}')


DO_CONVERT_TO_UM = True

if DO_CONVERT_TO_UM:
    from vox_to_um import load_and_convert

    out_path_um = out_path.replace(".pkl", "_um.pkl")

    load_and_convert(
        out_path,
        out_pkl_path_um=out_path_um,
        res_um_per_vox=(1.625, 1.625, 2.5)
    )