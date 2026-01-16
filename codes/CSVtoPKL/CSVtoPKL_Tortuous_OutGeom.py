"""
Build vascular graph with indexed geometry and robust tortuosity.
Includes edge-geometry annotation stored in data["annotation"] (point-wise).
Made by Ana.
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
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
MIN_STRAIGHT_DIST = 1.0  # µm

print("=== START CSV → PKL (OUTGEOM + annotation) ===")

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
edge_geometry_annotation_df = pd.read_csv(folder + "edge_geometry_annotation.csv", header=None)

edge_geometry_annotation_df = pd.read_csv(folder + "edge_geometry_annotation.csv", header=None)

print("CSVs loaded")

# ============================
# Create graph
# ============================

G = ig.Graph()
G.add_vertices(len(vertices_df))

G.vs["id"] = vertices_df[0].astype(int).tolist()

G.vs["coords"] = list(zip(
    coordinates_df[0].astype(np.float32),
    coordinates_df[1].astype(np.float32),
    coordinates_df[2].astype(np.float32),
))
<<<<<<< HEAD

G.vs["coords_image"] = list(zip(
    coordinates_images_df[0].astype(np.float32),
    coordinates_images_df[1].astype(np.float32),
    coordinates_images_df[2].astype(np.float32),
))

G.vs["annotation"] = annotation_vertex_df[0].astype(np.int32).tolist()
=======
G.vs["vertex_annotation"] = annotation_vertex_df[0].astype(int).tolist()
>>>>>>> 480ff5f (gt with edge annotation)
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()

# ============================
# Edges
# ============================

edges = []
edge_nkind = []
edge_radius = []
edge_length = []

for i, row in edges_df.iterrows():
    s, t = int(row[0]), int(row[1])

    nkind = 4
    if int(artery_df[0][i]) == 1:
        nkind = 2
    elif int(vein_df[0][i]) == 1:
        nkind = 3

    edges.append((s, t))
    edge_nkind.append(int(nkind))
    edge_radius.append(float(radii_df[0][i]))
    edge_length.append(float(length_df[0][i]))

G.add_edges(edges)

G.es["nkind"] = edge_nkind
G.es["radius"] = np.asarray(edge_radius, dtype=np.float32).tolist()
G.es["diameter"] = (2.0 * np.asarray(edge_radius, dtype=np.float32)).tolist()
G.es["length"] = np.asarray(edge_length, dtype=np.float32).tolist()

# ============================
# Geometry (indexed)
# ============================

x = edge_geometry_df[0].to_numpy(dtype=np.float32)
y = edge_geometry_df[1].to_numpy(dtype=np.float32)
z = edge_geometry_df[2].to_numpy(dtype=np.float32)

G.es["geom_start"] = geom_index_df[0].astype(np.int64).tolist()
G.es["geom_end"] = geom_index_df[1].astype(np.int64).tolist()

# ============================
# Edge-geometry annotation (point-wise)
# ============================

ann_geom = edge_geometry_annotation_df[0].to_numpy(dtype=np.int32)

if len(ann_geom) != len(x):
    raise ValueError(
        f"edge_geometry_annotation length ({len(ann_geom)}) != geometry coords length ({len(x)}). "
        "Check your CSV exports."
    )

# ============================
# Tortuosity (ROBUST)
# ============================

length_tortuous = np.zeros(G.ecount(), dtype=np.float32)
straight_dist = np.zeros(G.ecount(), dtype=np.float32)

for e in range(G.ecount()):
    s = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])

    if en - s < 2:
        continue

    dx = np.diff(x[s:en])
    dy = np.diff(y[s:en])
    dz = np.diff(z[s:en])

    length_tortuous[e] = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz))
    straight_dist[e] = np.sqrt(
        (x[en - 1] - x[s]) ** 2 +
        (y[en - 1] - y[s]) ** 2 +
        (z[en - 1] - z[s]) ** 2
    )

tortuosity = np.full(G.ecount(), np.nan, dtype=np.float32)
mask = straight_dist >= float(MIN_STRAIGHT_DIST)
tortuosity[mask] = length_tortuous[mask] / straight_dist[mask]

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()

print("Tortuosity computed")
print("  NaN:", int(np.sum(~mask)))
print("  Max:", float(np.nanmax(tortuosity)))

edge_annotation = edge_geometry_annotation_df[0].to_numpy(dtype=np.int32) # geom attr

# ============================
# Save
# ============================

data = {
    "graph": G,
    "coords": {"x": x, "y": y, "z": z},
<<<<<<< HEAD
    "annotation": ann_geom,  # <-- point-wise annotation for the geometry arrays
=======
    "edge_annotation": edge_annotation
>>>>>>> 480ff5f (gt with edge annotation)
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved graph to:", out_path)
print("=== DONE ===")
