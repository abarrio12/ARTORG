import pandas as pd
import igraph as ig
import numpy as np
import pickle
import os


graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"

# Bounding box (µm)
box = {
    "x": (1000, 2000),
    "y": (1500, 2500),
    "z": (1000, 2000),
}

vertices_df = pd.read_csv(folder + "vertices.csv", header=None)
coords_df   = pd.read_csv(folder + "coordinates_atlas.csv", header=None)

edges_df    = pd.read_csv(folder + "edges.csv", header=None)
length_df   = pd.read_csv(folder + "length.csv", header=None)
radii_df    = pd.read_csv(folder + "radii_edge.csv", header=None)
artery_df   = pd.read_csv(folder + "artery.csv", header=None)
vein_df     = pd.read_csv(folder + "vein.csv", header=None)

geom_idx_df = pd.read_csv(folder + "edge_geometry_indices.csv", header=None)

G = ig.Graph()
G.add_vertices(len(vertices_df))

G.vs["id"] = vertices_df[0].tolist()
G.vs["coords"] = list(zip(coords_df[0], coords_df[1], coords_df[2]))

edges = []
nkind = []
radius = []
lengths = []

for i, row in edges_df.iterrows():
    s, t = int(row[0]), int(row[1])
    edges.append((s, t))

    if artery_df[0][i] == 1:
        nkind.append(2)
    elif vein_df[0][i] == 1:
        nkind.append(3)
    else:
        nkind.append(4)

    radius.append(radii_df[0][i])
    lengths.append(length_df[0][i])

G.add_edges(edges)
G.es["nkind"] = nkind
G.es["radius"] = radius
G.es["diameter"] = [2*r for r in radius]
G.es["length"] = lengths

coords = np.array(G.vs["coords"])

inside_node = (
    (coords[:,0] >= box["x"][0]) & (coords[:,0] <= box["x"][1]) &
    (coords[:,1] >= box["y"][0]) & (coords[:,1] <= box["y"][1]) &
    (coords[:,2] >= box["z"][0]) & (coords[:,2] <= box["z"][1])
)

edges_inside = []
edges_across = []

for eid, (v0, v1) in enumerate(G.get_edgelist()):
    if inside_node[v0] and inside_node[v1]:
        edges_inside.append(eid)
    elif inside_node[v0] or inside_node[v1]:
        edges_across.append(eid)

edges_keep = np.unique(edges_inside + edges_across)
print("Edges kept:", len(edges_keep))

edge_geom_df = pd.read_csv(
    folder + "edge_geometry_coordinates.csv",
    header=None,
    names=["x","y","z"]
)

edge_geoms = {}

for eid in edges_keep:
    s, e = geom_idx_df.iloc[eid]
    geom = edge_geom_df.iloc[s:e].to_numpy(dtype=np.float32)
    edge_geoms[eid] = geom



G_sub = G.subgraph_edges(edges_keep, delete_vertices=True)


length_tort = []
tortuosity = []

for e in G_sub.es:
    old_eid = e.index  # cuidado: reindexado
    geom = edge_geoms[edges_keep[old_eid]]

    e["geometry"] = geom

    if len(geom) > 1:
        L = np.sum(np.linalg.norm(np.diff(geom, axis=0), axis=1))
        d = np.linalg.norm(geom[-1] - geom[0])
    else:
        L, d = 0.0, 0.0

    length_tort.append(L)
    tortuosity.append(L/d if d > 0 else np.nan)

G_sub.es["length_tortuous"] = length_tort
G_sub.es["tortuosity"] = tortuosity

out = f"/home/admin/Ana/MicroBrain/output{graph_number}/graph_{graph_number}_FULLGEOM_SUB.pkl"
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(out, "wb") as f:
    pickle.dump(G_sub, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved:", out)
print(G_sub.summary())
