from graph_tool.all import load_graph
import igraph as ig
import pickle
import numpy as np

# -----------------------------
# 1. LOAD THE GT GRAPH
# -----------------------------

gt_path = "/mnt/c/Users/Ana/OneDrive/Escritorio/ARTORG/igraph/my test codes/18_vessels_graph.gt"
g = load_graph(gt_path)

print("Loaded graph-tool graph:")
print("Nodes:", g.num_vertices())
print("Edges:", g.num_edges())


# -----------------------------
# 2. EXTRACT PROPERTIES FROM GT
# -----------------------------

# --- Vertex properties ---
coords          = g.vp["coordinates"]
coords_image    = g.vp["coordinates_atlas"]
radii_vertex    = g.vp["radii"]
annotation      = g.vp["annotation"]
dist_surface    = g.vp["distance_to_surface"]

# --- Edge properties ---
length          = g.ep["length"]
radius_edge     = g.ep["radii"]
artery          = g.ep["artery"]
vein            = g.ep["vein"]
geom_index      = g.ep["edge_geometry_indices"]

# --- GEOMETRY (graph property) ---
geom_all        = g.gp["edge_geometry_coordinates"]
# geom_all[i] = array Nx3 of points for curve i


# -----------------------------
# 3. BUILD IGRAPH GRAPH
# -----------------------------

G = ig.Graph()
G.add_vertices(g.num_vertices())

# NODE ATTRIBUTES
G.vs["coords"]             = [tuple(coords[v]) for v in g.vertices()]
G.vs["coords_image"]       = [tuple(coords_image[v]) for v in g.vertices()]
G.vs["radii"]              = [float(radii_vertex[v]) for v in g.vertices()]
G.vs["annotation"]         = [int(annotation[v]) for v in g.vertices()]
G.vs["distance_to_surface"]= [float(dist_surface[v]) for v in g.vertices()]

# ADD EDGES
edges = [(int(e.source()), int(e.target())) for e in g.edges()]
G.add_edges(edges)

# EDGE ATTRIBUTES
G.es["length"]  = [float(length[e]) for e in g.edges()]
G.es["radius"]  = [float(radius_edge[e]) for e in g.edges()]
G.es["diameter"] = [2*float(radius_edge[e]) for e in g.edges()]
G.es["artery"]  = [int(artery[e]) for e in g.edges()]
G.es["vein"]    = [int(vein[e]) for e in g.edges()]

# nkind (artery=2, vein=3, capillary=4)
nkind_list = []
for e in g.edges():
    if artery[e] == 1:
        nk = 2
    elif vein[e] == 1:
        nk = 3
    else:
        nk = 4
    nkind_list.append(nk)

G.es["nkind"] = nkind_list

# -----------------------------
# 4. ASSIGN GEOMETRY PER EDGE
# -----------------------------

edge_geometries = []

for e in g.edges():
    idx = int(geom_index[e])    # curve_id
    coords_array = geom_all[idx]   # array Nx3
    coords_list = [tuple(p) for p in coords_array]
    edge_geometries.append(coords_list)

G.es["geometry"] = edge_geometries

print("Example edge:", G.es[0].attributes())
print("Geometry points:", G.es["geometry"][0][:5])


# -----------------------------
# 5. SAVE AS PKL
# -----------------------------

out_path = "graph_from_gt.pkl"
with open(out_path, "wb") as f:
    pickle.dump(G, f)

print("Saved igraph graph to:", out_path)
