"""
Code written by Sofia to read the csv files extracted from Renier and transform them into a pickles file
"""
import pandas as pd
import igraph as ig
import pickle
import numpy as np

# import sys
# graph_number = sys.argv[1]

graph_number = 18

# Load CSV files
folder = "/Volumes/home/RenierDatasets/HalfBrain/082025-datasets/graph_" + str(graph_number) + "/CSV/"

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
# 3) POINTS
# ============================

# 3a) Curve ids to list
edge_to_curve = geom_index_df[0].tolist()  # len = num_edges, edge_to_curve[i] = curve_id de la arista i

# 3b) rename colums
# Asumo formato: col0 = curve_id, col1 = x, col2 = y, col3 = z
edge_geometry_df.columns = ["curve_id", "x", "y", "z"]

# 3c) agrupar: curve_id -> lista de (x,y,z)
geometry_dict = (
    edge_geometry_df
    .groupby("curve_id")
    .apply(lambda df: [tuple(p) for p in df[["x", "y", "z"]].values])
    .to_dict()
)

# 3d) construir lista de geometrías por arista (en el mismo orden que edges_df)
edge_geometries = []
for i in range(len(edges_df)):
    curve_id = edge_to_curve[i]          # qué curva le toca a esa arista
    points_edge = geometry_dict[curve_id]  # lista de (x,y,z) = POINTS
    edge_geometries.append(points_edge)

# Asignar al grafo >>>>>>> POINTS
G.es["geometry"] = edge_geometries

print(G.es[0].attributes())
print(G.es[1].attributes())

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

out_path = r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\igraph\my test codes" + str(graph_number) + f"/add_coord_{graph_number}_igraph.pkl"
with open(out_path, "wb") as f:
    pickle.dump(G, f)

print('Graph saved in pickle format at:', out_path)
