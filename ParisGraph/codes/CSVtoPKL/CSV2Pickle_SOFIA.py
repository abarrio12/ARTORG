"""
Code written by Sofia to read the csv files extracted from Renier and transform them into a pickles file
"""
import pandas as pd
import igraph as ig
import pickle
import random
import numpy as np

# import sys
# graph_number = sys.argv[1]


graph_number = 18


# Load CSV files
#folder = "/Volumes/home/RenierDatasets/HalfBrain/082025-datasets/graph_" + str(graph_number) + "/CSV/"
folder = "/home/admin/Ana/MicroBrain/CSV/"
vertices_df = pd.read_csv(folder + "vertices.csv", header=None)  # Nodes
coordinates_df = pd.read_csv(folder + "coordinates_atlas.csv", header=None)  # Coordinates nodes atlas verison
coordinates_images_df = pd.read_csv(folder + "coordinates.csv", header=None)  # Coordinates nodes from Image

edges_df = pd.read_csv(folder + "edges.csv", header=None)        # Edges
length_df = pd.read_csv(folder + "length.csv", header=None)     # Extra edge attributes
radii_df = pd.read_csv(folder + "radii_edge.csv", header=None)      # More edge attributes
radii_atlas_df = pd.read_csv(folder + "radii_atlas.csv", header=None)      # More edge attributes
vein_df = pd.read_csv(folder + "vein.csv", header=None)    # Label for vein edges
artery_df = pd.read_csv(folder + "artery.csv", header=None)         # Label for artery edges
radii_vertex_df = pd.read_csv(folder + "radii.csv", header=None)         # More edge attributes
annotation_vertex_df = pd.read_csv(folder + "annotation.csv", header=None)         # Annotation from ABA
distance_to_surface_df= pd.read_csv(folder + "distance_to_surface.csv", header=None) #distance to Surface


# Add the Tourtous Graph
edge_geometry_df = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None)        # Edges


print(np.mean([np.mean(radii_vertex_df[0]),np.mean(radii_df)]))

#pixel_dimension = 1.63 #[um/px]

# Create an empty graph
G = ig.Graph()

G.add_vertices(len(vertices_df))

# Assign unique IDs to each vertex
G.vs["id"] = vertices_df[0].tolist()

# Assign Coordinates x,y,z to Vertices
if coordinates_df.shape[1] >= 3:  # Ensure there are at least 3 columns (x, y, z)
    G.vs["coords"] = list(zip(coordinates_df[0], coordinates_df[1], coordinates_df[2]))
    G.vs["coords_image"] = list(zip(coordinates_images_df[0], coordinates_images_df[1], coordinates_images_df[2]))

G.vs["annotation"] = annotation_vertex_df[0].tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].tolist()
G.vs["radii"] = radii_vertex_df[0].tolist()

# Add edges to the graph
edges = []
edge_nkind = []  # Store nkind values for edges
radius = []   # Store diameters
lengths = []
for i, row in edges_df.iterrows():
    source = int(row[0])  # starting point of edge
    target = int(row[1])  # ending point of edge

    # Define nkind based on artery and vein files
    # 2: artery (DA), 3: vein (AV), 4: capillary (C)
    nkind = 4
    if artery_df[0][i] == 1:
        nkind = 2
    elif vein_df[0][i] == 1:
        nkind = 3

    # Collect edge attributes
    edges.append((source, target))
    edge_nkind.append(nkind)
    lengths.append(length_df[0][i])
    radius.append(radii_df[0][i])  # Handle missing data

# # Add edges to the igraph graph
G.add_edges(edges)

# Assign edge attributes
G.es["connectivity"] = edges
G.es["nkind"] = edge_nkind
G.es["radius"] = radius
G.es["diameter"] = [r * 2 for r in radius]
G.es["length"] = lengths

print(G.es[0].attributes())
print(G.es[1].attributes())

def check_column_types(series, col_name):
    types_found = set(type(x) for x in series if pd.notnull(x))
    if len(types_found) > 1:
        print(f"Column '{col_name}' has multiple types: {types_found}")
    else:
        print(f"Column '{col_name}' all values type: {types_found.pop()}")

# Example for your dataframes:
check_column_types(radii_vertex_df[0], "radii_vertex_df")
check_column_types(annotation_vertex_df[0], "annotation_vertex_df")
check_column_types(distance_to_surface_df[0], "distance_to_surface_df")
check_column_types(artery_df[0], "artery_df")
check_column_types(vein_df[0], "vein_df")
check_column_types(length_df[0], "length_df")
check_column_types(radii_df[0], "radii_df")
check_column_types(vertices_df[0], "vertices_df")

#out_path = "/Volumes/home/RenierDatasets/HalfBrain/082025-datasets/graph_" + str(graph_number) + "/add_coord_" + str(graph_number) +"_igraph.pkl"
out_path = "/home/admin/Ana/MicroBrain/output" + str(graph_number) + "igraph.pkl"
# save it in pickle to read with igraph
with open(out_path, "wb") as f:
    pickle.dump(G, f)
print('Graph saved in pickle format.')