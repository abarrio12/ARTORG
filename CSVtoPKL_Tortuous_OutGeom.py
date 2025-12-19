
"""
Implementation of Sofia's code to read the csv files extracted from Renier and transform them into a pickles file.
Made by Ana.

This script builds a full vascular graph from CSV files and stores it in a memory-efficient pickle format.

The graph structure (nodes and edges) is represented using `igraph`, while high-resolution vascular geometry is stored 
separately as global coordinate arrays. Each edge does not directly store its geometry; instead, it contains
integer indices (`geom_start`, `geom_end`) that reference contiguous segments in the global geometry arrays (`x`, `y`, `z`). 
This design avoids duplication of geometric data and significantly reduces memory usage for large graphs.

The script:
- Loads vertex, edge, and geometry information from CSV files 
- Constructs an igraph graph with vertex and edge attributes
- Stores vascular geometry in a memory-safe indexed format
- Computes tortuous length and tortuosity for each edge
- Saves the complete dataset (graph + geometry arrays) into a single pickle file
- Provides consistency and sanity checks to validate geometry–topology alignment

This data model is designed to scale to large vascular networks and is compatible with downstream analysis and visualization 
pipelines (e.g. ParaView / VTK).

"""

import pandas as pd
import igraph as ig
import numpy as np
import pickle
import os
import random
# ============================
# Parameters
# ============================

graph_number = 18
folder = "/home/admin/Ana/MicroBrain/CSV/"
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_full.pkl"

print("=== START CSV → PKL (OUTGEOM) ===")

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

print("CSVs loaded")

# ============================
# Create graph
# ============================

G = ig.Graph()
G.add_vertices(len(vertices_df))

# Graph structure:
# line = node
# column = attribute 
# 0 Ai Bi Ci --> node 1 + its attributes
# 1 Aj Bj Cj --> node 2 + its attributes
# ----------------------------
# Vertex attributes
# ----------------------------

G.vs["id"] = vertices_df[0].astype(int).tolist() #--> gives me the id (0, 1,...)

G.vs["coords"] = list(zip(
    coordinates_df[0], coordinates_df[1], coordinates_df[2]
))

G.vs["coords_image"] = list(zip(
    coordinates_images_df[0],
    coordinates_images_df[1],
    coordinates_images_df[2]
))

G.vs["annotation"] = annotation_vertex_df[0].astype(int).tolist()
G.vs["distance_to_surface"] = distance_to_surface_df[0].astype(np.float32).tolist()
G.vs["radii"] = radii_vertex_df[0].astype(np.float32).tolist()

# ----------------------------
# Edges
# ----------------------------

# Edges structure:
# v1, v2 --> not attributes, but connections between nodes
#  1 , 3
#  4 , 12
# edges_df[0] = origin nodes (1, 4)
# edges_df[1] = target nodes (3, 12)

# CSV does not create these connections just by reading. Only has info saying this connection should exist in the graph. 


edges = []
edge_nkind = []
edge_radius = []
edge_length = []

for i, row in edges_df.iterrows(): # reading csv lines
    s = int(row[0])
    t = int(row[1])

    nkind = 4
    if artery_df[0][i] == 1:
        nkind = 2
    elif vein_df[0][i] == 1:
        nkind = 3

    edges.append((s, t)) # edges list (not yet added to graph)
    edge_nkind.append(nkind)
    edge_radius.append(radii_df[0][i])
    edge_length.append(length_df[0][i])

G.add_edges(edges) # edge connection created. assigns internal order (!= csv, v1, v2)

G.es["nkind"] = edge_nkind
G.es["radius"] = np.array(edge_radius, dtype=np.float32).tolist()
G.es["diameter"] = (2 * np.array(edge_radius, dtype=np.float32)).tolist()
G.es["length"] = np.array(edge_length, dtype=np.float32).tolist()

# ----------------------------
# Geometry (MEMORY SAFE)
# Edge does not have the geometry information, but stores indexes (int) indicating the start and end of the points in the
# arrays.
# ----------------------------

# 3 arrays with the full geometry
x = edge_geometry_df[0].to_numpy(dtype=np.float32)
y = edge_geometry_df[1].to_numpy(dtype=np.float32)
z = edge_geometry_df[2].to_numpy(dtype=np.float32)

# indexes (int) per edge = range of points of the edge 
G.es["geom_start"] = geom_index_df[0].astype(np.int64).tolist()
G.es["geom_end"]   = geom_index_df[1].astype(np.int64).tolist()

# ----------------------------
# Calculate tortuosity
# ----------------------------
coords_v = np.array(G.vs["coords"])

length_tortuous = []
straight_dist = []

for e in range(G.ecount()):
    # Lenght of the tortuous should be computed using the edge geometry coordinates
    s = G.es[e]["geom_start"]
    e_ = G.es[e]["geom_end"]

    L = 0.0
    for i in range(s, e_ - 1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        dz = z[i+1] - z[i]
        L += np.sqrt(dx*dx + dy*dy + dz*dz)

    length_tortuous.append(L)

    # Straight distance between endpoints
    v0, v1 = G.es[e].tuple
    d = np.linalg.norm(coords_v[v0] - coords_v[v1])
    straight_dist.append(d)

length_tortuous = np.array(length_tortuous) #current length should be the sum of all the tortuous points.
straight_dist = np.array(straight_dist)

tortuosity = length_tortuous / straight_dist
tortuosity[straight_dist == 0] = np.nan

G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()


# ----------------------------
# Save EVERYTHING in ONE pkl
# ----------------------------

data = {
    "graph": G,
    "coords": {
        "x": x,
        "y": y,
        "z": z
    }
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved FULL graph to:", out_path)
print("=== DONE ===")




# =================== SANITY CHECKS =======================================
class AutomaticQualityCheck:
    def __init__(self, G, x, y, z, geom_start, geom_end):
        self.G = G #can't load graph directly, geometry is not inside object G
        self.x = x
        self.y = y
        self.z = z
        self.starts = np.asarray(geom_start)
        self.ends   = np.asarray(geom_end)
        self.coords_v = np.array(G.vs["coords"], dtype=np.float32)

    def run_all(self):
        print("Running quality checks...")
        self.check_indices_contiguous()
        self.check_min_geometry_points()
        self.check_geometry_matches_vertices()
        self.check_tortuous_vs_straight()
        print("All quality checks passed")


# ============== FRANCA'S QUALITY CHECK ===============================00
class ManualQualityCheck(object):
    def __init__(self, G, x, y, z):
        self.G = G
        self.x = x
        self.y = y
        self.z = z
        self.getRandomSamples()
        self.checkLength()
        self.checkRadii()

    def getRandomSamples(self):
        self.samples = [random.randint(0, self.G.ecount()) for i in range(1, 10)]
        print("Random edges to test the data: %s" %(', '.join([str(i) for i in self.samples])))

    def checkLength(self):
        for edge in self.samples:
            l = self.G.es["length"][edge] # length saved for for random edge --> igraph is already a list, no need to get_Array()
            edges = self.G.es[edge].tuple # u,v 
            A = numpy.array(self.G.vs["coords"][edges[0]]) #(xu,yu,zu)
            B = numpy.array(self.G.vs["coords"][edges[1]]) #(xv,yv,zv)
            l_comp = numpy.linalg.norm(A-B)
            print('%s, %.3f, %.3f'%(edge, l, l_comp))

    def checkRadii(self):
        for edge in self.samples:
            re = self.G.es["radii"][edge]
            edges = self.G.es[edge].tuple
            rv1 = self.G.vs["radii"][edges[0]]
            rv2 = self.G.vs["radii"][edges[1]]
            print('%s, %.3f, %.3f, %.3f'%(edge, re, rv1, rv2))
    
    def checkTortuosCoordinates(self):

        # edge geometry indices (from graph, not CSV)
        starts = np.asarray(self.G.es["geom_start"])
        ends   = np.asarray(self.G.es["geom_end"])

        # continuity check
        if not np.all(starts[1:] == ends[:-1]):
            print("Non matching geom_start / geom_end continuity")

        # vertex coordinates
        coords_v = np.asarray(self.G.vs["coords"], dtype=np.float32)

        # edge list
        edges = self.G.get_edgelist()

        for k, (s, e) in enumerate(zip(starts, ends)):
            v0, v1 = edges[k]
            # endpoints from vertices
            a = coords_v[[v0, v1]]   # u, v --> a = [[x_u, y_u, z_u], [x_v, y_v, z_v]]

            # endpoints from tortuous geometry
            b = np.array([
                [self.x[s],   self.y[s],   self.z[s]],
                [self.x[e-1], self.y[e-1], self.z[e-1]]
            ], dtype=np.float32)

            if not np.allclose(a, b) and not np.allclose(a, b[::-1]):
                print(f"Non matching tortuous coordinates in edge {k}")

qc = ManualQualityCheck(G, x, y, z)
qc.checkTortuosCoordinates()
