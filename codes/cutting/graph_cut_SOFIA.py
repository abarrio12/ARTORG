import numpy as np
import pickle
from collections import defaultdict

def cut_graph_to_bounding_box(G, xCoordsBox, yCoordsBox, zCoordsBox, output_pickle=None):
    """
    Cuts a graph to a bounding box: removes edges outside the box and border vertices, keeps only the largest component.

    Args:
        G (igraph.Graph): Input graph.
        xCoordsBox, yCoordsBox, zCoordsBox (list of 2 floats): [min, max] for each axis.
        output_pickle (str, optional): Path to save the processed graph. If None, graph is not saved.

    Returns:
        igraph.Graph: Processed graph restricted to the bounding box and largest component.
    """

    # Helper functions
    def is_inside_box(point):
        x, y, z = point
        return (xCoordsBox[0] <= x <= xCoordsBox[1] and
                yCoordsBox[0] <= y <= yCoordsBox[1] and
                zCoordsBox[0] <= z <= zCoordsBox[1])

    # Collect edges
    edges_in_box = []
    edges_across_border = []
    edges_outside_box = []
    border_vertices = []

    for e in G.es:
        vertices = [e.source, e.target]
        vertices_in_box = [is_inside_box(G.vs[v]['coords']) for v in vertices]

        if sum(vertices_in_box) == 2:
            edges_in_box.append(e.index)
        elif sum(vertices_in_box) == 1:
            edges_across_border.append(e.index)
            # Add vertex outside box to border vertices
            if not vertices_in_box[0]:
                border_vertices.append(vertices[0])
            else:
                border_vertices.append(vertices[1])
        else:
            edges_outside_box.append(e.index)

    # Delete edges outside box
    all_edges_to_keep = np.unique(edges_in_box + edges_across_border)
    edges_to_delete = list(set(range(G.ecount())) - set(all_edges_to_keep))
    G.delete_edges(edges_to_delete)

    # Delete border vertices
    G.delete_vertices(np.unique(border_vertices))

    # Delete any disconnected vertices
    disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
    G.delete_vertices(disconnected_vertices)

    # Update degree
    G.vs['degree'] = G.degree()

    # Keep only largest connected component
    largest_component = G.components().giant()

    if output_pickle:
        with open(output_pickle, "wb") as f:
            pickle.dump(largest_component, f)
        print(f"Graph saved to {output_pickle}")

    return largest_component




with open("C:/Users/Ana/OneDrive/Escritorio/ARTORG/igraph/half brain scripts/18_igraph.pkl", "rb") as f:
    G = pickle.load(f, encoding='latin1')

# These measurements should be set according to the desired bounding box in paraview (box option)
# In order to check which coordinates to use, select Hover Points (red dot with question mark) in paraview and click on points of interest (xmin, ymin, zmin)
# & (xmax, ymax, zmax) to see their coordinates.
xCoordsBox = [50, 150]
yCoordsBox = [70,170]
zCoordsBox = [110, 210]


G_cut = cut_graph_to_bounding_box(G, xCoordsBox, yCoordsBox, zCoordsBox, output_pickle="C:/Users/Ana/OneDrive/Escritorio/ARTORG/igraph/half brain scripts/18_igraph_CUT_SomatomotorArea.pkl")


print(G_cut.summary())
