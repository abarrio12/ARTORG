import numpy as np
from copy import deepcopy
import igraph
import lxml.etree as xml


# ------------------------------------------------------------------------------
def distance(v1, v2):
    """Distance between two vertices"""
    return np.linalg.norm(np.array(v1) - np.array(v2), ord=2)

# ------------------------------------------------------------------------------
def remap_connectivity(id_to_new_idx):
    """
    Returns a mapping from original 'id' to new index in the subgraph
    """
    remapped_vertex_ids = {}

    for i,idx  in enumerate(id_to_new_idx):
        remapped_vertex_ids[idx] = i

    return remapped_vertex_ids

# ------------------------------------------------------------------------------
def WriteOnFileVTP(filename, vertices_array, connectivity_array, point_data, cell_data, subgraph = False):
    """Write data to vtp file"""

    numPoints = len(vertices_array)
    numLines = len(connectivity_array)
    vtkTag = xml.Element("VTKFile", {
        "type": "PolyData",
        "version": "0.1",
        "byte_order": "LittleEndian",
    })
    polyDataTag = xml.SubElement(vtkTag, "PolyData")
    piece = xml.SubElement(polyDataTag, "Piece", {
        "NumberOfPoints": str(numPoints),
        "NumberOfVerts": "0",
        "NumberOfLines": str(numLines),
        "NumberOfStrips": "0",
        "NumberOfPolys": "0",
    })
    pointData = xml.SubElement(piece, "PointData")
    cellData = xml.SubElement(piece, "CellData")
    points = xml.SubElement(piece, "Points")
    lines = xml.SubElement(piece, "Lines")

    # point data
    for name, data_array in point_data.items():
        data = xml.SubElement(pointData, "DataArray",
                              {"type": "Float64", "Name": name, "NumberOfComponents": "1", "format": "ascii"})
        flattened_data_array = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
        data.text = " ".join(["{:10.15e}".format(p) for p in flattened_data_array])


    # cell data
    for name, data_array in cell_data.items():
        if name == "connectivity":
            data = xml.SubElement(cellData, "DataArray",
                                  {"type": "Float64", "Name": name, "NumberOfComponents": "2", "format": "ascii"})
            flattened_data_array = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
            data.text = "\n".join(["{:10.15e}".format(p) for p in flattened_data_array])


        else:
            data = xml.SubElement(cellData, "DataArray",
                                  {"type": "Float64", "Name": name, "NumberOfComponents": "1", "format": "ascii"})
            flattened_data_array = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
            data.text = "\n".join(["{:10.15e}".format(p) for p in flattened_data_array])


    # points
    coords = xml.SubElement(points, "DataArray", {"type": "Float64", "NumberOfComponents": "3", "format": "ascii"})
    coords.text = " ".join(["{:10.15e}".format(i) for c in vertices_array for i in c])

    # lines
    if subgraph == False:
        connectivity = xml.SubElement(lines, "DataArray", {"type": "Int32", "Name": "connectivity", "format": "ascii"})
        offsets = xml.SubElement(lines, "DataArray", {"type": "Int32", "Name": "offsets", "format": "ascii"})
        connectivity.text = " ".join(["{}".format(i) for c in connectivity_array for i in c])
        offsets.text = " ".join(["{}".format((i + 1) * 2) + " " for i in range(len(connectivity_array))])
    else:
        # If we are saving only a subgraph we need to rempap the connectivity array
        # Get the remapped edges (connectivity) and the id mapping
        dict_ids = remap_connectivity(point_data['index'])
        # Remap your connectivity_array using the ID mapping
        remapped_connectivity = []
        for src_id, tgt_id in connectivity_array:
            try:
                new_src = dict_ids[src_id]
                new_tgt = dict_ids[tgt_id]
                remapped_connectivity.append((new_src, new_tgt))
            except KeyError:
                # This happens if an edge points to a vertex not in the subgraph
                print(f"Skipping edge ({src_id}, {tgt_id}) — node not in subgraph")
        # Write to XML
        connectivity = xml.SubElement(lines, "DataArray", {"type": "Int32", "Name": "connectivity", "format": "ascii"})
        offsets = xml.SubElement(lines, "DataArray", {"type": "Int32", "Name": "offsets", "format": "ascii"})

        # Flatten the connectivity list for VTK
        connectivity.text = " ".join([str(i) for pair in remapped_connectivity for i in pair])
        offsets.text = " ".join([str((i + 1) * 2) for i in range(len(remapped_connectivity))])

    with open(filename, 'w') as vtkOutput:
        vtkOutput.write(xml.tostring(vtkTag, pretty_print=True, encoding="unicode"))


def write_vtp(graph, filename, subgraph_TF=False, penetrating_trees=False):
    """
        INPUT:
            graph: Graph in iGraph format
            filename: Name of the VTP file to be written.
            subgraph: True if we want to write a subgraph from a bigger graph, False otherwise
        OUTPUT:
            VTP file written to disk.

        ############################################################################
        #--> PAY ATTENTION TO THE UNITS FOR THE DEFINITION OF THE MINIMUM  !!!
        ############################################################################


    DETAILS:

    NON TORTUOUS GRAPH:

    - Point Data:
       --coords (x, y, z),
       --index,
       -- annotation: brain annotation raleted to the json file of the graph
       -- distance_to_surface

    - Cell Data: index, vessel_diameter, diameters, vessel_length, lengths2, vessel_nkind, radius
      -- connectivity # if subgraph requires remapping
      -- diameter: diameter of the vessel (edge), repeated for each sub_edge
      -- vessel_length: length of the vessel (edge), repeated for each sub_edge
      -- nkind: nkind of the vessel (edge), repeated for each sub_edge
      -- radius: vessel_diameter/2

    """

    # Make a copy of the graph to avoid modifying the original
    G = deepcopy(graph)

    if penetrating_trees:
        # Initialize the 'penetrating_trees' attribute for all edges
        WriteOnFileVTP(
            filename=filename,
            vertices_array=G.vs["coords"],
            connectivity_array=G.es["connectivity"],
            point_data={"index": G.vs["id"], "annotation": G.vs["annotation"],
                         "node_label": G.vs["node_label"]},
            cell_data={"connectivity": G.es["connectivity"], "radius": G.es["radius"],
                       "vessel_diameter": G.es["diameter"], "vessel_nkind": G.es["nkind"],
                       'penetrating_trees': G.es["penetrating_trees"],
                       "vessel_length": G.es["length"]}, subgraph=subgraph_TF)
    else:
        WriteOnFileVTP(
            filename=filename,
            vertices_array=G.vs["coords"],
            connectivity_array=G.es["connectivity"],
            point_data={"index": G.vs["id"], "annotation": G.vs["annotation"]
                      },
            cell_data={"connectivity": G.es["connectivity"], "radius": G.es["radius"],
                       "vessel_diameter": G.es["diameter"], "vessel_nkind": G.es["nkind"],
                       "vessel_length": G.es["length"]}, subgraph=subgraph_TF)
        
        
import pickle
# Load graph
with open("/home/admin/Ana/MicroBrain/output/cut_non_equiv.pkl", "rb") as f:
    G = pickle.load(f, encoding='latin1')

# Save as VTP file. If subgraph (cut) then subgraph_TF=True
write_vtp(G,
          "/home/admin/Ana/MicroBrain/output/18_igraph_nHcut3.vtp",
          subgraph_TF=True)
