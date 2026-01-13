import numpy as np
from copy import deepcopy
import lxml.etree as xml
import pickle


# ------------------------------------------------------------------------------
def remap_connectivity(id_to_new_idx):
    """
    Returns a mapping from original 'id' to new index in the subgraph
    (used only when subgraph=True, because VTK connectivity must be 0..N-1)
    """
    remapped_vertex_ids = {}
    for i, idx in enumerate(id_to_new_idx):
        remapped_vertex_ids[idx] = i
    return remapped_vertex_ids


# ------------------------------------------------------------------------------
def WriteOnFileVTP(filename, vertices_array, connectivity_array, point_data, cell_data, subgraph=False):
    """Write data to .vtp file (ASCII)"""

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

    pointDataTag = xml.SubElement(piece, "PointData")
    cellDataTag = xml.SubElement(piece, "CellData")
    pointsTag = xml.SubElement(piece, "Points")
    linesTag = xml.SubElement(piece, "Lines")

    # -------------------
    # PointData arrays
    # -------------------
    for name, data_array in point_data.items():
        data = xml.SubElement(
            pointDataTag, "DataArray",
            {"type": "Float64", "Name": name, "NumberOfComponents": "1", "format": "ascii"}
        )
        # Flatten tuples if any
        flattened = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
        data.text = " ".join(["{:10.15e}".format(p) for p in flattened])

    # -------------------
    # CellData arrays
    # -------------------
    for name, data_array in cell_data.items():
        if name == "connectivity":
            data = xml.SubElement(
                cellDataTag, "DataArray",
                {"type": "Float64", "Name": name, "NumberOfComponents": "2", "format": "ascii"}
            )
            flattened = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
            data.text = "\n".join(["{:10.15e}".format(p) for p in flattened])
        else:
            data = xml.SubElement(
                cellDataTag, "DataArray",
                {"type": "Float64", "Name": name, "NumberOfComponents": "1", "format": "ascii"}
            )
            flattened = [i for t in data_array for i in (t if isinstance(t, tuple) else (t,))]
            data.text = "\n".join(["{:10.15e}".format(p) for p in flattened])

    # -------------------
    # Points (Nx3)
    # -------------------
    coords = xml.SubElement(pointsTag, "DataArray", {"type": "Float64", "NumberOfComponents": "3", "format": "ascii"})
    coords.text = " ".join(["{:10.15e}".format(i) for c in vertices_array for i in c])

    # -------------------
    # Lines: connectivity + offsets
    # -------------------
    if not subgraph:
        connectivity = xml.SubElement(linesTag, "DataArray", {"type": "Int32", "Name": "connectivity", "format": "ascii"})
        offsets = xml.SubElement(linesTag, "DataArray", {"type": "Int32", "Name": "offsets", "format": "ascii"})

        connectivity.text = " ".join([str(i) for pair in connectivity_array for i in pair])
        offsets.text = " ".join([str((i + 1) * 2) for i in range(len(connectivity_array))])

    else:
        # If subgraph=True we must remap connectivity to 0..N-1 based on the points we actually write.
        # We assume point_data["index"] contains the ORIGINAL vertex ids used in edges.
        dict_ids = remap_connectivity(point_data["index"])

        remapped_connectivity = []
        for src_id, tgt_id in connectivity_array:
            if src_id in dict_ids and tgt_id in dict_ids:
                remapped_connectivity.append((dict_ids[src_id], dict_ids[tgt_id]))
            else:
                print(f"Skipping edge ({src_id}, {tgt_id}) — node not in subgraph")

        connectivity = xml.SubElement(linesTag, "DataArray", {"type": "Int32", "Name": "connectivity", "format": "ascii"})
        offsets = xml.SubElement(linesTag, "DataArray", {"type": "Int32", "Name": "offsets", "format": "ascii"})

        connectivity.text = " ".join([str(i) for pair in remapped_connectivity for i in pair])
        offsets.text = " ".join([str((i + 1) * 2) for i in range(len(remapped_connectivity))])

    with open(filename, "w") as vtkOutput:
        vtkOutput.write(xml.tostring(vtkTag, pretty_print=True, encoding="unicode"))

    print("Saved:", filename)


# ------------------------------------------------------------------------------
def write_vtp(graph, filename, subgraph_TF=False):
    """
    NON-TORTUOUS VTP export: straight segments between endpoints only.

    IMPORTANT for your project:
    - Use coords_image to match OUTGEOM x/y/z and the Box coords you take in ParaView.
    - Use G.get_edgelist() to avoid relying on G.es["connectivity"] existing.
    """
    G = deepcopy(graph)

    # --- endpoints in IMAGE coordinates (voxels) ---
    if "coords_image" not in G.vs.attributes():
        raise KeyError("Graph has no vertex attribute 'coords_image'. Use the correct PKL or change to 'coords'.")

    vertices_array = G.vs["coords_image"]

    # --- edges connectivity always available ---
    edges = G.get_edgelist()  # list of (source, target) using vertex indices 0..N-1

    # --- minimal checks ---
    needed_v_attrs = ["id", "annotation"]
    for a in needed_v_attrs:
        if a not in G.vs.attributes():
            raise KeyError(f"Graph missing vertex attribute '{a}' (required by this exporter).")

    needed_e_attrs = ["radius", "diameter", "nkind", "length"]
    for a in needed_e_attrs:
        if a not in G.es.attributes():
            raise KeyError(f"Graph missing edge attribute '{a}' (required by this exporter).")

    # If you want the "connectivity" also as CellData (like your original), we pass edges there too.
    WriteOnFileVTP(
        filename=filename,
        vertices_array=vertices_array,
        connectivity_array=edges,
        point_data={
            "index": G.vs["id"],
            "annotation": G.vs["annotation"],
        },
        cell_data={
            "connectivity": edges,
            "radius": G.es["radius"],
            "vessel_diameter": G.es["diameter"],
            "vessel_nkind": G.es["nkind"],
            "vessel_length": G.es["length"],
        },
        subgraph=subgraph_TF
    )


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Load graph
    in_pkl = "/home/admin/Ana/MicroBrain/18_igraph.pkl"
    out_vtp = "/home/admin/Ana/MicroBrain/18_igraph_NON_TORTUOUS.vtp"

    with open(in_pkl, "rb") as f:
        G = pickle.load(f, encoding="latin1")

    # Save as VTP file. If subgraph (cut) then subgraph_TF=True
    write_vtp(G, out_vtp, subgraph_TF=False)
