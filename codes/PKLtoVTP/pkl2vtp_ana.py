''' 
Script to convert an iGraph graph into a VTP file for visualization in Paraview.

Updated by Ana from pkl2vtp.py to be able to read from outgeom and mvn format.

Updated too so it checks attributes existance and handles None values.

'''

import numpy as np
from copy import deepcopy
import pickle
import igraph
import lxml.etree as xml
from collections import Counter, defaultdict

#------------------------------------------------------------------------------
def distance(v1, v2):
    """Distance between two vertices"""
    return np.linalg.norm(np.array(v1) - np.array(v2), ord=2)

#------------------------------------------------------------------------------
def safe_v_attr(G, name, default):
    """Devuelve atributo de vértice si existe; sustituye None por default."""
    if name not in G.vs.attribute_names():
        return None
    return [default if v is None else v for v in G.vs[name]]

#------------------------------------------------------------------------------
def safe_e_attr(G, name, default):
    """Devuelve atributo de edge si existe; sustituye None por default."""
    if name not in G.es.attribute_names():
        return None
    return [default if v is None else v for v in G.es[name]]

#------------------------------------------------------------------------------
def safe_e_value(edge, G, name, default):
    """Devuelve atributo de edge si existe; si no existe o es None, devuelve default."""
    if name not in G.es.attribute_names():
        return default
    value = edge[name]
    return default if value is None else value

#------------------------------------------------------------------------------
def safe_list(values, default):
    """Sustituye None dentro de una lista por default."""
    if values is None:
        return None
    return [default if v is None else v for v in values]

#------------------------------------------------------------------------------
def WriteOnFileVTP(filename, vertices_array, connectivity_array, point_data, cell_data):
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
        data = xml.SubElement(
            pointData,
            "DataArray",
            {
                "type": "Float64",
                "Name": name,
                "NumberOfComponents": "1",
                "format": "ascii",
            },
        )
        data.text = " ".join(["{:10.15e}".format(p) for p in data_array])

    # cell data
    for name, data_array in cell_data.items():
        if name == "connectivity":
            data = xml.SubElement(
                cellData,
                "DataArray",
                {
                    "type": "Float64",
                    "Name": name,
                    "NumberOfComponents": "2",
                    "format": "ascii",
                },
            )
            data.text = " ".join(["{:10.15e}".format(i) for c in connectivity_array for i in c])
        else:
            data = xml.SubElement(
                cellData,
                "DataArray",
                {
                    "type": "Float64",
                    "Name": name,
                    "NumberOfComponents": "1",
                    "format": "ascii",
                },
            )
            data.text = " ".join(["{:10.15e}".format(r) for r in data_array])

    # points
    coords = xml.SubElement(
        points,
        "DataArray",
        {
            "type": "Float64",
            "NumberOfComponents": "3",
            "format": "ascii",
        },
    )
    coords.text = " ".join(["{:10.15e}".format(i) for c in vertices_array for i in c])

    # lines
    connectivity = xml.SubElement(
        lines,
        "DataArray",
        {
            "type": "Int32",
            "Name": "connectivity",
            "format": "ascii",
        },
    )

    offsets = xml.SubElement(
        lines,
        "DataArray",
        {
            "type": "Int32",
            "Name": "offsets",
            "format": "ascii",
        },
    )

    connectivity.text = " ".join(["{}".format(i) for c in connectivity_array for i in c])
    offsets.text = " ".join(["{}".format((i + 1) * 2) + " " for i in range(len(connectivity_array))])

    with open(filename, "w") as vtkOutput:
        vtkOutput.write(xml.tostring(vtkTag, pretty_print=True, encoding="unicode"))

#------------------------------------------------------------------------------
def write_vtp(graph, filename, tortuous=True, verbose=False):
    """
        INPUT:
            graph: Graph in iGraph format
            filename: Name of the VTP file to be written.
            tortuous: Determines whether to plot the tortuous geometry (default=True).
            verbose: Whether to print if writing an array fails (default=False).
        OUTPUT:
            VTP file written to disk.

        ############################################################################
        #--> PAY ATTENTION TO THE UNITS FOR THE DEFINITION OF THE MINIMUM DISTANCE !!!
        ############################################################################


    DETAILS:

    NON TORTUOUS GRAPH:

    - Point Data:  coords (x, y, z), index, pBC ('None' substituted with -1000), pressure

    - Cell Data: connectivity, index, diameter (vessel), length (vessel), nkind


    TORTUOUS GRAPH:

    - Point Data:  coords (x, y, z), index, pBC ('None' substituted with -1000), pressure
      -- index: same index for original nodes and (vessel_index + step * local_point_index_in_edge)
         (step is 1/10 or 1/1000 depending on the total number of points per each edge)
         (local_point_index_in_edge goes from 1 to number of points per each edge -2,
         becuse the first and last one are excluded)
      -- pBC: original value for the nodes and '-1000' assigned to the points (points can't be on the boundaries)
      -- pressure: original value for the nodes and intermediate values (depending on number of points per each edge) for the points


    - Cell Data: index, vessel_diameter, diameters, vessel_length, lengths2, vessel_nkind, radius
      -- connectivity
      -- index: vessel_index + step * local_subedge_index_in_edge
         (step is 1/10 or 1/1000 depending on the total number of points per each edge -1 )
         (local_subedge_index_in_edge goes from 0 to number of points per each edge -1 )
      -- vessel_diameter: diameter of the vessel (edge), repeated for each sub_edge
      -- diameters: diameter of each sub_edge, calculated as the average between the two adjacent points
      -- vessel_length: length of the vessel (edge), repeated for each sub_edge
      -- lengths2: length of each sub_edge (extracted from 'lengths2', edge attribute)
      -- vessel_nkind: nkind of the vessel (edge), repeated for each sub_edge
      -- radius: vessel_diameter/2

    """

    # Make a copy of the graph to avoid modifying the original
    G = deepcopy(graph)

    # We maintain the index attributes, but we will rely on graph topology for connectivity
    G.vs["index"] = list(range(G.vcount()))
    if G.ecount() > 0:
        G.es["index"] = list(range(G.ecount()))

    # Delete self-loops
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

    ############################################################################
    # --> PAY ATTENTION TO THE UNITS FOR THE DEFINITION OF THE MINIMUM DISTANCE !!!
    ############################################################################

    # Select and delete the edges with length 0
    edges_to_remove = []

    for edge in G.es:
        v1, v2 = edge.tuple

        ps = safe_e_value(edge, G, "points", None)

        if ps is None:
            ps = [G.vs[v1]["coords"], G.vs[v2]["coords"]]

        if distance(G.vs[v1]["coords"], G.vs[v2]["coords"]) < 0.01 * 1e-6:
            ps = [G.vs[v1]["coords"], G.vs[v2]["coords"]]

            # Solo cambia length si existe como atributo
            if "length" in G.es.attribute_names():
                edge["length"] = 0.0

            edges_to_remove.append(edge.index)

        # remove possibly repeated points in points
        ps = [ps[0], *[p for i, p in enumerate(ps[1:]) if distance(p, ps[i]) > 0.01 * 1e-6]]

        if len(ps) < 2:
            ps = [G.vs[v1]["coords"], G.vs[v2]["coords"]]

        # Solo guarda points si existe como atributo
        if "points" in G.es.attribute_names():
            edge["points"] = ps

    G.delete_edges(edges_to_remove)

    ############################################################################

    # Convert the substance dictionary to arrays
    if "substance" in G.vs.attribute_names():
        substances = G.vs[0]["substance"].keys()
        for substance in substances:
            G.vs[substance] = [v["substance"][substance] for v in G.vs]
        del G.vs["substance"]

    # Find unconnected vertices and delete them
    disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
    G.delete_vertices(disconnected_vertices)

    # Prepare base point arrays from the cleaned graph
    vertices_array = list(G.vs["coords"])

    # Optional vertex attributes
    # Si no existen, no se escriben.
    # Si existen pero tienen None, se sustituyen.
    pBC_array = safe_v_attr(G, "pBC", -1000)
    pressure_array = safe_v_attr(G, "pressure", 0)

    new_vertex_index_array = [float(j) for j in range(len(vertices_array))]

    if tortuous:
        connectivity_array = []
        radius_array = []
        vessel_diameter_array = []
        vessel_length_array = []
        edge_index_array = []
        vessel_nkind_array = []
        diameters_edge_array = []
        lengths2_edge_array = []

        # Comprobamos una sola vez qué atributos existen
        has_points = "points" in G.es.attribute_names()
        has_length = "length" in G.es.attribute_names()
        has_diameter = "diameter" in G.es.attribute_names()
        has_nkind = "nkind" in G.es.attribute_names()
        has_diameters = "diameters" in G.es.attribute_names()
        has_lengths2 = "lengths2" in G.es.attribute_names()

        print("-- Processing tortuous edges...")

        for edge_obj in G.es:
            edge_index = edge_obj.index
            node_0 = edge_obj.source
            node_1 = edge_obj.target

            ps = safe_e_value(edge_obj, G, "points", None) if has_points else None
            vessel_length = safe_e_value(edge_obj, G, "length", 0) if has_length else None
            vessel_diameter = safe_e_value(edge_obj, G, "diameter", 0) if has_diameter else None
            nkind = safe_e_value(edge_obj, G, "nkind", -1) if has_nkind else None
            diameters = safe_e_value(edge_obj, G, "diameters", None) if has_diameters else None
            lengths2 = safe_e_value(edge_obj, G, "lengths2", None) if has_lengths2 else None

            if ps is None:
                ps = [G.vs[node_0]["coords"], G.vs[node_1]["coords"]]

            assert len(ps) >= 2

            firstIndex = len(vertices_array)
            R = 0.5 * vessel_diameter if vessel_diameter is not None else 0
            n_points = len(ps)

            if n_points >= 10:
                step = 1 / 1000
            else:
                step = 1 / 10

            # ORIENTATION CHECK: Ensure ps[0] is physically near node_0
            dist_0_start = distance(vertices_array[node_0], ps[0])
            dist_0_end = distance(vertices_array[node_0], ps[-1])

            if dist_0_start > dist_0_end:
                ps = ps[::-1]

            if n_points > 2:
                vertices_array += ps[1:-1]

                if pBC_array is not None:
                    pBC_array += [-1000] * (n_points - 2)

                new_vertex_index_array += [
                    edge_index + j * step for j in range(1, n_points - 1)
                ]

                if pressure_array is not None:
                    p1, p2 = pressure_array[node_0], pressure_array[node_1]
                    step_pr = (p2 - p1) / (n_points - 1.0)
                    pressure_array += [
                        p1 + j * step_pr for j in range(1, n_points - 1)
                    ]

            # append tortuous edges
            assert n_points >= 2

            if n_points == 2:
                new_cells = [(node_0, node_1)]
            elif n_points == 3:
                new_cells = [(node_0, firstIndex), (firstIndex, node_1)]
            else:
                new_cells = [
                    (node_0, firstIndex),
                    *[
                        (firstIndex + i, firstIndex + i + 1)
                        for i in range(n_points - 3)
                    ],
                    (firstIndex + n_points - 3, node_1),
                ]

            connectivity_array += new_cells
            num_new_segs = len(new_cells)

            # Siempre podemos calcular radius, pero solo lo escribiremos si tiene longitud correcta
            radius_array += [R] * num_new_segs

            # index siempre lo escribimos
            edge_index_array += [
                edge_index + j * step for j in range(num_new_segs)
            ]

            # Estos solo se rellenan si el atributo existe
            if has_diameter:
                vessel_diameter_array += [vessel_diameter] * num_new_segs

            if has_length:
                vessel_length_array += [vessel_length] * num_new_segs

            if has_nkind:
                vessel_nkind_array += [nkind] * num_new_segs

            # diameters: si el atributo existe pero falta algo, ponemos 0
            if has_diameters:
                if diameters is not None:
                    diameters = safe_list(diameters, 0)

                    local_diameters = [
                        (diameters[j] + diameters[j + 1]) / 2
                        for j in range(len(diameters) - 1)
                    ]

                    if len(local_diameters) == num_new_segs:
                        diameters_edge_array += local_diameters
                    else:
                        if verbose:
                            print("diameters mismatch en edge:", edge_index)
                            print("len(ps):", len(ps))
                            print("len(new_cells):", len(new_cells))
                            print("len(diameters):", len(diameters))
                            print("len(local_diameters):", len(local_diameters))
                        diameters_edge_array += [0] * num_new_segs
                else:
                    diameters_edge_array += [0] * num_new_segs

            # lengths2: si el atributo existe pero falta algo, ponemos 0
            if has_lengths2:
                if lengths2 is not None:
                    lengths2 = safe_list(lengths2, 0)

                    if len(lengths2) == num_new_segs:
                        lengths2_edge_array += lengths2
                    else:
                        if verbose:
                            print("lengths2 mismatch en edge:", edge_index)
                            print("len(ps):", len(ps))
                            print("len(new_cells):", len(new_cells))
                            print("len(lengths2):", len(lengths2))
                        lengths2_edge_array += [0] * num_new_segs
                else:
                    lengths2_edge_array += [0] * num_new_segs

        # Prepare final point data dict
        p_data = {
            "index": new_vertex_index_array,
        }

        if pBC_array is not None:
            p_data["pBC"] = pBC_array

        if pressure_array is not None:
            p_data["pressure"] = pressure_array

        # Cell data: attribute by attribute
        # If attr does not exist, is not written.
        # If it existed but had None, it would have been replaced by default.
        cell_data = {
            "connectivity": connectivity_array,
            "index": edge_index_array,
        }

        if len(radius_array) == len(connectivity_array):
            cell_data["radius"] = radius_array

        if has_diameter and len(vessel_diameter_array) == len(connectivity_array):
            cell_data["vessel_diameter"] = vessel_diameter_array

        if has_length and len(vessel_length_array) == len(connectivity_array):
            cell_data["vessel_length"] = vessel_length_array

        if has_nkind and len(vessel_nkind_array) == len(connectivity_array):
            cell_data["vessel_nkind"] = vessel_nkind_array

        if has_lengths2 and len(lengths2_edge_array) == len(connectivity_array):
            cell_data["lengths2"] = lengths2_edge_array

        if has_diameters and len(diameters_edge_array) == len(connectivity_array):
            cell_data["diameters"] = diameters_edge_array

        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_array,
            point_data=p_data,
            cell_data=cell_data,
        )

    else:
        # NON-TORTUOUS: Direct source -> target mapping
        connectivity_non_tortuous = [(e.source, e.target) for e in G.es]

        point_data = {
            "index": new_vertex_index_array,
        }

        if pBC_array is not None:
            point_data["pBC"] = pBC_array

        if pressure_array is not None:
            point_data["pressure"] = pressure_array

        cell_data = {
            "connectivity": connectivity_non_tortuous,
        }

        # Attribute by attribute.
        # If it does not exist, it is not written.
        # If it existed but had None, it would have been replaced by default.
        diameter_array = safe_e_attr(G, "diameter", 0)
        length_array = safe_e_attr(G, "length", 0)
        nkind_array = safe_e_attr(G, "nkind", -1)

        if diameter_array is not None:
            cell_data["diameter"] = diameter_array

        if length_array is not None:
            cell_data["length"] = length_array

        if nkind_array is not None:
            cell_data["nkind"] = nkind_array

        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_non_tortuous,
            point_data=point_data,
            cell_data=cell_data,
        )

#------------------------------------------------------------------------------
# Load a graph from a pickle file
input_igraph_pkl_path = "/storage/homefs/ab25c720/MicroBrain/ParisGraph/halfbrain_18_igraph.pkl"

# If .pkl contains directly an igraph.Graph:
graph = igraph.Graph.Read_Pickle(input_igraph_pkl_path)

# If your .pkl has the outgeom format {"graph": G, "vertex": ..., "geom": ...},
# use this instead of Graph.Read_Pickle:
#
# with open(input_igraph_pkl_path, "rb") as f:
#     data = pickle.load(f)
# graph = data["graph"]

# print(graph.summary())

output_path = "/storage/homefs/ab25c720/MicroBrain/ParisGraph/halfbrain_18_igraph.vtp"

write_vtp(
    graph,
    output_path,
    tortuous=True,
)
