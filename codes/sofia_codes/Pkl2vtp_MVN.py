import numpy as np
from copy import deepcopy
import pickle
import igraph
import lxml.etree as xml
from collections import Counter,defaultdict
#------------------------------------------------------------------------------
def distance(v1, v2):
    """Distance between two vertices"""
    return np.linalg.norm(np.array(v1)-np.array(v2), ord=2)
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
        data = xml.SubElement(pointData, "DataArray", {"type":"Float64", "Name":name, "NumberOfComponents":"1", "format":"ascii"})
        if name == "pBC":
            data.text = " ".join(["{:10.15e}".format(p) if p is not None else "-1000" for p in data_array])
        else:
            print(data_array)
            data.text = " ".join(["{:10.15e}".format(p) for p in data_array])

    # cell data
    for name, data_array in cell_data.items():
        if name == "connectivity":
            data = xml.SubElement(cellData, "DataArray",
                                  {"type": "Float64", "Name": name, "NumberOfComponents": "2", "format": "ascii"})
            data.text = " ".join(["{:10.15e}".format(i) for c in connectivity_array for i in c])
        else:
            data = xml.SubElement(cellData, "DataArray",
                                  {"type": "Float64", "Name": name, "NumberOfComponents": "1", "format": "ascii"})
            data.text = " ".join(["{:10.15e}".format(r) for r in data_array])

    # points
    coords = xml.SubElement(points, "DataArray", {"type":"Float64", "NumberOfComponents":"3", "format":"ascii"})
    coords.text = " ".join(["{:10.15e}".format(i) for c in vertices_array for i in c])

    # lines
    connectivity = xml.SubElement(lines, "DataArray", {"type":"Int32", "Name":"connectivity", "format":"ascii"})
    offsets = xml.SubElement(lines, "DataArray", {"type":"Int32", "Name":"offsets", "format":"ascii"})
    connectivity.text = " ".join(["{}".format(i) for c in connectivity_array for i in c])
    offsets.text = " ".join(["{}".format((i+1)*2) + " " for i in range(len(connectivity_array))])

    with open(filename, 'w') as vtkOutput:
        vtkOutput.write(xml.tostring(vtkTag, pretty_print=True, encoding="unicode"))

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
    G.vs['index'] = list(range(G.vcount()))
    if G.ecount() > 0:
        G.es['index'] = list(range(G.ecount()))

    # Delete self-loops
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

    ############################################################################
    # --> PAY ATTENTION TO THE UNITS FOR THE DEFINITION OF THE MINIMUM DISTANCE !!!
    ############################################################################

    ############################################################################
    # Select and delete the edges with length 0
    for edge in G.es:
        ps = G.es[edge.index]['points']
        v1 = G.es[edge.index]['connectivity'][0]
        v2 = G.es[edge.index]['connectivity'][1]
        if distance(G.vs[v1]['coords'], G.vs[v2]['coords']) < 0.01 * 1e-6:
            ps = [G.vs[v1]['coords'], G.vs[v2]['coords']]
            G.es[edge.index]['length'] = 0.0
        # remove possibly repeated points in points
        ps = [ps[0], *[p for i, p in enumerate(ps[1:]) if distance(p, ps[i]) > 0.01 * 1e-6]]
        if len(ps) < 2:
            ps = [G.vs[v1]['coords'], G.vs[v2]['coords']]
        G.es[edge.index]['points'] = ps

    edges_to_remove = [e.index for e in G.es if e['length'] == 0]
    G.delete_edges(edges_to_remove)
    ############################################################################

    # Convert the substance dictionary to arrays
    if 'substance' in G.vs.attribute_names():
        substances = G.vs[0]['substance'].keys()
        for substance in substances:
            G.vs[substance] = [v['substance'][substance] for v in G.vs]
        del G.vs['substance']

    # Find unconnected vertices (indices)
    disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
    G.delete_vertices(disconnected_vertices)

    vertices_array = G.vs["coords"]
    pBC_array = [-1000 if value is None else value for value in G.vs['pBC']]
    #degree_array = G.vs["degree"]
    pressure_array = G.vs["pressure"]
    new_vertex_index_array = [*[j for j in range(len(vertices_array))]]



    if tortuous:
        connectivity_array = []
        radius_array = []
        vessel_diameter_array = []
        vessel_length_array = []
        edge_index_array = []
        vessel_nkind_array = []
        diameters_edge_array = []
        lengths2_edge_array = []


        # go over direct edges (bifurcation->bifurcation) and add tortuous points for each segment
        #
        # B.___________________________________.B    direct edge
        #   \____                  ___    ____/
        #         \_______________/   \__/           tortuous edges
        #
        print("-- Processing tortuous edges...")
        edge_index = 0
        for edge, vessel_length, ps, vessel_diameter, nkind, diameters, lengths2 in zip(
                G.es["connectivity"], G.es["length"], G.es["points"], G.es["diameter"], G.es["nkind"],
                G.es["diameters"], G.es["lengths2"]):
            assert len(ps) >= 2
            assert len(edge) == 2

            firstIndex = len(vertices_array)
            R = 0.5 * vessel_diameter
            n_points = len(ps)

            if n_points >= 10:
                step = 1 / 1000
            else:
                step = 1 / 10

                # new points and point data only if there is tortuous segments
            if len(np.where(np.all(vertices_array == ps[0], axis=1))[0]) == 1:
                if edge[0] != int(np.where(np.all(vertices_array == ps[0], axis=1))[0]):
                    node_0 = edge[1]
                    node_1 = edge[0]
                else:
                    node_0 = edge[0]
                    node_1 = edge[1]
            elif len(np.where(np.all(vertices_array == ps[-1], axis=1))[0]) == 1:
                if edge[1] != int(np.where(np.all(vertices_array == ps[-1], axis=1))[0]):
                    node_0 = edge[1]
                    node_1 = edge[0]
                else:
                    node_0 = edge[0]
                    node_1 = edge[1]

            if n_points > 2:
                vertices_array += ps[
                                  1:-1]  # add points except first and last which is the bifurcation point (already in the list)
                pBC_array += [-1000, ] * (n_points - 2)
                #degree_array += [2, ] * (n_points - 2)
                new_vertex_index_array += [*[edge_index + j * step for j in range(1, n_points-1)]]  # adding also the bifurcation point (first and last)
                p1, p2 = pressure_array[node_0], pressure_array[node_1]
                step_pr = (p2 - p1) / (n_points - 1.0)
                pressure_array += [*[p1 + j * step_pr for j in range(1, n_points - 1)]]  # add points except first and last

            # append tortuous edges
            assert n_points >= 2
            if n_points == 2:
                new_cells = [(node_0, node_1)]
            elif n_points == 3:
                new_cells = [(node_0, firstIndex), (firstIndex, node_1)]
            else:
                new_cells = [(node_0, firstIndex),
                             *[(firstIndex + i, firstIndex + i + 1) for i in range(n_points - 3)],
                             (firstIndex + n_points - 3, node_1)]

            connectivity_array += new_cells
            radius_array += [R, ] * len(new_cells)
            vessel_diameter_array += [vessel_diameter, ] * len(new_cells)
            vessel_length_array += [vessel_length, ] * len(new_cells)
            vessel_nkind_array += [nkind, ] * len(new_cells)
            edge_index_array += [*[edge_index + j * step for j in range(len(new_cells))]]
            diameters_edge_array += [*[(diameters[j] + diameters[j + 1]) / 2 for j in range(len(diameters) - 1)]]
            lengths2_edge_array += [*[j for j in lengths2]]
            edge_index += 1

        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_array,
            point_data={"pBC": pBC_array, "pressure": pressure_array, #"degree": degree_array,
                        "index": new_vertex_index_array},
            cell_data={"connectivity": connectivity_array, "radius": radius_array,
                       "vessel_diameter": vessel_diameter_array, "vessel_nkind": vessel_nkind_array,
                       "vessel_length": vessel_length_array, "index": edge_index_array, "lengths2": lengths2_edge_array,
                       "diameters": diameters_edge_array},
        )

    else:
        connectivity_non_tortuous = G.es["connectivity"]
        nkind_non_tortuous = G.es["nkind"]
        length_non_tortuous = G.es["length"]
        diameter_non_tortuous = G.es["diameter"]


        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_non_tortuous,
            point_data={"pBC": pBC_array, "pressure": pressure_array,#"degree": degree_array,
                        "index": new_vertex_index_array},
            cell_data={"connectivity": connectivity_non_tortuous, "diameter": diameter_non_tortuous, "length": length_non_tortuous, "nkind": nkind_non_tortuous},
        )

# Load a graph from a pickle file
input_igraph_pkl_path = "C:/Users/Ana/OneDrive/Escritorio/ARTORG/igraph/MVN1_CUT.pkl"


graph = igraph.Graph.Read_Pickle(input_igraph_pkl_path)
print(graph.summary())


output_path = "C:/Users/Ana/OneDrive/Escritorio/ARTORG/igraph/"

write_vtp(graph, output_path+'MVN1_CUT.vtp', False)

