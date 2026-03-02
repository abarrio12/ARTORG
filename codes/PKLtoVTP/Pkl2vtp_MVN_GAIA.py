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
    
    # We maintain the index attributes, but we will rely on graph topology for connectivity
    G.vs['index'] = list(range(G.vcount()))
    if G.ecount() > 0:
        G.es['index'] = list(range(G.ecount())) 

    # Delete self-loops
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

    ############################################################################
    # --> PAY ATTENTION TO THE UNITS FOR THE DEFINITION OF THE MINIMUM DISTANCE !!!
    ############################################################################

    # Select and delete the edges with length 0
    # FIX: We use edge.tuple (live indices) instead of 'connectivity' attribute
    edges_to_remove = []
    for edge in G.es:
        v1, v2 = edge.tuple
        ps = edge['points']
        
        if distance(G.vs[v1]['coords'], G.vs[v2]['coords']) < 0.01 * 1e-6:
            ps = [G.vs[v1]['coords'], G.vs[v2]['coords']]
            edge['length'] = 0.0
            edges_to_remove.append(edge.index)
        
        # remove possibly repeated points in points
        ps = [ps[0], *[p for i, p in enumerate(ps[1:]) if distance(p, ps[i]) > 0.01 * 1e-6]]
        if len(ps) < 2:
            ps = [G.vs[v1]['coords'], G.vs[v2]['coords']]
        edge['points'] = ps

    G.delete_edges(edges_to_remove)
    ############################################################################

    # Convert the substance dictionary to arrays
    if 'substance' in G.vs.attribute_names():
        substances = G.vs[0]['substance'].keys()
        for substance in substances:
            G.vs[substance] = [v['substance'][substance] for v in G.vs]
        del G.vs['substance']

    # Find unconnected vertices (indices) and delete them
    disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
    G.delete_vertices(disconnected_vertices)

    # Prepare base point arrays from the cleaned graph
    vertices_array = list(G.vs["coords"])
    
    # Check for optional attributes safely
    pBC_array = None
    if 'pBC' in G.vs.attribute_names():
        pBC_array = [-1000 if value is None else value for value in G.vs['pBC']]
        
    pressure_array = None
    if 'pressure' in G.vs.attribute_names():
        pressure_array = list(G.vs["pressure"])

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

        # go over direct edges (bifurcation->bifurcation) and add tortuous points for each segment
        #
        # B.___________________________________.B    direct edge
        #   \____                  ___    ____/
        #         \_______________/   \__/           tortuous edges
        #
        print("-- Processing tortuous edges...")
        
        # FIX: iterate over edge objects to get LIVE source/target indices
        for edge_obj in G.es:
            ps = edge_obj['points']
            vessel_length = edge_obj['length']
            vessel_diameter = edge_obj['diameter']
            nkind = edge_obj['nkind']
            diameters = edge_obj['diameters']
            lengths2 = edge_obj['lengths2']
            edge_index = edge_obj.index # The current edge ID

            assert len(ps) >= 2
            
            node_0 = edge_obj.source
            node_1 = edge_obj.target

            firstIndex = len(vertices_array)
            R = 0.5 * vessel_diameter
            n_points = len(ps)

            if n_points >= 10:
                step = 1 / 1000
            else:
                step = 1 / 10

            # ORIENTATION CHECK: Ensure ps[0] is physically near node_0
            # This replaces the unreliable 'np.where' search
            dist_0_start = distance(vertices_array[node_0], ps[0])
            dist_0_end = distance(vertices_array[node_0], ps[-1])
            if dist_0_start > dist_0_end:
                ps = ps[::-1]

            if n_points > 2:
                vertices_array += ps[1:-1]  # add points except first and last
                if pBC_array is not None:
                    pBC_array += [-1000, ] * (n_points - 2)
                
                new_vertex_index_array += [edge_index + j * step for j in range(1, n_points-1)]
                
                if pressure_array is not None:
                    p1, p2 = pressure_array[node_0], pressure_array[node_1]
                    step_pr = (p2 - p1) / (n_points - 1.0)
                    pressure_array += [p1 + j * step_pr for j in range(1, n_points - 1)]

            # append tortuous edges (the connectivity "thread")
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
            num_new_segs = len(new_cells)
            
            radius_array += [R, ] * num_new_segs
            vessel_diameter_array += [vessel_diameter, ] * num_new_segs
            vessel_length_array += [vessel_length, ] * num_new_segs
            vessel_nkind_array += [nkind, ] * num_new_segs
            edge_index_array += [edge_index + j * step for j in range(num_new_segs)]
            
            if diameters is not None:
                diameters_edge_array += [(diameters[j] + diameters[j + 1]) / 2 for j in range(len(diameters) - 1)]
            if lengths2 is not None:
                lengths2_edge_array += [j for j in lengths2]

        # Prepare final point data dict
        p_data = {"index": new_vertex_index_array}
        if pBC_array is not None: p_data["pBC"] = pBC_array
        if pressure_array is not None: p_data["pressure"] = pressure_array

        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_array,
            point_data=p_data,
            cell_data={
                "connectivity": connectivity_array, 
                "radius": radius_array,
                "vessel_diameter": vessel_diameter_array, 
                "vessel_nkind": vessel_nkind_array,
                "vessel_length": vessel_length_array, 
                "index": edge_index_array, 
                "lengths2": lengths2_edge_array,
                "diameters": diameters_edge_array
            },
        )

    else:
        # NON-TORTUOUS: Direct source -> target mapping
        connectivity_non_tortuous = [(e.source, e.target) for e in G.es]
        
        WriteOnFileVTP(
            filename=filename,
            vertices_array=vertices_array,
            connectivity_array=connectivity_non_tortuous,
            point_data={"index": new_vertex_index_array},
            cell_data={
                "connectivity": connectivity_non_tortuous, 
                "diameter": G.es["diameter"], 
                "length": G.es["length"], 
                "nkind": G.es["nkind"]
            },
        )
# Load a graph from a pickle file
input_igraph_pkl_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"

graph = igraph.Graph.Read_Pickle(input_igraph_pkl_path)
#print(graph.summary())


output_path = "/home/admin/Ana/MicroBrain/output/vtp/"

write_vtp(graph, output_path+'graph_18_OutGeom_um_tortuous.vtp', tortuous=True)
write_vtp(graph, output_path+'graph_18_OutGeom_um_straight.vtp', tortuous=False)
