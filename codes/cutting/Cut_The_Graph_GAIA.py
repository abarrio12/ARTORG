import numpy as np
import pickle
import igraph
from collections import Counter, defaultdict

def intersect_with_plane(point1, point2, axis, value):
    # Linear interpolation to find intersection point
    # Assuming  axis is 0 for x, 1 for y, 2 for z
    if point2[axis] == point1[axis]:
        raise ValueError(
            f"Cannot calculate intersection when both points have the same {['x', 'y', 'z'][axis]} coordinate.")
    t = (value - point1[axis]) / (point2[axis] - point1[axis])
    return point1 + t * (point2 - point1)

def is_inside_box(point, xCoordsBox, yCoordsBox, zCoordsBox):
    x, y, z = point
    return xCoordsBox[0] <= x <= xCoordsBox[1] and yCoordsBox[0] <= y <= yCoordsBox[1] and zCoordsBox[0] <= z <= zCoordsBox[1]

def node_exists(graph, point):
    for v in graph.vs:
        if np.array_equal(v["coords"], point):
            return v.index  # Returns the node index if it exists
    return None

def distance(v1, v2):
    """Distance between two vertices"""
    return np.linalg.norm(np.array(v1)-np.array(v2), ord=2)
# --------------------------------------------------------------------------------------------------------------------------
# BOUNDING BOX - OPTION 1
# --------------------------------------------------------------------------------------------------------------------------

def get_edges_in_boundingBox_vertex_based(xCoordsBox, yCoordsBox,zCoordsBox):
    """ Outputs edges belonging to a given box.
    INPUT: xCoords = xmin,xmax
           yCoords = ymin,ymax
           zCoords = zmin,zmax

           !!! IN MICROMETERS !!!

    OUTPUT: edges_in_box: all edges completely in edge
            edges_across_border: edges with one vertex in box and one outside
            edges_outside_box: edges with both vertices outside
            border_vertices: vertices outside box
            new_edges_on_border: new edges delimited by old node (inside the box) and
                                 new node (intersection between edge across the border and box plane)
    """
    """
    The new inserted node is defined as follows:
    - coords: from the intersection
    - degree: 1
    - index: last of the list of nodes
    - pressure: linear distribution of the pressure along the vessel depending on the pressure in internal node and the one in the external node of the original edge 
                (across the border) 
    - pBC: defined as the 'pressure' above (the node is by definition on the boundary)
    
    The new inserted edge is defined as follows:
    - connectivity: (intersection, original_internal_node)
    - diameter: diameter from the original edge (vessel diameter)
    - nkind: nkind from the original edge (vessel nkind)
    - points: defined as intersection + internal_points_from_original_edge_across or 
              internal_points_from_original_edge_across + intersection (depending on which node is inside the box)
    - lengths2: defined as the distance between every consecutive point (= len(npoints) -1)
    - length: sum of the values in the attribute 'lengths2' of the new edge
    - diameters: for each internal_points_from_original_edge_across takes th value of the original edge and 
                 for the intersection takes the value of the closest internal_points_from_original_edge_across 
                 (one value for each point --> = len(npoints))
    - lengths: for each internal_points_from_original_edge_across takes th value of the original edge and 
               for the intersection takes the value of the closest internal_points_from_original_edge_across
               (one value for each point --> = len(npoints))
    
    PLEASE TAKE IN CONSIDERATION THE UNITS:
    -- pressure (mmHg)
    -- length/diameter (micrometers)
    """

    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []
    new_edges_on_border = []

    print("xCoordsBox: ", xCoordsBox)
    print("yCoordsBox: ", yCoordsBox)
    print("zCoordsBox: ", zCoordsBox)
    for e in G.es:
        points = e['points']
        #if G.vs[e.source]['coords'].all() != points[0].all(): coords is a tuple in my graph, points np array (N,3) can't use .all or == for float32
        coords_source = np.asarray(G.vs[e.source]["coords"], dtype=np.float32)
        p0 = np.asarray(e["points"][0], dtype=np.float32)

        if np.allclose(coords_source, p0, atol=1e-6):
            node_0 = e.source
            node_1 = e.target
        else:
            node_0 = e.target
            node_1 = e.source

        vertices = [node_0, node_1] #indices of the two vertices that are connected by the edge e
        vertices_in_box = [0, 0]

        for i, v in enumerate(vertices): #i is the index (0 or 1), and v is the vertex index (one of the 2 connected by the edge e)
            coords = G.vs[v]['coords']
            if is_inside_box(coords, xCoordsBox, yCoordsBox, zCoordsBox):
                vertices_in_box[i] = 1

        if np.sum(vertices_in_box) == 2:
            edges_in_box.append(e.index)
        elif np.sum(vertices_in_box) == 1:
            #print("vertices ", vertices ,"vertices in box", vertices_in_box[0], vertices_in_box[1], "coords", G.vs[e.source]['coords'], G.vs[e.target]['coords'] )
            #print("num vertici", len(G.vs))
            edges_across_border.append(e.index)
            save_index = []
            index = 0
            consecutive = []

            for p in points:
                if is_inside_box(p, xCoordsBox, yCoordsBox, zCoordsBox):
                    consecutive_index = 1
                else:
                    consecutive_index = 0
                save_index.append(index)
                consecutive.append(consecutive_index)
                index += 1

            frequency = []
            count = 1
            for i in range(1, len(consecutive)):
                if consecutive[i] == consecutive[i - 1]:
                    count += 1
                else:
                    frequency.append(count)
                    count = 1
            frequency.append(count)


            if vertices_in_box[0] == 0:  #first node (node_0) outside
                border_vertices.append(vertices[0])
                internal_point = points[sum(frequency[:-1])]
                external_point = points[sum(frequency[:-1])-1]
                internal_points = points[sum(frequency[:-1]):]
                save_index = save_index[sum(frequency[:-1]):]
            else: #second node (node_1) outside
                border_vertices.append(vertices[1])
                internal_point = points[frequency[0]-1]
                external_point = points[frequency[0]]
                internal_points = points[:frequency[0]]
                save_index = save_index[:frequency[0]]
            for axis, (min_val, max_val) in enumerate(zip([xCoordsBox[0], yCoordsBox[0], zCoordsBox[0]], [xCoordsBox[1], yCoordsBox[1], zCoordsBox[1]])):
                if external_point[axis] < min_val or external_point[axis] > max_val:

                    intersection_point = intersect_with_plane(internal_point, external_point, axis, min_val if external_point[axis] < min_val else max_val)
                    # check if the node is already in the nodes list before adding it
                    node_index = node_exists(G, intersection_point)
                    if node_index is None:
                        # adding the new vertex in the end of the vertices list
                        node_index = len(G.vs)
                        G.add_vertices(1)

                        # adding attributes for the new vertex
                        G.vs[-1]['coords'] = intersection_point
                        G.vs[-1]['degree'] = 1
                        G.vs[-1]['index'] = node_index

                        '''
                        step_pr = (G.vs[vertices[1]]['pressure'] - G.vs[vertices[0]]['pressure']) / (len(points)-1)
                        #linear distribution of the pressure along the vessel
                        G.vs[-1]['pBC'] = G.vs[vertices[0]]['pressure'] + step_pr * len(internal_points)
                        G.vs[-1]['pressure'] = G.vs[vertices[0]]['pressure'] + step_pr * len(internal_points)
                        '''


                    if vertices_in_box[0] == 0:  #first node (node_0) outside
                        # Define the points attribute
                        new_points = [intersection_point] + internal_points
                        #print(new_points[0], new_points[-1])
                        G.add_edge(node_index, node_1)
                        #print(G.vs[node_index]['coords'], G.vs[e.target]['coords'])
                        new_edge = G.es[-1]
                        new_edge["connectivity"] = (node_index, node_1)
                        #print(new_edge["connectivity"])
                        #print("new", new_points)
                        #print("old", points)
                    else:  # second node (node_1) outside
                        new_points = internal_points + [intersection_point]
                        #print(new_points[0], new_points[-1])
                        G.add_edge(node_0, node_index)
                        #print(G.vs[e.source]['coords'], G.vs[node_index]['coords'])
                        new_edge = G.es[-1]
                        new_edge["connectivity"] = (node_0, node_index)
                        #print(new_edge["connectivity"])
                        #print("new", new_points)
                        #print("old", points)

                    # Set the edge attributes
                    lengths2 = [distance(new_points[i], new_points[i+1]) for i in range(len(new_points)-1)]
                    diameters = []
                    lengths = []
                    if vertices_in_box[0] == 0:  # first node outside
                        diameters.append(e['diameters'][save_index[0]])
                        lengths.append(e['lengths'][save_index[0]])
                    for i in save_index:
                        diameters.append(e['diameters'][i])
                        lengths.append(e['lengths'][i])
                    if vertices_in_box[0] == 1:  # second node outside
                        diameters.append(e['diameters'][save_index[-1]])
                        lengths.append(e['lengths'][save_index[-1]])

                    new_edge["diameter"] = e['diameter']  # Copying the diameter from the original edge (vessel diameter)
                    new_edge["length"] = sum(lengths2)
                    new_edge["nkind"] = e['nkind'] # Copying the nkind from the original edge (vessel nkind)
                    new_edge["lengths2"] = lengths2
                    new_edge["points"] = new_points
                    new_edge["diameters"] = diameters
                    new_edge["lengths"] = lengths

                    if "hd" in G.es.attributes():
                        new_edge["hd"] = e['hd']

                    if "htt" in G.es.attributes():
                        new_edge["htt"] = e['htt']

                    if "flow" in G.es.attributes():
                        new_edge["flow"] = e['flow']

                    if "flow_rate" in G.es.attributes():
                        new_edge["flow_rate"] = e['flow_rate']

                    if "rbc_velocity" in G.es.attributes():
                        new_edge["rbc_velocity"] = e['rbc_velocity']

                    if "v" in G.es.attributes():
                        new_edge["v"] = e['v']

                    new_edges_on_border.append(new_edge.index)
                    break

        else:
            edges_outside_box.append(e.index)
            border_vertices.append(vertices[0])
            border_vertices.append(vertices[1])


    edges_in_box = np.unique(edges_in_box)
    edges_outside_box = np.unique(edges_outside_box)
    edges_across_border = np.unique(edges_across_border)
    border_vertices = np.unique(border_vertices)
    new_edges_on_border = np.unique(new_edges_on_border)

    return edges_in_box, edges_across_border, edges_outside_box, border_vertices, new_edges_on_border

# --------------------------------------------------------------------------------------------------------------------------
# BOUNDING BOX - OPTION 1 - END
# --------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------
# BOUNDING BOX - OPTION 2
# --------------------------------------------------------------------------------------------------------------------------

def get_edges_in_boundingBox_vertex_based_2(xCoordsBox=[1300,1700],yCoordsBox=[1400,1800],zCoordsBox=[400, 800]):
    """ Outputs edges belonging to a given box.
    INPUT: xCoords = xmin,xmax
           yCoords = ymin,ymax
           zCoords = zmin,zmax
    OUTPUT: edges_in_box: all edges completely in edge
            edges_across_border: edges with one vertex in box and one outside
            edges_outside_box: edges with both vertices outside
            border_vertices: vertices outside box
    """

    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []
    print("xCoordsBox: ", xCoordsBox)
    print("yCoordsBox: ", yCoordsBox)
    print("zCoordsBox: ", zCoordsBox)
    for e in G.es:
        vertices = [e.source, e.target] #indices of the two vertices that are connected by the edge e
        vertices_in_box = [0, 0]
        points = e['points']
        internal_points = [p for p in points if is_inside_box(p, xCoordsBox, yCoordsBox, zCoordsBox)]

        for i, v in enumerate(vertices):
            coords = G.vs[v]['coords']
            if coords[0] >= xCoordsBox[0] and coords[0] <= xCoordsBox[1]:
                if coords[1] >= yCoordsBox[0] and coords[1] <= yCoordsBox[1]:
                    if coords[2] >= zCoordsBox[0] and coords[2] <= zCoordsBox[1]:
                        vertices_in_box[i] = 1
        if np.sum(vertices_in_box) == 2:
            edges_in_box.append(e.index)
        elif np.sum(vertices_in_box) == 1:
            edges_across_border.append(e.index)
            if vertices_in_box[0] == 0:
                border_vertices.append(vertices[0])
            else:
                border_vertices.append(vertices[1])
        else:
            edges_outside_box.append(e.index)


    edges_in_box = np.unique(edges_in_box)
    edges_outside_box = np.unique(edges_outside_box)
    edges_across_border = np.unique(edges_across_border)
    border_vertices = np.unique(border_vertices)

    return edges_in_box, edges_across_border, edges_outside_box, border_vertices

# --------------------------------------------------------------------------------------------------------------------------
# BOUNDING BOX - OPTION 2 - END
# --------------------------------------------------------------------------------------------------------------------------



# ============= CODE EXECUTION ============


with open("/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl", "rb") as f:
    data = pickle.load(f)

G = data["graph"] # when graph as diccionary ~ outgeom
x = data["coords"]["x"]
y = data["coords"]["y"]
z = data["coords"]["z"]

coords_v = np.asarray(G.vs["coords"], dtype=float)
print("VERTEX coords bounds:", coords_v.min(axis=0), coords_v.max(axis=0))# bounds de los points de edges (esto es lo que ParaView suele mostrar)
allp = []
for e in G.es[:1000]:  # con 1000 ya basta para ver escala (evita petarlo)
    p = np.asarray(e["points"], dtype=float)
    allp.append(p)
allp = np.vstack(allp)
print("EDGE points bounds (sample):", allp.min(axis=0), allp.max(axis=0))
 


vertices_data = {attr: G.vs[attr] for attr in G.vs.attributes()}
edges_data = {attr: G.es[attr] for attr in G.es.attributes()}
edges_data["connectivity"] = G.get_edgelist()


'''
# Open the .pkl file in binary read mode ('rb') with specific protocol
with open("edges_Artemisia_Entire_MVN.pkl", 'rb') as file:
    # Load data from the .pkl file using the appropriate protocol
    edges_data = pickle.load(file, encoding='latin1')

with open("vertices_Artemisia_Entire_MVN.pkl", 'rb') as file:
    # Load data from the .pkl file using the appropriate protocol
    vertices_data = pickle.load(file, encoding='latin1')
'''


edge_list = np.array(edges_data['connectivity'])

G = igraph.Graph(edge_list.tolist())  # Generate igraph based on edge_list

# assign all edge and vertex attributes
for edge_attribute in edges_data:
    G.es[edge_attribute] = edges_data[edge_attribute]

for node_attribute in vertices_data:
    G.vs[node_attribute] = vertices_data[node_attribute]

print("Number of vertices:", len(G.vs))
print("Number of edges:", len(G.es))

print(G.summary())

coords = np.array(G.vs['coords'])

# Find the min and max for x, y, z
x_min, y_min, z_min = np.min(coords, axis=0)
x_max, y_max, z_max = np.max(coords, axis=0)

print("Original xCoordsBox: [", x_min, ",", x_max, "]")
print("Original yCoordsBox: [", y_min, ",", y_max, "]")
print("Original zCoordsBox: [", z_min, ",", z_max, "]")


coords = np.array(G.vs['coords'])
coords_img = np.array(G.vs['coords_image'])



    # Image resolution (µm / voxel)
res = np.array([1.625, 1.625, 2.5], dtype=float)

    # Box center (voxels)
center = np.array([2100, 4200, 750], dtype=float)

    # Box physical size (µm)
box_um = np.array([400, 400, 400], dtype=float)

box_vox = box_um / res
xBox = [center[0] - box_vox[0]/2, center[0] + box_vox[0]/2]
yBox = [center[1] - box_vox[1]/2, center[1] + box_vox[1]/2]
zBox = [center[2] - box_vox[2]/2, center[2] + box_vox[2]/2]

# get_edges_in_boundingBox_vertex_based

# Consistency check
coords_um = np.asarray(G.vs["coords"], float)
print("coords(um) bounds", coords_um.min(0), coords_um.max(0))
print("box (um)", xBox, yBox, zBox)


# =========================                     Execute option 1                =============================================

edges_in_box, edges_across_border, edges_outside_box, border_vertices, new_edges_on_border = get_edges_in_boundingBox_vertex_based(xCoordsBox=xBox, yCoordsBox=yBox, zCoordsBox=zBox)

# =========================                     Execute option 2                =============================================


#edges_in_box, edges_across_border, edges_outside_box, border_vertices = get_edges_in_boundingBox_vertex_based_2(xCoordsBox=[1300, 1700], yCoordsBox=[1400, 1800], zCoordsBox=[0, 400])

#all_edges = np.concatenate([edges_in_box])
all_edges = np.concatenate([edges_in_box, new_edges_on_border])
print("Number of edges in the box: ", len(edges_in_box))
print("Number of edges across the border: ", len(edges_across_border))
print("Number of edges outside the box: ", len(edges_outside_box))
print("Number of edges in the box + edges across the border: ", len(all_edges))
print("Number of new edges on the border: ", len(new_edges_on_border))
edges_to_delete = list(set(range(G.ecount())) - set(all_edges))
print("Number of edges to delete: ", len(edges_to_delete))
print("Number of border vertices: ", len(border_vertices))
G.delete_edges(edges_to_delete)
G.delete_vertices(border_vertices)
disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
G.delete_vertices(disconnected_vertices)
G.vs['degree'] = G.degree()




# Converti ogni numpy array in una tupla e mappa le coordinate agli indici
coord_to_indices = defaultdict(list)

# Popola il dizionario coord_to_indices con le coordinate come chiavi e gli indici come valori
for i, coord in enumerate(G.vs['coords']):
    coord_as_tuple = tuple(coord)  # Converte numpy array in tupla
    coord_to_indices[coord_as_tuple].append(i)

# Trova le coordinate duplicate, cioè quelle con più di un indice
#duplicate_indices = {coord: indices for coord, indices in coord_to_indices.items() if len(indices) > 1}

#if duplicate_indices:
    #    print("Nodes with coordinates duplicate:")
    #    for coord, indices in duplicate_indices.items():
    #       print(f"Coordinates {coord} found in the nodes with index: {indices}")
#else:
    #    print("No nodes with coordinates duplicate.")


vertices_data = {}
for attribute_name in G.vs.attributes():
    vertices_data[attribute_name] = G.vs[attribute_name]

edges_data = {}
for attribute_name in G.es.attributes():
    edges_data[attribute_name] = G.es[attribute_name]
edges_data['connectivity'] = G.get_edgelist()

print("New number of vertices:", len(G.vs))
print("New number of edges:", len(G.es))


print(G.summary())

total_vertices = len(G.vs)
total_edges = len(G.es)


# Get the last 290 vertices
#last_300_vertices = G.vs[total_vertices - 300:total_vertices]
#last_300_edges = G.es[total_edges - 300:total_edges]

# Print the last 290 vertices
#for vertex in last_300_vertices:
#    if is_inside_box(vertex["coords"], xCoordsBox=[1300,1700],yCoordsBox=[1400,1800],zCoordsBox=[0, 400]):
#        print(vertex.index, vertex.attributes(), "okay")
#    else:
#        print(vertex.index, vertex.attributes(), "ops")
#for edge in last_300_edges:
#   print(edge.index, edge['nkind'], edge['diameter'], len(edge['points']), edge['connectivity'], edge['points'])

'''
# Save vertices data to pickle file
with open('vertices_CUT_Artemisia.pkl', 'wb') as f:
    pickle.dump(vertices_data, f)

# Save edges data to pickle file
with open('edges_CUT_Artemisia.pkl', 'wb') as f:
    pickle.dump(edges_data, f)

'''

with open("vertices_18_graph.pkl", "wb") as f:
    pickle.dump(vertices_data, f)

with open("edges_18_graph.pkl", "wb") as f:
    pickle.dump(edges_data, f)


