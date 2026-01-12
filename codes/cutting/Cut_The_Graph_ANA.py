import numpy as np
import pickle
import igraph
from collections import defaultdict

# ---------------------------
# Helpers
# ---------------------------

def intersect_with_plane(point1, point2, axis, value):
    """Linear interpolation intersection between segment (point1->point2) and plane axis=value."""
    if point2[axis] == point1[axis]:
        raise ValueError(
            f"Cannot calculate intersection when both points have the same {['x','y','z'][axis]} coordinate."
        )
    t = (value - point1[axis]) / (point2[axis] - point1[axis])
    return point1 + t * (point2 - point1)

def is_inside_box(point, xCoordsBox, yCoordsBox, zCoordsBox):
    x, y, z = point
    return (xCoordsBox[0] <= x <= xCoordsBox[1] and
            yCoordsBox[0] <= y <= yCoordsBox[1] and
            zCoordsBox[0] <= z <= zCoordsBox[1])

def distance(v1, v2):
    return np.linalg.norm(np.asarray(v1, dtype=np.float64) - np.asarray(v2, dtype=np.float64), ord=2)

# CHANGE 1:  node_exists for float coords (intersection points)
def node_exists(graph, point, atol=1e-6):
    p = np.asarray(point, dtype=np.float64)
    for v in graph.vs:
        if np.allclose(np.asarray(v["coords"], dtype=np.float64), p, atol=atol):
            return v.index
    return None

# ---------------------------
# Bounding box option 1 (Gaia style)
# ---------------------------

def get_edges_in_boundingBox_vertex_based(xCoordsBox, yCoordsBox, zCoordsBox):
    """
    Outputs edges belonging to a given box.
    INPUT box is assumed to be in the SAME units as G.vs['coords'] and e['points'] checks (here: µm).
    """

    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []
    new_edges_on_border = []

    print("xCoordsBox:", xCoordsBox)
    print("yCoordsBox:", yCoordsBox)
    print("zCoordsBox:", zCoordsBox)

    for e in G.es:
        # points are stored in voxels -> scale to µm ON THE FLY (no big memory duplication)
        points = np.asarray(e["points"], dtype=np.float64) * scale

        #CHANGE 2:  endpoint orientation (use distance to points[0] vs points[-1])
        c_s = np.asarray(G.vs[e.source]["coords"], dtype=np.float64)
        p_first = points[0]
        p_last  = points[-1]

        if np.linalg.norm(c_s - p_first) <= np.linalg.norm(c_s - p_last):
            node_0, node_1 = e.source, e.target
        else:
            node_0, node_1 = e.target, e.source

        vertices = [node_0, node_1]
        vertices_in_box = [0, 0]

        for i, v in enumerate(vertices):
            coords = np.asarray(G.vs[v]["coords"], dtype=np.float64)
            if is_inside_box(coords, xCoordsBox, yCoordsBox, zCoordsBox):
                vertices_in_box[i] = 1

        if np.sum(vertices_in_box) == 2:
            edges_in_box.append(e.index)
            continue

        if np.sum(vertices_in_box) == 0:
            edges_outside_box.append(e.index)
            border_vertices.append(vertices[0])
            border_vertices.append(vertices[1])
            continue

        # across border
        edges_across_border.append(e.index)

        save_index = []
        consecutive = []
        for idx, p in enumerate(points):
            consecutive_index = 1 if is_inside_box(p, xCoordsBox, yCoordsBox, zCoordsBox) else 0
            save_index.append(idx)
            consecutive.append(consecutive_index)

        # run-length encoding lengths
        frequency = []
        count = 1
        for i in range(1, len(consecutive)):
            if consecutive[i] == consecutive[i - 1]:
                count += 1
            else:
                frequency.append(count)
                count = 1
        frequency.append(count)

        if vertices_in_box[0] == 0:  # first node outside (node_0 outside)
            border_vertices.append(vertices[0])
            internal_point  = points[sum(frequency[:-1])]
            external_point  = points[sum(frequency[:-1]) - 1]
            internal_points = points[sum(frequency[:-1]):]
            save_index      = save_index[sum(frequency[:-1]):]
        else:  # second node outside (node_1 outside)
            border_vertices.append(vertices[1])
            internal_point  = points[frequency[0] - 1]
            external_point  = points[frequency[0]]
            internal_points = points[:frequency[0]]
            save_index      = save_index[:frequency[0]]

        # determine which plane is crossed
        for axis, (min_val, max_val) in enumerate(
            zip([xCoordsBox[0], yCoordsBox[0], zCoordsBox[0]],
                [xCoordsBox[1], yCoordsBox[1], zCoordsBox[1]])
        ):
            if external_point[axis] < min_val or external_point[axis] > max_val:
                plane_val = min_val if external_point[axis] < min_val else max_val
                intersection_point = intersect_with_plane(internal_point, external_point, axis, plane_val)

                # check if already exists
                node_index = node_exists(G, intersection_point)
                if node_index is None:
                    node_index = len(G.vs)
                    G.add_vertices(1)
                    G.vs[-1]["coords"] = intersection_point.tolist()  # keep vertex coords in µm
                    G.vs[-1]["degree"] = 1
                    G.vs[-1]["index"]  = node_index

                if vertices_in_box[0] == 0:  # first node outside
                    new_points = np.vstack([intersection_point, internal_points])
                    G.add_edge(node_index, node_1)
                    new_edge = G.es[-1]
                    new_edge["connectivity"] = (node_index, node_1)
                    new_edge["points_um"] = True
                else:  # second node outside
                    new_points = np.vstack([internal_points, intersection_point])
                    G.add_edge(node_0, node_index)
                    new_edge = G.es[-1]
                    new_edge["connectivity"] = (node_0, node_index)
                    new_edge["points_um"] = True


                # edge attributes
                lengths2 = [distance(new_points[i], new_points[i + 1]) for i in range(len(new_points) - 1)]

                # diameters/lengths per-point based on original indexing
                diameters = []
                lengths = []

                if vertices_in_box[0] == 0:  # first node outside
                    diameters.append(e["diameters"][save_index[0]])
                    lengths.append(e["lengths"][save_index[0]])

                for i_idx in save_index:
                    diameters.append(e["diameters"][i_idx])
                    lengths.append(e["lengths"][i_idx])

                if vertices_in_box[0] == 1:  # second node outside
                    diameters.append(e["diameters"][save_index[-1]])
                    lengths.append(e["lengths"][save_index[-1]])

                new_edge["diameter"]  = e["diameter"]
                new_edge["length"]    = float(np.sum(lengths2))
                new_edge["nkind"]     = e["nkind"]
                new_edge["lengths2"]  = lengths2
                new_edge["points"]    = new_points.tolist()  # stored in µm for the new edges
                new_edge["diameters"] = diameters
                new_edge["lengths"]   = lengths

                # optional attributes if present
                for k in ("hd", "htt", "flow", "flow_rate", "rbc_velocity", "v"):
                    if k in G.es.attributes():
                        new_edge[k] = e[k]

                new_edges_on_border.append(new_edge.index)
                break  # done with this edge

    edges_in_box = np.unique(edges_in_box)
    edges_outside_box = np.unique(edges_outside_box)
    edges_across_border = np.unique(edges_across_border)
    border_vertices = np.unique(border_vertices)
    new_edges_on_border = np.unique(new_edges_on_border)

    return edges_in_box, edges_across_border, edges_outside_box, border_vertices, new_edges_on_border


# ---------------------------
# Bounding box option 2 (if you still want it)
# ---------------------------

def get_edges_in_boundingBox_vertex_based_2(xCoordsBox, yCoordsBox, zCoordsBox):
    """Simpler bbox classification. NOTE: uses points scaled to µm as well."""
    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []

    print("xCoordsBox:", xCoordsBox)
    print("yCoordsBox:", yCoordsBox)
    print("zCoordsBox:", zCoordsBox)

    for e in G.es:
        vertices = [e.source, e.target]
        vertices_in_box = [0, 0]

        # ✅ keep consistent units
        points = np.asarray(e["points"], dtype=np.float64) * scale
        _ = [p for p in points if is_inside_box(p, xCoordsBox, yCoordsBox, zCoordsBox)]  # optional

        for i, v in enumerate(vertices):
            coords = np.asarray(G.vs[v]["coords"], dtype=np.float64)
            if is_inside_box(coords, xCoordsBox, yCoordsBox, zCoordsBox):
                vertices_in_box[i] = 1

        if np.sum(vertices_in_box) == 2:
            edges_in_box.append(e.index)
        elif np.sum(vertices_in_box) == 1:
            edges_across_border.append(e.index)
            border_vertices.append(vertices[0] if vertices_in_box[0] == 0 else vertices[1])
        else:
            edges_outside_box.append(e.index)

    return (np.unique(edges_in_box),
            np.unique(edges_across_border),
            np.unique(edges_outside_box),
            np.unique(border_vertices))


# ---------------------------
# EXECUTION
# ---------------------------

with open("/home/admin/Ana/MicroBrain/output18/18_igraph_FULLGEOM.pkl", "rb") as f:
    G = pickle.load(f)

# conversion um/voxel
sx, sy, sz = 1.625, 1.625, 2.5
scale = np.array([sx, sy, sz], dtype=np.float64)

# IMPORTANT: vertex coords in µm (derived from coords_image)
coords_img = np.asarray(G.vs["coords_image"], dtype=np.float64)
G.vs["coords"] = (coords_img * scale).tolist()

# choose box from ParaView (voxels) -> convert to µm
xmin_pv, xmax_pv = 1000, 2000
ymin_pv, ymax_pv = 0, 1000
zmin_pv, zmax_pv = 1500, 2500

xCoordsBox_um = [xmin_pv * sx, xmax_pv * sx]
yCoordsBox_um = [ymin_pv * sy, ymax_pv * sy]
zCoordsBox_um = [zmin_pv * sz, zmax_pv * sz]

print("coords(µm) bounds:", np.min(np.asarray(G.vs["coords"]), axis=0), np.max(np.asarray(G.vs["coords"]), axis=0))
print("box (µm):", xCoordsBox_um, yCoordsBox_um, zCoordsBox_um)

edges_in_box, edges_across_border, edges_outside_box, border_vertices, new_edges_on_border = \
    get_edges_in_boundingBox_vertex_based(xCoordsBox_um, yCoordsBox_um, zCoordsBox_um)

all_edges = np.concatenate([edges_in_box, new_edges_on_border])
print("Number of edges in the box:", len(edges_in_box))
print("Number of edges across the border:", len(edges_across_border))
print("Number of edges outside the box:", len(edges_outside_box))
print("Number of new edges on the border:", len(new_edges_on_border))

edges_to_delete = list(set(range(G.ecount())) - set(all_edges))
print("Number of edges to delete:", len(edges_to_delete))
print("Number of border vertices:", len(border_vertices))

G.delete_edges(edges_to_delete)
G.delete_vertices(border_vertices)

disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
G.delete_vertices(disconnected_vertices)
G.vs["degree"] = G.degree()

# --- NORMALIZE: store all remaining edge points in µm ---
# Original edges: points were in voxels -> scale them now
# New border edges: already in µm -> do NOT scale again
if "points_um" not in G.es.attributes():
    # in case you run on a graph without new edges created
    G.es["points_um"] = [False] * G.ecount()
    for e in G.es:
        already_um = bool(e["points_um"])
        if not already_um:
            pts = np.asarray(e["points"], dtype=np.float64)
            e["points"] = (pts * scale).tolist()


# Save
vertices_data = {attr: G.vs[attr] for attr in G.vs.attributes()}

# Optional: remove helper flag before saving
if "points_um" in G.es.attributes():
    del G.es["points_um"]
    
edges_data = {attr: G.es[attr] for attr in G.es.attributes()}
edges_data["connectivity"] = G.get_edgelist()

with open("vertices_18_graph.pkl", "wb") as f:
    pickle.dump(vertices_data, f)

with open("edges_18_graph.pkl", "wb") as f:
    pickle.dump(edges_data, f)
