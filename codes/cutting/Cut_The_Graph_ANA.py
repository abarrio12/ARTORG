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
    return np.linalg.norm(
        np.asarray(v1, dtype=np.float64) - np.asarray(v2, dtype=np.float64),
        ord=2
    )

def node_exists(graph, point, atol=1e-6):
    """Find an existing vertex by float coords (intersection points)."""
    p = np.asarray(point, dtype=np.float64)
    for v in graph.vs:
        if np.allclose(np.asarray(v["coords"], dtype=np.float64), p, atol=atol):
            return v.index
    return None


# ---------------------------
# Bounding box option 1 (Gaia-style cut, but in µm space)
# ---------------------------

def get_edges_in_boundingBox_vertex_based(xCoordsBox, yCoordsBox, zCoordsBox):
    """
    Classify edges by bbox and create NEW border edges.
    Assumptions:
      - G.vs["coords"] are in µm
      - e["points"] for ORIGINAL edges are in voxels (image voxels), so we scale ON THE FLY here
      - NEW border edges created here will be stored in µm and tagged with e["points_um"]=True
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
        # ORIGINAL edges are voxels -> scale to µm ON THE FLY
        points = np.asarray(e["points"], dtype=np.float64) * scale

        # robust orientation: compare source vertex to first/last point
        c_s = np.asarray(G.vs[e.source]["coords"], dtype=np.float64)
        p_first = points[0]
        p_last = points[-1]

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

        s = int(np.sum(vertices_in_box))
        if s == 2:
            edges_in_box.append(e.index)
            continue

        if s == 0:
            edges_outside_box.append(e.index)
            border_vertices.append(vertices[0])
            border_vertices.append(vertices[1])
            continue

        # across border
        edges_across_border.append(e.index)

        save_index = []
        consecutive = []
        for idx, p in enumerate(points):
            inside = 1 if is_inside_box(p, xCoordsBox, yCoordsBox, zCoordsBox) else 0
            save_index.append(idx)
            consecutive.append(inside)

        # run-length encoding
        frequency = []
        count = 1
        for i in range(1, len(consecutive)):
            if consecutive[i] == consecutive[i - 1]:
                count += 1
            else:
                frequency.append(count)
                count = 1
        frequency.append(count)

        if vertices_in_box[0] == 0:  # node_0 outside
            border_vertices.append(vertices[0])
            internal_point  = points[sum(frequency[:-1])]
            external_point  = points[sum(frequency[:-1]) - 1]
            internal_points = points[sum(frequency[:-1]):]
            save_index      = save_index[sum(frequency[:-1]):]
        else:  # node_1 outside
            border_vertices.append(vertices[1])
            internal_point  = points[frequency[0] - 1]
            external_point  = points[frequency[0]]
            internal_points = points[:frequency[0]]
            save_index      = save_index[:frequency[0]]

        # find crossed plane + intersect
        for axis, (min_val, max_val) in enumerate(
            zip([xCoordsBox[0], yCoordsBox[0], zCoordsBox[0]],
                [xCoordsBox[1], yCoordsBox[1], zCoordsBox[1]])
        ):
            if external_point[axis] < min_val or external_point[axis] > max_val:
                plane_val = min_val if external_point[axis] < min_val else max_val
                intersection_point = intersect_with_plane(internal_point, external_point, axis, plane_val)

                node_index = node_exists(G, intersection_point)
                if node_index is None:
                    node_index = len(G.vs)
                    G.add_vertices(1)
                    G.vs[-1]["coords"] = intersection_point.tolist()  # µm
                    G.vs[-1]["degree"] = 1
                    G.vs[-1]["index"]  = node_index

                if vertices_in_box[0] == 0:  # first node outside
                    new_points = np.vstack([intersection_point, internal_points])
                    G.add_edge(node_index, node_1)
                    new_edge = G.es[-1]
                    new_edge["connectivity"] = (node_index, node_1)
                else:  # second node outside
                    new_points = np.vstack([internal_points, intersection_point])
                    G.add_edge(node_0, node_index)
                    new_edge = G.es[-1]
                    new_edge["connectivity"] = (node_0, node_index)

                # Mark: this new edge already has points in µm
                new_edge["points_um"] = True

                # compute per-segment lengths in µm
                lengths2 = np.linalg.norm(np.diff(new_points, axis=0), axis=1)
                new_edge["lengths2"] = lengths2.tolist()

                # per-point "lengths" vector in Gaia-style (N), last repeats last segment
                if new_points.shape[0] >= 2:
                    lengths_vec = np.empty(new_points.shape[0], dtype=np.float64)
                    lengths_vec[:-1] = lengths2
                    lengths_vec[-1]  = lengths2[-1]
                else:
                    lengths_vec = np.zeros(new_points.shape[0], dtype=np.float64)

                new_edge["lengths"] = lengths_vec.tolist()
                new_edge["length"]  = float(lengths2.sum())

                # diameters/lengths per-point based on original indexing
                diameters = []
                lengths_old = []  # if you still want to preserve original per-point lengths from input

                if vertices_in_box[0] == 0:
                    diameters.append(e["diameters"][save_index[0]])
                    lengths_old.append(e["lengths"][save_index[0]])

                for i_idx in save_index:
                    diameters.append(e["diameters"][i_idx])
                    lengths_old.append(e["lengths"][i_idx])

                if vertices_in_box[0] == 1:
                    diameters.append(e["diameters"][save_index[-1]])
                    lengths_old.append(e["lengths"][save_index[-1]])

                new_edge["diameter"]  = e["diameter"]
                new_edge["nkind"]     = e["nkind"]
                new_edge["points"]    = new_points.tolist()  # µm
                new_edge["diameters"] = diameters

                # keep original "lengths" if you want, but it's in original units.
                # better to NOT keep it to avoid confusion. If you do, rename:
                new_edge["lengths_input"] = lengths_old

                # optional attributes
                for k in ("hd", "htt", "flow", "flow_rate", "rbc_velocity", "v"):
                    if k in G.es.attributes():
                        new_edge[k] = e[k]

                new_edges_on_border.append(new_edge.index)
                break

    return (np.unique(edges_in_box),
            np.unique(edges_across_border),
            np.unique(edges_outside_box),
            np.unique(border_vertices),
            np.unique(new_edges_on_border))


# ---------------------------
# EXECUTION
# ---------------------------

with open("/home/admin/Ana/MicroBrain/output18/18_igraph_FULLGEOM.pkl", "rb") as f:
    G = pickle.load(f)

# voxel -> µm scaling (image resolution)
sx, sy, sz = 1.625, 1.625, 2.5
scale = np.array([sx, sy, sz], dtype=np.float64)

# Make vertex coords be µm (derived from coords_image voxels)
coords_img = np.asarray(G.vs["coords_image"], dtype=np.float64)
G.vs["coords"] = (coords_img * scale).tolist()

# ParaView box in VOXELS -> convert to µm
xmin_pv, xmax_pv = 1000, 2000
ymin_pv, ymax_pv = 0, 1000
zmin_pv, zmax_pv = 1500, 2500

xCoordsBox_um = [xmin_pv * sx, xmax_pv * sx]
yCoordsBox_um = [ymin_pv * sy, ymax_pv * sy]
zCoordsBox_um = [zmin_pv * sz, zmax_pv * sz]

print("coords(µm) bounds:",
      np.min(np.asarray(G.vs["coords"], dtype=np.float64), axis=0),
      np.max(np.asarray(G.vs["coords"], dtype=np.float64), axis=0))
print("box (µm):", xCoordsBox_um, yCoordsBox_um, zCoordsBox_um)

# Ensure points_um attribute exists (False for all original edges)
if "points_um" not in G.es.attributes():
    G.es["points_um"] = [False] * G.ecount()

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

# Remove isolated vertices
disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
G.delete_vertices(disconnected_vertices)
G.vs["degree"] = G.degree()

# ---------------------------
# OPTION B (your request):
# Normalize final graph so ALL remaining edges store points in µm,
# AND recompute lengths2 / lengths / length consistently in µm,
# BUT do NOT double-scale new border edges (points_um=True)
# ---------------------------
for e in G.es:
    already_um = bool(e["points_um"]) if "points_um" in G.es.attributes() else False

    pts = np.asarray(e["points"], dtype=np.float64)

    # If original edge -> convert to µm now
    if not already_um:
        pts = pts * scale
        e["points"] = pts.tolist()
        e["points_um"] = True  # now it's µm too

    # Recompute lengths in µm (for EVERY edge now)
    if pts.shape[0] >= 2:
        lengths2_um = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        e["lengths2"] = lengths2_um.tolist()

        lengths_um = np.empty(pts.shape[0], dtype=np.float64)
        lengths_um[:-1] = lengths2_um
        lengths_um[-1]  = lengths2_um[-1]
        e["lengths"] = lengths_um.tolist()

        e["length"] = float(lengths2_um.sum())
    else:
        e["lengths2"] = []
        e["lengths"] = [0.0] * int(pts.shape[0])
        e["length"] = 0.0

# OPTIONAL: remove helper flag before saving (if you don't want it)
if "points_um" in G.es.attributes():
    del G.es["points_um"]

# OPTIONAL: if you created lengths_input and you don't want it, remove it:
if "lengths_input" in G.es.attributes():
    del G.es["lengths_input"]

# Save output as dicts (your format)
vertices_data = {attr: G.vs[attr] for attr in G.vs.attributes()}
edges_data = {attr: G.es[attr] for attr in G.es.attributes()}
edges_data["connectivity"] = G.get_edgelist()

with open("vertices_18_graph.pkl", "wb") as f:
    pickle.dump(vertices_data, f)

with open("edges_18_graph.pkl", "wb") as f:
    pickle.dump(edges_data, f)

print("Saved: vertices_18_graph.pkl and edges_18_graph.pkl")
