'''
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
    
    # ---- DEBUG: how many vertices are inside the box? (done once) ----
    V = np.array(G.vs["coords"], dtype=np.float64)
    inside_v = (
        (xCoordsBox[0] <= V[:, 0]) & (V[:, 0] <= xCoordsBox[1]) &
        (yCoordsBox[0] <= V[:, 1]) & (V[:, 1] <= yCoordsBox[1]) &
        (zCoordsBox[0] <= V[:, 2]) & (V[:, 2] <= zCoordsBox[1])
    )
    
    both_in = 0
    one_in = 0
    for e in G.es:
        a, b = e.tuple
        ia, ib = inside_v[a], inside_v[b]
        if ia and ib:
            both_in += 1
        elif ia or ib:
            one_in += 1
    
    print("Edges with BOTH endpoints inside:", both_in)
    print("Edges with ONE endpoint inside:", one_in)

    # ---------------------------------------------------------------

    for e in G.es:
        # ORIGINAL edges are voxels -> scale to µm ON THE FLY
        points = np.asarray(e["points"], dtype=np.float64) * scale
        # any polyline point inside?
        any_point_inside = np.any(
            (xCoordsBox[0] <= points[:,0]) & (points[:,0] <= xCoordsBox[1]) &
            (yCoordsBox[0] <= points[:,1]) & (points[:,1] <= yCoordsBox[1]) &
            (zCoordsBox[0] <= points[:,2]) & (points[:,2] <= zCoordsBox[1])
        )
        
        if any_point_inside:
            edges_in_box.append(e.index)
        else:
            edges_outside_box.append(e.index)
        continue

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
            #border_vertices.append(vertices[0])
            #border_vertices.append(vertices[1])

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
                
                # Points (µm)
                new_edge["points"] = new_points.tolist()
                
                # Recompute lengths in µm from new_points
                if new_points.shape[0] >= 2:
                    lengths2_um = np.linalg.norm(np.diff(new_points, axis=0), axis=1)
                    new_edge["lengths2"] = lengths2_um.tolist()
                
                    lengths_um = np.empty(new_points.shape[0], dtype=np.float64)
                    lengths_um[:-1] = lengths2_um
                    lengths_um[-1]  = lengths2_um[-1]
                    new_edge["lengths"] = lengths_um.tolist()
                
                    new_edge["length"] = float(lengths2_um.sum())
                else:
                    new_edge["lengths2"] = []
                    new_edge["lengths"]  = [0.0] * int(new_points.shape[0])
                    new_edge["length"]   = 0.0
                
                # Diameters: copy from original by indices, and add one for the intersection
                diameters = []
                
                if vertices_in_box[0] == 0:
                    # intersection at start -> use first internal point’s diameter
                    diameters.append(e["diameters"][save_index[0]])
                for i_idx in save_index:
                    diameters.append(e["diameters"][i_idx])
                if vertices_in_box[0] == 1:
                    # intersection at end -> use last internal point’s diameter
                    diameters.append(e["diameters"][save_index[-1]])
                
                new_edge["diameters"] = diameters
                
                # Scalar attrs copied
                new_edge["diameter"] = e["diameter"]
                new_edge["nkind"]    = e["nkind"]

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

with open("/home/admin/Ana/MicroBrain/output18/18_igraph_FULLGEOM_SUB.pkl", "rb") as f:
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

# Ensure points_um attribute exists for all remaining edges(False for all original edges)
# Flag: original edges are not in um yet (they are in voxels)
if "points_um" not in G.es.attributes():
    G.es["points_um"] = [False] * G.ecount()

# ========================================================
# ---- CUT (this function scales points on-the-fly) ----
# =========================================================
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
#G.delete_vertices(border_vertices)

# Remove isolated vertices
disconnected_vertices = [v.index for v in G.vs if G.degree(v) == 0]
G.delete_vertices(disconnected_vertices)
G.vs["degree"] = G.degree()

# ---- OPTION B: normalize ALL remaining edges to µm + recompute lengths ----
for e in G.es:
    already_um = bool(e["points_um"])
    pts = np.asarray(e["points"], dtype=np.float64)

    if not already_um:
        pts = pts * scale
        e["points"] = pts.tolist()
        e["points_um"] = True

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
        e["lengths"]  = [0.0] * int(pts.shape[0])
        e["length"]   = 0.0


# OPTIONAL: remove helper flag before saving (if you don't want it)
# Keep points_um if you want (recommended for debugging)
# If you dont want to remove it, comment:
if "points_um" in G.es.attributes():
    del G.es["points_um"]

# OPTIONAL: if you created lengths_input and you don't want it, remove it:
if "lengths_input" in G.es.attributes():
    del G.es["lengths_input"]

with open("graph_18_CUT.pkl", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


# Save output as dicts (your format)
vertices_data = {attr: G.vs[attr] for attr in G.vs.attributes()}
edges_data = {attr: G.es[attr] for attr in G.es.attributes()}
edges_data["connectivity"] = G.get_edgelist()

with open("vertices_18_graph.pkl", "wb") as f:
    pickle.dump(vertices_data, f)

with open("edges_18_graph.pkl", "wb") as f:
    pickle.dump(edges_data, f)

print("Saved: graph_18_CUT.pkl")

print("Saved: vertices_18_graph.pkl and edges_18_graph.pkl")

'''

import numpy as np
import pickle
import igraph as ig

def intersect_with_plane(p1, p2, axis, value):
    if p2[axis] == p1[axis]:
        raise ValueError("Segment parallel to plane.")
    t = (value - p1[axis]) / (p2[axis] - p1[axis])
    return p1 + t * (p2 - p1), t

def is_inside_box(p, xB, yB, zB):
    return (xB[0] <= p[0] <= xB[1] and
            yB[0] <= p[1] <= yB[1] and
            zB[0] <= p[2] <= zB[1])

def node_exists_coords_um(G, p_um, atol=1e-6):
    p = np.asarray(p_um, dtype=np.float64)
    for v in G.vs:
        if np.allclose(np.asarray(v["coords_um"], dtype=np.float64), p, atol=atol):
            return v.index
    return None

def recompute_lengths_um(points_um):
    pts = np.asarray(points_um, dtype=np.float64)
    if pts.shape[0] >= 2:
        lengths2 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        lengths = np.empty(pts.shape[0], dtype=np.float64)
        lengths[:-1] = lengths2
        lengths[-1]  = lengths2[-1]
        return lengths2.tolist(), lengths.tolist(), float(lengths2.sum())
    else:
        return [], [0.0]*int(pts.shape[0]), 0.0

def get_edges_in_boundingBox_vertex_based(G, xB, yB, zB, scale):
    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []
    new_edges_on_border = []

    for e in G.es:
        # endpoints in µm (image space)
        c0 = np.asarray(G.vs[e.source]["coords_um"], dtype=np.float64)
        c1 = np.asarray(G.vs[e.target]["coords_um"], dtype=np.float64)

        in0 = is_inside_box(c0, xB, yB, zB)
        in1 = is_inside_box(c1, xB, yB, zB)

        if in0 and in1:
            edges_in_box.append(e.index)
            continue

        if (not in0) and (not in1):
            edges_outside_box.append(e.index)
            border_vertices.extend([e.source, e.target])
            continue

        # exactly one endpoint inside
        edges_across_border.append(e.index)

        # decide inside/outside node
        if in0:
            node_in, node_out = e.source, e.target
        else:
            node_in, node_out = e.target, e.source
        border_vertices.append(node_out)

        # points in µm (image space)
        pts_um = np.asarray(e["points"], dtype=np.float64) * scale

        # orient polyline so it starts at inside node (closest end)
        cin = np.asarray(G.vs[node_in]["coords_um"], dtype=np.float64)
        if np.linalg.norm(cin - pts_um[0]) > np.linalg.norm(cin - pts_um[-1]):
            pts_um = pts_um[::-1].copy()
            diams = np.asarray(e["diameters"], dtype=np.float64)[::-1].copy()
        else:
            diams = np.asarray(e["diameters"], dtype=np.float64).copy()

        # find first crossing from inside -> outside along the polyline
        inside_flags = np.array([is_inside_box(p, xB, yB, zB) for p in pts_um], dtype=bool)

        # We want an index j such that inside[j] == True and inside[j+1] == False
        cross_idx = None
        for j in range(len(inside_flags)-1):
            if inside_flags[j] and (not inside_flags[j+1]):
                cross_idx = j
                break

        # If we didn't find a clean inside->outside transition, fallback:
        # keep as "in box" if any point inside, else outside.
        if cross_idx is None:
            if inside_flags.any():
                edges_in_box.append(e.index)
            else:
                edges_outside_box.append(e.index)
            continue

        p_in  = pts_um[cross_idx]
        p_out = pts_um[cross_idx+1]

        # Determine which plane is crossed (like Gaia)
        intersection = None
        for axis, (mn, mx) in enumerate([(xB[0], xB[1]), (yB[0], yB[1]), (zB[0], zB[1])]):
            if p_out[axis] < mn or p_out[axis] > mx:
                plane_val = mn if p_out[axis] < mn else mx
                intersection, t = intersect_with_plane(p_in, p_out, axis, plane_val)
                break

        if intersection is None:
            # should be rare; fallback keep edge
            edges_in_box.append(e.index)
            continue

        # create/find border node at intersection (coords_um)
        node_i = node_exists_coords_um(G, intersection)
        if node_i is None:
            node_i = len(G.vs)
            G.add_vertices(1)
            G.vs[-1]["coords_um"] = intersection.tolist()
            G.vs[-1]["degree"] = 1
            G.vs[-1]["index"] = node_i

        # build new_points: from inside part + intersection
        internal_pts = pts_um[:cross_idx+1]  # includes p_in
        internal_d   = diams[:cross_idx+1]

        new_points_um = np.vstack([internal_pts, intersection])  # end at boundary
        # diameters: copy internal + add diameter for intersection = last internal diameter (simple)
        new_diams = np.concatenate([internal_d, [internal_d[-1]]])

        # add new edge (inside node -> intersection node)
        G.add_edge(node_in, node_i)
        new_edge = G.es[-1]
        new_edge["connectivity"] = (node_in, node_i)
        new_edge["points_um"] = True
        new_edge["points"] = new_points_um.tolist()
        new_edge["diameters"] = new_diams.tolist()

        # copy scalars
        new_edge["diameter"] = e["diameter"]
        new_edge["nkind"] = e["nkind"]
        if "radius" in e.attributes():
            new_edge["radius"] = e["radius"]

        # recompute lengths in µm
        l2, lvec, L = recompute_lengths_um(new_points_um)
        new_edge["lengths2"] = l2
        new_edge["lengths"]  = lvec
        new_edge["length"]   = L

        new_edges_on_border.append(new_edge.index)

    return (np.unique(edges_in_box),
            np.unique(edges_across_border),
            np.unique(edges_outside_box),
            np.unique(border_vertices),
            np.unique(new_edges_on_border))

# ---------------------------
# Run
# ---------------------------
with open("/home/admin/Ana/MicroBrain/output18/18_igraph_FULLGEOM_SUB.pkl", "rb") as f:
    G = pickle.load(f)

sx, sy, sz = 1.625, 1.625, 2.5
scale = np.array([sx, sy, sz], dtype=np.float64)

# coords_um from coords_image
coords_img = np.asarray(G.vs["coords_image"], dtype=np.float64)
G.vs["coords_um"] = (coords_img * scale).tolist()

# box in voxels -> µm
xmin_pv, xmax_pv = 1000, 2000
ymin_pv, ymax_pv = 0, 1000
zmin_pv, zmax_pv = 1500, 2500

xB = [xmin_pv*sx, xmax_pv*sx]
yB = [ymin_pv*sy, ymax_pv*sy]
zB = [zmin_pv*sz, zmax_pv*sz]

if "points_um" not in G.es.attributes():
    G.es["points_um"] = [False]*G.ecount()

edges_in_box, edges_across, edges_out, border_vertices, new_edges = \
    get_edges_in_boundingBox_vertex_based(G, xB, yB, zB, scale)

all_edges = np.concatenate([edges_in_box, new_edges])
edges_to_delete = list(set(range(G.ecount())) - set(all_edges))

G.delete_edges(edges_to_delete)
# borrar nodos fuera + aislados (si quieres)
# G.delete_vertices(border_vertices)
disconnected = [v.index for v in G.vs if G.degree(v) == 0]
G.delete_vertices(disconnected)
G.vs["degree"] = G.degree()

# normalize remaining ORIGINAL edges to µm + recompute lengths
for e in G.es:
    pts = np.asarray(e["points"], dtype=np.float64)
    if not bool(e["points_um"]):
        pts = pts * scale
        e["points"] = pts.tolist()
        e["points_um"] = True

    l2, lvec, L = recompute_lengths_um(pts)
    e["lengths2"] = l2
    e["lengths"]  = lvec
    e["length"]   = L

with open("graph_18_CUT.pkl", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved: graph_18_CUT.pkl")

