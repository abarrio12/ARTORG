"""
GEOM-GRAPH BOX CLIPPER (OUTGEOM VERSION)

This script filters and clips a 3D graph based on a defined Bounding Box.
It is based on Cut_The_Graph from Gaia, but adapted to graphs that do not have
the geometry store within the edges. 

LOGIC AND WORKFLOW:
1. NODE-BASED FILTERING: 
   The script first evaluates the position of the logical endpoints (source/target) 
   of each edge relative to the box.
   
2. PATH RECONSTRUCTION & CLIPPING:
   - If both nodes are INSIDE: The edge is kept as is.
   - If both nodes are OUTSIDE: The edge is discarded (even if the path 
     temporarily crosses the box).
   - If the edge CROSSES the boundary (one node in, one out): 
     a) The "tortuous" path (polyline geometry) is reconstructed using the 
        indices in 'geom_start' and 'geom_end'.
     b) The script identifies the segment where the path exits/enters the box.
     c) It calculates the intersection point with the box plane.
     d) A new "Border Node" is created at the intersection.
     e) The path is truncated: only the points within the box plus the new 
        intersection point are kept.

3. OUTGEOM MEMORY MANAGEMENT:
   Instead of storing points inside the edge object, the new clipped geometry 
   is appended to the global coordinate arrays (x, y, z). The 'geom_start' 
   and 'geom_end' pointers are updated to reference these new positions.

4. CLEANUP:
   After clipping, orphaned nodes and redundant edges are removed to maintain 
   topological integrity.

# ==============================================================================
# AUTHOR: Ana Barrio
# DATE:  10-02-26 
# ==============================================================================
"""

import numpy as np
import igraph as ig
import pickle


# --------------------------
# Helpers
# --------------------------

def intersect_with_plane(point1, point2, axis, value):
    if point2[axis] == point1[axis]:
        raise ValueError(
            f"Cannot calculate intersection when both points have the same {['x','y','z'][axis]} coordinate."
        )
    t = (value - point1[axis]) / (point2[axis] - point1[axis])
    return point1 + t * (point2 - point1)

def is_inside_box_point(p, xBox, yBox, zBox):
    return (xBox[0] <= p[0] <= xBox[1] and
            yBox[0] <= p[1] <= yBox[1] and
            zBox[0] <= p[2] <= zBox[1])

def node_exists_coords(graph, point, tol=0.0):
    """
    Gaia original usa array_equal sobre v['coords'].
    En flotantes, array_equal es frágil. Aquí puedes:
      - usar array_equal si estás seguro que cae exacto,
      - o usar allclose con tol.
    Para mantenerte fiel a Gaia, dejo tol=0 por defecto.
    """
    P = np.asarray(point)
    for v in graph.vs:
        C = np.asarray(v["coords"])
        if tol == 0.0:
            if np.array_equal(C, P):
                return v.index
        else:
            if np.allclose(C, P, atol=tol):
                return v.index
    return None


# --------------------------
# Gaia-like cut (OUTGEOM) - memory safe
# --------------------------

def get_edges_in_boundingBox_vertex_based_outgeom(
    data, xBox, yBox, zBox,
    use_vertex_attr="coords_image",   # decide in/out (voxels)
    node_dedup_tol=0.0,               # if you want allclose, write 1e-6
    compact=False                     # do not compact by default (avoid OOM)
):
    """
    SAME IDEA as Gaia (function 1):
    - Decides in/out by endpoints
    - If both are inside -> keep edge (does NOT touch geometry)
    - If 1 in 1 out -> creates intersection, new edge with geom_start/end pointing to new points
    - If both are outside -> discards (does NOT consider both outside but the path inside)

    GRAPH INFO DIFFERENCE:
    - There are no e['points']; the geometry is in data['coords']['x/y/z'] + geom_start/end
    - For the new edge, we append points to the end of x/y/z and set geom_start/end

    Returns an OUTGEOM "cut" dict (main function expects it).
    """

    G = data["graph"]

    # coords globales como numpy (NO convertir a list gigante)
    x = np.asarray(data["coords"]["x"], dtype=np.float64)
    y = np.asarray(data["coords"]["y"], dtype=np.float64)
    z = np.asarray(data["coords"]["z"], dtype=np.float64)

    # Para nuevos puntos, acumulamos en listas pequeñas y concatenamos al final
    new_x, new_y, new_z = [], [], []

    # Precompute inside_v (barato)
    V = np.asarray(G.vs[use_vertex_attr], dtype=np.float64)
    inside_v = (
        (V[:, 0] >= xBox[0]) & (V[:, 0] <= xBox[1]) &
        (V[:, 1] >= yBox[0]) & (V[:, 1] <= yBox[1]) &
        (V[:, 2] >= zBox[0]) & (V[:, 2] <= zBox[1])
    )

    edges_in_box = []
    edges_outside_box = []
    edges_across_border = []
    border_vertices = []
    new_edges_on_border = []

    print("xBox:", xBox)
    print("yBox:", yBox)
    print("zBox:", zBox)

    
    # Iterate by index of original edge 
    original_ecount = G.ecount()

    for ei in range(original_ecount):
        e = G.es[ei]
        u = int(e.source)
        v = int(e.target)

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # CASE A: both in -> keep
        if u_in and v_in:
            edges_in_box.append(ei)
            continue

        # CASE B: bot out -> discard 
        if (not u_in) and (not v_in):
            edges_outside_box.append(ei)
            border_vertices.append(u)
            border_vertices.append(v)
            continue

        # CASE C: 1 in 1 out -> intersec
        edges_across_border.append(ei)
        border_vertices.append(u if not u_in else v)

        # reconstruct geometry
        s = int(e["geom_start"])
        en = int(e["geom_end"])
        if en - s < 2:
            continue

        xs = x[s:en]
        ys = y[s:en]
        zs = z[s:en]

        # In Gaia, points[0] should be source coords
        # In OUTGEOM, we assume xs[0],ys[0],zs[0] as source, if not, invert

        p0 = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
        cu = np.asarray(G.vs[u][use_vertex_attr], dtype=np.float64)

        if not np.allclose(cu, p0, atol=1e-6):
            # invert polyline and swap endpoints 
            xs = xs[::-1]
            ys = ys[::-1]
            zs = zs[::-1]
            u, v = v, u
            u_in, v_in = v_in, u_in

        # First point out ? 
        inside_mask = (
            (xs >= xBox[0]) & (xs <= xBox[1]) &
            (ys >= yBox[0]) & (ys <= yBox[1]) &
            (zs >= zBox[0]) & (zs <= zBox[1])
        )

        # If not points inside, we can't cut
        if not np.any(inside_mask):
            continue

        # We look for the "change" 1->0 (goes out) or 0->1 (goes in), depending on endpoint
        # - if u_in True, we look for 1->0
        # - if u_in False, we look for 0->1

        diff = np.diff(inside_mask.astype(np.int8))
        if u_in and (not v_in):
            cut_idx = np.where(diff == -1)[0]   # 1->0
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[0])  # last in is i
            internal_point = np.array([xs[i], ys[i], zs[i]], dtype=np.float64)
            external_point = np.array([xs[i+1], ys[i+1], zs[i+1]], dtype=np.float64)
            internal_points = np.column_stack([xs[:i+1], ys[:i+1], zs[:i+1]])
            # intersection with plane
            inter = None
            for axis, (mn, mx) in enumerate([xBox, yBox, zBox]):
                if external_point[axis] < mn or external_point[axis] > mx:
                    inter = intersect_with_plane(internal_point, external_point, axis, mn if external_point[axis] < mn else mx)
                    break
            if inter is None:
                continue
            new_points = np.vstack([internal_points, inter])

            # intersection node
            node_index = node_exists_coords(G, inter, tol=node_dedup_tol)
            if node_index is None:
                node_index = G.vcount()
                G.add_vertices(1)
                if "coords" in G.vs.attributes():
                    G.vs[-1]["coords"] = tuple(map(float, inter))
                if use_vertex_attr in G.vs.attributes():
                    G.vs[-1][use_vertex_attr] = tuple(map(float, inter))
                if "degree" in G.vs.attributes():
                    G.vs[-1]["degree"] = 1
                if "index" in G.vs.attributes():
                    G.vs[-1]["index"] = node_index

            # new edge: u (inside) -> node_index (border)
            G.add_edge(u, node_index)
            new_edge = G.es[-1]
            new_edge["connectivity"] = (u, node_index)

        else:
            # u_in False, v_in True (goes in)
            cut_idx = np.where(diff == 1)[0]    # 0->1
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[0])  # i out, i+1 in
            internal_point = np.array([xs[i+1], ys[i+1], zs[i+1]], dtype=np.float64)
            external_point = np.array([xs[i], ys[i], zs[i]], dtype=np.float64)
            internal_points = np.column_stack([xs[i+1:], ys[i+1:], zs[i+1:]])
            inter = None
            for axis, (mn, mx) in enumerate([xBox, yBox, zBox]):
                if external_point[axis] < mn or external_point[axis] > mx:
                    inter = intersect_with_plane(internal_point, external_point, axis, mn if external_point[axis] < mn else mx)
                    break
            if inter is None:
                continue
            new_points = np.vstack([inter, internal_points])

            node_index = node_exists_coords(G, inter, tol=node_dedup_tol)
            if node_index is None:
                node_index = G.vcount()
                G.add_vertices(1)
                if "coords" in G.vs.attributes():
                    G.vs[-1]["coords"] = tuple(map(float, inter))
                if use_vertex_attr in G.vs.attributes():
                    G.vs[-1][use_vertex_attr] = tuple(map(float, inter))
                if "degree" in G.vs.attributes():
                    G.vs[-1]["degree"] = 1
                if "index" in G.vs.attributes():
                    G.vs[-1]["index"] = node_index

            # new edge: node_index (border) -> v (inside)
            G.add_edge(node_index, v)
            new_edge = G.es[-1]
            new_edge["connectivity"] = (node_index, v)

        # OUTGEOM: save geom of new edge in new arrays (not touching x/y/z yet)
        s_new = len(x) + len(new_x)
        new_x.extend(new_points[:, 0].tolist())
        new_y.extend(new_points[:, 1].tolist())
        new_z.extend(new_points[:, 2].tolist())
        en_new = len(x) + len(new_x)

        if "geom_start" in G.es.attributes():
            new_edge["geom_start"] = int(s_new)
        if "geom_end" in G.es.attributes():
            new_edge["geom_end"] = int(en_new)

        # copy attributes(diameter, nkind, etc.)
        for attr in G.es.attributes():
            if attr in ["connectivity", "geom_start", "geom_end"]:
                continue
            try:
                new_edge[attr] = e[attr]
            except KeyError:
                pass

        new_edges_on_border.append(new_edge.index)

    # concatenate new points of polyline
    if len(new_x):
        x2 = np.concatenate([x, np.asarray(new_x, dtype=np.float64)])
        y2 = np.concatenate([y, np.asarray(new_y, dtype=np.float64)])
        z2 = np.concatenate([z, np.asarray(new_z, dtype=np.float64)])
    else:
        x2, y2, z2 = x, y, z

    # save in data
    data["coords"]["x"] = x2
    data["coords"]["y"] = y2
    data["coords"]["z"] = z2

    # apply cleanup
    edges_in_box = np.unique(edges_in_box)
    new_edges_on_border = np.unique(new_edges_on_border)
    border_vertices = np.unique(border_vertices)

    all_edges = np.concatenate([edges_in_box, new_edges_on_border]) if len(new_edges_on_border) else np.array(edges_in_box, dtype=int)
    edges_to_delete = list(set(range(G.ecount())) - set(all_edges))

    print("keep edges:", len(all_edges), " / total edges:", G.ecount())
    print("delete edges:", len(edges_to_delete))
    print("border vertices:", len(border_vertices))

    G.delete_edges(edges_to_delete)
    if len(border_vertices):
        # OJO: after deleting edges, vertex indices are still valid. 
        # Vertex not yet deleted.
        G.delete_vertices(border_vertices)

    disconnected_vertices = [vv.index for vv in G.vs if G.degree(vv) == 0]
    if disconnected_vertices:
        G.delete_vertices(disconnected_vertices)

    if "degree" in G.vs.attributes():
        G.vs["degree"] = G.degree()

    # Optional: compact (High in RAM, can be killed in really big datasets !!!)
    if compact:
        G2 = data["graph"]
        xx = np.asarray(data["coords"]["x"], dtype=np.float64)
        yy = np.asarray(data["coords"]["y"], dtype=np.float64)
        zz = np.asarray(data["coords"]["z"], dtype=np.float64)

        nx, ny, nz = [], [], []
        new_s, new_en = [], []

        for e in G2.es:
            s = int(e["geom_start"])
            en = int(e["geom_end"])
            if en - s < 2:
                start = len(nx); end = len(nx)
                new_s.append(start); new_en.append(end)
                continue
            start = len(nx)
            nx.extend(xx[s:en].tolist())
            ny.extend(yy[s:en].tolist())
            nz.extend(zz[s:en].tolist())
            end = len(nx)
            new_s.append(start); new_en.append(end)

        if "geom_start" in G2.es.attributes():
            G2.es["geom_start"] = list(map(int, new_s))
        if "geom_end" in G2.es.attributes():
            G2.es["geom_end"] = list(map(int, new_en))

        data["coords"]["x"] = np.asarray(nx, dtype=np.float64)
        data["coords"]["y"] = np.asarray(ny, dtype=np.float64)
        data["coords"]["z"] = np.asarray(nz, dtype=np.float64)

    return data


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3.pkl"

    data = pickle.load(open(in_path, "rb"))

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

    print("BOX (voxels):", xBox, yBox, zBox)

    cut = get_edges_in_boundingBox_vertex_based_outgeom(
        data, xBox, yBox, zBox,
        use_vertex_attr="coords_image",
        node_dedup_tol=1e-6,    
        compact=False          
    )

    # Sanity check: show points per edge (not coords)
    Gc = cut["graph"]
    xx = np.asarray(cut["coords"]["x"], dtype=np.float64)
    yy = np.asarray(cut["coords"]["y"], dtype=np.float64)
    zz = np.asarray(cut["coords"]["z"], dtype=np.float64)

    n_out = 0
    n_tot = 0
    sample_edges = min(5000, Gc.ecount())
    for ei in range(sample_edges):
        e = Gc.es[ei]
        s = int(e["geom_start"]); en = int(e["geom_end"])
        if en - s < 2:
            continue
        Px = xx[s:en]; Py = yy[s:en]; Pz = zz[s:en]
        inside = (
            (Px >= xBox[0]) & (Px <= xBox[1]) &
            (Py >= yBox[0]) & (Py <= yBox[1]) &
            (Pz >= zBox[0]) & (Pz <= zBox[1])
        )
        n_out += int(np.sum(~inside))
        n_tot += int(len(inside))
    print(f"Sanity(sample {sample_edges} edges): outside points = {n_out} / {n_tot}")

    with open(out_path, "wb") as f:
        pickle.dump(cut, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path,
          "Vertices:", cut["graph"].vcount(),
          "Edges:", cut["graph"].ecount())