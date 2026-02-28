'''
This code reproduces the same cut as Gaias cut the graph code, but implements only her first function. 
Also, as input uses the formatted graph from outgeometry. 
For the cut, it uses the center from Paraview , converts it to um and then reproduces the cut. 
Be aware that the center is really in voxels, other wise, you could directly do 
center_um and take the center from Paraview. 
Paraview does not convert units, it represents the units of the input file, so take units into 
consideration
'''
import numpy as np
import igraph as ig
from collections import Counter


# -----------------------------
# Box builder (your requested way)
# -----------------------------
def make_box_from_center_vox(center_vox, box_um, res_um_per_vox):
    center_vox = np.asarray(center_vox, float)
    box_um = np.asarray(box_um, float)
    res_um_per_vox = np.asarray(res_um_per_vox, float)

    center_um = center_vox * res_um_per_vox
    half = box_um / 2.0

    return {
        "xmin": float(center_um[0] - half[0]), "xmax": float(center_um[0] + half[0]),
        "ymin": float(center_um[1] - half[1]), "ymax": float(center_um[1] + half[1]),
        "zmin": float(center_um[2] - half[2]), "zmax": float(center_um[2] + half[2]),
    }


def is_inside_point(p, box):
    return (box["xmin"] <= p[0] <= box["xmax"] and
            box["ymin"] <= p[1] <= box["ymax"] and
            box["zmin"] <= p[2] <= box["zmax"])


def points_inside_mask(P, box):
    P = np.asarray(P, float)
    return (
        (P[:, 0] >= box["xmin"]) & (P[:, 0] <= box["xmax"]) &
        (P[:, 1] >= box["ymin"]) & (P[:, 1] <= box["ymax"]) &
        (P[:, 2] >= box["zmin"]) & (P[:, 2] <= box["zmax"])
    )


def intersect_with_plane(p_in, p_out, axis, value):
    p_in = np.asarray(p_in, float)
    p_out = np.asarray(p_out, float)
    if p_out[axis] == p_in[axis]:
        return None
    t = (value - p_in[axis]) / (p_out[axis] - p_in[axis])
    if t < 0.0 or t > 1.0:
        return None
    p = p_in + t * (p_out - p_in)
    p[axis] = value
    return p


def segment_box_intersection(p_in, p_out, box):
    """
    Gaia-like: pick the first axis where p_out is outside -> intersect with that boundary plane.
    Returns (p_intersection, face_name) or (None, None).
    """
    bounds = [
        (0, box["xmin"], box["xmax"], "x_min", "x_max"),
        (1, box["ymin"], box["ymax"], "y_min", "y_max"),
        (2, box["zmin"], box["zmax"], "z_min", "z_max"),
    ]

    for axis, mn, mx, fmin, fmax in bounds:
        if p_out[axis] < mn:
            p = intersect_with_plane(p_in, p_out, axis, mn)
            if p is None:
                continue
            if is_inside_point(p, box):
                return p, fmin
        if p_out[axis] > mx:
            p = intersect_with_plane(p_in, p_out, axis, mx)
            if p is None:
                continue
            if is_inside_point(p, box):
                return p, fmax

    return None, None


def euclid(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b))


# -----------------------------
# Debug: near-plane counts
# -----------------------------
def debug_near_plane_counts(G, box, coords_attr="coords", eps_um=5.0):
    P = np.asarray(G.vs[coords_attr], float)
    faces = [
        ("x_min", 0, box["xmin"]), ("x_max", 0, box["xmax"]),
        ("y_min", 1, box["ymin"]), ("y_max", 1, box["ymax"]),
        ("z_min", 2, box["zmin"]), ("z_max", 2, box["zmax"]),
    ]
    print("\n=== DEBUG: vertices close to each face plane ===")
    print("coords bounds:", P.min(axis=0), "->", P.max(axis=0))
    print("box:", box)
    for name, axis, val in faces:
        dist = np.abs(P[:, axis] - float(val))
        print(name, "min_dist", float(dist.min()), "n_within_eps", int((dist <= eps_um).sum()))


# -----------------------------
# Main cutter: formatted Gaia graph
# -----------------------------
def cut_formatted_graph_gaia_like(
    G_in: ig.Graph,
    box: dict,
    coords_attr="coords",
    atol_orient=1e-6,
    reuse_border_vertices=False,   # Gaia uses array_equal (almost no reuse). Default False = safest for degree-1 BC.
):
    if coords_attr not in G_in.vs.attributes():
        raise KeyError(f"Graph has no vertex attr '{coords_attr}'. Available: {G_in.vs.attributes()}")
    if "points" not in G_in.es.attributes():
        raise KeyError("Graph has no edge attr 'points' (expected formatted graph).")

    # Work on a copy (so you can re-run safely)
    G = G_in.copy()

    # ensure border markers exist
    if "is_border" not in G.vs.attributes():
        G.vs["is_border"] = [0] * G.vcount()
    if "border_face" not in G.vs.attributes():
        G.vs["border_face"] = [None] * G.vcount()

    coords_v = np.asarray(G.vs[coords_attr], float)
    inside_v = points_inside_mask(coords_v, box)

    edges_in_box = []
    edges_across_border = []
    edges_outside_box = []
    border_vertices = []
    new_edges_on_border = []

    # optional reuse map (exact tuple match)
    coord_to_vid = {}
    if reuse_border_vertices:
        for vid, c in enumerate(G.vs[coords_attr]):
            coord_to_vid[tuple(map(float, c))] = vid

    def add_border_vertex(p, face, inherit_vid):
        p = tuple(map(float, p))

        if reuse_border_vertices and p in coord_to_vid:
            vid = coord_to_vid[p]
        else:
            vid = G.vcount()
            G.add_vertices(1)

            # inherit vertex attrs (Gaia-like)
            for a in G.vs.attributes():
                if a == coords_attr:
                    continue
                if a == "index":
                    G.vs[vid][a] = -1
                else:
                    G.vs[vid][a] = G.vs[inherit_vid][a]

            G.vs[vid][coords_attr] = p
            G.vs[vid]["is_border"] = 1
            G.vs[vid]["border_face"] = str(face)

            if reuse_border_vertices:
                coord_to_vid[p] = vid

        return vid

    # iterate over ORIGINAL edges only (important because we will add edges)
    nE0 = G.ecount()
    for ei in range(nE0):
        e = G.es[ei]
        u = int(e.source)
        v = int(e.target)

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        pts = np.asarray(e["points"], float)
        if pts.shape[0] < 2:
            continue

        # orient points so pts[0] matches source coords
        if not np.allclose(pts[0], coords_v[u], atol=atol_orient):
            pts = pts[::-1].copy()
            u, v = v, u
            u_in, v_in = v_in, u_in

        # classify by endpoints
        if u_in and v_in:
            edges_in_box.append(ei)
            continue
        if (not u_in) and (not v_in):
            edges_outside_box.append(ei)
            border_vertices.extend([u, v])
            continue

        # across border: one in, one out
        edges_across_border.append(ei)
        if not u_in and v_in:
            # flip to make u inside
            pts = pts[::-1].copy()
            u, v = v, u
            u_in, v_in = v_in, u_in

        # now u_in == True, v_in == False
        border_vertices.append(v)

        mask = points_inside_mask(pts, box)
        if not np.any(mask):
            continue

        # find first leaving: 1 -> 0
        diff = np.diff(mask.astype(np.int8))
        cut_idx = np.where(diff == -1)[0]
        if cut_idx.size == 0:
            continue
        i = int(cut_idx[0])

        p_in = pts[i]
        p_out = pts[i + 1]

        inter, face = segment_box_intersection(p_in, p_out, box)
        if inter is None:
            continue

        # internal points + intersection
        internal_pts = pts[: i + 1]
        new_points = [tuple(map(float, p)) for p in internal_pts] + [tuple(map(float, inter))]

        # create new border node
        border_vid = add_border_vertex(inter, face, inherit_vid=u)

        # add new edge between inside node and border node
        G.add_edge(int(u), int(border_vid))
        new_e = G.es[-1]

        # copy edge attrs from original edge, then overwrite geometry attrs
        for a in G.es.attributes():
            if a in ("points", "diameters", "lengths2", "length", "connectivity"):
                continue
            new_e[a] = e[a]

        # diameters along points (if available)
        if "diameters" in G.es.attributes() and e["diameters"] is not None:
            dpts = np.asarray(e["diameters"], float)
            if dpts.shape[0] == pts.shape[0]:
                kept_d = dpts[: i + 1]
                new_diam = [float(x) for x in kept_d] + [float(kept_d[-1])]
            else:
                new_diam = [float("nan")] * len(new_points)
        else:
            new_diam = [float("nan")] * len(new_points)

        # lengths2 + length
        L2 = [euclid(new_points[j], new_points[j + 1]) for j in range(len(new_points) - 1)]
        L = float(np.sum(L2))

        new_e["points"] = new_points
        new_e["diameters"] = new_diam
        new_e["lengths2"] = L2
        new_e["length"] = L
        new_edges_on_border.append(int(new_e.index))

    # keep only edges in box + new border edges
    all_edges_keep = set(edges_in_box) | set(new_edges_on_border)
    edges_to_delete = [i for i in range(G.ecount()) if i not in all_edges_keep]
    G.delete_edges(edges_to_delete)

    # delete original outside vertices (border_vertices are original outside endpoints)
    border_vertices = np.unique(np.asarray(border_vertices, dtype=int))
    border_vertices = border_vertices[border_vertices < G.vcount()]
    if border_vertices.size:
        G.delete_vertices(border_vertices.tolist())

    # delete isolated
    iso = [v.index for v in G.vs if G.degree(v) == 0]
    if iso:
        G.delete_vertices(iso)

    # recompute bookkeeping
    G.vs["degree"] = G.degree()
    G.vs["index"] = list(range(G.vcount()))
    G.es["connectivity"] = [tuple(map(int, ed.tuple)) for ed in G.es]
    if "length_steps" in G.es.attributes():
        G.es["length_steps"] = [len(l2) for l2 in G.es["lengths2"]]

    return G


def qc_border_nodes(G, coords_attr="coords"):
    if "is_border" not in G.vs.attributes():
        print("[QC] no is_border attr")
        return
    is_border = np.asarray(G.vs["is_border"], int) == 1
    ids = np.where(is_border)[0]
    print("\n=== QC: border nodes (created by cut) ===")
    print("border nodes:", int(ids.size))
    if ids.size:
        deg = np.asarray(G.degree(ids), int)
        print("border degree histogram:", dict(Counter(deg)))
        print("border degree>1:", int((deg > 1).sum()))
        if "border_face" in G.vs.attributes():
            bf = np.asarray(G.vs["border_face"], object)
            c = Counter([bf[i] for i in ids])
            print("border_face counts:", dict(c))


if __name__ == "__main__":
    in_path  = "/home/ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted.pkl"
    out_path = "/home/ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted_Hcut_gaia_like.pkl"

    # your requested box definition
    res_um_per_vox = np.array([1.625, 1.625, 2.5], dtype=float)
    center_vox = np.array([2100, 4200, 750], dtype=float)
    box_um = np.array([400, 400, 400], dtype=float)

    box = make_box_from_center_vox(center_vox, box_um, res_um_per_vox)

    G = ig.Graph.Read_Pickle(in_path)

    # debug: near-plane (optional sanity)
    debug_near_plane_counts(G, box, coords_attr="coords", eps_um=5.0)

    H = cut_formatted_graph_gaia_like(
        G,
        box,
        coords_attr="coords",
        reuse_border_vertices=False,   # keep False if you want BC nodes to stay terminal (degree ~1)
    )

    qc_border_nodes(H, coords_attr="coords")

    H.write_pickle(out_path)
    print("\nSaved:", out_path)
    print("unit:", H["unit"] if "unit" in H.attributes() else None)
    print("V/E:", H.vcount(), H.ecount())
    print("V attrs:", H.vs.attributes())
    print("E attrs:", H.es.attributes())
