"""
Cut a vessel graph inside a 3D box (ROI) in micrometers (µm).

Explanation
-----------------
You start from an OutGeom PKL where you already converted voxel coordinates to µm
(using convert_outgeom_vox2um script). This file then:

1) Selects the vertices that are inside the ROI box (in µm).
2) Goes through each vessel edge (which is stored as a polyline: many points).
3) Keeps vessel parts that are inside the box:
   - If a vessel goes out of the box, it is clipped at the box boundary.
   - A new "border vertex" is created at the intersection point.
4) Builds a new graph + new geometry arrays for the cut.
5) Recomputes:
   - lengths2_R (distance to the next point, last = 0 to comply with indexing format)
   - length_R per edge (sum of lengths2_R inside the edge range)
   - tortuosity_R (length_R / straight distance), dimensionless

Input must contain (already in µm)
----------------------------------
- data["vertex_R"]["coords_image_R"] : vertex coordinates in µm
- data["geom_R"]["x_R","y_R","z_R"]  : polyline points in µm
- data["graph"] with edge attrs: "geom_start", "geom_end"
- data["geom"]["annotation"] 
- data["geom_R"]["diameters_atlas_geom_R"] 

Output
------
Returns a dict:
- out["graph"]    : new igraph.Graph
- out["vertex_R"] : new vertex arrays in µm
- out["geom_R"]   : new geometry arrays in µm
"""


import pickle
from typing import Dict, Optional, Tuple

import igraph as ig
import numpy as np


# -----------------------------------------------------------------------------
# Small helper functions
# -----------------------------------------------------------------------------

def is_inside_box(
    points: np.ndarray,
    x_box: Tuple[float, float],
    y_box: Tuple[float, float],
    z_box: Tuple[float, float],
) -> np.ndarray:
    """Return True/False for each point: is it inside the ROI box?"""
    return (
        (points[:, 0] >= x_box[0]) & (points[:, 0] <= x_box[1]) &
        (points[:, 1] >= y_box[0]) & (points[:, 1] <= y_box[1]) &
        (points[:, 2] >= z_box[0]) & (points[:, 2] <= z_box[1])
    )


def make_reuse_key(point: np.ndarray, tol: float = 1e-6) -> Tuple[int, int, int]:
    """
    Turn a point into a stable key so we can reuse the same border vertex
    if multiple vessels hit the box at (almost) the same place.
    """
    p = np.asarray(point, dtype=np.float64)
    return tuple(np.round(p / tol).astype(np.int64))


def segment_hits_box(
    p0: np.ndarray,
    p1: np.ndarray,
    x_box: Tuple[float, float],
    y_box: Tuple[float, float],
    z_box: Tuple[float, float],
) -> Optional[np.ndarray]:
    """
    If the segment p0->p1 crosses the ROI boundary, return the intersection point.
    If it does not cross in a clean way, return None.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0

    best_t = None
    best_p = None

    # test intersection with the 6 planes of the box
    for axis, (mn, mx) in enumerate([x_box, y_box, z_box]):
        if abs(d[axis]) < 1e-15:
            continue

        for bound in (mn, mx):
            t = (bound - p0[axis]) / d[axis]
            if t < 0.0 or t > 1.0:
                continue

            p = p0 + t * d
            p[axis] = bound  # snap exactly to the plane

            # check if the intersection is still inside the other two box ranges
            if (
                x_box[0] <= p[0] <= x_box[1] and
                y_box[0] <= p[1] <= y_box[1] and
                z_box[0] <= p[2] <= z_box[1]
            ):
                if best_t is None or t < best_t:
                    best_t = t
                    best_p = p

    return None if best_p is None else best_p


# -----------------------------------------------------------------------------
# Simple checks (optional, but useful)
# -----------------------------------------------------------------------------

def analyze_long_vessels(
    data_um: Dict,
    x_box_um: Tuple[float, float],
    y_box_um: Tuple[float, float],
    z_box_um: Tuple[float, float],
    long_factor: float = 2.0,
) -> Tuple[int, int]:
    """
    Print how many vessels cross the ROI, and how many are "long".

    "Long" here means: vessel polyline length > long_factor * average box size.
    """
    g = data_um["geom_R"]
    v = data_um["vertex_R"]
    G = data_um["graph"]

    x = np.asarray(g["x_R"], dtype=np.float64)
    y = np.asarray(g["y_R"], dtype=np.float64)
    z = np.asarray(g["z_R"], dtype=np.float64)

    coords_v = np.asarray(v["coords_image_R"], dtype=np.float64)

    inside_v = is_inside_box(coords_v, x_box_um, y_box_um, z_box_um)

    box_size = np.array(
        [x_box_um[1] - x_box_um[0], y_box_um[1] - y_box_um[0], z_box_um[1] - z_box_um[0]],
        dtype=float,
    )
    long_threshold_um = float(long_factor) * float(box_size.mean())

    n_crossing = 0
    n_long_crossing = 0

    for e in G.es:
        u, w = int(e.source), int(e.target)
        if bool(inside_v[u]) == bool(inside_v[w]):
            continue

        s = int(e["geom_start"])
        en = int(e["geom_end"])
        if en - s < 2:
            continue

        P = np.column_stack([x[s:en], y[s:en], z[s:en]])
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))

        n_crossing += 1
        if L > long_threshold_um:
            n_long_crossing += 1

    print("Box size (µm):", box_size)
    print("Long threshold (µm):", long_threshold_um)
    print("Crossing edges:", n_crossing)
    print("Long crossing edges:", n_long_crossing)
    return n_crossing, n_long_crossing


def check_box_margin(
    data_um: Dict,
    x_box_um: Tuple[float, float],
    y_box_um: Tuple[float, float],
    z_box_um: Tuple[float, float],
    margin_um: Tuple[float, float, float] = (10.0, 10.0, 10.0),
) -> None:
    """
    Quick check: make sure the ROI is not too close to the dataset border.
    Uses vertex_R coords in µm to estimate dataset bounds.
    """
    V = np.asarray(data_um["vertex_R"]["coords_image_R"], dtype=np.float64)
    minV = V.min(axis=0)
    maxV = V.max(axis=0)

    mx, my, mz = map(float, margin_um)
    ok = (
        (x_box_um[0] - minV[0] >= mx) and (maxV[0] - x_box_um[1] >= mx) and
        (y_box_um[0] - minV[1] >= my) and (maxV[1] - y_box_um[1] >= my) and
        (z_box_um[0] - minV[2] >= mz) and (maxV[2] - z_box_um[1] >= mz)
    )
    if not ok:
        raise RuntimeError(f"ROI too close to dataset border (< {margin_um} µm).")
    print(f"ROI is safely inside the dataset (margin ≥ {margin_um} µm).")


# -----------------------------------------------------------------------------
# Main cutting function (in µm)
# -----------------------------------------------------------------------------

def cut_outgeom_in_um(
    data_um: Dict,
    x_box_um: Tuple[float, float],
    y_box_um: Tuple[float, float],
    z_box_um: Tuple[float, float],
    tol: float = 1e-6,
    min_straight_dist_um: float = 1.0,
) -> Dict:
    """
    Cut the OutGeom dataset using ROI box bounds in µm.
    
    How it works
    ------------
    - Keep vertices whose coords_image_R are inside the ROI.
    - For each edge polyline:
        * If both endpoints are inside: keep the part inside the box (largest connected block).
        * If one endpoint is inside: clip at the box boundary and add a border vertex.
        * If both endpoints are outside: skip (same behavior as your current script).

    Returns a new dict with:
      - graph:      new igraph.Graph
      - vertex_R:   new vertex coordinates in µm
      - geom_R:     new polyline points in µm + recomputed lengths/diameters
    """

    G = data_um["graph"]
    gR = data_um["geom_R"]
    vR = data_um["vertex_R"]

    # geometry points (polylines)
    x = np.asarray(gR["x_R"], dtype=np.float64)
    y = np.asarray(gR["y_R"], dtype=np.float64)
    z = np.asarray(gR["z_R"], dtype=np.float64)

    # vertex coordinates
    coords_v = np.asarray(vR["coords_image_R"], dtype=np.float64)
    inside_v = is_inside_box(coords_v, x_box_um, y_box_um, z_box_um)

    # keep annotation if present in the *original* geom dict
    ann_geom = None
    if "geom" in data_um and isinstance(data_um["geom"], dict) and "annotation" in data_um["geom"]:
        ann_geom = np.asarray(data_um["geom"]["annotation"], dtype=np.int32)
        if len(ann_geom) != len(x):
            raise ValueError("geom['annotation'] length must match geom_R length.")

    #  keep diameters in µm if present
    diam_geom_um = None
    if "diameters_atlas_geom_R" in gR:
        diam_geom_um = np.asarray(gR["diameters_atlas_geom_R"], dtype=np.float32)
        if len(diam_geom_um) != len(x):
            raise ValueError("geom_R['diameters_atlas_geom_R'] length must match geom_R length.")

    # -------------------------------------------------------------------------
    # Build new graph H with only vertices inside the ROI
    # -------------------------------------------------------------------------
    H = ig.Graph()
    H.add_vertices(0)

    inside_old_ids = np.where(inside_v)[0].astype(int)
    H.add_vertices(len(inside_old_ids))

    old2new = {int(old_i): int(new_i) for new_i, old_i in enumerate(inside_old_ids)}

    # Copy any igraph vertex attributes if they exist (safe even if empty)
    v_attrs = list(G.vs.attributes())
    for a in v_attrs:
        H.vs[a] = [G.vs[int(i)][a] for i in inside_old_ids]

    # New vertex arrays (µm)
    H_vertex_R: Dict[str, np.ndarray] = {
        "coords_image_R": coords_v[inside_old_ids].copy().astype(np.float32)
    }

    # -------------------------------------------------------------------------
    # Border vertices (created when a vessel is clipped)
    # -------------------------------------------------------------------------
    border_key_to_new: Dict[Tuple[int, int, int], int] = {}

    def add_border_vertex(inter_point_um: np.ndarray, inherit_from_idx: int) -> int:
        """
        Create or reuse a border vertex at the intersection point.
        We inherit all vertex attributes from an existing vertex inside the cut,
        to keep downstream code simple.
        """
        key = make_reuse_key(inter_point_um, tol=tol)
        if key in border_key_to_new:
            return border_key_to_new[key]

        new_idx = H.vcount()
        H.add_vertices(1)
        border_key_to_new[key] = new_idx

        # copy igraph vertex attributes from a "real" vertex
        for a in v_attrs:
            H.vs[new_idx][a] = H.vs[inherit_from_idx][a]

        # add to vertex array
        H_vertex_R["coords_image_R"] = np.vstack(
            [H_vertex_R["coords_image_R"], inter_point_um.reshape(1, 3).astype(np.float32)]
        )
        return new_idx

    # -------------------------------------------------------------------------
    # New geometry arrays (store everything globally)
    # -------------------------------------------------------------------------
    new_x: list = []
    new_y: list = []
    new_z: list = []
    new_ann: list = []
    new_diameters: list = []
    new_lengths2: list = []  # "distance to next point", last is 0

    def append_polyline(
        P_keep_um: np.ndarray,
        A_keep: Optional[np.ndarray],
        D_keep: Optional[np.ndarray],
    ) -> Tuple[int, int]:
        """Append points to geom_R arrays and return (start, end) indices."""
        start = len(new_x)
        n = int(P_keep_um.shape[0])

        new_x.extend(P_keep_um[:, 0].tolist())
        new_y.extend(P_keep_um[:, 1].tolist())
        new_z.extend(P_keep_um[:, 2].tolist())

        if A_keep is not None:
            new_ann.extend(A_keep.tolist())

        if D_keep is None:
            new_diameters.extend(np.full(n, np.nan, dtype=np.float32).tolist())
        else:
            D_keep = np.asarray(D_keep, dtype=np.float32).reshape(-1)
            if len(D_keep) != n:
                raise ValueError("Diameter array length mismatch while appending polyline.")
            new_diameters.extend(D_keep.tolist())

        if n < 2:
            new_lengths2.extend([0.0] * n)
            return start, start + n

        seg = np.linalg.norm(np.diff(P_keep_um, axis=0), axis=1).astype(np.float32)  # (n-1,)
        new_lengths2.extend(np.concatenate([seg, [0.0]]).astype(np.float32).tolist())
        return start, start + n

    # new edges and their geometry ranges
    new_edges: list = []
    new_geom_start: list = []
    new_geom_end: list = []
    orig_eid: list = []

    # carry original edge attrs except geom_start/geom_end (we rebuild those)
    e_attrs = list(G.es.attributes())
    new_edge_attr: Dict[str, list] = {a: [] for a in e_attrs if a not in ["geom_start", "geom_end"]}

    start_idx = np.asarray(G.es["geom_start"], dtype=np.int64)
    end_idx = np.asarray(G.es["geom_end"], dtype=np.int64)

    # -------------------------------------------------------------------------
    # Walk through all original edges and decide how to keep/clip them
    # -------------------------------------------------------------------------
    for e in G.es:
        u = int(e.source)
        v = int(e.target)

        s0 = int(start_idx[e.index])
        e0 = int(end_idx[e.index])
        if e0 - s0 < 2:
            continue

        P = np.column_stack([x[s0:e0], y[s0:e0], z[s0:e0]]).astype(np.float64, copy=False)
        A = ann_geom[s0:e0].copy() if ann_geom is not None else None
        D = diam_geom_um[s0:e0].copy() if diam_geom_um is not None else None

        # Make sure P[0] matches vertex u (orientation)
        if not np.allclose(P[0], coords_v[u], atol=1e-4):
            P = P[::-1].copy()
            if A is not None:
                A = A[::-1].copy()
            if D is not None:
                D = D[::-1].copy()
            u, v = v, u

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # both endpoints inside: keep the inside part
        if u_in and v_in:
            uu = old2new[u]
            vv = old2new[v]

            mask = is_inside_box(P, x_box_um, y_box_um, z_box_um)
            if np.all(mask):
                P_keep, A_keep, D_keep = P, A, D
            else:
                # keep the biggest continuous inside chunk
                idx = np.where(mask)[0]
                if len(idx) < 2:
                    continue
                gaps = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[0, gaps + 1]
                ends = np.r_[gaps, len(idx) - 1]
                k = int(np.argmax(ends - starts + 1))
                i0 = int(idx[starts[k]])
                i1 = int(idx[ends[k]]) + 1

                P_keep = P[i0:i1]
                A_keep = A[i0:i1] if A is not None else None
                D_keep = D[i0:i1] if D is not None else None

            gs, ge = append_polyline(P_keep, A_keep, D_keep)

            new_edges.append((uu, vv))
            new_geom_start.append(gs)
            new_geom_end.append(ge)
            orig_eid.append(int(e.index))
            for a in new_edge_attr:
                new_edge_attr[a].append(e[a])
            continue

        # both endpoints outside: skip (same as your current behavior)
        if (not u_in) and (not v_in):
            continue

        # one endpoint inside, one outside: clip and create border vertex
        mask = is_inside_box(P, x_box_um, y_box_um, z_box_um)
        if not np.any(mask):
            continue

        # leaving the box
        if u_in:
            diff = np.diff(mask.astype(np.int8))
            cut_idx = np.where(diff == -1)[0]
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[0])

            inter = segment_hits_box(P[i], P[i + 1], x_box_um, y_box_um, z_box_um)
            if inter is None:
                continue

            P_keep = np.vstack([P[:i + 1], inter])
            A_keep = np.concatenate([A[:i + 1], [int(A[i])]]).astype(np.int32) if A is not None else None
            D_keep = np.concatenate([D[:i + 1], [float(D[i])]]).astype(np.float32) if D is not None else None

            uu = old2new[u]
            ww = add_border_vertex(inter, inherit_from_idx=uu)

            gs, ge = append_polyline(P_keep, A_keep, D_keep)

            new_edges.append((uu, ww))
            new_geom_start.append(gs)
            new_geom_end.append(ge)
            orig_eid.append(int(e.index))
            for a in new_edge_attr:
                new_edge_attr[a].append(e[a])

        # entering the box
        else:
            diff = np.diff(mask.astype(np.int8))
            cut_idx = np.where(diff == 1)[0]
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[-1])

            inter = segment_hits_box(P[i], P[i + 1], x_box_um, y_box_um, z_box_um)
            if inter is None:
                continue

            P_keep = np.vstack([inter, P[i + 1:]])
            A_keep = np.concatenate([[int(A[i + 1])], A[i + 1:]]).astype(np.int32) if A is not None else None
            D_keep = np.concatenate([[float(D[i + 1])], D[i + 1:]]).astype(np.float32) if D is not None else None

            vv = old2new[v]
            ww = add_border_vertex(inter, inherit_from_idx=vv)

            gs, ge = append_polyline(P_keep, A_keep, D_keep)

            new_edges.append((ww, vv))
            new_geom_start.append(gs)
            new_geom_end.append(ge)
            orig_eid.append(int(e.index))
            for a in new_edge_attr:
                new_edge_attr[a].append(e[a])

    # -------------------------------------------------------------------------
    # Build final output
    # -------------------------------------------------------------------------
    if not new_edges:
        out = {
            "graph": H,
            "vertex_R": {"coords_image_R": np.empty((0, 3), dtype=np.float32)},
            "geom_R": {
                "x_R": np.empty(0, np.float64),
                "y_R": np.empty(0, np.float64),
                "z_R": np.empty(0, np.float64),
                "lengths2_R": np.empty(0, np.float32),
                "diameters_R": np.empty(0, np.float32),
            },
        }
        if ann_geom is not None:
            out["geom_R"]["annotation"] = np.empty(0, np.int32)
        return out

    H.add_edges(new_edges)
    H.es["geom_start"] = list(map(int, new_geom_start))
    H.es["geom_end"] = list(map(int, new_geom_end))
    H.es["orig_eid"] = orig_eid
    for a, vals in new_edge_attr.items():
        H.es[a] = vals

    nx = np.asarray(new_x, dtype=np.float64)
    ny = np.asarray(new_y, dtype=np.float64)
    nz = np.asarray(new_z, dtype=np.float64)
    L2 = np.asarray(new_lengths2, dtype=np.float32)
    Dp = np.asarray(new_diameters, dtype=np.float32)

    # Edge length_R = sum of lengths2_R along its points
    edge_len = np.zeros(H.ecount(), dtype=np.float64)
    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])
        edge_len[ei] = 0.0 if (en - s < 2) else float(np.sum(L2[s:en]))
    H.es["length_R"] = edge_len.astype(np.float32).tolist()

    # Tortuosity_R = length_R / straight distance
    sd = np.zeros(H.ecount(), dtype=np.float64)
    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])
        if en - s < 2:
            continue
        sd[ei] = float(np.linalg.norm([nx[en - 1] - nx[s], ny[en - 1] - ny[s], nz[en - 1] - nz[s]]))

    tort = np.full(H.ecount(), np.nan, dtype=np.float64)
    m = sd >= float(min_straight_dist_um)
    tort[m] = edge_len[m] / sd[m]
    H.es["tortuosity_R"] = tort.astype(np.float32).tolist()

    # Remove isolated vertices (and keep the vertex array aligned)
    iso = [vv.index for vv in H.vs if H.degree(vv) == 0]
    if iso:
        keep = np.ones(H.vcount(), dtype=bool)
        keep[np.array(iso, dtype=int)] = False
        H.delete_vertices(iso)
        H_vertex_R["coords_image_R"] = H_vertex_R["coords_image_R"][keep]

    out = {
        "graph": H,
        "vertex_R": H_vertex_R,
        "geom_R": {
            "x_R": nx,
            "y_R": ny,
            "z_R": nz,
            "lengths2_R": L2,
            "diameters_R": Dp,
        },
    }
    if ann_geom is not None:
        out["geom_R"]["annotation"] = np.asarray(new_ann, dtype=np.int32)

    return out


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um_Hcut.pkl"

    data_um = pickle.load(open(in_path, "rb"))

    # Example: if you know the ROI center in voxels, convert it to µm first.
    res_um_per_vox = np.array([1.625, 1.625, 2.5], dtype=float)
    center_vox = np.array([2100, 4200, 750], dtype=float)
    center_um = center_vox * res_um_per_vox
   
    # Box size in µm (physical size)
    box_um = np.array([400, 400, 400], dtype=float)

    x_box = (float(center_um[0] - box_um[0] / 2), float(center_um[0] + box_um[0] / 2))
    y_box = (float(center_um[1] - box_um[1] / 2), float(center_um[1] + box_um[1] / 2))
    z_box = (float(center_um[2] - box_um[2] / 2), float(center_um[2] + box_um[2] / 2))

    print("ROI box (µm):", x_box, y_box, z_box)

    analyze_long_vessels(data_um, x_box, y_box, z_box, long_factor=2.0)
    check_box_margin(data_um, x_box, y_box, z_box, margin_um=(10.0, 10.0, 10.0))

    cut_um = cut_outgeom_in_um(
        data_um,
        x_box, y_box, z_box,
        tol=1e-6,
        min_straight_dist_um=1.0,
    )

    # Final sanity check: the cut points should all lie inside the ROI
    Pcut = np.column_stack([cut_um["geom_R"]["x_R"], cut_um["geom_R"]["y_R"], cut_um["geom_R"]["z_R"]])
    inside_cut = is_inside_box(Pcut, x_box, y_box, z_box)
    print("Cut points outside ROI:", int(np.sum(~inside_cut)), "/", len(Pcut))

    with open(out_path, "wb") as f:
        pickle.dump(cut_um, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path)
    print("Vertices:", cut_um["graph"].vcount(), "Edges:", cut_um["graph"].ecount())
