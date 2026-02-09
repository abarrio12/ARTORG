import numpy as np
import igraph as ig
import pickle

# --------------------------
# Helpers
# --------------------------
def is_inside_box_np(P, xBox, yBox, zBox):
    return (
        (P[:, 0] >= xBox[0]) & (P[:, 0] <= xBox[1]) &
        (P[:, 1] >= yBox[0]) & (P[:, 1] <= yBox[1]) &
        (P[:, 2] >= zBox[0]) & (P[:, 2] <= zBox[1])
    )

def quant_key(p, tol=1e-6):
    p = np.asarray(p, dtype=np.float64)
    return tuple(np.round(p / tol).astype(np.int64))

def segment_box_intersection(p0, p1, xBox, yBox, zBox):
    """
    Intersect segment p0->p1 with AABB boundary. Returns (3,) or None.
    Snaps axis exactly to the plane.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0

    best_t = None
    best_p = None

    for axis, (mn, mx) in enumerate([xBox, yBox, zBox]):
        if abs(d[axis]) < 1e-15:
            continue
        for bound in (mn, mx):
            t = (bound - p0[axis]) / d[axis]
            if t < 0.0 or t > 1.0:
                continue
            p = p0 + t * d
            p[axis] = bound  # snap
            if (xBox[0] <= p[0] <= xBox[1] and
                yBox[0] <= p[1] <= yBox[1] and
                zBox[0] <= p[2] <= zBox[1]):
                if best_t is None or t < best_t:
                    best_t = t
                    best_p = p

    return None if best_p is None else best_p



def restore_edges_gaia_like(cut, store_points=False):
    """
    Reconstruye atributos estilo Gaia en cut["graph"].es:
      - points (opcional)
      - lengths2 (n-1)
      - length (sum)
      - diameters (n)
      - lengths (n)  -> aquí usamos geom["lengths"] (arclen)
    """
    H = cut["graph"]
    x = cut["geom"]["x"]
    y = cut["geom"]["y"]
    z = cut["geom"]["z"]

    # ya los tienes globales (recalculados en el cut)
    L2p = cut["geom"].get("lengths2", None)     # per point (n) con último=0
    Lp  = cut["geom"].get("lengths", None)      # per point (n) arclen
    Dp  = cut["geom"].get("diameters", None)    # per point (n)

    if L2p is None or Lp is None or Dp is None:
        raise ValueError("Faltan geom['lengths2'] o geom['lengths'] o geom['diameters'] en el cut.")

    edge_points   = []
    edge_lengths2 = []
    edge_lengths  = []
    edge_diams    = []
    edge_len      = []

    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])

        if en - s < 2:
            edge_len.append(0.0)
            edge_lengths2.append([])
            edge_diams.append([])
            edge_lengths.append([])
            edge_points.append([] if store_points else None)
            continue

        # lengths2 Gaia: n-1 (quitamos el último 0)
        l2 = np.asarray(L2p[s:en-1], dtype=float).tolist()
        edge_lengths2.append(l2)
        edge_len.append(float(np.sum(l2)))

        # diameters per point
        edge_diams.append(np.asarray(Dp[s:en], dtype=float).tolist())

        # lengths per point (arclen)
        edge_lengths.append(np.asarray(Lp[s:en], dtype=float).tolist())

        # points (opcional, pesa)
        if store_points:
            pts = list(zip(map(float, x[s:en]), map(float, y[s:en]), map(float, z[s:en])))
            edge_points.append(pts)
        else:
            edge_points.append(None)

    # set attrs like Gaia
    if store_points:
        H.es["points"] = edge_points
    H.es["lengths2"] = edge_lengths2
    H.es["diameters"] = edge_diams
    H.es["lengths"] = edge_lengths
    H.es["length"] = edge_len

    return cut

# --------------------------
# Optional: analyze long vessels cut
# --------------------------
def analyze_cut_edges(data, xBox, yBox, zBox, res_um, long_factor=2.0):
    G = data["graph"]

    x = data["geom"]["x"]
    y = data["geom"]["y"]
    z = data["geom"]["z"]

    coords_v = data["vertex"]["coords_image"]

    inside_v = (
        (coords_v[:, 0] >= xBox[0]) & (coords_v[:, 0] <= xBox[1]) &
        (coords_v[:, 1] >= yBox[0]) & (coords_v[:, 1] <= yBox[1]) &
        (coords_v[:, 2] >= zBox[0]) & (coords_v[:, 2] <= zBox[1])
    )

    box_size_vox = np.array([xBox[1]-xBox[0], yBox[1]-yBox[0], zBox[1]-zBox[0]], dtype=float)
    box_size_um = box_size_vox * np.asarray(res_um, float)
    long_threshold_um = float(long_factor) * float(box_size_um.mean())

    n_crossing = 0
    n_long_crossing = 0
   

    for e in G.es:
        u, v = int(e.source), int(e.target)
        u_in, v_in = bool(inside_v[u]), bool(inside_v[v])
        if u_in == v_in:
            continue

        s = int(e["geom_start"])
        en = int(e["geom_end"])
        if en - s < 2:
            continue

        P = np.column_stack([x[s:en], y[s:en], z[s:en]]).astype(np.float64)
        dP = np.diff(P, axis=0)
        dP_um = dP * np.asarray(res_um, float)
        L_um = float(np.sum(np.linalg.norm(dP_um, axis=1)))

        n_crossing += 1
        if L_um > long_threshold_um:
            n_long_crossing += 1

    print("Box size (µm):", box_size_um)
    print("Long vessel threshold (µm):", long_threshold_um)
    print("Edges crossing box:", n_crossing)
    print("Long edges crossing box:", n_long_crossing)
    return n_crossing, n_long_crossing


def check_box_margin_simple(data, xBox, yBox, zBox, res_um, margin_um=10.0):
    V = np.asarray(data["vertex"]["coords_image"], dtype=np.float64)
    minV = V.min(axis=0)
    maxV = V.max(axis=0)

    margin_vox = margin_um / np.asarray(res_um, dtype=float)

    ok = (
        (xBox[0] - minV[0] >= margin_vox[0]) and
        (maxV[0] - xBox[1] >= margin_vox[0]) and
        (yBox[0] - minV[1] >= margin_vox[1]) and
        (maxV[1] - yBox[1] >= margin_vox[1]) and
        (zBox[0] - minV[2] >= margin_vox[2]) and
        (maxV[2] - zBox[1] >= margin_vox[2])
    )

    if not ok:
        raise RuntimeError(f"ROI box too close to dataset surface (< {margin_um} µm).")
    print(f"Box is ≥ {margin_um} µm away from surface")


# --------------------------
# Main cut (Gaia-like info but global arrays)
# --------------------------
def cut_outgeom_gaia_like(data, xBox, yBox, zBox, tol=1e-6, min_straight_dist=1.0):
    G = data["graph"]

    # --------------------------
    # input: geom (p)
    # --------------------------
    x = np.asarray(data["geom"]["x"], dtype=np.float64)
    y = np.asarray(data["geom"]["y"], dtype=np.float64)
    z = np.asarray(data["geom"]["z"], dtype=np.float64)

    ann_geom = data["geom"].get("annotation", None)
    if ann_geom is not None:
        ann_geom = np.asarray(ann_geom, dtype=np.int32)
        if len(ann_geom) != len(x):
            raise ValueError("geom['annotation'] length must match geom x/y/z length.")

    r_geom = data["geom"].get("radii", None)
    if r_geom is not None:
        r_geom = np.asarray(r_geom, dtype=np.float32)
        if len(r_geom) != len(x):
            raise ValueError("geom['radii'] length must match geom x/y/z length.")

    # --------------------------
    # input: vertex (v)
    # --------------------------
    Vdict = data["vertex"]
    coords_v = np.asarray(Vdict["coords_image"], dtype=np.float64)  # required
    nV = coords_v.shape[0]

    # optional per-vertex
    vid      = np.asarray(Vdict["id"], dtype=np.int64) if "id" in Vdict else None
    v_coords = np.asarray(Vdict["coords"], dtype=np.float64) if "coords" in Vdict else None
    v_ann    = np.asarray(Vdict["vertex_annotation"], dtype=np.int32) if "vertex_annotation" in Vdict else None
    v_dist   = np.asarray(Vdict["distance_to_surface"], dtype=np.float32) if "distance_to_surface" in Vdict else None
    v_radii  = np.asarray(Vdict["radii"], dtype=np.float32) if "radii" in Vdict else None

    def _check_len(name, arr):
        if arr is not None and len(arr) != nV:
            raise ValueError(f"vertex['{name}'] length must match number of vertices ({nV}).")
    _check_len("id", vid)
    if v_coords is not None and v_coords.shape[0] != nV:
        raise ValueError("vertex['coords'] shape[0] must match number of vertices.")
    _check_len("vertex_annotation", v_ann)
    _check_len("distance_to_surface", v_dist)
    _check_len("radii", v_radii)

    inside_v = (
        (coords_v[:, 0] >= xBox[0]) & (coords_v[:, 0] <= xBox[1]) &
        (coords_v[:, 1] >= yBox[0]) & (coords_v[:, 1] <= yBox[1]) &
        (coords_v[:, 2] >= zBox[0]) & (coords_v[:, 2] <= zBox[1])
    )

    # --------------------------
    # new graph H
    # --------------------------
    H = ig.Graph()
    H.add_vertices(0)

    v_attrs = list(G.vs.attributes())
    e_attrs = list(G.es.attributes())

    inside_old_ids = np.where(inside_v)[0].astype(int)
    H.add_vertices(len(inside_old_ids))
    old2new = {int(old_i): int(new_i) for new_i, old_i in enumerate(inside_old_ids)}

    # copy igraph vertex attrs (whatever exists)
    for a in v_attrs:
        H.vs[a] = [G.vs[int(i)][a] for i in inside_old_ids]

    # --------------------------
    # output: vertex arrays for H (start with inside vertices)
    # --------------------------
    H_vertex = {"coords_image": coords_v[inside_old_ids].copy()}
    if vid is not None:
        H_vertex["id"] = vid[inside_old_ids].copy()
    if v_coords is not None:
        H_vertex["coords"] = v_coords[inside_old_ids].copy()
    if v_ann is not None:
        H_vertex["vertex_annotation"] = v_ann[inside_old_ids].copy()
    if v_dist is not None:
        H_vertex["distance_to_surface"] = v_dist[inside_old_ids].copy()
    if v_radii is not None:
        H_vertex["radii"] = v_radii[inside_old_ids].copy()

    # --------------------------
    # new vertices on border
    # --------------------------
    border_key2new = {}

    def add_border_vertex(inter_point, inherit_from_h_idx):
        """
        Create/reuse border vertex.
        coords_image = inter_point.
        Others inherited from inherit_from_h_idx.
        """
        key = quant_key(inter_point, tol=tol)
        if key in border_key2new:
            return border_key2new[key]

        ww = H.vcount()
        H.add_vertices(1)
        border_key2new[key] = ww

        # igraph vertex attrs
        for a in v_attrs:
            H.vs[ww][a] = H.vs[inherit_from_h_idx][a]

        # vertex arrays
        H_vertex["coords_image"] = np.vstack([H_vertex["coords_image"], inter_point.reshape(1, 3)])
        for k in list(H_vertex.keys()):
            if k == "coords_image":
                continue
            arr = H_vertex[k]
            if arr.ndim == 2:
                H_vertex[k] = np.vstack([arr, arr[inherit_from_h_idx:inherit_from_h_idx+1]])
            else:
                H_vertex[k] = np.concatenate([arr, arr[inherit_from_h_idx:inherit_from_h_idx+1]])

        return ww

    # --------------------------
    # new geom arrays (p) + Gaia-like pointwise traces (global arrays)
    # --------------------------
    new_x, new_y, new_z = [], [], []
    new_ann = []
    new_r = []

    # Gaia definitions stored globally:
    # lengths2: distance to next point (npoints-1), stored as npoints with last=0
    # length(edge) = sum(lengths2[s:en])
    # diameters: per point (len=npoints)
    # lengths: per point (len=npoints) -> we store arclen (cumulative)
    new_lengths2 = []
    new_lengths = []
    new_diameters = []

    def append_geom_block(P_keep, A_keep, R_keep):
        start_idx = len(new_x)
        n = int(P_keep.shape[0])

        new_x.extend(P_keep[:, 0].tolist())
        new_y.extend(P_keep[:, 1].tolist())
        new_z.extend(P_keep[:, 2].tolist())

        if A_keep is not None:
            new_ann.extend(A_keep.tolist())

        if R_keep is not None:
            # diameters per point
            D = (2.0 * np.asarray(R_keep, dtype=np.float32))
            new_r.extend(np.asarray(R_keep, dtype=np.float32).tolist())
        else:
            D = np.full(n, np.nan, dtype=np.float32)

        new_diameters.extend(D.astype(np.float32).tolist())

        if n < 2:
            new_lengths2.extend([0.0] * n)
            new_lengths.extend([0.0] * n)
            return start_idx, start_idx + n

        dP = np.diff(P_keep, axis=0)
        seg = np.linalg.norm(dP, axis=1).astype(np.float32)                  # (n-1,)
        lengths2_per_point = np.concatenate([seg, [0.0]]).astype(np.float32) # (n,)
        arclen = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)  # (n,)

        new_lengths2.extend(lengths2_per_point.tolist())
        new_lengths.extend(arclen.tolist())
        return start_idx, start_idx + n

    new_geom_start, new_geom_end = [], []
    new_edges = []
    new_edge_attr = {a: [] for a in e_attrs if a not in ["geom_start", "geom_end"]}
    orig_eid = []
    # --------------------------
    # iterate edges
    # --------------------------

    for e in G.es:
        u = int(e.source)
        v = int(e.target)

        s0 = int(e["geom_start"])
        e0 = int(e["geom_end"])
        if e0 - s0 < 2:
            continue

        P = np.column_stack([x[s0:e0], y[s0:e0], z[s0:e0]]).astype(np.float64, copy=False)
        A = ann_geom[s0:e0].copy() if ann_geom is not None else None
        R = r_geom[s0:e0].copy() if r_geom is not None else None

        # orient so P[0] matches coords_image[u]
        cu = coords_v[u]
        if not np.allclose(P[0], cu, atol=1e-6):
            P = P[::-1].copy()
            if A is not None:
                A = A[::-1].copy()
            if R is not None:
                R = R[::-1].copy()
            u, v = v, u

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # A: both inside
        if u_in and v_in:
            uu = old2new[u]
            vv = old2new[v]

            inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
            if np.all(inside_mask):
                P_keep, A_keep, R_keep = P, A, R
            else:
                idx = np.where(inside_mask)[0]
                if len(idx) < 2:
                    continue
                cuts = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[0, cuts + 1]
                ends = np.r_[cuts, len(idx) - 1]
                k = int(np.argmax(ends - starts + 1))
                i0 = int(idx[starts[k]])
                i1 = int(idx[ends[k]]) + 1
                P_keep = P[i0:i1]
                A_keep = A[i0:i1] if A is not None else None
                R_keep = R[i0:i1] if R is not None else None

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep)

            new_edges.append((uu, vv))
            orig_eid.append(int(e.index))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)
            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])
            continue

        # B: both outside (skip as per your current)
        if (not u_in) and (not v_in):
            continue

        # C: one in, one out
        inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
        if not np.any(inside_mask):
            continue

        # leaving: u in, v out
        if u_in:
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == -1)[0]
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[0])

            inter = segment_box_intersection(P[i], P[i+1], xBox, yBox, zBox)
            if inter is None:
                continue

            P_keep = np.vstack([P[:i+1], inter])

            A_keep = None
            if A is not None:
                inter_ann = int(A[i])  # closest internal point
                A_keep = np.concatenate([A[:i+1], np.array([inter_ann], np.int32)])

            R_keep = None
            if R is not None:
                inter_r = float(R[i])  # closest internal point
                R_keep = np.concatenate([R[:i+1], np.array([inter_r], np.float32)])

            uu = old2new[u]
            ww = add_border_vertex(inter, inherit_from_h_idx=uu)

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep)

            new_edges.append((uu, ww))
            orig_eid.append(int(e.index))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)
            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])

        # entering: u out, v in
        else:
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == 1)[0]
            if len(cut_idx) == 0:
                continue
            i = int(cut_idx[-1])

            inter = segment_box_intersection(P[i], P[i+1], xBox, yBox, zBox)
            if inter is None:
                continue

            P_keep = np.vstack([inter, P[i+1:]])

            A_keep = None
            if A is not None:
                inter_ann = int(A[i+1])  # closest internal point
                A_keep = np.concatenate([np.array([inter_ann], np.int32), A[i+1:]])

            R_keep = None
            if R is not None:
                inter_r = float(R[i+1])  # closest internal point
                R_keep = np.concatenate([np.array([inter_r], np.float32), R[i+1:]])

            vv = old2new[v]
            ww = add_border_vertex(inter, inherit_from_h_idx=vv)

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep)

            new_edges.append((ww, vv))
            orig_eid.append(int(e.index))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)
            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])

    # --------------------------
    # finalize output
    # --------------------------
    if len(new_edges) == 0:
        out = {
            "graph": H,
            "vertex": {k: (np.empty((0,3), dtype=v.dtype) if v.ndim == 2 else np.empty((0,), dtype=v.dtype))
                       for k, v in H_vertex.items()},
            "geom": {
                "x": np.empty(0, np.float64),
                "y": np.empty(0, np.float64),
                "z": np.empty(0, np.float64),
                "lengths2": np.empty(0, np.float32),
                "lengths": np.empty(0, np.float32),
                "diameters": np.empty(0, np.float32),
            },
        }
        if ann_geom is not None:
            out["geom"]["annotation"] = np.empty(0, np.int32)
        if r_geom is not None:
            out["geom"]["radii"] = np.empty(0, np.float32)
        return out

    H.add_edges(new_edges)
    H.es["geom_start"] = list(map(int, new_geom_start))
    H.es["geom_end"] = list(map(int, new_geom_end))
    H.es["orig_eid"] = orig_eid

    for a, vals in new_edge_attr.items():
        H.es[a] = vals

    nx = np.asarray(new_x, np.float64)
    ny = np.asarray(new_y, np.float64)
    nz = np.asarray(new_z, np.float64)

    # pointwise arrays
    L2 = np.asarray(new_lengths2, dtype=np.float32)
    Lp = np.asarray(new_lengths, dtype=np.float32)
    Dp = np.asarray(new_diameters, dtype=np.float32)

    # sanity lengths
    nP = len(nx)
    if len(L2) != nP or len(Lp) != nP or len(Dp) != nP:
        raise RuntimeError("geom pointwise arrays length mismatch with x/y/z.")

    # recompute length(edge) as sum(lengths2)  (Gaia definition)
    edge_len = np.zeros(H.ecount(), dtype=np.float64)
    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])
        if en - s < 2:
            edge_len[ei] = 0.0
        else:
            edge_len[ei] = float(np.sum(L2[s:en]))
    H.es["length"] = edge_len.tolist()

    # recompute tortuosity for cut (still useful)
    lt = np.zeros(H.ecount(), dtype=np.float64)
    sd = np.zeros(H.ecount(), dtype=np.float64)
    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])
        if en - s < 2:
            continue
        dx = np.diff(nx[s:en])
        dy = np.diff(ny[s:en])
        dz = np.diff(nz[s:en])
        lt[ei] = np.sum(np.sqrt(dx*dx + dy*dy + dz*dz))
        sd[ei] = np.sqrt((nx[en-1]-nx[s])**2 + (ny[en-1]-ny[s])**2 + (nz[en-1]-nz[s])**2)

    tort = np.full(H.ecount(), np.nan, dtype=np.float64)
    m = sd >= float(min_straight_dist)
    tort[m] = lt[m] / sd[m]
    H.es["length_tortuous"] = lt.tolist()
    H.es["tortuosity"] = tort.tolist()

    # delete isolated + keep arrays aligned
    iso = [vv.index for vv in H.vs if H.degree(vv) == 0]
    if iso:
        keep = np.ones(H.vcount(), dtype=bool)
        keep[np.array(iso, int)] = False
        H.delete_vertices(iso)
        for k in H_vertex:
            H_vertex[k] = H_vertex[k][keep]

    out = {
        "graph": H,
        "vertex": H_vertex,
        "geom": {
            "x": nx,
            "y": ny,
            "z": nz,
            "lengths2": L2,       # per point (last of each edge = 0)
            "lengths": Lp,        # per point (arclen)
            "diameters": Dp,      # per point
        },
    }

    if ann_geom is not None:
        out["geom"]["annotation"] = np.asarray(new_ann, np.int32)
        if len(out["geom"]["annotation"]) != len(nx):
            raise RuntimeError("Cut annotation length mismatch with geom length.")

    if r_geom is not None:
        out["geom"]["radii"] = np.asarray(new_r, np.float32)
        if len(out["geom"]["radii"]) != len(nx):
            raise RuntimeError("Cut radii length mismatch with geom length.")

    return out


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/18_igraph_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/18_igraph_Hcut3.pkl"

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

    analyze_cut_edges(data, xBox, yBox, zBox, res_um=res, long_factor=2.0)
    check_box_margin_simple(data, xBox, yBox, zBox, res_um=res, margin_um=10.0)

    cut = cut_outgeom_gaia_like(
        data,
        xBox, yBox, zBox,
        tol=1e-6,
        min_straight_dist=1.0
    )

    # cut = restore_edges_gaia_like(cut, store_points=False) => edge["points"], edge["diameters"], edge["lenghts2"] ... Computational High !!

    # strict sanity check (geom, not coords)
    Pcut = np.column_stack([cut["geom"]["x"], cut["geom"]["y"], cut["geom"]["z"]])
    inside_cut = (
        (Pcut[:, 0] >= xBox[0]) & (Pcut[:, 0] <= xBox[1]) &
        (Pcut[:, 1] >= yBox[0]) & (Pcut[:, 1] <= yBox[1]) &
        (Pcut[:, 2] >= zBox[0]) & (Pcut[:, 2] <= zBox[1])
    )
    n_out = int(np.sum(~inside_cut))
    print("Cut points outside box:", n_out, "/", len(Pcut))

    with open(out_path, "wb") as f:
        pickle.dump(cut, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path,
          "Vertices:", cut["graph"].vcount(),
          "Edges:", cut["graph"].ecount())
