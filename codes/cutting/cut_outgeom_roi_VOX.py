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

def analyze_long_vessel(data, xBox, yBox, zBox, long_factor=2.0):
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

    box_size = np.array([xBox[1]-xBox[0], yBox[1]-yBox[0], zBox[1]-zBox[0]], dtype=float)
    long_threshold_um = float(long_factor) * float(box_size.mean())

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
        L = float(np.sum(np.linalg.norm(dP, axis=1)))

        n_crossing += 1
        if L > long_threshold_um:
            n_long_crossing += 1

    print("Box size (µm):", box_size)
    print("Long vessel threshold (µm):", long_threshold_um)
    print("Edges crossing box:", n_crossing)
    print("Long edges crossing box:", n_long_crossing)
    return n_crossing, n_long_crossing

def check_box_margin_simple(data, xBox, yBox, zBox, margin_um=10.0, res_um_per_vox=(1.625, 1.625, 2.5)):
    """
    xBox,yBox,zBox están en VOXELS (coords_image).
    margin_um se da en µm (float o (3,))
    """
    if isinstance(margin_um, (int, float)):
        margin_um = np.array([margin_um, margin_um, margin_um], dtype=np.float64)
    else:
        margin_um = np.asarray(margin_um, dtype=np.float64)
        if margin_um.shape != (3,):
            raise ValueError("margin_um must be float or (3,)")

    res = np.asarray(res_um_per_vox, dtype=np.float64)
    margin_vox = margin_um / res

    V = np.asarray(data["vertex"]["coords_image"], dtype=np.float64)
    minV = V.min(axis=0)
    maxV = V.max(axis=0)

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
    print(f"Box is ≥ {margin_um} µm away from surface (≈ {margin_vox} vox)")

# --------------------------
# Main cut (keeps ALL attributes from the input PKL where possible)
# --------------------------
def cut_outgeom_gaia_like(data, xBox, yBox, zBox, tol=1e-6, min_straight_dist=1.0):
    """
    Keeps:
      - graph.vs attrs: all existing (copied for inside + inherited for border)
      - graph.es attrs: all existing (copied) + overwrites geom_start/geom_end + recomputes length/tortuosity
      - vertex dict: copies ALL keys (arrays) that match nV (and coords-like matrices (nV,3))
      - geom dict: copies x,y,z + re-slices annotation/radii/radii_atlas_geom if present
      - additionally writes geom["diameters"] per-point (Gaia style) from geom["radii"] if present

    IMPORTANT:
      - border vertices are created at intersections and inherit vertex attributes from nearest inside vertex.
      - edges are cut and geometry is cropped accordingly.
    """
    G = data["graph"]

    # --------------------------
    # input: geom (p)
    # --------------------------
    geom_in = data["geom"]

    x = np.asarray(geom_in["x"], dtype=np.float64)
    y = np.asarray(geom_in["y"], dtype=np.float64)
    z = np.asarray(geom_in["z"], dtype=np.float64)

    ann_geom = geom_in.get("annotation", None)
    if ann_geom is not None:
        ann_geom = np.asarray(ann_geom, dtype=np.int32)
        if len(ann_geom) != len(x):
            raise ValueError("geom['annotation'] length must match geom x/y/z length.")

    r_geom = geom_in.get("radii", None)
    if r_geom is not None:
        r_geom = np.asarray(r_geom, dtype=np.float32)
        if len(r_geom) != len(x):
            raise ValueError("geom['radii'] length must match geom x/y/z length.")

    r_atlas_geom = geom_in.get("radii_atlas_geom", None)
    if r_atlas_geom is not None:
        r_atlas_geom = np.asarray(r_atlas_geom, dtype=np.float32)
        if len(r_atlas_geom) != len(x):
            raise ValueError("geom['radii_atlas_geom'] length must match geom x/y/z length.")

    # --------------------------
    # input: vertex (v)
    # --------------------------
    Vdict_in = data["vertex"]
    coords_v = np.asarray(Vdict_in["coords_image"], dtype=np.float64)  # required
    nV = coords_v.shape[0]

    # Determine which vertex dict keys we can copy generically
    # - If arr is (nV,) or (nV,k), we keep it.
    # - For border vertices, we will append rows by inheriting from an inside vertex.
    v_keys_copy = []
    for k, arr in Vdict_in.items():
        if not isinstance(arr, np.ndarray):
            # sometimes id might be list; convert to np for consistency
            try:
                arr = np.asarray(arr)
            except Exception:
                continue
        if arr.ndim == 1 and arr.shape[0] == nV:
            v_keys_copy.append(k)
        elif arr.ndim == 2 and arr.shape[0] == nV:
            v_keys_copy.append(k)

    if "coords_image" not in v_keys_copy:
        raise KeyError("vertex['coords_image'] missing or wrong shape.")

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
    # output: vertex arrays for H (start with inside vertices), copy ALL keys we can
    # --------------------------
    H_vertex = {}
    for k in v_keys_copy:
        arr = np.asarray(Vdict_in[k])
        H_vertex[k] = arr[inside_old_ids].copy()

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

        # vertex arrays: append one row/value for each key
        for k in list(H_vertex.keys()):
            if k == "coords_image":
                # override with intersection point
                H_vertex[k] = np.vstack([H_vertex[k], inter_point.reshape(1, 3)])
                continue

            arr = H_vertex[k]
            if arr.ndim == 2:
                H_vertex[k] = np.vstack([arr, arr[inherit_from_h_idx:inherit_from_h_idx+1]])
            else:
                H_vertex[k] = np.concatenate([arr, arr[inherit_from_h_idx:inherit_from_h_idx+1]])

        return ww

    # --------------------------
    # new geom arrays (p): we keep x,y,z, lengths2, and copy any per-point arrays (annotation, radii, radii_atlas_geom)
    # --------------------------
    new_x, new_y, new_z = [], [], []
    new_ann = []            # annotation
    new_r = []              # radii
    new_r_atlas = []        # radii_atlas_geom

    new_lengths2 = []       # per-point, last=0
    new_diameters = []      # per-point (from radii), last is fine (still diameter at that point)

    def append_geom_block(P_keep, A_keep, R_keep, R_atlas_keep):
        start_idx = len(new_x)
        n = int(P_keep.shape[0])

        new_x.extend(P_keep[:, 0].tolist())
        new_y.extend(P_keep[:, 1].tolist())
        new_z.extend(P_keep[:, 2].tolist())

        if A_keep is not None:
            new_ann.extend(A_keep.tolist())

        # radii + diameters per point
        if R_keep is not None:
            R_keep_f = np.asarray(R_keep, dtype=np.float32)
            new_r.extend(R_keep_f.tolist())
            D = (2.0 * R_keep_f).astype(np.float32)
        else:
            D = np.full(n, np.nan, dtype=np.float32)
        new_diameters.extend(D.tolist())

        # radii_atlas_geom per point
        if R_atlas_keep is not None:
            new_r_atlas.extend(np.asarray(R_atlas_keep, dtype=np.float32).tolist())

        # lengths2 per point
        if n < 2:
            new_lengths2.extend([0.0] * n)
            return start_idx, start_idx + n

        dP = np.diff(P_keep, axis=0)
        seg = np.linalg.norm(dP, axis=1).astype(np.float32)                  # (n-1,)
        lengths2_per_point = np.concatenate([seg, [0.0]]).astype(np.float32) # (n,)
        new_lengths2.extend(lengths2_per_point.tolist())

        return start_idx, start_idx + n

    new_geom_start, new_geom_end = [], []
    new_edges = []

    # copy ALL edge attrs except geom_start/geom_end (we will overwrite those)
    new_edge_attr = {a: [] for a in e_attrs if a not in ["geom_start", "geom_end"]}
    orig_eid = []

    # --------------------------
    # iterate edges
    # --------------------------
    for e in G.es:
        u = int(e.source)
        v = int(e.target)

        if "geom_start" not in e.attributes() or "geom_end" not in e.attributes():
            raise KeyError("Input graph edges must have geom_start/geom_end")

        s0 = int(e["geom_start"])
        e0 = int(e["geom_end"])
        if e0 - s0 < 2:
            continue

        P = np.column_stack([x[s0:e0], y[s0:e0], z[s0:e0]]).astype(np.float64, copy=False)
        A = ann_geom[s0:e0].copy() if ann_geom is not None else None
        R = r_geom[s0:e0].copy() if r_geom is not None else None
        R_at = r_atlas_geom[s0:e0].copy() if r_atlas_geom is not None else None

        # orient so P[0] matches coords_image[u]
        cu = coords_v[u]
        if not np.allclose(P[0], cu, atol=1e-6):
            P = P[::-1].copy()
            if A is not None:
                A = A[::-1].copy()
            if R is not None:
                R = R[::-1].copy()
            if R_at is not None:
                R_at = R_at[::-1].copy()
            u, v = v, u

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # A: both inside
        if u_in and v_in:
            uu = old2new[u]
            vv = old2new[v]

            inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
            if np.all(inside_mask):
                P_keep, A_keep, R_keep, R_at_keep = P, A, R, R_at
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
                R_at_keep = R_at[i0:i1] if R_at is not None else None

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep, R_at_keep)

            new_edges.append((uu, vv))
            orig_eid.append(int(e.index))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)
            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])
            continue

        # B: both outside (skip)
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
                inter_ann = int(A[i])
                A_keep = np.concatenate([A[:i+1], np.array([inter_ann], np.int32)])

            R_keep = None
            if R is not None:
                inter_r = float(R[i])
                R_keep = np.concatenate([R[:i+1], np.array([inter_r], np.float32)])

            R_at_keep = None
            if R_at is not None:
                inter_r_at = float(R_at[i])
                R_at_keep = np.concatenate([R_at[:i+1], np.array([inter_r_at], np.float32)])

            uu = old2new[u]
            ww = add_border_vertex(inter, inherit_from_h_idx=uu)

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep, R_at_keep)

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
                inter_ann = int(A[i+1])
                A_keep = np.concatenate([np.array([inter_ann], np.int32), A[i+1:]])

            R_keep = None
            if R is not None:
                inter_r = float(R[i+1])
                R_keep = np.concatenate([np.array([inter_r], np.float32), R[i+1:]])

            R_at_keep = None
            if R_at is not None:
                inter_r_at = float(R_at[i+1])
                R_at_keep = np.concatenate([np.array([inter_r_at], np.float32), R_at[i+1:]])

            vv = old2new[v]
            ww = add_border_vertex(inter, inherit_from_h_idx=vv)

            start_idx, end_idx = append_geom_block(P_keep, A_keep, R_keep, R_at_keep)

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
                "diameters": np.empty(0, np.float32),
            },
        }
        if ann_geom is not None:
            out["geom"]["annotation"] = np.empty(0, np.int32)
        if r_geom is not None:
            out["geom"]["radii"] = np.empty(0, np.float32)
        if r_atlas_geom is not None:
            out["geom"]["radii_atlas_geom"] = np.empty(0, np.float32)
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

    # per-point arrays
    L2 = np.asarray(new_lengths2, dtype=np.float32)
    Dp = np.asarray(new_diameters, dtype=np.float32)

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

    # recompute tortuosity for cut (uses the cut geometry)
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
    H.es["tortuosity"] = tort.tolist()

    # delete isolated + keep vertex arrays aligned
    iso = [vv.index for vv in H.vs if H.degree(vv) == 0]
    if iso:
        keep = np.ones(H.vcount(), dtype=bool)
        keep[np.array(iso, int)] = False
        H.delete_vertices(iso)
        for k in H_vertex:
            H_vertex[k] = H_vertex[k][keep]

    # Build output geom dict:
    # - always include x,y,z,lengths2
    # - include lengths2, diameters
    # - include any original per-point arrays if present (annotation/radii/radii_atlas_geom)
    geom_out = {
        "x": nx,
        "y": ny,
        "z": nz,
        "lengths2": L2,        # per point
        "diameters": Dp,       # per point (from radii)
    }
    if ann_geom is not None:
        geom_out["annotation"] = np.asarray(new_ann, np.int32)
        if len(geom_out["annotation"]) != len(nx):
            raise RuntimeError("Cut annotation length mismatch with geom length.")
    if r_geom is not None:
        geom_out["radii"] = np.asarray(new_r, np.float32)
        if len(geom_out["radii"]) != len(nx):
            raise RuntimeError("Cut radii length mismatch with geom length.")
    if r_atlas_geom is not None:
        geom_out["radii_atlas_geom"] = np.asarray(new_r_atlas, np.float32)
        if len(geom_out["radii_atlas_geom"]) != len(nx):
            raise RuntimeError("Cut radii_atlas_geom length mismatch with geom length.")

    out = {
        "graph": H,
        "vertex": H_vertex,
        "geom": geom_out,
    }

    # keep a unit tag so later you know what you cut in
    # here coords_image/xBox are vox, so set vox
    out["unit"] = data.get("unit", "vox")

    return out


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3.pkl"

    print("Loading:", in_path)
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

    analyze_long_vessel(data, xBox, yBox, zBox, long_factor=2.0)
    check_box_margin_simple(data, xBox, yBox, zBox, margin_um=10.0, res_um_per_vox=(1.625, 1.625, 2.5))

    cut = cut_outgeom_gaia_like(
        data,
        xBox, yBox, zBox,
        tol=1e-6,
        min_straight_dist=1.0
    )

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
          "Unit:", cut.get("unit", None),
          "Vertices:", cut["graph"].vcount(),
          "Edges:", cut["graph"].ecount())

    # quick key summary
    print("\n=== Cut summary ===")
    print("graph.vs attrs:", cut["graph"].vs.attributes())
    print("graph.es attrs:", cut["graph"].es.attributes())
    print("vertex keys:", list(cut["vertex"].keys()))
    print("geom keys:", list(cut["geom"].keys()))

