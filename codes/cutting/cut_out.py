import numpy as np
import igraph as ig
import pickle

# --------------------------
# Helpers
# --------------------------
def is_inside_box_np(P, xBox, yBox, zBox):
    """Strict inside: inclusive bounds, no tolerance."""
    return (
        (P[:, 0] >= xBox[0]) & (P[:, 0] <= xBox[1]) &
        (P[:, 1] >= yBox[0]) & (P[:, 1] <= yBox[1]) &
        (P[:, 2] >= zBox[0]) & (P[:, 2] <= zBox[1])
    )

def quant_key(p, tol=1e-6):
    """Quantize point for stable dict key."""
    p = np.asarray(p, dtype=np.float64)
    return tuple(np.round(p / tol).astype(np.int64))

def segment_box_intersection(p0, p1, xBox, yBox, zBox):
    """
    Intersect segment p0->p1 with AABB boundary.
    Returns intersection point (3,) or None.

    IMPORTANT: Snap axis coordinate exactly to boundary plane to avoid tiny outside points.
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
            p[axis] = bound  # SNAP to plane

            if (xBox[0] <= p[0] <= xBox[1] and
                yBox[0] <= p[1] <= yBox[1] and
                zBox[0] <= p[2] <= zBox[1]):
                if best_t is None or t < best_t:
                    best_t = t
                    best_p = p

    return None if best_p is None else best_p


# --------------------------
# Optional: analyze long vessels cut
# --------------------------
def analyze_cut_edges(data, xBox, yBox, zBox, res_um, long_factor=2.0):
    G = data["graph"]

    # geometry points (p)
    x = data["geom"]["x"]          # np.ndarray, shape (nP,)
    y = data["geom"]["y"]          # np.ndarray, shape (nP,)
    z = data["geom"]["z"]          # np.ndarray, shape (nP,)


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
    """
    Simple check: ROI box must be at least margin_um from global dataset bounds.
    Uses coords_image bounds (vertex coords).
    """
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
# Main cut (A/B/C structure, fixed)
# --------------------------
def cut_outgeom_gaia_like(data, xBox, yBox, zBox, tol=1e-6, min_straight_dist=1.0):
    """
    data: dict con {"graph":G, "coords":{"x":x,"y":y,"z":z}, "annotation":ann_geom}
    Devuelve data_cut en el mismo formato OUTGEOM, recortando también annotation.

    FIX CRITICO:
      - En Case A, guardar P_keep (no P).
      - Intersections snapped to boundary plane.
      - Strict inside check at the end.
    """
    G = data["graph"]

    # geometry points (p)
    x = data["geom"]["x"]          # np.ndarray, shape (nP,)
    y = data["geom"]["y"]          # np.ndarray, shape (nP,)
    z = data["geom"]["z"]          # np.ndarray, shape (nP,)


    ann_geom = data["geom"].get("annotation", None)
    if ann_geom is not None and len(ann_geom) != len(x):
        raise ValueError("geom['annotation'] length must match geom x/y/z length.")

    r_geom = data["geom"].get("radii", None)   # (nP,) o None

    # --- nuevo grafo H ---
    H = ig.Graph()
    H.add_vertices(0)

    v_attrs = list(G.vs.attributes())
    e_attrs = list(G.es.attributes())

    coords_v = data["vertex"]["coords_image"]
    inside_v = (
        (coords_v[:, 0] >= xBox[0]) & (coords_v[:, 0] <= xBox[1]) &
        (coords_v[:, 1] >= yBox[0]) & (coords_v[:, 1] <= yBox[1]) &
        (coords_v[:, 2] >= zBox[0]) & (coords_v[:, 2] <= zBox[1])
    )

    inside_old_ids = np.where(inside_v)[0].astype(int)
    H.add_vertices(len(inside_old_ids))

    old2new = {}
    for new_i, old_i in enumerate(inside_old_ids):
        old2new[int(old_i)] = int(new_i)

    # Copiar atributos de vertices interiores
    for a in v_attrs:
        H.vs[a] = [G.vs[int(i)][a] for i in inside_old_ids]

    border_key2new = {}

    new_x, new_y, new_z = [], [], []
    new_ann = []
    new_geom_start, new_geom_end = [], []

    new_edges = []
    new_edge_attr = {a: [] for a in e_attrs if a not in ["geom_start", "geom_end"]}

    # ---------------------------------------------------------
    # iterar edges
    # ---------------------------------------------------------
    for e in G.es:
        u = int(e.source)
        v = int(e.target)
        s = int(e["geom_start"])
        en = int(e["geom_end"])
        if en - s < 2:
            continue

        # slice de puntos del edge
        P = np.column_stack([x[s:en], y[s:en], z[s:en]]).astype(np.float64, copy=False)
        A = ann_geom[s:en].copy() if ann_geom is not None else None

        # Orientación coherente: P[0] debería ser coords_image del source
        cu = coords_v[u].astype(np.float64, copy=False)
        if not np.allclose(P[0], cu, atol=1e-6):
            P = P[::-1].copy()
            if A is not None:
                A = A[::-1].copy()
            u, v = v, u  # swap

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # --------------------------
        # A: ambos dentro
        # --------------------------
        if u_in and v_in:
            uu = old2new[u]
            vv = old2new[v]

            inside_mask = is_inside_box_np(P, xBox, yBox, zBox)

            if np.all(inside_mask):
                P_keep = P
                A_keep = A
            else:
                idx = np.where(inside_mask)[0]
                if len(idx) < 2:
                    continue
                cuts = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[0, cuts + 1]
                ends = np.r_[cuts, len(idx) - 1]
                seg_lengths = (ends - starts + 1)
                k = int(np.argmax(seg_lengths))
                i0 = int(idx[starts[k]])
                i1 = int(idx[ends[k]]) + 1
                P_keep = P[i0:i1]
                A_keep = A[i0:i1] if A is not None else None

            # FIX CRITICO: guardar P_keep, no P
            start_idx = len(new_x)
            new_x.extend(P_keep[:, 0].tolist())
            new_y.extend(P_keep[:, 1].tolist())
            new_z.extend(P_keep[:, 2].tolist())
            if A_keep is not None:
                new_ann.extend(A_keep.tolist())
            end_idx = len(new_x)

            new_edges.append((uu, vv))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)

            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])
            continue

        
        # --------------------------
        # B: ambos fuera
        # --------------------------

        if (not u_in) and (not v_in):
            continue

        '''
        # B: ambos fuera pero cruza la caja (entra y sale)
        if (not u_in) and (not v_in):
            inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
            diff = np.diff(inside_mask.astype(np.int8))

            enter_idx = np.where(diff == 1)[0]   # 0 -> 1
            exit_idx  = np.where(diff == -1)[0]  # 1 -> 0

            if len(enter_idx) == 0 or len(exit_idx) == 0:
                continue

            i_ent = int(enter_idx[0])
            i_ext = int(exit_idx[-1])
            if i_ent >= i_ext:
                continue

            # entry intersection: outside -> inside  (i_ent -> i_ent+1)
            inter_in = segment_box_intersection(P[i_ent], P[i_ent+1], xBox, yBox, zBox)
            if inter_in is None:
                continue

            # exit intersection: inside -> outside (i_ext -> i_ext+1)
            inter_out = segment_box_intersection(P[i_ext], P[i_ext+1], xBox, yBox, zBox)
            if inter_out is None:
                continue

            P_mid = P[i_ent + 1 : i_ext + 1]  # inside points
            P_keep = np.vstack([inter_in, P_mid, inter_out])

            if A is not None:
                ann_in = int(A[i_ent + 1])
                ann_out = int(A[i_ext])
                A_mid = A[i_ent + 1 : i_ext + 1]
                A_keep = np.concatenate([np.array([ann_in], np.int32), A_mid, np.array([ann_out], np.int32)])
            else:
                A_keep = None

            key_in = quant_key(inter_in, tol=tol)
            key_out = quant_key(inter_out, tol=tol)

            def get_or_create_border_vertex(key, inter, inherit_from_old):
                if key in border_key2new:
                    return border_key2new[key]
                ww = H.vcount()
                H.add_vertices(1)
                border_key2new[key] = ww
                for a in v_attrs:
                    if a == "coords_image":
                        H.vs[ww][a] = tuple(map(float, inter))
                    else:
                        H.vs[ww][a] = G.vs[int(inherit_from_old)][a]
                return ww

            w_in = get_or_create_border_vertex(key_in, inter_in, inherit_from_old=u)
            w_out = get_or_create_border_vertex(key_out, inter_out, inherit_from_old=v)

            start_idx = len(new_x)
            new_x.extend(P_keep[:, 0].tolist())
            new_y.extend(P_keep[:, 1].tolist())
            new_z.extend(P_keep[:, 2].tolist())
            if A_keep is not None:
                new_ann.extend(A_keep.tolist())
            end_idx = len(new_x)

            new_edges.append((w_in, w_out))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)

            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])

            continue
            '''
        # --------------------------
        # C: cruza borde (uno dentro, otro fuera)
        # --------------------------
        inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
        if not np.any(inside_mask):
            continue

        # C1) u_in True, v_in False  (sale)
        if u_in:
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == -1)[0]  # 1->0
            if len(cut_idx) == 0:
                continue

            i = int(cut_idx[0])  # last inside is i
            inter = segment_box_intersection(P[i], P[i+1], xBox, yBox, zBox)
            if inter is None:
                continue

            P_keep = np.vstack([P[:i+1], inter])

            if A is not None:
                inter_ann = int(A[i])
                A_keep = np.concatenate([A[:i+1], np.array([inter_ann], np.int32)])
            else:
                A_keep = None

            uu = old2new[u]
            key = quant_key(inter, tol=tol)

            if key in border_key2new:
                ww = border_key2new[key]
            else:
                ww = H.vcount()
                H.add_vertices(1)
                border_key2new[key] = ww
                for a in v_attrs:
                    if a == "coords_image":
                        H.vs[ww][a] = tuple(map(float, inter))
                    else:
                        H.vs[ww][a] = H.vs[uu][a]

            start_idx = len(new_x)
            new_x.extend(P_keep[:, 0].tolist())
            new_y.extend(P_keep[:, 1].tolist())
            new_z.extend(P_keep[:, 2].tolist())
            if A_keep is not None:
                new_ann.extend(A_keep.tolist())
            end_idx = len(new_x)

            new_edges.append((uu, ww))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)

            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])

        # C2) v_in True, u_in False (entra)
        else:
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == 1)[0]  # 0->1
            if len(cut_idx) == 0:
                continue

            i = int(cut_idx[-1])  # first inside is i+1
            inter = segment_box_intersection(P[i], P[i+1], xBox, yBox, zBox)
            if inter is None:
                continue

            P_keep = np.vstack([inter, P[i+1:]])

            if A is not None:
                inter_ann = int(A[i+1])
                A_keep = np.concatenate([np.array([inter_ann], np.int32), A[i+1:]])
            else:
                A_keep = None

            vv = old2new[v]
            key = quant_key(inter, tol=tol)

            if key in border_key2new:
                ww = border_key2new[key]
            else:
                ww = H.vcount()
                H.add_vertices(1)
                border_key2new[key] = ww
                for a in v_attrs:
                    if a == "coords_image":
                        H.vs[ww][a] = tuple(map(float, inter))
                    else:
                        H.vs[ww][a] = H.vs[vv][a]

            start_idx = len(new_x)
            new_x.extend(P_keep[:, 0].tolist())
            new_y.extend(P_keep[:, 1].tolist())
            new_z.extend(P_keep[:, 2].tolist())
            if A_keep is not None:
                new_ann.extend(A_keep.tolist())
            end_idx = len(new_x)

            new_edges.append((ww, vv))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)

            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])

    # --------------------------
    # finalize
    # --------------------------
    if len(new_edges) == 0:
        out = {
            "graph": H,
            "geom": {
                "x": np.empty(0, dtype=np.float64),
                "y": np.empty(0, dtype=np.float64),
                "z": np.empty(0, dtype=np.float64),
            }
        }
        if ann_geom is not None:
            out["geom"]["annotation"] = np.empty(0, dtype=np.int32)
        return out


    H.add_edges(new_edges)

    H.es["geom_start"] = list(map(int, new_geom_start))
    H.es["geom_end"] = list(map(int, new_geom_end))

    for a, vals in new_edge_attr.items():
        H.es[a] = vals

    nx = np.asarray(new_x, dtype=np.float64)
    ny = np.asarray(new_y, dtype=np.float64)
    nz = np.asarray(new_z, dtype=np.float64)

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
        lt[ei] = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz))
        sd[ei] = np.sqrt((nx[en-1] - nx[s])**2 + (ny[en-1] - ny[s])**2 + (nz[en-1] - nz[s])**2)

    tort = np.full(H.ecount(), np.nan, dtype=np.float64)
    mask = sd >= float(min_straight_dist)
    tort[mask] = lt[mask] / sd[mask]
    H.es["length_tortuous"] = lt.tolist()
    H.es["tortuosity"] = tort.tolist()

    iso = [vv.index for vv in H.vs if H.degree(vv) == 0]
    if iso:
        H.delete_vertices(iso)

    if len(new_edges) == 0:
        out = {
            "graph": H,
            "geom": {
                "x": np.empty(0, dtype=np.float64),
                "y": np.empty(0, dtype=np.float64),
                "z": np.empty(0, dtype=np.float64),
            }
        }
        if ann_geom is not None:
            out["geom"]["annotation"] = np.empty(0, dtype=np.int32)
        return out

    return out


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3_B.pkl"

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

    # strict sanity check
    Pcut = np.column_stack([cut["coords"]["x"], cut["coords"]["y"], cut["coords"]["z"]])
    inside_cut = (
        (Pcut[:, 0] >= xBox[0]) & (Pcut[:, 0] <= xBox[1]) &
        (Pcut[:, 1] >= yBox[0]) & (Pcut[:, 1] <= yBox[1]) &
        (Pcut[:, 2] >= zBox[0]) & (Pcut[:, 2] <= zBox[1])
    )
    n_out = int(np.sum(~inside_cut))
    print("Cut points outside box:", n_out, "/", len(Pcut))
    if n_out == 0:
        print("No outside points. All good.")

    with open(out_path, "wb") as f:
        pickle.dump(cut, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path,
          "Vertices:", cut["graph"].vcount(),
          "Edges:", cut["graph"].ecount())
