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

def quant_key(p, tol=1e-4):
    return tuple(np.round(np.asarray(p) / tol).astype(np.int64))

def segment_box_intersection(p_in, p_out, xBox, yBox, zBox):
    """
    Intersección del segmento (p_in -> p_out) con el borde del AABB.
    p_in está dentro, p_out está fuera.
    Devuelve punto de intersección (3,) o None.
    """
    p_in = np.asarray(p_in, float)
    p_out = np.asarray(p_out, float)
    d = p_out - p_in

    t_candidates = []
    for axis, (mn, mx) in enumerate([xBox, yBox, zBox]):
        if abs(d[axis]) < 1e-12:
            continue

        for bound in (mn, mx):
            t = (bound - p_in[axis]) / d[axis]
            if 0.0 <= t <= 1.0:
                p = p_in + t * d
                if (xBox[0] - 1e-6 <= p[0] <= xBox[1] + 1e-6 and
                    yBox[0] - 1e-6 <= p[1] <= yBox[1] + 1e-6 and
                    zBox[0] - 1e-6 <= p[2] <= zBox[1] + 1e-6):
                    t_candidates.append((t, p))

    if not t_candidates:
        return None

    t_candidates.sort(key=lambda a: a[0])
    return t_candidates[0][1]

# --------------------------
# Main cut
# --------------------------
def cut_outgeom_gaia_like(data, xBox, yBox, zBox, tol=1e-4, min_straight_dist=1.0):
    """
    data: dict con {"graph":G, "coords":{"x":x,"y":y,"z":z}, "annotation":ann_geom}
    Devuelve data_cut en el mismo formato OUTGEOM, recortando también annotation.
    """
    G = data["graph"]
    x = np.asarray(data["coords"]["x"], dtype=np.float32)
    y = np.asarray(data["coords"]["y"], dtype=np.float32)
    z = np.asarray(data["coords"]["z"], dtype=np.float32)

    ann_geom = None
    if "annotation" in data:
        ann_geom = np.asarray(data["annotation"], dtype=np.int32)
        if len(ann_geom) != len(x):
            raise ValueError("data['annotation'] length must match coords arrays length.")

    # --- nuevo grafo H ---
    H = ig.Graph()
    H.add_vertices(0)

    v_attrs = list(G.vs.attributes())
    e_attrs = list(G.es.attributes())

    old2new = {}

    coords_v = np.asarray(G.vs["coords_image"], dtype=np.float32)
    inside_v = (
        (coords_v[:, 0] >= xBox[0]) & (coords_v[:, 0] <= xBox[1]) &
        (coords_v[:, 1] >= yBox[0]) & (coords_v[:, 1] <= yBox[1]) &
        (coords_v[:, 2] >= zBox[0]) & (coords_v[:, 2] <= zBox[1])
    )

    inside_old_ids = np.where(inside_v)[0]
    H.add_vertices(len(inside_old_ids))

    for new_i, old_i in enumerate(inside_old_ids):
        old2new[int(old_i)] = int(new_i)

    # Copiar atributos de vertices interiores
    for a in v_attrs:
        H.vs[a] = [G.vs[int(i)][a] for i in inside_old_ids]

    border_key2new = {}

    new_x, new_y, new_z = [], [], []
    new_ann = []  # point-wise annotation recortada (si existe)
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
        P = np.column_stack([x[s:en], y[s:en], z[s:en]]).astype(np.float32, copy=False)
        A = ann_geom[s:en].copy() if ann_geom is not None else None

        # Orientación coherente: P[0] debería ser coords_image del source
        cu = np.asarray(G.vs[u]["coords_image"], dtype=np.float32)
        if not np.allclose(P[0], cu, atol=1e-5):
            P = P[::-1].copy()
            if A is not None:
                A = A[::-1].copy()

        u_in = bool(inside_v[u])
        v_in = bool(inside_v[v])

        # A: ambos dentro
        if u_in and v_in:
            uu = old2new[u]
            vv = old2new[v]

            start_idx = len(new_x)
            new_x.extend(P[:, 0].tolist())
            new_y.extend(P[:, 1].tolist())
            new_z.extend(P[:, 2].tolist())
            if A is not None:
                new_ann.extend(A.tolist())
            end_idx = len(new_x)

            new_edges.append((uu, vv))
            new_geom_start.append(start_idx)
            new_geom_end.append(end_idx)

            for a in new_edge_attr.keys():
                new_edge_attr[a].append(e[a])
            continue

        # B: ambos fuera
        if (not u_in) and (not v_in):
            continue

        # C: cruza borde
        inside_mask = is_inside_box_np(P, xBox, yBox, zBox)
        if not np.any(inside_mask):
            continue

        if u_in:
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == -1)[0]

            if len(cut_idx) == 0:
                P_keep = P
                A_keep = A
                inter = None
            else:
                i = int(cut_idx[0])
                p_in = P[i]
                p_out = P[i + 1]
                inter = segment_box_intersection(p_in, p_out, xBox, yBox, zBox)
                if inter is None:
                    continue

                P_keep = np.vstack([P[:i + 1], inter.astype(np.float32)])

                if A is not None:
                    # anotación del punto nuevo: copiamos la del último punto dentro (P[i])
                    inter_ann = int(A[i])
                    A_keep = np.concatenate([A[:i + 1], np.array([inter_ann], dtype=np.int32)])
                else:
                    A_keep = None

            uu = old2new[u]

            if inter is None:
                inter = P_keep[-1]
            key = quant_key(inter, tol=tol)

            if key in border_key2new:
                ww = border_key2new[key]
            else:
                ww = H.vcount()
                H.add_vertices(1)
                border_key2new[key] = ww
                for a in v_attrs:
                    if a == "coords":
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

        else:
            # v_in == True
            diff = np.diff(inside_mask.astype(np.int8))
            cut_idx = np.where(diff == 1)[0]  # 0->1 entrando

            if len(cut_idx) == 0:
                P_keep = P
                A_keep = A
                inter = None
            else:
                i = int(cut_idx[-1])
                p_out = P[i]
                p_in = P[i + 1]
                inter = segment_box_intersection(p_in, p_out, xBox, yBox, zBox)
                if inter is None:
                    continue

                P_keep = np.vstack([inter.astype(np.float32), P[i + 1:]])

                if A is not None:
                    inter_ann = int(A[i + 1])  # primer punto dentro
                    A_keep = np.concatenate([np.array([inter_ann], dtype=np.int32), A[i + 1:]])
                else:
                    A_keep = None

            vv = old2new[v]

            if inter is None:
                inter = P_keep[0]
            key = quant_key(inter, tol=tol)

            if key in border_key2new:
                ww = border_key2new[key]
            else:
                ww = H.vcount()
                H.add_vertices(1)
                border_key2new[key] = ww
                for a in v_attrs:
                    if a == "coords":
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

    # Añadir edges
    if len(new_edges) == 0:
        out = {"graph": H, "coords": {"x": np.array([], np.float32),
                                     "y": np.array([], np.float32),
                                     "z": np.array([], np.float32)}}
        if ann_geom is not None:
            out["annotation"] = np.array([], np.int32)
        return out

    H.add_edges(new_edges)

    # geom indices
    H.es["geom_start"] = list(map(int, new_geom_start))
    H.es["geom_end"] = list(map(int, new_geom_end))

    # copiar atributos edge
    for a, vals in new_edge_attr.items():
        H.es[a] = vals

    # Recalcular length_tortuous / tortuosity
    nx = np.asarray(new_x, dtype=np.float32)
    ny = np.asarray(new_y, dtype=np.float32)
    nz = np.asarray(new_z, dtype=np.float32)

    lt = np.zeros(H.ecount(), dtype=np.float32)
    sd = np.zeros(H.ecount(), dtype=np.float32)

    for ei in range(H.ecount()):
        s = int(H.es[ei]["geom_start"])
        en = int(H.es[ei]["geom_end"])
        if en - s < 2:
            continue
        dx = np.diff(nx[s:en])
        dy = np.diff(ny[s:en])
        dz = np.diff(nz[s:en])
        lt[ei] = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz))
        sd[ei] = np.sqrt((nx[en - 1] - nx[s]) ** 2 + (ny[en - 1] - ny[s]) ** 2 + (nz[en - 1] - nz[s]) ** 2)

    tort = np.full(H.ecount(), np.nan, dtype=np.float32)
    mask = sd >= float(min_straight_dist)
    tort[mask] = lt[mask] / sd[mask]
    H.es["length_tortuous"] = lt.tolist()
    H.es["tortuosity"] = tort.tolist()

    # borrar vértices aislados
    iso = [v.index for v in H.vs if H.degree(v) == 0]
    if iso:
        H.delete_vertices(iso)

    out = {
        "graph": H,
        "coords": {"x": nx, "y": ny, "z": nz},
    }
    if ann_geom is not None:
        out["annotation"] = np.asarray(new_ann, dtype=np.int32)

        # sanity check
        if len(out["annotation"]) != len(out["coords"]["x"]):
            raise RuntimeError("Cut annotation length mismatch with coords arrays.")

    return out


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"

    data = pickle.load(open(in_path, "rb"))
    G = data["graph"]

    # box micrometers
    xBox = [1500 / 1.625, 2500 / 1.625]
    yBox = [1500 / 1.625, 2500 / 1.625]
    zBox = [1500 / 2.5,   2500 / 2.5]

    # box paraview for ROI cut -> direct Hover points measurements
    #xBox = [1500, 2500]
    #yBox = [1500, 2500]
    #zBox = [1500,   2500]

    coords_img = np.asarray(G.vs["coords_image"], float)
    print("coords_image bounds:", coords_img.min(0), coords_img.max(0))

    x = np.asarray(data["coords"]["x"], float)
    y = np.asarray(data["coords"]["y"], float)
    z = np.asarray(data["coords"]["z"], float)
    print("edge-geometry (x,y,z) bounds:", [x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()])
    print("BOX (Paraview):", xBox, yBox, zBox)

    cut = cut_outgeom_gaia_like(data, xBox, yBox, zBox, tol=1e-3, min_straight_dist=1.0)

    with open(out_path, "wb") as f:
        pickle.dump(cut, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path, "Vertices:", cut["graph"].vcount(), "Edges:", cut["graph"].ecount())
    if "annotation" in cut:
        print("Cut annotation length:", len(cut["annotation"]))
