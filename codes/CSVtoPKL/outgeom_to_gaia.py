import numpy as np
import igraph as ig


def outgeom_to_igraph_materialized(data, space="um"):
    G = data["graph"]
    G2 = G.copy()

    Vvox = data.get("vertex", {})
    Vum  = data.get("vertex_R", {})
    nV = G.vcount()
    nE = G.ecount()

    # -------------------------
    # select attr according to space
    # -------------------------
    if space == "um":
        geom = data["geom_R"]
        Vsrc = Vum
        coords_attr = "coords_image_R"
        v_r_key = "radii_atlas_R"          # vertex radii in µm

        xk, yk, zk = "x_R", "y_R", "z_R"
        l2k = "lengths2_R"

        # Prefer diameters per-point already computed in the UM converter
        d_p = np.asarray(geom["diameters_atlas_geom_R"], float) if "diameters_atlas_geom_R" in geom else None

        # Fallback (if diameters are not present): radii per-point (µm)
        g_r_key = "radii_atlas_geom_R"
        r_p = np.asarray(geom[g_r_key], float) if (d_p is None and g_r_key in geom) else None

        # edge scalars (copy)
        e_len_key  = "length_R" if "length_R" in G.es.attributes() else None

        # robust diameter scalar fallback chain
        e_diam_key = ("diameter_atlas_R" if "diameter_atlas_R" in G.es.attributes() else
                      "diameter_atlas"   if "diameter_atlas"   in G.es.attributes() else
                      "diameter"         if "diameter"         in G.es.attributes() else None)

    else:
        geom = data["geom"]
        Vsrc = Vvox
        coords_attr = "coords_image"
        v_r_key = "radii_atlas"            # vertex radii in vox

        xk, yk, zk = "x", "y", "z"
        l2k = "lengths2"

        d_p = None
        g_r_key = "radii_atlas_geom"
        r_p = np.asarray(geom[g_r_key], float) if g_r_key in geom else None

        e_len_key  = "length" if "length" in G.es.attributes() else None
        e_diam_key = ("diameter_atlas" if "diameter_atlas" in G.es.attributes() else
                      "diameter"       if "diameter"       in G.es.attributes() else None)

    # -------------------------
    # gaia vertex attrs
    # -------------------------
    if coords_attr not in Vsrc:
        raise KeyError(f"Missing {coords_attr} in vertex dict for space='{space}'")

    coords = np.asarray(Vsrc[coords_attr], float)
    G2.vs["coords"] = [tuple(map(float, row)) for row in coords]
    G2.vs["index"] = list(range(nV))

    # annotation
    if "vertex_annotation" in G.vs.attributes() and len(G.vs["vertex_annotation"]) == nV:
        G2.vs["annotation"] = list(G.vs["vertex_annotation"])
    elif "vertex_annotation" in Vsrc and len(Vsrc["vertex_annotation"]) == nV:
        G2.vs["annotation"] = list(Vsrc["vertex_annotation"])
    else:
        G2.vs["annotation"] = [None] * nV

    # diameter(v) = 2 * radii_atlas(_R)
    if v_r_key in Vsrc and len(Vsrc[v_r_key]) == nV:
        vr = np.asarray(Vsrc[v_r_key], float)
        G2.vs["diameter"] = (2.0 * vr).astype(float).tolist()
    else:
        G2.vs["diameter"] = [float("nan")] * nV

    # -------------------------
    # gaia edge attrs
    # -------------------------
    # nkind
    if "nkind" in G.es.attributes():
        G2.es["nkind"] = list(G.es["nkind"])
    else:
        G2.es["nkind"] = [None] * nE

    # connectivity
    G2.es["connectivity"] = [tuple(map(int, e.tuple)) for e in G2.es]

    # geom indices
    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Missing geom_start/geom_end in edges")
    G2.es["geom_start"] = list(G.es["geom_start"])
    G2.es["geom_end"]   = list(G.es["geom_end"])

    gs = np.asarray(G.es["geom_start"], np.int64)
    ge = np.asarray(G.es["geom_end"], np.int64)

    # polyline arrays
    x = np.asarray(geom[xk], float); y = np.asarray(geom[yk], float); z = np.asarray(geom[zk], float)
    L2 = np.asarray(geom[l2k], float)

    # per-edge lists
    points, lengths2, diameters = [], [], []

    for eid in range(nE):
        s = int(gs[eid]); t = int(ge[eid])

        pts = np.stack([x[s:t], y[s:t], z[s:t]], axis=1)
        points.append([tuple(map(float, row)) for row in pts])

        l2 = [float(v) for v in L2[s:t-1]]   # n_points-1
        lengths2.append(l2)

        # per-point diameters
        if d_p is not None:
            diameters.append([float(v) for v in d_p[s:t]])              # already diameter
        elif r_p is not None:
            diameters.append([float(v) for v in (2.0 * r_p[s:t])])      # from radius
        else:
            diameters.append([float("nan")] * (t - s))

    G2.es["points"] = points
    G2.es["lengths2"] = lengths2
    G2.es["diameters"] = diameters    # per-point diameters (Gaia)

    G2["unit"] = "um" if space == "um" else "vox"  # keep track of units

    # edge scalar length: copy from graph if exists, else sum(lengths2)
    if e_len_key is not None:
        G2.es["length"] = [float(v) for v in G.es[e_len_key]]
    else:
        G2.es["length"] = [float(np.sum(v)) if len(v) else float("nan") for v in lengths2]

    # edge scalar diameter: copy from graph if exists, else mean(per-point diameters) WITHOUT warnings
    if e_diam_key is not None:
        G2.es["diameter"] = [float(v) for v in G.es[e_diam_key]]
    else:
        diam_edge = []
        for v in diameters:
            arr = np.asarray(v, float)
            valid = arr[~np.isnan(arr)]
            diam_edge.append(float(valid.mean()) if valid.size else float("nan"))
        G2.es["diameter"] = diam_edge

    # -------------------------
    # only keep Gaia attributes (remove others)
    # -------------------------
    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {"connectivity", "nkind", "diameter", "diameters",
              "length", "lengths2", "points",
              "geom_start", "geom_end"}

    for a in list(G2.vs.attributes()):
        if a not in keep_v:
            del G2.vs[a]
    for a in list(G2.es.attributes()):
        if a not in keep_e:
            del G2.es[a]

    return G2

import pickle

if __name__ == "__main__":
    in_path  = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3_um.pkl"   
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3_um_gaia.pkl"

    # load graph (cut)
    with open(in_path, "rb") as f:
        data = pickle.load(f)

    #materialize to Gaia format (change space if wanted)
    G_gaia = outgeom_to_igraph_materialized(data, space="um")

    # save igraph pkl
    G_gaia.write_pickle(out_path)

    #minimal print to check
    print("Saved:", out_path)
    print("Units:", G_gaia["unit"])
    print("V/E:", G_gaia.vcount(), G_gaia.ecount())
    print("V attrs:", G_gaia.vs.attributes())
    print("E attrs:", G_gaia.es.attributes())