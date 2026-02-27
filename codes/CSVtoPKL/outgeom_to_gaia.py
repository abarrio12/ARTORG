'''
This code reformats the attributes into Gaia's structure so data is stored at edge/vertex level. 

Be aware of the space you are working with. Depending on space, different attributes are considered. 

This code exports in the unit you pass as parameter. THere is no conversion here. 

Author: Ana
Updated: 27 Feb 2026
'''
import numpy as np
import igraph as ig

def _has_um(data):
    return ("geom_R" in data) and ("vertex_R" in data)

def _pick_first(existing, candidates):
    """Return first key in candidates that exists in dict-like `existing`, else None."""
    for k in candidates:
        if k in existing:
            return k
    return None

def outgeom_to_igraph_materialized(data, space="auto", verbose=True):
    """
    Convert outgeom pseudo-json -> Gaia-like igraph where each edge stores:
      - connectivity, nkind, length, diameter scalars
      - points, lengths2 (per segment), diameters (per point)
    Handles both VOX and UM outgeom files with correct attribute names.
    """
    G = data["graph"]
    G2 = G.copy()
    nV = G.vcount()
    nE = G.ecount()

    # -------------------------
    # Decide space
    # -------------------------
    if space == "auto":
        space = "um" if _has_um(data) else "vox"
    if space not in ("um", "vox"):
        raise ValueError("space must be 'auto', 'um', or 'vox'")

    # -------------------------
    # Select sources by space
    # -------------------------
    if space == "um":
        if "geom_R" not in data or "vertex_R" not in data:
            raise KeyError("Requested space='um' but geom_R/vertex_R not found in data.")
        geom = data["geom_R"]
        Vsrc = data["vertex_R"]

        coords_attr = "coords_image_R"
        xk, yk, zk = "x_R", "y_R", "z_R"
        l2k = "lengths2_R"

        # Per-point atlas diameter in µm: prefer diam_atlas_geom_R, else radii_atlas_geom_R * 2
        dpts_key = _pick_first(geom, ["diam_atlas_geom_R", "diameters_atlas_geom_R"]) # supporting both names just in case (they are the same)
        rpts_key = _pick_first(geom, ["radii_atlas_geom_R"])

        # Edge scalar length in µm
        e_len_key = _pick_first({k: True for k in G.es.attributes()}, ["length_R", "length_um"])
        # Edge scalar atlas diameter in µm
        e_diam_key = _pick_first({k: True for k in G.es.attributes()}, ["diameter_atlas_R"])

        # Vertex atlas radii in µm
        v_r_key = "radii_atlas_R"

        unit_str = "um"

    else:  # vox
        geom = data["geom"]
        Vsrc = data["vertex"]

        coords_attr = "coords_image"
        xk, yk, zk = "x", "y", "z"
        l2k = "lengths2"  

        # Per-point atlas diameter in atlas vox units: prefer diam_atlas_geom, else radii_atlas_geom * 2
        dpts_key = _pick_first(geom, ["diam_atlas_geom"])
        rpts_key = _pick_first(geom, ["radii_atlas_geom"])

        # Edge scalar length in vox: your build stores arc length in G.es["length"]
        e_len_key = _pick_first({k: True for k in G.es.attributes()}, ["length"])
        e_len_steps_key = _pick_first({k: True for k in G.es.attributes()}, ["length_steps"])
        # Edge scalar atlas diameter in atlas vox units
        e_diam_key = _pick_first({k: True for k in G.es.attributes()}, ["diameter_atlas"])

        # Vertex atlas radii in atlas vox units
        v_r_key = "radii_atlas"

        unit_str = "vox"

    # -------------------------
    # Vertex attributes (Gaia)
    # -------------------------
    if coords_attr not in Vsrc:
        raise KeyError(f"Missing vertex['{coords_attr}'] for space='{space}'")

    coords = np.asarray(Vsrc[coords_attr], dtype=np.float64)
    if coords.shape != (nV, 3):
        raise ValueError(f"{coords_attr} shape {coords.shape} != ({nV}, 3)")

    G2.vs["coords"] = [tuple(map(float, row)) for row in coords]
    G2.vs["index"] = list(range(nV))

    # annotation: prefer data["vertex"]["vertex_annotation"] (it is not in G.vs in your outgeom)
    ann = None
    if "vertex" in data and "vertex_annotation" in data["vertex"] and len(data["vertex"]["vertex_annotation"]) == nV:
        ann = data["vertex"]["vertex_annotation"]
    elif "vertex_annotation" in Vsrc and len(Vsrc["vertex_annotation"]) == nV:
        ann = Vsrc["vertex_annotation"]

    G2.vs["annotation"] = list(ann) if ann is not None else [None] * nV

    # vertex diameter from atlas radii (whatever unit is in this space)
    if v_r_key in Vsrc and len(Vsrc[v_r_key]) == nV:
        vr = np.asarray(Vsrc[v_r_key], dtype=np.float64)
        G2.vs["diameter"] = (2.0 * vr).astype(float).tolist()
    else:
        G2.vs["diameter"] = [float("nan")] * nV

    # -------------------------
    # Edge attributes (Gaia)
    # -------------------------
    # nkind
    G2.es["nkind"] = list(G.es["nkind"]) if "nkind" in G.es.attributes() else [None] * nE

    # connectivity
    G2.es["connectivity"] = [tuple(map(int, e.tuple)) for e in G2.es]

    # geom indices
    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Missing geom_start/geom_end in edges")

    G2.es["geom_start"] = list(G.es["geom_start"])
    G2.es["geom_end"]   = list(G.es["geom_end"])

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    # polyline arrays
    for k in (xk, yk, zk, l2k):
        if k not in geom:
            raise KeyError(f"Missing geom['{k}'] for space='{space}'")

    x = np.asarray(geom[xk], dtype=np.float64)
    y = np.asarray(geom[yk], dtype=np.float64)
    z = np.asarray(geom[zk], dtype=np.float64)
    L2 = np.asarray(geom[l2k], dtype=np.float64)

    # per-point diameters (atlas)
    d_p = None
    r_p = None
    if dpts_key is not None:
        d_p = np.asarray(geom[dpts_key], dtype=np.float64)
    elif rpts_key is not None:
        r_p = np.asarray(geom[rpts_key], dtype=np.float64)

    # Per-edge lists for Gaia
    points, lengths2_list, diameters_list = [], [], []

    for eid in range(nE):
        s = int(gs[eid]); t = int(ge[eid])
        if t <= s:
            points.append([])
            lengths2_list.append([])
            diameters_list.append([])
            continue

        pts = np.stack([x[s:t], y[s:t], z[s:t]], axis=1)
        points.append([tuple(map(float, row)) for row in pts])

        # segment lengths: s .. t-2 (t-1 segments)
        if (t - s) >= 2:
            l2 = [float(v) for v in L2[s:t-1]]
        else:
            l2 = []
        lengths2_list.append(l2)

        # per-point diameters: one per point
        if d_p is not None:
            diameters_list.append([float(v) for v in d_p[s:t]])
        elif r_p is not None:
            diameters_list.append([float(v) for v in (2.0 * r_p[s:t])])
        else:
            diameters_list.append([float("nan")] * (t - s))

    G2.es["points"] = points
    G2.es["lengths2"] = lengths2_list
    G2.es["diameters"] = diameters_list
    G2["unit"] = unit_str

    # edge scalar length
    if e_len_key is not None:
        G2.es["length"] = [float(v) for v in np.asarray(G.es[e_len_key], dtype=np.float64)]
    else:
        G2.es["length"] = [float(np.sum(v)) if len(v) else float("nan") for v in lengths2_list]

    # optional length_steps in vox mode
    if space == "vox" and 'e_len_steps_key' in locals() and e_len_steps_key is not None:
        G2.es["length_steps"] = [float(v) for v in np.asarray(G.es[e_len_steps_key], dtype=np.float64)]

    # edge scalar diameter: prefer edge attribute; else mean(per-point diameters)
    if e_diam_key is not None:
        G2.es["diameter"] = [float(v) for v in np.asarray(G.es[e_diam_key], dtype=np.float64)]
    else:
        diam_edge = []
        for vlist in diameters_list:
            arr = np.asarray(vlist, dtype=np.float64)
            valid = arr[np.isfinite(arr)]
            diam_edge.append(float(valid.mean()) if valid.size else float("nan"))
        G2.es["diameter"] = diam_edge

    # -------------------------
    # Only keep Gaia attributes
    # -------------------------
    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {"connectivity", "nkind", "diameter", "diameters",
              "length", "lengths2", "points",
              "geom_start", "geom_end", "length_steps"}

    for a in list(G2.vs.attributes()):
        if a not in keep_v:
            del G2.vs[a]
    for a in list(G2.es.attributes()):
        if a not in keep_e:
            del G2.es[a]

    # -------------------------
    # Sanity print
    # -------------------------
    if verbose:
        L = np.asarray(G2.es["length"], dtype=np.float64)
        D = np.asarray(G2.es["diameter"], dtype=np.float64)
        print(f"[Gaia materialized] space={space} unit={G2['unit']} V={G2.vcount():,} E={G2.ecount():,}")
        print(f"  length: min={float(np.nanmin(L)):.3f}  med={float(np.nanmedian(L)):.3f}  max={float(np.nanmax(L)):.3f}")
        print(f"  diam  : min={float(np.nanmin(D)):.3f}  med={float(np.nanmedian(D)):.3f}  max={float(np.nanmax(D)):.3f}")

    return G2


import pickle

in_path  = "/home/ana/MicroBrain/output/um/graph_18_OutGeom_um.pkl"
out_path = "/home/ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted.pkl"

with open(in_path, "rb") as f:
    data = pickle.load(f)

G_gaia = outgeom_to_igraph_materialized(data, space="auto", verbose=True)
G_gaia.write_pickle(out_path)

print("Saved:", out_path)
print("Units:", G_gaia["unit"])
print("V/E:", G_gaia.vcount(), G_gaia.ecount())
print("V attrs:", G_gaia.vs.attributes())
print("E attrs:", G_gaia.es.attributes())
