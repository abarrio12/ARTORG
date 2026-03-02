'''
This code reformats the attributes into Gaia’s structure so data is stored at edge/vertex level.

Be aware of the space you are working with. Depending on the requested output space, different
geometry attributes are used (VOX: geom; UM: geom_R).

Diameter source:
- We use TubeMap radii from image space (data["geom"]["radii"] / data["vertex"]["radii"]).
- We do NOT use atlas-derived radii/diameters (radii_atlas*, diameter_atlas*).

Unit handling (UPDATED):
- Space controls COORDINATES + LENGTHS:
    * space="vox" -> coords/lengths from geom + vertex (vox)
    * space="um"  -> coords/lengths from geom_R + vertex_R (µm)

- diameter_unit controls DIAMETERS ONLY (independent from space):
    * diameter_unit="vox": store diameters in voxel units (diam_vox = 2 * radii_vox)
    * diameter_unit="um" : store diameters in µm using voxel spacing (diam_um ≈ 2 * radii_vox * sx)

Note:
- We assume TubeMap radii are defined in the image XY plane. When sx == sy (like here), using sx is
  consistent. If sx != sy, we use sqrt(sx*sy). This is an approximation.

Author: Ana
Updated: 27 Feb 2026 (revised for separate diameter_unit)
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

# BE AWARE OF DIAMETER UNIT (now is autom to vox due to wanting the analysis in vox)

def outgeom_to_igraph_materialized(data, space="auto", diameter_unit="vox", verbose=True):
    """
    Convert outgeom pseudo-json -> Gaia-like igraph where each edge stores:
      - connectivity, nkind, length, diameter scalars
      - points, lengths2 (per segment), diameters (per point)

    Handles both VOX and UM outgeom files with correct attribute names.

    Parameters
    ----------
    space : {"auto","um","vox"}
        Controls which geometry arrays are used for coords/lengths.
    diameter_unit : {"vox","um"}
        Controls units of stored diameters (edge + per-point + vertex).
        Independent from `space`.
    """
    G = data["graph"]
    G2 = G.copy()
    nV = G.vcount()
    nE = G.ecount()

    # -------------------------
    # Decide space (coords/lengths)
    # -------------------------
    if space == "auto":
        space = "um" if _has_um(data) else "vox"
    if space not in ("um", "vox"):
        raise ValueError("space must be 'auto', 'um', or 'vox'")

    # -------------------------
    # Decide diameter unit (diameters only)
    # -------------------------
    if diameter_unit not in ("vox", "um"):
        raise ValueError("diameter_unit must be 'vox' or 'um'")

    # -------------------------
    # Pick geometry source
    # -------------------------
    if space == "um":
        if "geom_R" not in data or "vertex_R" not in data:
            raise KeyError("Requested space='um' but geom_R/vertex_R not found in data.")
        geom_pts = data["geom_R"]         # x_R,y_R,z_R,lengths2_R (µm)
        Vsrc     = data["vertex_R"]       # coords_image_R (µm)
        coords_attr = "coords_image_R"
        xk, yk, zk, l2k = "x_R", "y_R", "z_R", "lengths2_R"
        unit_str = "um"
    else:
        if "geom" not in data or "vertex" not in data:
            raise KeyError("Requested space='vox' but geom/vertex not found in data.")
        geom_pts = data["geom"]           # x,y,z,lengths2 (vox)
        Vsrc     = data["vertex"]         # coords_image (vox)
        coords_attr = "coords_image"
        xk, yk, zk, l2k = "x", "y", "z", "lengths2"
        unit_str = "vox"

    # -------------------------
    # TubeMap radii source (always from VOX geom)
    # -------------------------
    if "geom" not in data:
        raise KeyError("data missing 'geom' (required for TubeMap radii)")
    geom_vox = data["geom"]
    rpts_key_vox = _pick_first(geom_vox, ["radii", "radii_geom"])
    if rpts_key_vox is None:
        raise KeyError("No TubeMap radii found in data['geom'] (expected 'radii' or 'radii_geom').")

    r_vox = np.asarray(geom_vox[rpts_key_vox], dtype=np.float64)

    # ---------------------------------------------------------
    # Convert per-point diameters using diameter_unit (independent of space)
    # ---------------------------------------------------------
    if diameter_unit == "um":
        sx, sy, sz = map(float, data.get("spacing_um_per_voxel", (1.625, 1.625, 2.5)))
        scale_r = sx if abs(sx - sy) < 1e-9 else np.sqrt(sx * sy)
        d_p = 2.0 * (r_vox * scale_r)     # diameters in µm
    else:
        d_p = 2.0 * r_vox                 # diameters in vox

    # -------------------------
    # Vertex attributes (coords in chosen space)
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

    # vertex diameter from TubeMap vertex radii (stored in voxel vertex dict) -> controlled by diameter_unit
    if "vertex" in data and "radii" in data["vertex"] and len(data["vertex"]["radii"]) == nV:
        vr_vox = np.asarray(data["vertex"]["radii"], dtype=np.float64)
        if diameter_unit == "um":
            sx, sy, sz = map(float, data.get("spacing_um_per_voxel", (1.625, 1.625, 2.5)))
            scale_r = sx if abs(sx - sy) < 1e-9 else np.sqrt(sx * sy)
            G2.vs["diameter"] = (2.0 * vr_vox * scale_r).astype(float).tolist()
        else:
            G2.vs["diameter"] = (2.0 * vr_vox).astype(float).tolist()
    else:
        G2.vs["diameter"] = [float("nan")] * nV

    # -------------------------
    # Edge attributes
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
        if k not in geom_pts:
            raise KeyError(f"Missing geom['{k}'] for space='{space}'")

    x = np.asarray(geom_pts[xk], dtype=np.float64)
    y = np.asarray(geom_pts[yk], dtype=np.float64)
    z = np.asarray(geom_pts[zk], dtype=np.float64)
    L2 = np.asarray(geom_pts[l2k], dtype=np.float64)

    # alignment check (tube diameters must index same concatenated point array)
    if d_p.shape[0] != x.shape[0]:
        raise ValueError(f"TubeMap diameters length {d_p.shape[0]} != geometry length {x.shape[0]} (not aligned).")

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

        # segment lengths: one per segment => (t-s-1)
        lengths2_list.append([float(v) for v in L2[s:t-1]] if (t - s) >= 2 else [])

        # per-point diameters: one per point => (t-s)
        diameters_list.append([float(v) for v in d_p[s:t]])

    G2.es["points"] = points
    G2.es["lengths2"] = lengths2_list
    G2.es["diameters"] = diameters_list

    # store metadata
    G2["unit"] = unit_str                 # coords/length unit
    G2["diameter_unit"] = diameter_unit   # diameter unit

    # scalar edge length
    e_len_key = _pick_first({k: True for k in G.es.attributes()},
                            ["length_R", "length_um"] if unit_str == "um" else ["length"])

    if e_len_key is not None:
        G2.es["length"] = [float(v) for v in np.asarray(G.es[e_len_key], dtype=np.float64)]
    else:
        G2.es["length"] = [float(np.sum(v)) if len(v) else float("nan") for v in lengths2_list]

    # scalar edge diameter from per-point diameters
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
        print(f"[Gaia materialized] space={space} unit={G2['unit']} diameter_unit={G2['diameter_unit']} V={G2.vcount():,} E={G2.ecount():,}")
        print(f"  length: min={float(np.nanmin(L)):.3f}  med={float(np.nanmedian(L)):.3f}  max={float(np.nanmax(L)):.3f}")
        print(f"  diam  : min={float(np.nanmin(D)):.3f}  med={float(np.nanmedian(D)):.3f}  max={float(np.nanmax(D)):.3f}")

    return G2


import pickle

in_path  = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um_formatted.pkl"

with open(in_path, "rb") as f:
    data = pickle.load(f)

# coords/lengths in UM (auto detects geom_R), but diameters stored in VOX:
G_gaia = outgeom_to_igraph_materialized(data, space="auto", diameter_unit="vox", verbose=True)
G_gaia.write_pickle(out_path)

print("Saved:", out_path)
print("Units (coords/length):", G_gaia["unit"])
print("Units (diameter):", G_gaia["diameter_unit"])
print("V/E:", G_gaia.vcount(), G_gaia.ecount())
print("V attrs:", G_gaia.vs.attributes())
print("E attrs:", G_gaia.es.attributes())