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
import os
import pickle
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

def print_dict_contents(name, d, sample_n=3):
    print(f"\n=== {name} ===")
    for k, v in d.items():
        try:
            print(f"{k}: type={type(v).__name__}, len={len(v)}")
            if len(v) > 0:
                print(f"  first {sample_n}: {v[:sample_n]}")
        except TypeError:
            print(f"{k}: type={type(v).__name__}, value={v}")

def outgeom_to_gaia_dicts(data, space="auto", diameter_unit="vox", verbose=True):
    """
    Convert outgeom pseudo-json into Gaia-like dictionaries:
      - vertices_dict
      - edges_dict
      - graph_dict

    Parameters
    ----------
    space : {"auto","um","vox"}
        Controls which geometry arrays are used for coords/lengths.
    diameter_unit : {"vox","um"}
        Controls units of stored diameters (edge + per-point + vertex).
        Independent from `space`.
    """
    G = data["graph"]
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
    # Decide diameter unit
    # -------------------------
    if diameter_unit not in ("vox", "um"):
        raise ValueError("diameter_unit must be 'vox' or 'um'")

    # -------------------------
    # Pick geometry source
    # -------------------------
    if space == "um":
        if "geom_R" not in data or "vertex_R" not in data:
            raise KeyError("Requested space='um' but geom_R/vertex_R not found in data.")
        geom_pts = data["geom_R"]
        Vsrc = data["vertex_R"]
        coords_attr = "coords_image_R"
        xk, yk, zk, l2k = "x_R", "y_R", "z_R", "lengths2_R"
        unit_str = "um"
    else:
        if "geom" not in data or "vertex" not in data:
            raise KeyError("Requested space='vox' but geom/vertex not found in data.")
        geom_pts = data["geom"]
        Vsrc = data["vertex"]
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

    # -------------------------
    # Per-point diameters
    # -------------------------
    if diameter_unit == "um":
        sx, sy, sz = map(float, data.get("spacing_um_per_voxel", (1.625, 1.625, 2.5)))
        scale_r = sx if abs(sx - sy) < 1e-9 else np.sqrt(sx * sy)
        d_p = 2.0 * (r_vox * scale_r)
    else:
        d_p = 2.0 * r_vox

    # -------------------------
    # Vertex attributes
    # -------------------------
    if coords_attr not in Vsrc:
        raise KeyError(f"Missing vertex['{coords_attr}'] for space='{space}'")

    coords = np.asarray(Vsrc[coords_attr], dtype=np.float64)
    if coords.shape != (nV, 3):
        raise ValueError(f"{coords_attr} shape {coords.shape} != ({nV}, 3)")

    ann = None
    if "vertex" in data and "vertex_annotation" in data["vertex"] and len(data["vertex"]["vertex_annotation"]) == nV:
        ann = data["vertex"]["vertex_annotation"]
    elif "vertex_annotation" in Vsrc and len(Vsrc["vertex_annotation"]) == nV:
        ann = Vsrc["vertex_annotation"]

    if "vertex" in data and "radii" in data["vertex"] and len(data["vertex"]["radii"]) == nV:
        vr_vox = np.asarray(data["vertex"]["radii"], dtype=np.float64)
        if diameter_unit == "um":
            sx, sy, sz = map(float, data.get("spacing_um_per_voxel", (1.625, 1.625, 2.5)))
            scale_r = sx if abs(sx - sy) < 1e-9 else np.sqrt(sx * sy)
            vdiam = (2.0 * vr_vox * scale_r).astype(float)
        else:
            vdiam = (2.0 * vr_vox).astype(float)
    else:
        vdiam = np.full(nV, np.nan, dtype=np.float64)

    vertices_dict = {
        "coords": [tuple(map(float, row)) for row in coords],
        "index": list(range(nV)),
        "annotation": list(ann) if ann is not None else [None] * nV,
        "diameter": vdiam.tolist(),
    }

    # -------------------------
    # Edge attributes
    # -------------------------
    nkind = list(G.es["nkind"]) if "nkind" in G.es.attributes() else [None] * nE
    connectivity = [tuple(map(int, e.tuple)) for e in G.es]

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Missing geom_start/geom_end in edges")

    geom_start = list(G.es["geom_start"])
    geom_end   = list(G.es["geom_end"])

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    for k in (xk, yk, zk, l2k):
        if k not in geom_pts:
            raise KeyError(f"Missing geom['{k}'] for space='{space}'")

    x = np.asarray(geom_pts[xk], dtype=np.float64)
    y = np.asarray(geom_pts[yk], dtype=np.float64)
    z = np.asarray(geom_pts[zk], dtype=np.float64)
    L2 = np.asarray(geom_pts[l2k], dtype=np.float64)

    if d_p.shape[0] != x.shape[0]:
        raise ValueError(f"TubeMap diameters length {d_p.shape[0]} != geometry length {x.shape[0]} (not aligned).")

    points = []
    lengths2_list = []
    diameters_list = []

    for eid in range(nE):
        s = int(gs[eid])
        t = int(ge[eid])

        if t <= s:
            points.append([])
            lengths2_list.append([])
            diameters_list.append([])
            continue

        pts = np.stack([x[s:t], y[s:t], z[s:t]], axis=1)
        points.append([tuple(map(float, row)) for row in pts])

        lengths2_list.append([float(v) for v in L2[s:t-1]] if (t - s) >= 2 else [])
        diameters_list.append([float(v) for v in d_p[s:t]])

    # scalar edge length
    e_len_key = _pick_first({k: True for k in G.es.attributes()},
                            ["length_R", "length_um"] if unit_str == "um" else ["length"])

    if e_len_key is not None:
        edge_length = [float(v) for v in np.asarray(G.es[e_len_key], dtype=np.float64)]
    else:
        edge_length = [float(np.sum(v)) if len(v) else float("nan") for v in lengths2_list]

    # scalar edge diameter
    edge_diameter = []
    for vlist in diameters_list:
        arr = np.asarray(vlist, dtype=np.float64)
        valid = arr[np.isfinite(arr)]
        edge_diameter.append(float(valid.mean()) if valid.size else float("nan"))

    edges_dict = {
        "connectivity": connectivity,
        "nkind": nkind,
        "diameter": edge_diameter,
        "diameters": diameters_list,
        "length": edge_length,
        "lengths2": lengths2_list,
        "points": points,
        "geom_start": geom_start,
        "geom_end": geom_end,
    }

    graph_dict = {
        "unit": unit_str,
        "diameter_unit": diameter_unit,
        "n_vertices": nV,
        "n_edges": nE,
    }

    if verbose:
        L = np.asarray(edge_length, dtype=np.float64)
        D = np.asarray(edge_diameter, dtype=np.float64)
        print(f"[Gaia dicts] space={space} unit={unit_str} diameter_unit={diameter_unit} V={nV:,} E={nE:,}")
        print(f"  length: min={float(np.nanmin(L)):.3f}  med={float(np.nanmedian(L)):.3f}  max={float(np.nanmax(L)):.3f}")
        print(f"  diam  : min={float(np.nanmin(D)):.3f}  med={float(np.nanmedian(D)):.3f}  max={float(np.nanmax(D)):.3f}")

    return vertices_dict, edges_dict, graph_dict


def save_gaia_dicts(data, out_dir, base_name="graph_18_OutGeom_um_formatted", space="auto", diameter_unit="vox", verbose=True):
    vertices_dict, edges_dict, graph_dict = outgeom_to_gaia_dicts(
        data=data,
        space=space,
        diameter_unit=diameter_unit,
        verbose=verbose,
    )

    os.makedirs(out_dir, exist_ok=True)

    v_path = os.path.join(out_dir, f"{base_name}_verticesDict.pkl")
    e_path = os.path.join(out_dir, f"{base_name}_edgesDict.pkl")
    g_path = os.path.join(out_dir, f"{base_name}_graphDict.pkl")

    with open(v_path, "wb") as f:
        pickle.dump(vertices_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(e_path, "wb") as f:
        pickle.dump(edges_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(g_path, "wb") as f:
        pickle.dump(graph_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:")
    print(" ", v_path)
    print(" ", e_path)
    print(" ", g_path)

    return vertices_dict, edges_dict, graph_dict, v_path, e_path, g_path


# =========================
# Example
# =========================
import os
import re
import pickle

# =========================
# Example
# =========================
name = "graph_18_OutGeom_Hcut2_um.pkl"

base_input_dir = "/home/admin/Ana/MicroBrain/output/um"
base_output_dir = "/home/admin/Ana/MicroBrain/output/dictOut"

in_path = os.path.join(base_input_dir, name)

# sacar Hcut1 del nombre
m = re.search(r"(Hcut\d+)", name)
subfolder = m.group(1) if m else "no_cut"

out_dir = os.path.join(base_output_dir, subfolder)

# nombre base de salida sin .pkl
base_name = os.path.splitext(name)[0] + "_formatted"

with open(in_path, "rb") as f:
    data = pickle.load(f)

vertices_dict, edges_dict, graph_dict, v_path, e_path, g_path = save_gaia_dicts(
    data,
    out_dir=out_dir,
    base_name=base_name,
    space="auto",
    diameter_unit="vox",
    verbose=True,
)

print_dict_contents("VERTICES DICT", vertices_dict)
print_dict_contents("EDGES DICT", edges_dict)
print_dict_contents("GRAPH DICT", graph_dict)