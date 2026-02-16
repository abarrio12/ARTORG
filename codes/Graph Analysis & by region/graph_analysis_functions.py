"""
graph_analysis_functions.py
Faces-only BC + Gaia-style vessel density + radii sanity + redundancy via edge-disjoint paths

✅ Classic graph analysis (diameter/length/degrees/duplicates/loops)
✅ BC detection + analysis PER FACE only (NO per-box totals)
✅ Cube-net plot + 3D cube views for BC points
✅ Spatial plots: nkind map + HDN map + diameter-by-nkind plots
✅ Gaia-style vessel density using MICRO-SEGMENTS between polyline points (OutGeom-like data dict)
✅ Sanity checks for polyline radii consistency
✅ Redundancy: max number of EDGE-DISJOINT arteriole→venule paths inside the box (maxflow)
✅ Degree distribution checks + spatial maps for any degree band
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import igraph as ig


# ====================================================================================================================
#                                                   GLOBAL CONSTANTS
# ====================================================================================================================

EDGE_NKIND_TO_LABEL = {
    2: "arteriole",
    3: "venule",
    4: "capillary"
}

FACES = ("x_min","x_max","y_min","y_max","z_min","z_max") 

DEFAULT_DEPTH_BINS_UM = [
    ("superficial",   0.0,  20.0),
    ("superficial_2", 20.0, 50.0),
    ("middle",        50.0, 200.0),
    ("deep",          200.0, np.inf),
]

VESSEL_COLORS = {
    "arteriole": "#ff2828ff",   
    "venule": "#0072c4ff",      
    "capillary": "#7f7f7fff",  
}

# Voxels of image to micrometers resolution
res_um_per_vox = np.array([1.625, 1.625, 2.5])


# dict with faces of box, axis to use and corresponding value 
FACES_DEF = {
    "x_min": (0, "xmin"),
    "x_max": (0, "xmax"),
    "y_min": (1, "ymin"),
    "y_max": (1, "ymax"),
    "z_min": (2, "zmin"),
    "z_max": (2, "zmax"),
}

# ====================================================================================================================
#                                                    LOAD / SAVE
# ====================================================================================================================

def load_graph(path):
    """Load an igraph Graph object from a pickle file."""
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph


def dump_graph(graph, out_path):
    """Save an igraph Graph object into a pickle file."""
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"\nGraph successfully saved to: {out_path}")


def activate_space_inplace(data, space="vox"):
    """
    Makes analysis code unit-agnostic.
    After activation, functions always use:

        data["geom"]
        data["vertex"]
        G.es["length"]
        G.es["tortuosity"]

    regardless of original storage.
    """

    G = data["graph"]

    if space == "vox":
        # Nothing to change (original already in place)
        return data

    elif space == "um":

        if "geom_R" not in data or "vertex_R" not in data:
            raise ValueError("Micrometer data not found (geom_R / vertex_R missing).")

        # ---- Replace geom ----
        gR = data["geom_R"]
        g_new = {}

        # Strip _R suffix automatically
        for k, v in gR.items():
            if k.endswith("_R"):
                g_new[k[:-2]] = v
            else:
                g_new[k] = v

        data["geom"] = g_new

        # ---- Replace vertex ----
        vR = data["vertex_R"]
        v_new = {}

        for k, v in vR.items():
            if k.endswith("_R"):
                v_new[k[:-2]] = v
            else:
                v_new[k] = v

        data["vertex"] = v_new

        # ---- Replace edge attributes ----
        if "length_R" in G.es.attributes():
            G.es["length"] = G.es["length_R"]

        if "tortuosity_R" in G.es.attributes():
            G.es["tortuosity"] = G.es["tortuosity_R"]

        return data

    else:
        raise ValueError("space must be 'vox' or 'um'")


def sync_vertex_attributes(data):
    """
    Copy all arrays from data["vertex"] = pseudo json structure of pkl file 
    into data["graph"].vs (keep compliance and escalability)
    Must be aligned with current graph.
    """
    G = data["graph"]
    for key, arr in data["vertex"].items():
        G.vs[key] = arr



def make_box(
    center_vox,
    box_um,
    res_um_per_vox = res_um_per_vox,
    as_float=True,
    sanity_check=True
):
    """
    Create a box dict (xmin,xmax,...) in VOXELS from:
      - center in VOXELS
      - box size in µm
      - image resolution in µm/voxel

    This is the STANDARD way to build boxes when coords_attr = 'coords_image'.

    Parameters
    ----------
    center_vox : array-like, shape (3,)
        Box center in VOXELS (e.g. from ParaView).
    box_um : array-like, shape (3,)
        Physical box size in µm (e.g. [400,400,400]).
    res_um_per_vox : array-like, shape (3,)
        Image resolution in µm / voxel (e.g. [1.625,1.625,2.5]).
    as_float : bool, default True
        Force output box values to float.
    sanity_check : bool, default True
        Print box size in µm to verify correctness.

    Returns
    -------
    box : dict
        {
          'xmin','xmax',
          'ymin','ymax',
          'zmin','zmax'
        }   (ALL IN VOXELS)
    """

    center_vox = np.asarray(center_vox, dtype=float)
    box_um = np.asarray(box_um, dtype=float)
    res_um_per_vox = np.asarray(res_um_per_vox, dtype=float)

    if center_vox.shape != (3,):
        raise ValueError("center_vox must be length-3 (x,y,z) in voxels")
    if box_um.shape != (3,):
        raise ValueError("box_um must be length-3 (µm)")
    if res_um_per_vox.shape != (3,):
        raise ValueError("res_um_per_vox must be length-3 (µm/voxel)")

    # Convert physical size → voxels
    box_vox = box_um / res_um_per_vox

    box = {
        "xmin": center_vox[0] - box_vox[0] / 2,
        "xmax": center_vox[0] + box_vox[0] / 2,
        "ymin": center_vox[1] - box_vox[1] / 2,
        "ymax": center_vox[1] + box_vox[1] / 2,
        "zmin": center_vox[2] - box_vox[2] / 2,
        "zmax": center_vox[2] + box_vox[2] / 2,
    }

    if as_float:
        box = {k: float(v) for k, v in box.items()}

    if sanity_check:
        sx = (box["xmax"] - box["xmin"]) * res_um_per_vox[0]
        sy = (box["ymax"] - box["ymin"]) * res_um_per_vox[1]
        sz = (box["zmax"] - box["zmin"]) * res_um_per_vox[2]
        print(f"[make_box] size check (µm): "
              f"x={sx:.2f}, y={sy:.2f}, z={sz:.2f}")

    return box








# ====================================================================================================================
#                                                   BASIC HELPERS 
# ====================================================================================================================

def get_coords(graph, coords_attr):
    """Return Nx3 float array from graph.vs[coords_attr]."""
    check_attr(graph, coords_attr, "vs")
    P = np.asarray(graph.vs[coords_attr], dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got {P.shape}.")
    return P


def validate_box_faces(box):
    """Raise if box misses required keys."""
    req = ("xmin","xmax","ymin","ymax","zmin","zmax")
    missing = [k for k in req if k not in box]
    if missing:
        raise ValueError(f"box missing keys: {missing}. Required: {list(req)}")



def check_attr(graph, names, where="edge"):
    """
    Check that required attribute(s) exist in the graph.

    names : str or list of str
    where : 'edge'/'es' or 'vertex'/'vs'
    """
    if isinstance(names, str):
        names = [names]

    where = where.lower()

    if where in ("edge", "es"):
        existing = set(graph.es.attributes())
        missing = [name for name in names if name not in existing]
        if missing:
            raise ValueError(f"Missing edge attribute(s): {missing}",
                             f"Available: {graph.es.attributes()}")
        return

    if where in ("vertex", "vs"):
        existing = set(graph.vs.attributes())
        missing = [name for name in names if name not in existing]
        if missing:
            raise ValueError(f"Missing vertex attribute(s): {missing}",
                             f"Available: {graph.vs.attributes()}")
        return

    raise ValueError("where must be one of: 'edge'/'es' or 'vertex'/'vs'")



def duplicated_edge_stats(G):
    edges = G.get_edgelist()
    pairs = [tuple(sorted(e)) for e in edges]
    c = Counter(pairs)

    n_pairs_duplicated = sum(v > 1 for v in c.values())
    n_extra_edges = sum(v - 1 for v in c.values() if v > 1)

    return {
        "n_pairs_duplicated": int(n_pairs_duplicated),
        "n_extra_edges": int(n_extra_edges),
        "perc_extra_edges": float(100 * n_extra_edges / G.ecount()) if G.ecount() else 0.0
    }



def loop_edge_stats(G):
    """
    Return statistics about self-loops (edges where source == target).
    """

    loop_idx = [e.index for e in G.es if e.source == e.target]
    n_loops = len(loop_idx)
    n_edges = G.ecount()

    return {
        "n_loops": int(n_loops),
        "perc_loops": float(100 * n_loops / n_edges) if n_edges else 0.0,
        "loop_indices": loop_idx
    }



def get_geom_from_data(data, want_radii=False):
    G = data["graph"]
    g = data["geom"]
    x = np.asarray(g["x"], float)
    y = np.asarray(g["y"], float)
    z = np.asarray(g["z"], float)
    L2 = np.asarray(g["lengths2"], float)
    r  = np.asarray(g["radii"], float) if (want_radii and "radii" in g) else None
    return G, x, y, z, L2, r

def get_range_from_data(data, eid):
    G = data["graph"]
    return int(G.es[eid]["geom_start"]), int(G.es[eid]["geom_end"])

def get_edge_attr_from_data(data, eid, name):
    return data["graph"].es[eid][name]














# ====================================================================================================================
#                                               CLASSIC GRAPH ANALYSIS (biological metrics)
# ====================================================================================================================

def single_connected_component(graph):
    """
    Ensure graph is single connected component. If not, keep the largest one.

    Returns:
        is_single (bool)
        n_components (int)
        graph (igraph.Graph)
    """
    comps = graph.components()
    n_components = len(comps)
    is_single = (n_components == 1)

    if is_single:
        print("The graph is a single connected component.")
    else:
        print(f"The graph has {n_components} components. Keeping the largest one.")
        graph = comps.giant()

    return is_single, n_components, graph



def get_edges_types(graph, label_dict=EDGE_NKIND_TO_LABEL, return_dict=True):
    """
    Count how many edges belong to each nkind type.
    Returns counts and proportions.
    """
    check_attr(graph, "nkind", "es")

    edge_types = np.asarray(graph.es["nkind"], dtype=int)
    unique, counts = np.unique(edge_types, return_counts=True)

    total = len(edge_types)

    results = {}

    print("\nEdge types:\n")

    for k, n in zip(unique, counts):
        name = label_dict.get(k, str(k)) if label_dict else str(k)
        perc = 100 * n / total

        print(f" - {name} (nkind={k}): {n} edges ({perc:.1f}%)")

        results[k] = {
            "name": name,
            "count": int(n),
            "percentage": float(perc)
        }

    if return_dict:
        return results

    return unique, counts



def get_avg_length_nkind(graph):
    """Compute mean edge length per nkind."""
    check_attr(graph, ["length", "nkind"], "es")
        
    length_att = np.array(graph.es["length"], dtype=float)
    nkind = np.array(graph.es["nkind"], dtype=int)

    unique = np.unique(nkind)
    l = []
    print("\nAverage length by nkind:\n")
    for k in unique:
        mean_l = float(np.mean(length_att[nkind == k]))
        l.append(mean_l)
        print(f"nkind = {k}: average length (att) = {mean_l:.6f} µm")
    return unique, l



def get_avg_diameter_nkind(graph):                                              
    """
    Compute mean edge diameter (non tortuous) per nkind.
    Reminder: In 18_graph.pkl the edge radii from non tortuous is computed as max(tortuous points/vertices radii)
    Therefore, this diameter is 2*max(points radii)
    """
    check_attr(graph, ["diameter", "nkind"], "es")

    diam = np.asarray(graph.es["diameter"], dtype=float)
    nkind = np.asarray(graph.es["nkind"], dtype=int)
    unique = np.unique(nkind)

    mean_diameters = np.array([diam[nkind == k].mean() for k in unique])

    print("\nAverage diameter by nkind:\n")
    for k, m in zip(unique, mean_diameters):
        print(f"nkind = {k}: average diameter = {m:.6f} µm")
    return unique, mean_diameters




def diameter_stats_nkind(
    graph,
    label_dict=None,
    ranges=None,
    plot=True,
    verbose=False,
):
    """
    Compute diameter statistics grouped by nkind.

    Returns
    -------
    stats_dict : dict
        { nkind: {
            "name": str,
            "n": int,
            "mean": float,
            "median": float,
            "p5": float,
            "p95": float,
            "perc_in_range": float|None,
            "range": (low, high)|None
        } }
    """
    check_attr(graph, ["diameter", "nkind"], "es")

    diam = np.asarray(graph.es["diameter"], dtype=float)
    nkind = np.asarray(graph.es["nkind"], dtype=int)

    stats_dict = {}
    for k in np.unique(nkind):
        subset = diam[nkind == k]
        if subset.size == 0:
            continue

        name = label_dict.get(int(k), str(k)) if label_dict else str(k)

        mean = float(np.mean(subset))
        median = float(np.median(subset))
        p5 = float(np.percentile(subset, 5))
        p95 = float(np.percentile(subset, 95))

        perc_in_range = None
        rng = None
        if ranges is not None and k in ranges:
            low, high = ranges[k]
            rng = (float(low), float(high))
            perc_in_range = float(np.mean((subset >= low) & (subset <= high)) * 100.0)

        stats_dict[int(k)] = {
            "name": name,
            "n": int(subset.size),
            "mean": mean,
            "median": median,
            "p5": p5,
            "p95": p95,
            "perc_in_range": perc_in_range,
            "range": rng,
        }

    # Console output
    if verbose:
        print("\n=== Diameter statistics (by nkind) ===\n")
        for k in sorted(stats_dict.keys()):
            s = stats_dict[k]
            print(f"{s['name']} (nkind={k}, n={s['n']}):")
            print(f"  mean:   {s['mean']:.2f} µm")
            print(f"  median: {s['median']:.2f} µm")
            print(f"  p5–p95: {s['p5']:.2f} – {s['p95']:.2f} µm")
            if s["perc_in_range"] is not None:
                lo, hi = s["range"]
                print(f"  % in range ({lo}–{hi}): {s['perc_in_range']:.1f}%")
            print()

    # General plot
    if plot:
        plot_violin_box_by_category(
            diam,
            nkind,
            label_dict=EDGE_NKIND_TO_LABEL,
            xlabel="Vessel type",
            ylabel="Diameter (µm)",
            title="Diameter distribution by vessel type"
        )

    return stats_dict






def get_degrees(graph, threshold=4):
    """
     Compute node degrees and high-degree-node indices (deg >= threshold).

    Parameters
    ----------
    G : igraph.Graph
    threshold : int

    Returns
    -------
    unique_degrees : np.ndarray
    high_degree_idx : np.ndarray
        Indices (in current graph) of high-degree nodes.
    """

    deg = np.asarray(graph.degree(), dtype=int)
    graph.vs["degree"] = deg
    mask = deg >= int(threshold)
    graph.vs["high_degree_node"] = np.where(mask, deg, 0)
    hdn_idx = np.where(mask)[0]

    print("Unique degrees:", np.unique(deg))
    print(f"HDN (>= {threshold}): {hdn_idx.size}")

    return np.unique(deg), hdn_idx




def distance_to_surface_stats(
    graph,
    nodes,
    depth_attr="distance_to_surface",
    depth_bins=DEFAULT_DEPTH_BINS_UM,
):
    """
    Compute summary stats and named depth-bin distribution for graph.vs[depth_attr]
    over the provided nodes.


    Note: distance_to_surface is stored in VOXELS and was computed assuming isotropic voxels.
    Since our image resolution is anisotropic (1.625, 1.625, 2.5 µm),
    we approximate the physical depth by multiplying with the XY resolution (1.625 µm/voxel).
    This keeps consistency across regions, although it is an approximation.
    """

    check_attr(graph, depth_attr, "vs")

    nodes = np.asarray(nodes, dtype=int)
    if nodes.size == 0:
        return None

    d = np.asarray(graph.vs[depth_attr], dtype=float)
    vals = d[nodes]

    out = {
        "n": int(vals.size),
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "bins": {},
    }

    n = vals.size
    for label, low, high in depth_bins:
        mask = (vals >= low) & (vals < high)
        c = int(np.sum(mask))
        out["bins"][label] = {
            "range_um": [float(low), float(high)],
            "count": c,
            "proportion": float(c / n),
        }

    return out



def analyze_hdn_pattern_in_box(
    graph,
    degree_thr=4,
    box=None,                          # dict xmin/xmax/ymin/ymax/zmin/zmax (vox) optional
    coords_attr="coords_image",
    depth_attr="distance_to_surface",
    depth_bins=(("0–20", 0, 20), ("20–50", 20, 50), ("50–200", 50, 200), (">200", 200, np.inf)),
    vessel_type_map=EDGE_NKIND_TO_LABEL,              # nkind->label
    verbose=True,
):
    """
    Summarize HDN pattern: abundance, type composition, depth, wall bias, spatial concentration.
    Returns a dict (and optionally prints a readable summary).
    """
    # --- HDN indices ---
    deg = np.asarray(graph.degree(), dtype=int)
    hdn = np.where(deg >= int(degree_thr))[0]

    out = {
        "degree_thr": int(degree_thr),
        "n_nodes": int(graph.vcount()),
        "n_hdn": int(hdn.size),
        "hdn_fraction": float(hdn.size / graph.vcount()) if graph.vcount() else 0.0,
    }

    if hdn.size == 0:
        if verbose:
            print(f"[HDN] No HDN found for deg >= {degree_thr}.")
        return out

    # --- Coords for spatial pattern ---
    if coords_attr in graph.vs.attributes():
        P = np.asarray(graph.vs[coords_attr], float)
        Ph = P[hdn]

        # Concentration: compare average distance to centroid (HDN vs all)
        c_all = P.mean(axis=0)
        c_hdn = Ph.mean(axis=0)
        d_all = np.linalg.norm(P - c_all, axis=1)
        d_hdn = np.linalg.norm(Ph - c_hdn, axis=1)

        out["spatial"] = {
            "centroid_all": c_all.tolist(),
            "centroid_hdn": c_hdn.tolist(),
            "centroid_shift": (c_hdn - c_all).tolist(),
            "mean_dist_to_centroid_all": float(d_all.mean()),
            "mean_dist_to_centroid_hdn": float(d_hdn.mean()),
            "concentration_ratio_hdn_over_all": float(d_hdn.mean() / d_all.mean()) if d_all.mean() else None,
        }
        # Interpretation: ratio < 1 => HDN more concentrated than all nodes
    else:
        out["spatial"] = None

    # --- Type composition (by incident edges' nkind) ---
    type_comp = None
    if vessel_type_map is not None and ("nkind" in graph.es.attributes()):
        labels = np.array([infer_node_type_from_incident_edges(graph, v, vessel_type_map) for v in hdn], dtype=object)
        uniq, cnt = np.unique(labels, return_counts=True)
        type_comp = {str(u): {"count": int(c), "proportion": float(c / labels.size)} for u, c in zip(uniq, cnt)}
    out["hdn_type_composition"] = type_comp

    # --- Depth (distance_to_surface) summary + bins ---
    depth_out = None
    if depth_attr in graph.vs.attributes():
        d = np.asarray(graph.vs[depth_attr], float)
        vals = d[hdn] * res_um_per_vox[0]   # distance to surface in voxels by default. Passing to micrometers.
        depth_out = {
            "n": int(vals.size),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
            "bins": {},
        }
        n = vals.size
        for label, low, high in depth_bins:
            mask = (vals >= low) & (vals < high)
            c = int(mask.sum())
            depth_out["bins"][label] = {"range_um": [float(low), float(high)], "count": c, "proportion": float(c / n)}
    out["depth_hdn"] = depth_out

    # --- Wall/face bias (requires box + coords) ---
    face_bias = None
    if box is not None and coords_attr in graph.vs.attributes():
        xmin, xmax = float(box["xmin"]), float(box["xmax"])
        ymin, ymax = float(box["ymin"]), float(box["ymax"])
        zmin, zmax = float(box["zmin"]), float(box["zmax"])
        # eps: 2 vox por defecto (ajusta según resolución)
        eps = 2.0

        face_bias = {
            "x_min": float(np.mean(np.abs(P[hdn, 0] - xmin) <= eps)),
            "x_max": float(np.mean(np.abs(P[hdn, 0] - xmax) <= eps)),
            "y_min": float(np.mean(np.abs(P[hdn, 1] - ymin) <= eps)),
            "y_max": float(np.mean(np.abs(P[hdn, 1] - ymax) <= eps)),
            "z_min": float(np.mean(np.abs(P[hdn, 2] - zmin) <= eps)),
            "z_max": float(np.mean(np.abs(P[hdn, 2] - zmax) <= eps)),
        }
        face_bias["max_face_bias"] = float(max(face_bias.values()))
    out["face_bias_hdn"] = face_bias

    # --- Pretty print (optional) ---
    if verbose:
        print(f"\n=== HDN fingerprint (deg ≥ {degree_thr}) ===")
        print(f"HDN: {out['n_hdn']} / {out['n_nodes']}  ({100*out['hdn_fraction']:.2f}%)")

        if out["hdn_type_composition"] is not None:
            print("\nHDN node-type composition (inferred from incident edge nkind):")
            for k, v in sorted(out["hdn_type_composition"].items(), key=lambda kv: -kv[1]["proportion"]):
                print(f"  {k:10s}: {v['count']:4d}  ({100*v['proportion']:.1f}%)")

        if out["depth_hdn"] is not None:
            dh = out["depth_hdn"]
            print(f"\nHDN depth ({depth_attr}) µm: median={dh['median']:.2f}, mean={dh['mean']:.2f}, min={dh['min']:.2f}, max={dh['max']:.2f}")
            print("Depth bins:")
            for lab, b in dh["bins"].items():
                print(f"  {lab:>7s}: {b['count']:4d} ({100*b['proportion']:.1f}%)")

        if out["face_bias_hdn"] is not None:
            fb = out["face_bias_hdn"]
            print("\nFace bias (fraction of HDN within ~2 vox of each face):")
            for face in ["x_min","x_max","y_min","y_max","z_min","z_max"]:
                print(f"  {face}: {fb[face]:.3f}")
            print(f"  max_face_bias: {fb['max_face_bias']:.3f}")

        if out["spatial"] is not None:
            sp = out["spatial"]
            r = sp["concentration_ratio_hdn_over_all"]
            if r is not None:
                print(f"\nSpatial concentration ratio (HDN/all) = {r:.3f}  (<1 => HDN more concentrated)")

    return out


                                                               
















# ====================================================================================================================
# BC DETECTION (faces-only)
# ====================================================================================================================

def bc_nodes_on_face(graph, axis, value, box, coords_attr, eps=2.0):   
    """                                                                
    Returns node indices that are:
      (1) within eps of the face plane (axis=value)
      (2) strictly inside the face rectangle on the other 2 axes
    """
    C = get_coords(graph, coords_attr).astype(float)

    bounds = np.array([
        [box["xmin"], box["xmax"]],
        [box["ymin"], box["ymax"]],
        [box["zmin"], box["zmax"]],
    ], dtype=float)

    # plane condition (with eps)
    on_plane = np.abs(C[:, axis] - float(value)) <= float(eps)

    # rest of axis (no eps)
    other_axes = [i for i in range(3) if i != axis]
    inside = (
        (C[:, other_axes[0]] >= bounds[other_axes[0], 0]) &
        (C[:, other_axes[0]] <= bounds[other_axes[0], 1]) &
        (C[:, other_axes[1]] >= bounds[other_axes[1], 0]) &
        (C[:, other_axes[1]] <= bounds[other_axes[1], 1])
    )

    return np.where(on_plane & inside)[0]




def infer_node_type_from_incident_edges(graph, node_id, vessel_type_map = EDGE_NKIND_TO_LABEL):
    check_attr(graph, "nkind", "es")
    inc = graph.incident(int(node_id))
    nk = [graph.es[e]["nkind"] for e in inc]
    nk = [int(x) for x in nk if x is not None]

    if not nk:
        return "unknown"

    n_type = Counter(nk).most_common(1)[0][0]
    return vessel_type_map.get(n_type, f"nkind_{n_type}")




def analyze_bc_faces(
    graph,
    box,
    coords_attr="coords_image",
    eps=2.0,
    degree_thr=4,
    compute_types=True,
    compute_depth=True,
    return_node_ids=False,
    verbose=True,
):
    """
    BC analysis PER FACE (faces-only).
    Returns dict: face -> metrics
    """
    validate_box_faces(box)

    deg = np.asarray(graph.degree(), dtype=int)
    has_d2s = compute_depth and ("distance_to_surface" in graph.vs.attributes())
    has_nkind = compute_types and ("nkind" in graph.es.attributes())

    if verbose:
        print("\n=== BC ANALYSIS (PER FACE) ===")
        print(f"Graph: {graph.vcount()} vertices, {graph.ecount()} edges")
        print(f"coords_attr='{coords_attr}' | eps={eps} | degree_thr={degree_thr}")

    results = {}

    for face, (axis, key) in FACES_DEF.items():
        value = float(box[key])
        nodes = bc_nodes_on_face(graph, axis, value, box, coords_attr, eps=eps).astype(int)
        n = int(nodes.size)

        # Degree metrics
        deg_counts = Counter(deg[nodes]) if n else Counter()
        high_mask = (deg[nodes] >= int(degree_thr)) if n else np.array([], dtype=bool)
        high_n = int(high_mask.sum()) if n else 0

        out = {
            "count": n,
            "degree_counts": dict(deg_counts),
            "high_degree_count": high_n,
            "high_degree_percent": float(100.0 * high_n / n) if n else 0.0,
        }

        # Type metrics (optional)
        if has_nkind and n:
            labels = [infer_node_type_from_incident_edges(graph, int(v)) for v in nodes]
            tc = Counter(labels)
            out["type_counts"] = dict(tc)
            out["type_percent"] = {k: 100.0 * v / n for k, v in tc.items()}
        else:
            out["type_counts"] = {}
            out["type_percent"] = {}

        # Depth metrics (optional)
        out["distance_to_surface_stats"] = distance_to_surface_stats(graph, nodes) if (has_d2s and n) else None

        # Optional: return node ids (heavy)
        if return_node_ids:
            out["nodes"] = nodes
            out["high_degree_nodes"] = nodes[high_mask] if n else np.array([], dtype=int)

        results[face] = out

        if verbose:
            print(f"\n--- {face} ---")
            print(f"BC nodes: {n}")
            if n:
                deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(deg_counts.items())])
                print(f"Degree (d:count): {deg_str}")
                print(f"High-degree (>= {degree_thr}): {high_n} ({out['high_degree_percent']:.2f}%)")
            if out["type_counts"]:
                for k, v in sorted(out["type_counts"].items(), key=lambda kv: -kv[1]):
                    print(f"  {k}: {v} ({out['type_percent'][k]:.1f}%)")
            if out["distance_to_surface_stats"] is not None:
                dstat = out["distance_to_surface_stats"]
                print("distance_to_surface (µm): "
                      f"min={dstat['min']:.2f}, mean={dstat['mean']:.2f}, "
                      f"median={dstat['median']:.2f}, max={dstat['max']:.2f}")

    return results



# ====================================================================================================================
# PLOTTING: GENERAL
# ====================================================================================================================

def plot_bar_by_category_general(categ, attribute_toplot, label_dict=None,
                        xlabel="Category", ylabel="Value",
                        title="Category statistics",
                        show_values=True, value_fmt="{:.2f}"):
    labels = [label_dict.get(c, c) if label_dict else c for c in categ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, attribute_toplot, color="steelblue", edgecolor="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_values:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                value_fmt.format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()


def plot_hist_by_category_general(
    values,
    category,
    label_dict=None,
    bins=40,
    layout="horizontal",
    density=True,
    show_mean=True,
    variable_name="Value",
    category_name="Category",
    main_title=None
):
    values = np.asarray(values, float)
    category = np.asarray(category)

    m = np.isfinite(values)
    values = values[m]
    category = category[m]

    cats = np.unique(category)
    N_total = int(values.size)   
    
    lo, hi = values.min(), values.max()
    edges = np.linspace(lo, hi, bins + 1)

    # Layout
    if layout == "horizontal":
        fig, axes = plt.subplots(1, len(cats),
                                 figsize=(4 * len(cats), 4),
                                 sharex=True)
    else:
        fig, axes = plt.subplots(len(cats), 1,
                                 figsize=(6, 3 * len(cats)),
                                 sharex=True)

    if len(cats) == 1:
        axes = [axes]

    if main_title is None:
        main_title = f"{variable_name} distribution by {category_name}"

    fig.suptitle(main_title)
    

    for ax, c in zip(axes, cats):

        subset = values[category == c]
        n = int(subset.size)
        pct = (100.0 * n / N_total) if N_total else 0.0 
        pct_decimals=1

        ax.hist(subset, bins=edges, density=density, alpha=0.7)

        if show_mean and len(subset):
            mean_val = subset.mean()
            ax.axvline(mean_val, linestyle="--", color = "red")
            ax.legend([f"Mean = {mean_val:.2f}"])

        name = label_dict.get(c, str(c)) if label_dict else str(c)
        ax.set_title(f"{name} (n={n}: {pct:.{pct_decimals}f}%)")

        ax.set_xlabel(variable_name)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_violin_box_by_category(values, category, label_dict=None,
                                xlabel="Category", ylabel="Value",
                                title="Distribution by category"):

    values = np.asarray(values, float)
    category = np.asarray(category)

    cats = np.unique(category)
    try:
        cats = np.array(sorted(cats, key=lambda x: int(x)))
    except Exception:
        cats = np.array(sorted(cats, key=lambda x: str(x)))

    data = [values[category == c] for c in cats]
    labels = [label_dict.get(int(c), str(c)) if label_dict else str(c) for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(1, len(cats) + 1)

    # --- Violin ---
    parts = ax.violinplot(data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        name = labels[i].lower()
        color = VESSEL_COLORS.get(name, "lightgray")
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # --- Boxplot ---
    ax.boxplot(data,
               positions=positions,
               widths=0.18,
               showfliers=False,
               patch_artist=False,
               medianprops=dict(color="aqua", linewidth=2))

    # --- Scatter jittered ---
    for i, subset in enumerate(data):
        x_center = positions[i]
        jitter = np.random.normal(x_center, 0.03, size=len(subset))
        ax.scatter(jitter, subset,
                   color="black",
                   s=12,
                   alpha=0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Estética más limpia
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()




# ====================================================================================================================
# PLOTTING: BC CUBES (faces-only)
# ====================================================================================================================



def plot_bc_3_cubes_tinted(
    G, box, coords_attr="coords_image", eps=2.0,
    elev=18, azim=35,
    face_alpha=0.10,
    point_alpha=0.85,
    point_size=8,
    sample_max=20000,
    show_vessel_legend=True
):
    """
    3 panels:
      A: x_min + z_min
      B: x_max + z_max
      C: y_min + y_max
    with tinted faces for easier visualization.
    """
    validate_box_faces(box)
    coords = get_coords(G, coords_attr).astype(float)

    # --- cube geometry (corners, edges, face polygons) ---
    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=float)

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    face_polys = {
        "x_min": [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        "x_max": [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
        "y_min": [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        "y_max": [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        "z_min": [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        "z_max": [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
    }

    # --- compute BC nodes per face (independiente de analyze_bc_faces) ---
    face_nodes = {}
    for face, (axis, key) in FACES_DEF.items():
        face_nodes[face] = bc_nodes_on_face(
            G, axis, float(box[key]), box, coords_attr, eps=eps
        ).astype(int)

    def nodes_for_faces(faces_subset):
        ids = np.unique(np.concatenate([face_nodes[f] for f in faces_subset])) if faces_subset else np.array([], int)
        if sample_max is not None and ids.size > sample_max:
            ids = np.random.choice(ids, size=int(sample_max), replace=False)
        return ids

    vessel_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}

    # face tint colors (suaves)
    face_colors = {
        "x_min": "tab:orange",
        "x_max": "tab:orange",
        "y_min": "tab:green",
        "y_max": "tab:green",
        "z_min": "tab:purple",
        "z_max": "tab:purple",
    }

    def draw_panel(ax, faces_subset, title):
        # cube wireframe
        for a, b in edges:
            ax.plot([corners[a,0], corners[b,0]],
                    [corners[a,1], corners[b,1]],
                    [corners[a,2], corners[b,2]], linewidth=1.0)

        # tinted faces
        polys = [face_polys[f] for f in faces_subset]
        cols  = [face_colors.get(f, "lightgray") for f in faces_subset]
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.6, alpha=face_alpha)
        ax.add_collection3d(pc)

        # scatter BC nodes (colored by inferred type)
        ids = nodes_for_faces(faces_subset)
        if ids.size:
            pts = coords[ids]
            labs = np.array([infer_node_type_from_incident_edges(G, int(v)) for v in ids], dtype=object)
            for lab, col in vessel_colors.items():
                m = (labs == lab)
                if np.any(m):
                    ax.scatter(pts[m,0], pts[m,1], pts[m,2],
                               s=point_size, alpha=point_alpha, color=col)

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # --- fixed face pairing as you requested ---
    faces_A = ("x_min", "z_min")
    faces_B = ("x_max", "z_max")
    faces_C = ("y_min", "y_max")

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    draw_panel(ax1, faces_A, "View A: x_min + z_min")
    draw_panel(ax2, faces_B, "View B: x_max + z_max")
    draw_panel(ax3, faces_C, "View C: y_min + y_max")

    if show_vessel_legend:
        handles = [
            Line2D([0], [0], marker='o', linestyle='', markersize=7,
                   markerfacecolor=vessel_colors[k], markeredgecolor='none', label=k)
            for k in ["arteriole", "venule", "capillary", "unknown"]
        ]
        ax3.legend(handles=handles, title="Vessel type", loc="upper left")

    plt.tight_layout()
    plt.show()
    return 



def plot_bc_cube_net(
    res,
    title="BC composition per face (cube net)",
    face_alpha=0.35,
    fontsize=10,
    pct_decimals=1,
    show_unknown=False,
    show_high_degree=True
):
    """
    Cube net for results from analyze_bc_faces().
    Expects:
      res[face]["count"]
      res[face]["type_percent"]
      res[face]["high_degree_percent"]
      res[face]["high_degree_count"]
      
    """
    # Standard cube net:
    #         y_max
    # x_min   z_min   x_max   z_max
    #         y_min
    layout = {
        "y_max": (1, 2),
        "x_min": (0, 1),
        "z_min": (1, 1),
        "x_max": (2, 1),
        "z_max": (3, 1),
        "y_min": (1, 0),
    }

    face_colors = {
        "z_max": "#9ECAE1",
        "y_max": "#A1D99B",
        "x_min": "#FDD0A2",
        "x_max": "#FCBBA1",
        "y_min": "#C7C1E3",
        "z_min": "#D9D9D9",
    }

    vessel_order = ["arteriole", "venule", "capillary"]
    if show_unknown:
        vessel_order.append("unknown")

    def face_text(face):
        if face not in res:
            return f"{face}\n(no data)"

        n = int(res[face].get("count", 0))
        tp = res[face].get("type_percent", {}) or {}
        tc = res[face].get("type_counts", {}) or {}

        lines = [f"{face}", f"{n} total nodes (100%)"]

        for k in vessel_order:
            if k in tp:
                pct = float(tp[k])
                cnt = int(tc.get(k, 0))
                if pct < 1e-9:
                    continue
                lines.append(f"{k}: {cnt} ({pct:.{pct_decimals}f}%)")

        if show_high_degree and "high_degree_percent" in res[face]:
            pct_hd = float(res[face].get("high_degree_percent", 0.0))
            n_hd = int(res[face].get("high_degree_count", 0))
            lines.append(f"HD(≥4): {n_hd} ({pct_hd:.{pct_decimals}f}%)")

        if len(lines) <= 2:
            lines.append("(no types)")
        return "\n".join(lines)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 2, pad=12)

    for face, (gx, gy) in layout.items():
        rect = Rectangle(
            (gx, gy), 1, 1,
            facecolor=face_colors.get(face, "#EEEEEE"),
            edgecolor="black",
            linewidth=1.2,
            alpha=face_alpha
        )
        ax.add_patch(rect)
        ax.text(gx + 0.5, gy + 0.5, face_text(face),
                ha="center", va="center", fontsize=fontsize)

    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 3.2)
    plt.tight_layout()
    plt.show()




# ====================================================================================================================
# PLOTTING: SPATIAL NKIND / HDN / DIAMETER-BY-NKIND
# ====================================================================================================================

def plot_nkind_spatial_edges(graph, coords_attr="coords_image", sample_edges=8000,
                                   title=None, ax=None, alpha=0.85, linewidth=0.6, seed=0):
    P = get_coords(graph, coords_attr)
    check_attr(graph, "nkind", "es")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    m = graph.ecount()
    if sample_edges is None:
        edge_ids = np.arange(m)
    else:
        rng = np.random.default_rng(seed)
        edge_ids = rng.choice(m, size=min(int(sample_edges), m), replace=False)

    nk = np.asarray(graph.es["nkind"], dtype=int)

    for eid in edge_ids:
        e = graph.es[int(eid)]
        u, v = e.tuple
        lab = EDGE_NKIND_TO_LABEL.get(int(nk[int(eid)]), "unknown")
        col = VESSEL_COLORS.get(lab, "black")

        ax.plot([P[u,0], P[v,0]], [P[u,1], P[v,1]], [P[u,2], P[v,2]],
                color=col, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or "Spatial vessel-type map (edges)")

    ax.legend(handles=[
        Line2D([0],[0], color=VESSEL_COLORS["arteriole"], lw=2, label="arteriole"),
        Line2D([0],[0], color=VESSEL_COLORS["venule"], lw=2, label="venule"),
        Line2D([0],[0], color=VESSEL_COLORS["capillary"], lw=2, label="capillary"),
    ], loc="best")

    plt.tight_layout()
    plt.show()
    return fig, ax



def plot_high_degree_nodes_by_type(
    graph,
    coords_attr="coords_image",
    high_degree_threshold=4,
    title=None,
    ax=None,
    s_all=3,
    s_hdn=35,
    alpha_all=0.25,
    alpha_hdn=0.95,
    vessel_colors=None
):
    """3D scatter: all nodes in gray, HDN colored by inferred vessel type from incident edges."""
    if vessel_colors is None:
        vessel_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}

    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), dtype=int)
    hdn = np.where(deg >= int(high_degree_threshold))[0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # all nodes (background)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s_all, c="lightgray", alpha=alpha_all, depthshade=False)

    # HDN colored by inferred type
    if hdn.size:
        labels = np.array([infer_node_type_from_incident_edges(graph, int(v)) for v in hdn], dtype=object)
        for lab, col in vessel_colors.items():
            m = (labels == lab)
            if np.any(m):
                pts = P[hdn[m]]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           s=s_hdn, c=col, alpha=alpha_hdn, depthshade=False, label=f"HDN {lab}")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"HDN (deg ≥ {high_degree_threshold}) colored by inferred type")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax







# ====================================================================================================================
# BC TABLES (faces-only)
# ====================================================================================================================

def bc_faces_table(res, box_name="Box"):                               
    """
    Create a pandas table from analyze_bc_faces() results (faces only).
    Columns: Face, count, % arteriole/venule/capillary/unknown, high degree %
    """
    import pandas as pd

    rows = []
    for face, face_data in res.items():
        total = int(face_data.get("count", 0))
        tp = face_data.get("type_percent", {}) or {}

        rows.append({
            "Box": box_name,
            "Face": face,
            "BC nodes": total,
            "% Arteriole": float(tp.get("arteriole", 0.0)),
            "% Venule": float(tp.get("venule", 0.0)),
            "% Capillary": float(tp.get("capillary", 0.0)),
            "% Unknown": float(tp.get("unknown", 0.0)),
            "High degree %": float(face_data.get("high_degree_percent", 0.0)),
        })

    return pd.DataFrame(rows)


# ====================================================================================================================
# GAIA-STYLE MICRO-SEGMENTS + VESSEL DENSITY (requires OutGeom-like data dict)
# ====================================================================================================================

import numpy as np

class PolylineBackend:
    """
    Minimum interface:
      - coords arrays globales x,y,z (len = n_points)
      - optional radii per point r (len = n_points) or None
      - optional lengths2 per segment L2 (len = n_points-1) or None
      - edge_geom_range(eid) -> (s,t) indices in global arrays
      - edge_attr(eid, name) -> edge value (i.e nkind)
    """
    def __init__(self, G):
        self.G = G

    def edge_geom_range(self, eid): raise NotImplementedError
    def edge_attr(self, eid, name): raise NotImplementedError
    @property
    def x(self): raise NotImplementedError
    @property
    def y(self): raise NotImplementedError
    @property
    def z(self): raise NotImplementedError
    @property
    def r(self): return None
    @property
    def lengths2(self): return None


class OutGeomBackend(PolylineBackend):
    def __init__(self, data):
        super().__init__(data["graph"])
        self.data = data
        c = data["coords"]
        self._x = np.asarray(c["x"], float)
        self._y = np.asarray(c["y"], float)
        self._z = np.asarray(c["z"], float)
        self._r = np.asarray(data["radii_geom"], float) if "radii_geom" in data else None

        # tu caso: lengths2 está aquí (ajusta si tu ruta es distinta)
        self._L2 = None
        if "geom" in data and "lengths2" in data["geom"]:
            self._L2 = np.asarray(data["geom"]["lengths2"], float)

    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def z(self): return self._z
    @property
    def r(self): return self._r
    @property
    def lengths2(self): return self._L2

    def edge_geom_range(self, eid):
        s = int(self.G.es[eid]["geom_start"])
        t = int(self.G.es[eid]["geom_end"])
        return s, t

    def edge_attr(self, eid, name):
        return self.G.es[eid][name]




def microsegments(source, get_geom = get_geom_from_data, get_range = get_range_from_data, get_edge_attr = get_edge_attr_from_data,
                       nkind_attr="nkind", want_radii=False):
    
    # Note: 
    G, x, y, z, L2, r = get_geom(source, want_radii=want_radii)

    mids, lens, nk, r0s, r1s = [], [], [], [], []

    for eid in range(G.ecount()):
        s, t = get_range(source, eid)
        if (t - s) < 2:
            continue

        nkind_e = int(get_edge_attr(source, eid, nkind_attr))

        for i in range(s, t - 1):
            L = float(L2[i])  
            if L <= 0:
                continue

            mids.append(((x[i]+x[i+1])*0.5, (y[i]+y[i+1])*0.5, (z[i]+z[i+1])*0.5))
            lens.append(L)
            nk.append(nkind_e)

            if want_radii and r is not None:
                r0s.append(float(r[i])); r1s.append(float(r[i+1]))
            else:
                r0s.append(np.nan); r1s.append(np.nan)

    return {
        "midpoints": np.asarray(mids, float),
        "lengths": np.asarray(lens, float),
        "nkind": np.asarray(nk, int),
        "r0": np.asarray(r0s, float),
        "r1": np.asarray(r1s, float),
    }



# Sanity check 
def count_microsegments_by_nkind(ms, label_map=None, verbose=True):
    """Counts micro-segments per nkind."""
    if label_map is None:
        label_map = EDGE_NKIND_TO_LABEL

    nk = ms["nkind"]
    out = {int(k): int(np.sum(nk == k)) for k in np.unique(nk)}

    if verbose:
        print("\n=== Micro-segment counts (Gaia-style) ===")
        for k in sorted(out.keys()):
            print(f"  nkind={k} ({label_map.get(k, k)}): {out[k]}")
        print(f"  TOTAL micro-segments: {len(nk)}")
    return out


def vessel_density_slabs_in_box(ms, box, slab=50.0, axis="z",                               # REPASAR, YO CREO QUE NO ESTÁ BIEN. PORQ HAY DOS METODOS ?? 
                               use_volume_fraction=False, verbose=True):
    """
    Compute vessel density inside a box, split into slabs along an axis.
    Uses MICRO-SEGMENTS (midpoint inside box).

    Metrics:
      use_volume_fraction=False -> length density = sum(L) / tissue_vol  (1/µm^2)
      use_volume_fraction=True  -> volume fraction = sum(pi*r^2*L) / tissue_vol (unitless) needs radii

    Returns pandas DataFrame with per-slab densities total + by nkind.
    """
    import pandas as pd

    mids = ms["midpoints"]
    L = ms["lengths"]
    nk = ms["nkind"]
    r0 = ms["r0"]
    r1 = ms["r1"]

    ax_i = {"x": 0, "y": 1, "z": 2}[axis]
    d = mids[:, ax_i]

    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    inside = (
        (mids[:, 0] >= xmin) & (mids[:, 0] <= xmax) &
        (mids[:, 1] >= ymin) & (mids[:, 1] <= ymax) &
        (mids[:, 2] >= zmin) & (mids[:, 2] <= zmax)
    )

    mids = mids[inside]
    d = d[inside]
    L = L[inside]
    nk = nk[inside]
    r0 = r0[inside]
    r1 = r1[inside]

    dmin = float({"x": xmin, "y": ymin, "z": zmin}[axis])
    dmax = float({"x": xmax, "y": ymax, "z": zmax}[axis])

    edges = np.arange(dmin, dmax + slab, slab)
    if edges[-1] < dmax:
        edges = np.append(edges, dmax)

    # cross-section area perpendicular to axis
    if axis == "x":
        A = (ymax - ymin) * (zmax - zmin)
    elif axis == "y":
        A = (xmax - xmin) * (zmax - zmin)
    else:
        A = (xmax - xmin) * (ymax - ymin)

    if use_volume_fraction:
        if not (np.all(np.isfinite(r0)) and np.all(np.isfinite(r1))):
            raise ValueError("Need data['radii_geom'] in microsegments to compute volume fraction.")
        rmean = 0.5 * (r0 + r1)
        amount = np.pi * (rmean ** 2) * L
        metric = "vol_frac"
    else:
        amount = L
        metric = "len_density"

    rows = []
    kinds = np.unique(nk)

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (d >= lo) & (d < hi) if i < (len(edges) - 2) else (d >= lo) & (d <= hi)

        tissue_vol = A * (hi - lo)
        tot = float(np.sum(amount[m]))

        row = {
            "slab_lo": lo,
            "slab_hi": hi,
            "tissue_vol": tissue_vol,
            f"total_{metric}": (tot / tissue_vol) if tissue_vol > 0 else np.nan
        }

        for k in kinds:
            vv = float(np.sum(amount[m & (nk == k)]))
            row[f"{EDGE_NKIND_TO_LABEL.get(k, k)}_{metric}"] = (vv / tissue_vol) if tissue_vol > 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n=== Vessel density slabs (axis={axis}, slab={slab} µm, metric={metric}) ===")
        print(df)

    return df




# ======================================================================================================================================================
#                                                               GAIA COMPLIANCE
# ======================================================================================================================================================
import numpy as np
import matplotlib.pyplot as plt




def edge_radius_consistency_report_MAX(                                 # NOTA: EXTREMADAMENTE COMPLICADO, LARGO, NO NECESITO TANTA INFO, HACERLO PARA MUCHOS MENOS EDGES
    data,
    sample=10000,             # Reducido un poco para que el loop de búsqueda sea rápido
    seed=0,
    bins=100,
    tol_abs_list=(0.0, 1e-3, 1e-2, 5e-2, 1e-1),
    use_abs_delta=True
):
    """
    Compara el radio del eje (r_edge) contra el MÁXIMO de su geometría:
        r_ref = max(geom['radii'][start : end])
    """

    G = data["graph"]
    rg = np.asarray(data["geom"]["radii"], dtype=np.float32)

    # 1. Obtener radios de los ejes
    if "radius" in G.es.attributes():
        r_edge = np.asarray(G.es["radius"], dtype=np.float32)
    else:
        raise ValueError("No se encontró G.es['radius'].")

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    nE = G.ecount()
    # Muestreo
    if sample is None or sample >= nE:
        idx = np.arange(nE, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(nE, size=int(sample), replace=False)

    # 2. CÁLCULO DE LA REFERENCIA POR MÁXIMO (Aquí está el cambio clave)
    r_ref = np.zeros(len(idx), dtype=np.float32)
    
    print(f"Calculando el máximo de geometría para {len(idx)} ejes...")
    for i, edge_idx in enumerate(idx):
        start = gs[edge_idx]
        end = ge[edge_idx]
        if end > start:
            # Buscamos el máximo en el segmento real de la geometría
            r_ref[i] = np.max(rg[start:end])
        else:
            r_ref[i] = rg[start] # Caso de un solo punto

    r_edge_i = r_edge[idx]

    # 3. Diferencias
    delta = (r_edge_i - r_ref).astype(np.float64)
    delta_abs = np.abs(delta)

    # Estadísticas básicas
    print(f"\n--- REPORTE DE CONSISTENCIA (POLÍTICA: MÁXIMO) ---")
    print(f"Δr stats: mean={np.mean(delta):.6g} | std={np.std(delta):.6g}")
    print(f"|Δr| <= 0.0 (Coincidencia exacta): {100.0 * np.mean(delta_abs <= 1e-7):.2f}%")

    # Porcentajes de error
    print("\n% dentro de tolerancia ABSOLUTA:")
    for t in tol_abs_list:
        pct = 100.0 * np.mean(delta_abs <= t)
        print(f"  |Δr| <= {t:g}: {pct:.2f}%")

    # Histograma
    plt.figure(figsize=(10,6))
    plt.hist(delta, bins=bins, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.xlabel("Error (Radio_Eje - Max_Geometría)")
    plt.ylabel("Número de ejes")
    plt.title("Distribución del error respecto al MÁXIMO")
    plt.show()

    return delta




# ====================================================================================================================
# RADII SANITY CHECKS (for microbloom / consistency)
# ====================================================================================================================

def check_polyline_radii_match_endnodes(                                             # ADAPTAR A ATRIBUTOS QUE TENGO AHORA, NO HACE FALTA TANTA PARAFERNALIA 
    data,                                                                           # SACAR LOS PUNTOS CON LOS INDICES Y COMPARAR, OJO CON LOS ACCESOS A RADII
    node_r_attr_candidates=("radii", "radius", "radius_point"),
    tol=1e-3,
    verbose=True
):
    """
    Check that polyline endpoint radii match radii at end nodes A/B.
    Needs:
      data["graph"] with edge attrs geom_start/geom_end
      data["radii_geom"] radii per polyline point
      graph.vs has a node radii attribute (one of candidates)
    """
    G = data["graph"]
    if "radii_geom" not in data:
        raise ValueError("data['radii_geom'] missing.")
    r_geom = np.asarray(data["radii_geom"], float)

    node_attr = None
    for cand in node_r_attr_candidates:
        if cand in G.vs.attributes():
            node_attr = cand
            break
    if node_attr is None:
        raise ValueError(f"No node radius attr found in graph.vs. Tried: {node_r_attr_candidates}")

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end' to check polyline endpoints.")

    r_node = np.asarray(G.vs[node_attr], float)

    bad = 0
    max_diff = 0.0
    examples = []

    for e in range(G.ecount()):
        s = int(G.es[e]["geom_start"])
        t = int(G.es[e]["geom_end"])
        if (t - s) < 2:
            continue

        a, b = G.es[e].tuple
        rA = float(r_node[a])
        rB = float(r_node[b])

        r0 = float(r_geom[s])
        r1 = float(r_geom[t - 1])

        d0 = abs(r0 - rA)
        d1 = abs(r1 - rB)
        md = max(d0, d1)

        if md > tol:
            bad += 1
            if len(examples) < 10:
                examples.append((e, md, rA, r0, rB, r1))
        max_diff = max(max_diff, md)

    if verbose:
        print("\n=== Radii check: polyline endpoints vs node radii ===")
        print(f"Node radius attr used: '{node_attr}' | tol={tol}")
        print(f"Mismatching edges: {bad} / {G.ecount()} | max diff: {max_diff:.6g}")
        if examples:
            print("Examples (edge, maxdiff, rA, r_poly0, rB, r_polyEnd):")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": bad, "max_diff": max_diff, "examples": examples, "node_attr": node_attr}

# ============================================================================================================================================0



#  LOS TRES SIGUIENTES CODIGOS PODRÍA ELIMINARLOS, NO TIENEN MUCHO SENTIDO NI TP ME DAN APENAS INFO DE ANALISIS DE BOX 

def check_radii_endpoints_allow_swap(data, tol=1e-3, max_examples=15, use_coords_check=False, coords_tol=1e-6):   # QUITARLO, METER COMPROBACIÓN DE ORDEN COMO CHECK EN CODIGO ANTERIOR 
    """
    Compara radii de vértices vs radii de geom en endpoints de cada edge.
    Permite geometría invertida (swap start/end) y reporta cuántos swaps hay.

    data format:
      data["graph"]  -> igraph Graph (edges have geom_start/geom_end)
      data["vertex"]["radii"] -> (nV,)
      data["geom"]["radii"]   -> (nP,)
      data["geom"]["x/y/z"]   -> (nP,)

    Si use_coords_check=True, además intenta decidir orientación comparando P[0] con coords_image[u].
    """
    G = data["graph"]

    # --- vertex radii ---
    if "vertex" not in data or "radii" not in data["vertex"]:
        raise KeyError("Falta data['vertex']['radii']")
    vr = np.asarray(data["vertex"]["radii"], dtype=np.float32)

    if len(vr) != G.vcount():
        raise ValueError(f"vertex['radii'] len={len(vr)} != G.vcount()={G.vcount()}")

    # --- geom radii ---
    if "geom" not in data or "radii" not in data["geom"]:
        raise KeyError("Falta data['geom']['radii']")
    gr = np.asarray(data["geom"]["radii"], dtype=np.float32)

    # optional geom coords for sanity / orientation check
    x = np.asarray(data["geom"]["x"], dtype=np.float64) if "x" in data["geom"] else None
    y = np.asarray(data["geom"]["y"], dtype=np.float64) if "y" in data["geom"] else None
    z = np.asarray(data["geom"]["z"], dtype=np.float64) if "z" in data["geom"] else None
    if x is not None and len(x) != len(gr):
        raise ValueError(f"geom['x'] len={len(x)} != geom['radii'] len={len(gr)}")

    coords_v = None
    if use_coords_check:
        if "coords_image" not in data["vertex"]:
            raise KeyError("use_coords_check=True pero falta data['vertex']['coords_image']")
        coords_v = np.asarray(data["vertex"]["coords_image"], dtype=np.float64)
        if coords_v.shape[0] != G.vcount() or coords_v.shape[1] != 3:
            raise ValueError("vertex['coords_image'] debe ser (nV,3)")

        if x is None or y is None or z is None:
            raise KeyError("use_coords_check=True requiere geom['x','y','z']")

    bad = 0
    swapped = 0
    max_diff = 0.0
    examples = []

    for ei in range(G.ecount()):
        s = int(G.es[ei]["geom_start"])
        en = int(G.es[ei]["geom_end"])

        if en - s < 2:
            continue
        if s < 0 or en > len(gr) or en <= s:
            raise ValueError(f"Edge {ei}: geom_start/end fuera de rango: start={s}, end={en}, nP={len(gr)}")

        u = int(G.es[ei].source)
        v = int(G.es[ei].target)

        # --- direct match (u->start, v->end) ---
        du = float(abs(vr[u] - gr[s]))
        dv = float(abs(vr[v] - gr[en - 1]))
        d_direct = max(du, dv)

        # --- swapped match (u->end, v->start) ---
        du2 = float(abs(vr[u] - gr[en - 1]))
        dv2 = float(abs(vr[v] - gr[s]))
        d_swap = max(du2, dv2)

        # decide best
        d_best = d_direct
        mode = "DIRECT"
        if d_swap < d_direct:
            d_best = d_swap
            mode = "SWAP"
            swapped += 1

        # optional: compare to coordinates to see if geometry order matches u
        coord_mode = None
        if use_coords_check:
            P0 = np.array([x[s], y[s], z[s]], dtype=np.float64)
            Pend = np.array([x[en-1], y[en-1], z[en-1]], dtype=np.float64)
            cu = coords_v[u]
            # if P0 ~ cu => direct, if Pend ~ cu => swapped
            if np.allclose(P0, cu, atol=coords_tol):
                coord_mode = "DIRECT"
            elif np.allclose(Pend, cu, atol=coords_tol):
                coord_mode = "SWAP"
            else:
                coord_mode = "UNKNOWN"

        max_diff = max(max_diff, d_best)

        if d_best > tol:
            bad += 1
            if len(examples) < max_examples:
                row = (
                    ei, d_best,
                    u, float(vr[u]), float(gr[s]),
                    v, float(vr[v]), float(gr[en - 1]),
                    mode
                )
                if use_coords_check:
                    row = row + (coord_mode,)
                examples.append(row)

    print("=== Radii endpoint check (vertex vs geom endpoints, allow swap) ===")
    msg = f"tol={tol} | bad_edges={bad}/{G.ecount()} | swapped_edges={swapped}/{G.ecount()} | max_diff={max_diff:.6g}"
    if use_coords_check:
        msg += f" | coords_tol={coords_tol}"
    print(msg)

    if examples:
        if use_coords_check:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end], best_mode, coord_mode)")
        else:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end], best_mode)")
        for ex in examples:
            print(" ", ex)

    return {"bad": bad, "swapped": swapped, "max_diff": max_diff, "examples": examples}




def check_polyline_radii_variation(data, tol_rel=0.05, verbose=True):
    """
    For each edge polyline, checks relative variation of radii along points:
      rel = (max-min)/mean
    Flags edges above tol_rel (e.g. 0.05 = 5%)
    """
    G = data["graph"]
    data["radii_geom"] = data["geom"]["radii"]

    if "radii_geom" not in data:
        raise ValueError("data['radii_geom'] missing.")
    r = np.asarray(data["radii_geom"], float)

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end' to check polyline radii variation.")

    bad = 0
    worst = 0.0
    examples = []

    for e in range(G.ecount()):
        s = int(G.es[e]["geom_start"])
        t = int(G.es[e]["geom_end"])
        if (t - s) < 2:
            continue

        rr = r[s:t]
        if len(rr) < 2:
            continue

        m = float(np.nanmean(rr))
        if not np.isfinite(m) or m == 0:
            continue

        rel = float((np.nanmax(rr) - np.nanmin(rr)) / m)
        if rel > tol_rel:
            bad += 1
            if len(examples) < 10:
                examples.append((e, rel, float(np.nanmin(rr)), float(np.nanmax(rr)), m))
        worst = max(worst, rel)

    if verbose:
        print("\n=== Radii variation along polyline ===")
        print(f"Bad edges (rel var > {tol_rel}): {bad} / {G.ecount()} | worst={worst:.3f}")
        if examples:
            print("Examples (edge, rel_var, rmin, rmax, rmean):")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": bad, "worst_rel": worst, "examples": examples}



def debug_edge(data, ei, k=3):
    G = data["graph"]
    cv = np.asarray(data["vertex"]["coords_image"], float)
    vr = np.asarray(data["vertex"]["radii"], float)
    x = np.asarray(data["geom"]["x"], float)
    y = np.asarray(data["geom"]["y"], float)
    z = np.asarray(data["geom"]["z"], float)
    gr = np.asarray(data["geom"]["radii"], float)

    e = G.es[ei]
    u, v = int(e.source), int(e.target)
    s, en = int(e["geom_start"]), int(e["geom_end"])
    P0 = np.array([x[s], y[s], z[s]])
    P1 = np.array([x[en-1], y[en-1], z[en-1]])

    du0 = np.linalg.norm(P0 - cv[u]); dv0 = np.linalg.norm(P0 - cv[v])
    du1 = np.linalg.norm(P1 - cv[u]); dv1 = np.linalg.norm(P1 - cv[v])

    print(f"EDGE {ei} | u={u}, v={v} | npts={en-s}")
    print("coords distances:")
    print(f"  |Pstart-u|={du0:.6g}  |Pstart-v|={dv0:.6g}")
    print(f"  |Pend-u|  ={du1:.6g}  |Pend-v|  ={dv1:.6g}")

    print("endpoint radii:")
    print(f"  vr[u]={vr[u]:.6g}  gr[start]={gr[s]:.6g}  gr[end]={gr[en-1]:.6g}")
    print(f"  vr[v]={vr[v]:.6g}  gr[start]={gr[s]:.6g}  gr[end]={gr[en-1]:.6g}")

    # mira algunos puntos cercanos al start/end por si el nodo no coincide exactamente con s/en-1
    print("\nnear start radii:")
    for j in range(s, min(en, s+k)):
        print(f"  j={j-s:2d}  gr={gr[j]:.6g}  P=({x[j]:.3f},{y[j]:.3f},{z[j]:.3f})")
    print("near end radii:")
    for j in range(max(s, en-k), en):
        print(f"  j={j-(en-k):2d}  gr={gr[j]:.6g}  P=({x[j]:.3f},{y[j]:.3f},{z[j]:.3f})")

        import numpy as np

def classify_edge_endpoint_coords(data, tol=1e-6):
    G = data["graph"]
    cv = np.asarray(data["vertex"]["coords_image"], float)

    x = np.asarray(data["geom"]["x"], float)
    y = np.asarray(data["geom"]["y"], float)
    z = np.asarray(data["geom"]["z"], float)

    counts = {"OK_DIRECT":0, "OK_SWAP":0, "BAD_BOTH":0, "BAD_AMBIG":0}
    examples = {"OK_SWAP":[], "BAD_BOTH":[]}

    for ei in range(G.ecount()):
        e = G.es[ei]
        u, v = int(e.source), int(e.target)
        s, en = int(e["geom_start"]), int(e["geom_end"])
        if en - s < 2:
            continue

        P0 = np.array([x[s], y[s], z[s]])
        P1 = np.array([x[en-1], y[en-1], z[en-1]])

        mu0 = np.allclose(P0, cv[u], atol=tol)
        mv0 = np.allclose(P0, cv[v], atol=tol)
        mu1 = np.allclose(P1, cv[u], atol=tol)
        mv1 = np.allclose(P1, cv[v], atol=tol)

        direct = mu0 and mv1
        swap   = mv0 and mu1

        if direct and not swap:
            counts["OK_DIRECT"] += 1
        elif swap and not direct:
            counts["OK_SWAP"] += 1
            if len(examples["OK_SWAP"]) < 10:
                examples["OK_SWAP"].append((ei,u,v,s,en))
        elif direct and swap:
            counts["BAD_AMBIG"] += 1
        else:
            counts["BAD_BOTH"] += 1
            if len(examples["BAD_BOTH"]) < 10:
                # guarda distancias para debug
                du0 = np.linalg.norm(P0-cv[u]); dv0=np.linalg.norm(P0-cv[v])
                du1 = np.linalg.norm(P1-cv[u]); dv1=np.linalg.norm(P1-cv[v])
                examples["BAD_BOTH"].append((ei,u,v,du0,dv0,du1,dv1,s,en))

    print(counts)
    print("Examples OK_SWAP (ei,u,v,s,en):", examples["OK_SWAP"])
    print("Examples BAD_BOTH (ei,u,v,du0,dv0,du1,dv1,s,en):", examples["BAD_BOTH"])
    return counts, examples



# ESTOS CHECKS QUE INFO ME DAN?????? OSEA ME INTERESA ALGO MÁS ALLÁ DE HDN ? QUE YA LO HE COMPROBADO
# ====================================================================================================================
# DEGREE CHECKS: distribution + by type + spatial mapping for any degree band
# ====================================================================================================================

def degree_summary(graph, max_degree_to_print=None):
    deg = np.asarray(graph.degree(), dtype=int)
    c = Counter(deg.tolist())
    print("\n=== Degree distribution (all nodes) ===")
    for d in sorted(c.keys()):
        if max_degree_to_print is not None and d > max_degree_to_print:
            continue
        print(f"  degree {d}: {c[d]}")
    print(f"  max degree: {int(deg.max()) if deg.size else 0}")
    return c


def degree_summary_by_type(graph):
    deg = np.asarray(graph.degree(), dtype=int)
    labels = np.array([infer_node_type_from_incident_edges(graph, v.index) for v in graph.vs], dtype=object)

    out = {}
    print("\n=== Degree distribution by vessel-type (node label) ===")
    for lab in ["arteriole", "venule", "capillary", "unknown"]:
        m = labels == lab
        c = Counter(deg[m].tolist())
        out[lab] = c
        total = int(np.sum(m))
        print(f"\n  [{lab}] n={total}")
        for d in sorted(c.keys()):
            print(f"    degree {d}: {c[d]}")
    return out


def plot_degree_nodes_spatial(
    graph, coords_attr="coords_image",
    degree_min=4, degree_max=None,
    by_type=True,
    s_all=2, s_sel=30,
    alpha_all=0.15, alpha_sel=0.95,
    title=None
):
    """
    Shows in 3D the nodes whose degree is in [degree_min, degree_max].
    If by_type=True, colors selected nodes by type (arteriole red, venule blue, capillary gray).
    """
    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), dtype=int)

    if degree_max is None:
        sel = np.where(deg >= int(degree_min))[0]
        crit = f"deg ≥ {degree_min}"
    else:
        sel = np.where((deg >= int(degree_min)) & (deg <= int(degree_max)))[0]
        crit = f"{degree_min} ≤ deg ≤ {degree_max}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s_all, c="lightgray", alpha=alpha_all, depthshade=False)

    if len(sel) == 0:
        ax.set_title(title or f"No nodes with {crit}")
        plt.show()
        return fig, ax

    if not by_type:
        ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], s=s_sel, c="black", alpha=alpha_sel, depthshade=False)
    else:
        labs = np.array([infer_node_type_from_incident_edges(graph, int(v)) for v in sel], dtype=object)
        col = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}
        for lab in ["arteriole", "venule", "capillary", "unknown"]:
            m = labs == lab
            if np.any(m):
                ax.scatter(P[sel[m], 0], P[sel[m], 1], P[sel[m], 2],
                           s=s_sel, c=col[lab], alpha=alpha_sel, depthshade=False, label=lab)
        ax.legend(loc="best", title=f"{crit} nodes")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"Spatial distribution of {crit}")
    plt.show()
    return fig, ax


# ====================================================================================================================
# REDUNDANCY: edge-disjoint arteriole->venule paths inside the box
# ====================================================================================================================

def induced_subgraph_box(graph, box, coords_attr="coords_image", node_eps=0.0, edge_mode="both"):
    """
    Creates subgraph induced by nodes inside the box.

    edge_mode:
      - "both": keep edges only if BOTH endpoints are inside (recommended)
      - "any":  alias of "both" here to avoid silent wrong graphs (see note in comments)

    Returns:
      sub (igraph.Graph), sub_to_orig (np.array), orig_to_sub (dict)
    """
    P = get_coords(graph, coords_attr)

    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    inside = (
        (P[:, 0] >= xmin - node_eps) & (P[:, 0] <= xmax + node_eps) &
        (P[:, 1] >= ymin - node_eps) & (P[:, 1] <= ymax + node_eps) &
        (P[:, 2] >= zmin - node_eps) & (P[:, 2] <= zmax + node_eps)
    )
    keep = np.where(inside)[0]
    if keep.size == 0:
        return None, None, None

    sub = graph.induced_subgraph(keep)
    sub_to_orig = np.array(keep, dtype=int)
    orig_to_sub = {int(o): i for i, o in enumerate(sub_to_orig)}

    if edge_mode not in ("both", "any"):
        raise ValueError("edge_mode must be 'both' or 'any'")

    # NOTE: true "any" needs adding boundary nodes; for safety we keep "both".
    return sub, sub_to_orig, orig_to_sub


def _nodes_by_label_in_subgraph(sub, vessel_type_map=EDGE_NKIND_TO_LABEL):              # YA TENGO UN CODIGO Q HACE ESTO NO? PARA QUÉ QUIERO OTRO?????
    labels = [infer_node_type_from_incident_edges(sub, v.index, vessel_type_map=EDGE_NKIND_TO_LABEL) for v in sub.vs]
    out = {}
    for lab in ["arteriole", "venule", "capillary", "unknown"]:
        out[lab] = np.where(np.array(labels, dtype=object) == lab)[0]
    return out


def edge_disjoint_av_paths_in_box(
    graph, box, coords_attr="coords_image", node_eps=0.0,
    vessel_type_map=EDGE_NKIND_TO_LABEL,
    connect_cap=10**9,
):
    """
    Max number of EDGE-DISJOINT paths connecting any arteriole -> any venule
    inside the induced subgraph of the box.

    Robust for UNDIRECTED graphs:
      - edge-splitting transformation: each original edge becomes a capacity-1 gadget.

    Returns dict.
    """
    sub, _, _ = induced_subgraph_box(
        graph, box, coords_attr=coords_attr, node_eps=node_eps, edge_mode="both"
    )
    if sub is None or sub.ecount() == 0:
        return {"edge_disjoint_paths": 0, "nA": 0, "nV": 0,
                "n_nodes_sub": 0, "n_edges_sub": 0, "note": "empty subgraph"}

    groups = _nodes_by_label_in_subgraph(sub, vessel_type_map=vessel_type_map)
    A = groups.get("arteriole", np.array([], dtype=int))
    V = groups.get("venule", np.array([], dtype=int))

    if len(A) == 0 or len(V) == 0:
        return {"edge_disjoint_paths": 0, "nA": int(len(A)), "nV": int(len(V)),
                "n_nodes_sub": int(sub.vcount()), "n_edges_sub": int(sub.ecount()),
                "note": "no arteriole or no venule nodes in box"}

    n = sub.vcount()
    m = sub.ecount()

    # Directed auxiliary graph with edge-splitting nodes
    g2 = ig.Graph(directed=True)

    edge_in = np.arange(n, n + m, dtype=int)
    edge_out = np.arange(n + m, n + 2 * m, dtype=int)
    s = n + 2 * m
    t = s + 1
    g2.add_vertices(t + 1)

    edges2 = []
    caps = []

    # Capacity 1 through each original edge gadget: edge_in -> edge_out
    for e in range(m):
        edges2.append((int(edge_in[e]), int(edge_out[e])))
        caps.append(1.0)

    edgelist = sub.get_edgelist()
    for e, (u, v) in enumerate(edgelist):
        ein = int(edge_in[e])
        eout = int(edge_out[e])
        u = int(u)
        v = int(v)

        # u/v -> edge_in (big cap)
        edges2 += [(u, ein), (v, ein)]
        caps += [float(connect_cap), float(connect_cap)]

        # edge_out -> u/v (big cap)
        edges2 += [(eout, u), (eout, v)]
        caps += [float(connect_cap), float(connect_cap)]

    # super source to all A, all V to super sink
    for a in A:
        edges2.append((s, int(a)))
        caps.append(float(connect_cap))
    for v in V:
        edges2.append((int(v), t))
        caps.append(float(connect_cap))

    g2.add_edges(edges2)
    flow = g2.maxflow(s, t, capacity=caps)

    return {
        "edge_disjoint_paths": int(flow.value),
        "nA": int(len(A)),
        "nV": int(len(V)),
        "n_nodes_sub": int(sub.vcount()),
        "n_edges_sub": int(sub.ecount()),
        "note": "edge-disjoint paths via maxflow (edge-splitting, cap=1 per original edge)"
    }





# ====================================================================================================================
#                            NEW-PKL ONLY HELPERS (data["graph"], data["vertex"], data["geom"])
# ====================================================================================================================

def select_largest_component_data(data, verbose=True):            # LO PUEDO ELIMINAR, SIMPLEMENTE AÑADIR EN SINGLE CONNECTED, QUE ESCOJA EL BCC
    """
    NEW-PKL ONLY.
    Input: data = {"graph":G, "vertex":{...}, "geom":{...}}
    Output: same structure but keeping only the largest connected component (WEAK).
    geom is kept as-is (polylines still refer to original geom indices; that's OK for analysis that uses
    only node/edge topology; if you need geom consistency too, you'd do a geom cut, different step).
    """
    if not (isinstance(data, dict) and "graph" in data and "vertex" in data and "geom" in data):
        raise ValueError("Expected NEW-PKL dict with keys: 'graph', 'vertex', 'geom'.")

    G = data["graph"]
    comps = G.components(mode="WEAK")
    if len(comps) == 0:
        return data

    keep = np.asarray(comps[np.argmax(comps.sizes())], dtype=int)
    H = G.induced_subgraph(keep)

    if verbose:
        print("\n=== GIANT COMPONENT (NEW-PKL) ===")
        print("Original:", G.vcount(), "V", G.ecount(), "E")
        print("GC      :", H.vcount(), "V", H.ecount(), "E")

    V = data["vertex"]
    V2 = {}
    for k, arr in V.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == G.vcount():
            V2[k] = arr[keep].copy()
        elif arr.ndim == 2 and arr.shape[0] == G.vcount():
            V2[k] = arr[keep, :].copy()
        else:
            # leave untouched if not per-vertex sized
            V2[k] = arr

    return {"graph": H, "vertex": V2, "geom": data["geom"]}


def attach_vertex_attrs_from_data(data, verbose=False):                         # ESTO NO LO NECESITO, SOLAMENTE TENGO QUE ACTUALIZAR LOS ACCESOS EN LOS OTROS
    """
    NEW-PKL ONLY.
    Copies heavy per-vertex arrays from data["vertex"] into graph.vs attributes,
    so all your existing functions (that expect graph.vs["coords_image"], etc.) work.
    """
    if not (isinstance(data, dict) and "graph" in data and "vertex" in data):
        raise ValueError("attach_vertex_attrs_from_data expects NEW-PKL data dict.")

    G = data["graph"]
    V = data["vertex"]

    # required for box + plotting
    if "coords_image" not in V:
        raise ValueError("data['vertex']['coords_image'] is required.")
    G.vs["coords_image"] = [tuple(map(float, p)) for p in np.asarray(V["coords_image"], float)]

    # optional commonly used
    if "coords" in V:
        G.vs["coords"] = [tuple(map(float, p)) for p in np.asarray(V["coords"], float)]
    if "distance_to_surface" in V:
        G.vs["distance_to_surface"] = np.asarray(V["distance_to_surface"], float).tolist()
    if "radii" in V:
        G.vs["radii"] = np.asarray(V["radii"], float).tolist()
    if "vertex_annotation" in V:
        G.vs["vertex_annotation"] = np.asarray(V["vertex_annotation"], int).tolist()
    if "id" in V:
        G.vs["id"] = np.asarray(V["id"], int).tolist()

    if verbose:
        print("[attach_vertex_attrs_from_data] graph.vs attrs:", G.vs.attributes())

    return data


# ====================================================================================================================
#                                  RADII CHECK: vertex endpoints vs geom endpoints
# ====================================================================================================================


                                                # TIENE SENTIDO OTRO CODIGO A PARTE? SOLICITADO POR GAIA, PERO PODRÍA METERLO DONDE COMPRUEBO EL MAX Y YA NO? 

def check_endpoint_radii_vertex_vs_geom(data, tol=1e-3, verbose=True, max_print=10):
    """
    NEW-PKL ONLY.
    Checks: geom radii at first/last polyline point == vertex radii at nodes A/B.

    Needs:
      data["vertex"]["radii"] (nV,)
      data["geom"]["radii"]   (nP,)
      edges have geom_start/geom_end
    """
    if not (isinstance(data, dict) and "graph" in data and "vertex" in data and "geom" in data):
        raise ValueError("Expected NEW-PKL dict with keys: 'graph', 'vertex', 'geom'.")

    G = data["graph"]

    if "radii" not in data["vertex"]:
        raise ValueError("Missing data['vertex']['radii'] for endpoint radii check.")
    if "radii" not in data["geom"]:
        raise ValueError("Missing data['geom']['radii'] for endpoint radii check.")

    vr = np.asarray(data["vertex"]["radii"], float)
    gr = np.asarray(data["geom"]["radii"], float)

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end'.")

    bad = 0
    maxdiff = 0.0
    examples = []

    for ei in range(G.ecount()):
        e = G.es[ei]
        s = int(e["geom_start"]); en = int(e["geom_end"])
        if en - s < 2:
            continue

        u, v = e.tuple

        d0 = abs(float(gr[s])    - float(vr[u]))
        d1 = abs(float(gr[en-1]) - float(vr[v]))
        md = max(d0, d1)

        if md > tol:
            bad += 1
            if len(examples) < max_print:
                examples.append((ei, md, u, float(vr[u]), float(gr[s]), v, float(vr[v]), float(gr[en-1])))
        maxdiff = max(maxdiff, md)

    if verbose:
        print("\n=== Radii endpoint check (vertex vs geom endpoints) ===")
        print(f"tol={tol} | bad_edges={bad}/{G.ecount()} | max_diff={maxdiff:.6g}")
        if examples:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end])")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": int(bad), "max_diff": float(maxdiff), "examples": examples}


# ====================================================================================================================
#                             REDUNDANCY: return used subgraph edges (for viz)
# ====================================================================================================================


                                                        # NO TIENE SENTIDO ESTE CODIGO AQUI? PARA QUE QUEIRO ESTE CODIGO? "GAIA SOLICITED"
def edge_disjoint_av_paths_in_box_with_used_edges(
    graph, box, coords_attr="coords_image", node_eps=0.0,
    vessel_type_map=EDGE_NKIND_TO_LABEL,
    connect_cap=10**9,
):
    """
    Same as edge_disjoint_av_paths_in_box but also returns:
      used_edges_sub: list of edge IDs in the induced subgraph that are saturated by the maxflow solution.

    Note: this is ONE valid maximum-flow solution (not unique).
    """
    validate_box_faces(box)
    sub, sub_to_orig, _ = induced_subgraph_box(
        graph, box, coords_attr=coords_attr, node_eps=node_eps, edge_mode="both"
    )
    if sub is None or sub.ecount() == 0:
        return {"edge_disjoint_paths": 0, "used_edges_sub": [], "used_edges_orig": [],
                "nA": 0, "nV": 0, "n_nodes_sub": 0, "n_edges_sub": 0, "note": "empty subgraph"}

    groups = _nodes_by_label_in_subgraph(sub, vessel_type_map=vessel_type_map)
    A = groups.get("arteriole", np.array([], dtype=int))
    V = groups.get("venule", np.array([], dtype=int))

    if len(A) == 0 or len(V) == 0:
        return {"edge_disjoint_paths": 0, "used_edges_sub": [], "used_edges_orig": [],
                "nA": int(len(A)), "nV": int(len(V)),
                "n_nodes_sub": int(sub.vcount()), "n_edges_sub": int(sub.ecount()),
                "note": "no arteriole or no venule nodes in box"}

    n = sub.vcount()
    m = sub.ecount()

    g2 = ig.Graph(directed=True)
    edge_in = np.arange(n, n + m, dtype=int)
    edge_out = np.arange(n + m, n + 2*m, dtype=int)
    s = n + 2*m
    t = s + 1
    g2.add_vertices(t + 1)

    edges2 = []
    caps = []

    # cap=1 edges come FIRST -> flow.flow[0:m] corresponds to these.
    for e in range(m):
        edges2.append((int(edge_in[e]), int(edge_out[e])))
        caps.append(1.0)

    edgelist = sub.get_edgelist()
    for e, (u, v) in enumerate(edgelist):
        ein = int(edge_in[e]); eout = int(edge_out[e])
        u = int(u); v = int(v)
        edges2 += [(u, ein), (v, ein), (eout, u), (eout, v)]
        caps  += [float(connect_cap)] * 4

    for a in A:
        edges2.append((s, int(a))); caps.append(float(connect_cap))
    for v in V:
        edges2.append((int(v), t)); caps.append(float(connect_cap))

    g2.add_edges(edges2)
    flow = g2.maxflow(s, t, capacity=caps)

    f = np.asarray(flow.flow, float)
    used_edges_sub = [int(eid) for eid in range(m) if f[eid] > 0.5]

    # map sub-edge id -> orig-edge id
    # sub_to_orig maps sub-vertex -> orig-vertex. For edge mapping use original endpoints:
    used_edges_orig = []
    for eid in used_edges_sub:
        u_sub, v_sub = sub.es[eid].tuple
        u0 = int(sub_to_orig[u_sub]); v0 = int(sub_to_orig[v_sub])
        # find edge id in original graph between (u0,v0) (igraph: get_eid)
        try:
            used_edges_orig.append(int(graph.get_eid(u0, v0, directed=False, error=True)))
        except:
            pass

    return {
        "edge_disjoint_paths": int(flow.value),
        "used_edges_sub": used_edges_sub,
        "used_edges_orig": used_edges_orig,
        "nA": int(len(A)),
        "nV": int(len(V)),
        "n_nodes_sub": int(sub.vcount()),
        "n_edges_sub": int(sub.ecount()),
        "note": "used_edges_sub: saturated edges in one maxflow solution (subgraph edge IDs)"
    }



                                                                                            # LO MISMO DE ANTES, PARA Q USO ESTO ? Q INFO DA? GAIA
def plot_redundancy_used_edges(graph, box, used_edges_orig, coords_attr="coords_image",
                              sample_edges=8000, title=None):
    """
    Simple 3D viz:
      - light gray: random background edges in box-subgraph
      - red: edges used by maxflow solution
    """

    P = get_coords(graph, coords_attr)

    sub, sub_to_orig, _ = induced_subgraph_box(graph, box, coords_attr=coords_attr, node_eps=0.0, edge_mode="both")
    if sub is None or sub.ecount() == 0:
        print("[plot_redundancy_used_edges] empty subgraph")
        return

    # background edges (orig ids)
    bg_orig = []
    for e in sub.es:
        u, v = e.tuple
        u0 = int(sub_to_orig[u]); v0 = int(sub_to_orig[v])
        try:
            bg_orig.append(int(graph.get_eid(u0, v0, directed=False, error=True)))
        except:
            pass
    bg_orig = np.asarray(bg_orig, int)

    if len(bg_orig) > sample_edges:
        rng = np.random.default_rng(0)
        bg_orig = rng.choice(bg_orig, size=sample_edges, replace=False)

    used_edges_orig = np.asarray(list(set(used_edges_orig)), int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # background
    for eid in bg_orig:
        u, v = graph.es[int(eid)].tuple
        ax.plot([P[u,0], P[v,0]], [P[u,1], P[v,1]], [P[u,2], P[v,2]], color="lightgray", alpha=0.15, linewidth=0.6)

    # used
    for eid in used_edges_orig:
        u, v = graph.es[int(eid)].tuple
        ax.plot([P[u,0], P[v,0]], [P[u,1], P[v,1]], [P[u,2], P[v,2]], color="red", alpha=0.95, linewidth=2.0)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"Redundancy: edges used by maxflow solution (k={len(used_edges_orig)})")
    plt.show()
    return fig, ax


