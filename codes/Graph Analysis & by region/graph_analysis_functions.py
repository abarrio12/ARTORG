"""
Vascular graph analysis module.

This module includes:

• Classic graph metrics:
    - Edge length
    - Diameter
    - Degree distribution
    - Loop and duplicate detection

• Boundary condition (BC) detection:
    - Face-specific analysis only (no per-box aggregation)
    - Cube-net and 3D visualization

• Spatial analysis:
    - Vessel type maps
    - High-degree node (HDN) analysis
    - Diameter distribution by nkind

• Gaia-style vessel density:
    - Micro-segment based computation
    - Volume fraction using atlas radii
    - Slab-based analysis

• Redundancy metric:
    - Maximum number of edge-disjoint arteriole → venule paths (maxflow)

SPACE CONVENTION
----------------
space="vox" → use voxel coordinates (coords_image)
space="um"  → use micrometer coordinates (coords_image_R)

IMPORTANT:
- eps is ALWAYS specified in VOXELS.
- If space="um", eps is automatically converted using resolution.
- To switch units, change ONLY the "space" argument.

Author: Ana Barrio
Date: 17 Feb 2026
"""


# ====================================================================================================================
#                                                    IMPORTS
# ====================================================================================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig

from collections import Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle



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
#                                           SPACE RESOLUTION HELPERS (voxels vs micrometers)
# ====================================================================================================================
def resolve_vertex(data, space="vox"):
    return data["vertex"] if space == "vox" else data["vertex_R"]

def resolve_geom(data, space="vox"):
    return data["geom"] if space == "vox" else data["geom_R"]

def resolve_coords_attr(space="vox"):
    return "coords_image" if space == "vox" else "coords_image_R"

def resolve_depth_attr(space="vox"):
    return "distance_to_surface" if space == "vox" else "distance_to_surface_R"
def resolve_length_attr(space="vox"):
    return "length" if space == "vox" else "length_R"

def resolve_tortuosity_attr(space="vox"):
    return "tortuosity" if space == "vox" else "tortuosity_R"

def resolve_lengths2(data, space="vox"):
    if space == "vox":
        return np.asarray(data["geom"]["lengths2"], float)
    else:
        return np.asarray(data["geom_R"]["lengths2_R"], float)

def resolve_geom_coords(data, space="vox"):
    if space == "vox":
        g = data["geom"]
        return (
            np.asarray(g["x"], float),
            np.asarray(g["y"], float),
            np.asarray(g["z"], float),
        )
    else:
        g = data["geom_R"]
        return (
            np.asarray(g["x_R"], float),
            np.asarray(g["y_R"], float),
            np.asarray(g["z_R"], float),
        )
    

def resolve_eps(eps=2.0, space="vox", res_um_per_vox=res_um_per_vox, axis=0):
    '''
    Convention:
    - eps is ALWAYS specified in VOXELS
    - If space="um", eps is converted: eps_um = eps_vox * res_um_per_vox[axis].
    '''
    eps = float(eps)
    if space == "um":
        eps *= float(res_um_per_vox[axis])
    return eps


# ====================================================================================================================
#                                                  ATTRIBUTE VALIDATION
# ====================================================================================================================

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
            raise ValueError(f"Missing edge attribute(s): {missing}. Available: {graph.es.attributes()}")
        return

    if where in ("vertex", "vs"):
        existing = set(graph.vs.attributes())
        missing = [name for name in names if name not in existing]
        if missing:
            raise ValueError(f"Missing vertex attribute(s): {missing}. Available: {graph.vs.attributes()}")
        return

    raise ValueError("'where' parameter must be one of: 'edge'/'es' or 'vertex'/'vs'")



# ====================================================================================================================
#                                              GEOMETRY ACCESS FROM DATA DICT 
# ====================================================================================================================
def get_geom_from_data(data, space="vox"):

    G = data["graph"]

    if space == "vox":
        g = data["geom"]
        x = np.asarray(g["x"], float)
        y = np.asarray(g["y"], float)
        z = np.asarray(g["z"], float)
        L2 = np.asarray(g["lengths2"], float)
        r = np.asarray(g["radii_atlas_geom"], float) if "radii_atlas_geom" in g else None
    else:
        gR = data["geom_R"]
        x = np.asarray(gR["x_R"], float)
        y = np.asarray(gR["y_R"], float)
        z = np.asarray(gR["z_R"], float)
        L2 = np.asarray(gR["lengths2_R"], float)
        r = np.asarray(gR["radii_atlas_geom_R"], float) if "radii_atlas_geom_R" in gR else None

    return G, x, y, z, L2, r


def get_range_from_data(data, eid):
    G = data["graph"]
    return int(G.es[eid]["geom_start"]), int(G.es[eid]["geom_end"])

def get_edge_attr_from_data(data, eid, name):
    return data["graph"].es[eid][name]

# ====================================================================================================================
#                                                  SYNC ATTRIBUTES INTO GRAPH
# ====================================================================================================================

def resolve_space_and_attrs(
    graph,
    space=None,
    coords_attr=None,
    depth_attr=None,
    require_space=True,
    require_coords=True,
    require_depth=False,
):
    """
    Enforce the space/attribute contract.

    Rules:
    - If require_space: space must be "vox" or "um" (no defaults).
    - If coords_attr is None: infer from space (coords_image vs coords_image_R).
    - If depth_attr is None: infer from space (distance_to_surface vs distance_to_surface_R).
    - Validate that chosen attrs exist in graph.vs.

    Returns
    -------
    space, coords_attr, depth_attr
    """
    if require_space:
        if space not in ("vox", "um"):
            raise ValueError("space must be explicitly set to 'vox' or 'um' (no default).")
    else:
        if space is not None and space not in ("vox", "um"):
            raise ValueError("space must be None, 'vox', or 'um'.")

    # infer attrs only if space is provided
    if coords_attr is None:
        if require_coords:
            if space is None:
                raise ValueError("coords_attr is None and space is None. Provide one of them.")
            coords_attr = resolve_coords_attr(space=space)

    if depth_attr is None:
        if require_depth:
            if space is None:
                raise ValueError("depth_attr is None and space is None. Provide one of them.")
            depth_attr = resolve_depth_attr(space=space)
        else:
            # optional depth: only infer if space is given
            if space is not None:
                depth_attr = resolve_depth_attr(space=space)

    # validate attrs
    if require_coords and coords_attr is not None and coords_attr not in graph.vs.attributes():
        raise ValueError(f"coords_attr='{coords_attr}' not found in graph.vs. "
                         f"Available: {graph.vs.attributes()}")

    if require_depth and depth_attr is not None and depth_attr not in graph.vs.attributes():
        raise ValueError(f"depth_attr='{depth_attr}' not found in graph.vs. "
                         f"Available: {graph.vs.attributes()}")

    return space, coords_attr, depth_attr

def sync_vertex_attributes_safe(data, space="vox"):
    """
    Copy only per-vertex arrays into graph.vs, respecting space.
    - coords Nx3 -> list of tuples (igraph friendly)
    - 1D length nV -> list
    """
    G = data["graph"]
    V = resolve_vertex(data, space=space)
    nV = G.vcount()

    copied = []
    for k, arr in V.items():
        a = np.asarray(arr)

        # per-vertex 1D
        if a.ndim == 1 and a.shape[0] == nV:
            G.vs[k] = a.tolist()
            copied.append(k)

        # per-vertex Nx3 (coords)
        elif a.ndim == 2 and a.shape[0] == nV and a.shape[1] == 3:
            G.vs[k] = [tuple(map(float, row)) for row in a]
            copied.append(k)

        else:
            # ignore non per-vertex arrays
            continue
    print(f"[sync_vertex_attributes_safe] copied {len(copied)} attrs:", copied)
    return data



# ====================================================================================================================
#                                                    LOAD / SAVE
# ====================================================================================================================

def load_data(path):
    """Load an igraph Graph object from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def dump_data(data, out_path):
    """Save an igraph Graph object into a pickle file."""
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nData successfully saved to: {out_path}")



# ====================================================================================================================
#                                                    BOX BUILDERS
# ====================================================================================================================


def make_box_in_vox(
    center_vox,
    box_size_um,
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
    box_um = np.asarray(box_size_um, dtype=float)
    res_um_per_vox = np.asarray(res_um_per_vox, dtype=float)

    if center_vox.shape != (3,):
        raise ValueError("center_vox must be length-3 (x,y,z) in voxels")
    if box_um.shape != (3,):
        raise ValueError("box_size_um must be length-3 (µm)")
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


def make_box_in_um(center_vox, box_size_um, res_um_per_vox=res_um_per_vox):
    '''
    Return a box dict in µm from center in voxels (from Paraview), box size in µm, and resolution.
    Note: this is just a helper to build the box in micrometers if !!! coords_attr='coords_image_R' !!!!
    '''
    center_um = np.asarray(center_vox) * res_um_per_vox
    box_um = np.array(box_size_um, dtype=float)
    return {
        "xmin": float(center_um[0] - box_um[0]/2), "xmax": float(center_um[0] + box_um[0]/2),
        "ymin": float(center_um[1] - box_um[1]/2), "ymax": float(center_um[1] + box_um[1]/2),
        "zmin": float(center_um[2] - box_um[2]/2), "zmax": float(center_um[2] + box_um[2]/2),
    }








# ====================================================================================================================
#                                                   BASIC GRAPH HELPERS 
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


# ====================================================================================================================
#                                                  GRAPH STATS
# ====================================================================================================================

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







# ====================================================================================================================
#                                              CONNECTED COMPONENTS
# ====================================================================================================================

def single_connected_component(data):
    """
    Ensure graph is single connected component. If not, keep the largest one.

    Returns:
        is_single (bool)
        n_components (int)
        graph (igraph.Graph)
    """
    G = data["graph"]
    comps = G.components()
    n_components = len(comps)
    is_single = (n_components == 1)

    if is_single:
        print("The graph is a single connected component.")
        return True, 1, data
    
    print(f"The graph has {n_components} components. Keeping the largest one.")
    data2 = keep_giant_component_data(data)
    return False, n_components, data2 



def keep_giant_component_data(data):
    """
    Keep largest connected component and keep vertex arrays aligned.
    Geom arrays remain global (OK because surviving edges still reference global indices).
    """
    G = data["graph"]
    comps = G.components()
    if len(comps) <= 1:
        return data

    keep = np.asarray(comps[np.argmax(comps.sizes())], dtype=int)
    H = G.induced_subgraph(keep)
    print("\n=== GIANT COMPONENT ===")
    print("Original:", G.vcount(), "V", G.ecount(), "E")
    print("Giant   :", H.vcount(), "V", H.ecount(), "E")

    V2 = {}
    V = data.get("vertex", {})
    for k, arr in V.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == G.vcount():
            V2[k] = arr[keep].copy()
        elif arr.ndim == 2 and arr.shape[0] == G.vcount():
            V2[k] = arr[keep, :].copy()
        else:
            V2[k] = arr  # no per-vertex

    out = dict(data)
    out["graph"] = H
    out["vertex"] = V2
    return out



# ====================================================================================================================
#                                              LENGTH & DIAMETER ANALYSIS 
# ====================================================================================================================

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



def get_avg_length_nkind(graph, space="um"):
    """Compute mean edge length per nkind."""
    length_attr = resolve_length_attr(space=space)
    check_attr(graph, [length_attr, "nkind"], "es")

    length_att = np.array(graph.es[length_attr], dtype=float)
    nkind = np.array(graph.es["nkind"], dtype=int)

    unique = np.unique(nkind)
    l = []
    unit = "µm" if space == "um" else "vox"

    print(f"\nAverage length by nkind (space={space}):\n")
    for k in unique:
        mean_l = float(np.mean(length_att[nkind == k]))
        l.append(mean_l)
        print(f"nkind = {k}: average length ({length_attr}) = {mean_l:.6f} {unit}")

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



# ====================================================================================================================
#                                              DEGREES & HIGH-DEGREE NODES (HDN)
# ====================================================================================================================


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



def analyze_hdn_pattern_in_box(
    graph,
    space=None,
    coords_attr=None,
    depth_attr=None,
    degree_thr=4,
    box=None,  # must be in SAME units as coords_attr/space
    depth_bins_um=(("0–20", 0, 20), ("20–50", 20, 50), ("50–200", 50, 200), (">200", 200, np.inf)),
    vessel_type_map=EDGE_NKIND_TO_LABEL,
    eps_vox=2.0,  # ALWAYS in vox; converted internally if space="um"
):
    """
    HDN fingerprint (no prints). Strict unit compliance.

    Contract:
    - You MUST set space to "vox" or "um" (no default).
    - If coords_attr is None, it is inferred from space.
    - If depth_attr is None, it is inferred from space.
    - box must be in SAME units as coords_attr/space.
    - eps_vox is ALWAYS specified in vox; if space="um" it's converted per axis when applied.

    Returns dict with:
      - counts, type composition
      - spatial centroid/concentration (if coords available)
      - depth stats in µm (if depth available)
      - face bias (if box provided)
    """
    space, coords_attr, depth_attr = resolve_space_and_attrs(
        graph,
        space=space,
        coords_attr=coords_attr,
        depth_attr=depth_attr,
        require_space=True,
        require_coords=True,
        require_depth=False,  # depth is optional
    )

    deg = np.asarray(graph.degree(), dtype=int)
    hdn = np.where(deg >= int(degree_thr))[0]

    out = {
        "space": space,
        "coords_attr": coords_attr,
        "depth_attr": depth_attr,
        "eps_vox": float(eps_vox),
        "degree_thr": int(degree_thr),
        "n_nodes": int(graph.vcount()),
        "n_hdn": int(hdn.size),
        "hdn_fraction": float(hdn.size / graph.vcount()) if graph.vcount() else 0.0,
    }

    if hdn.size == 0:
        out["spatial"] = None
        out["hdn_type_composition"] = None
        out["depth_hdn"] = None
        out["face_bias_hdn"] = None
        return out

    # --- spatial pattern (coords required) ---
    P = np.asarray(graph.vs[coords_attr], float)
    Ph = P[hdn]

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

    # --- type composition ---
    type_comp = None
    if vessel_type_map is not None and ("nkind" in graph.es.attributes()):
        labels = np.array(
            [infer_node_type_from_incident_edges(graph, int(v), vessel_type_map) for v in hdn],
            dtype=object,
        )
        uniq, cnt = np.unique(labels, return_counts=True)
        type_comp = {str(u): {"count": int(c), "proportion": float(c / labels.size)} for u, c in zip(uniq, cnt)}
    out["hdn_type_composition"] = type_comp

    # --- depth (always reported in µm) ---
    depth_out = None
    if depth_attr is not None and depth_attr in graph.vs.attributes():
        d = np.asarray(graph.vs[depth_attr], float)[hdn]

        if space == "vox":
            # Paris convention: distance_to_surface stored in vox, isotropic assumption -> scale by XY
            vals_um = d * float(res_um_per_vox[0])
        else:
            vals_um = d

        depth_out = {
            "unit": "µm",
            "n": int(vals_um.size),
            "min": float(np.min(vals_um)),
            "mean": float(np.mean(vals_um)),
            "median": float(np.median(vals_um)),
            "max": float(np.max(vals_um)),
            "bins": {},
        }

        n = vals_um.size
        for label, low, high in depth_bins_um:
            mask = (vals_um >= float(low)) & (vals_um < float(high))
            c = int(np.sum(mask))
            depth_out["bins"][label] = {
                "range_um": [float(low), float(high)],
                "count": c,
                "proportion": float(c / n) if n else 0.0,
            }

    out["depth_hdn"] = depth_out

    # --- face bias (box must be same units as coords_attr/space) ---
    face_bias = None
    if box is not None:
        validate_box_faces(box)
        xmin, xmax = float(box["xmin"]), float(box["xmax"])
        ymin, ymax = float(box["ymin"]), float(box["ymax"])
        zmin, zmax = float(box["zmin"]), float(box["zmax"])

        # eps is ALWAYS in vox; convert to the coordinate unit if needed
        eps_x = resolve_eps(eps_vox, space=space, axis=0)
        eps_y = resolve_eps(eps_vox, space=space, axis=1)
        eps_z = resolve_eps(eps_vox, space=space, axis=2)

        face_bias = {
            "x_min": float(np.mean(np.abs(Ph[:, 0] - xmin) <= eps_x)),
            "x_max": float(np.mean(np.abs(Ph[:, 0] - xmax) <= eps_x)),
            "y_min": float(np.mean(np.abs(Ph[:, 1] - ymin) <= eps_y)),
            "y_max": float(np.mean(np.abs(Ph[:, 1] - ymax) <= eps_y)),
            "z_min": float(np.mean(np.abs(Ph[:, 2] - zmin) <= eps_z)),
            "z_max": float(np.mean(np.abs(Ph[:, 2] - zmax) <= eps_z)),
        }
        face_bias["max_face_bias"] = float(max(face_bias.values()))

    out["face_bias_hdn"] = face_bias
    return out



# ====================================================================================================================
#                                             DISTANCE TO SURFACE ANALYSIS
# ====================================================================================================================


def distance_to_surface_stats(
    graph,
    nodes,
    space="vox",  # "vox" or "um"
    depth_attr_vox="distance_to_surface",
    depth_attr_um="distance_to_surface_R",
    depth_bins_um=DEFAULT_DEPTH_BINS_UM,
    res_um_per_vox=res_um_per_vox,
):
    """
    Stats + depth-bin counts for distance-to-surface.

    - space="vox": uses graph.vs[depth_attr_vox] (vox). Converts the UM bins -> VOX using XY resolution.
    - space="um" : uses graph.vs[depth_attr_um]  (µm). Uses bins in µm directly.

    Note: Your conversion script defines:
      distance_to_surface_R = distance_to_surface * sx
    where sx = res_um_per_vox[0] (XY resolution).
    So we use that same convention here.
    """
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size == 0:
        return None

    if space == "um":
        depth_attr = depth_attr_um
        unit = "µm"
        # bins already in µm
        bins = depth_bins_um
        check_attr(graph, depth_attr, "vs")
        vals = np.asarray(graph.vs[depth_attr], dtype=float)[nodes]

        range_key = "range_um"

    elif space == "vox":
        depth_attr = depth_attr_vox
        unit = "vox"
        check_attr(graph, depth_attr, "vs")
        vals = np.asarray(graph.vs[depth_attr], dtype=float)[nodes]

        # Convert µm bins -> vox bins using XY scaling (sx), to stay consistent with your outgeom_um.py
        sx = float(res_um_per_vox[0])
        bins = []
        for label, low_um, high_um in depth_bins_um:
            low_v = float(low_um) / sx
            high_v = float(high_um) / sx if np.isfinite(high_um) else np.inf
            bins.append((label, low_v, high_v))

        range_key = "range_vox"

    else:
        raise ValueError("space must be 'vox' or 'um'")

    out = {
        "space": space,
        "unit": unit,
        "depth_attr": depth_attr,
        "n": int(vals.size),
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "bins": {},
    }

    n = vals.size
    for label, low, high in bins:
        mask = (vals >= low) & (vals < high)
        c = int(np.sum(mask))
        out["bins"][label] = {
            range_key: [float(low), float(high)],
            "count": c,
            "proportion": float(c / n),
        }

    return out




                                                               


# ====================================================================================================================
#                                   BOUNDARY NODES ANALYSIS (BC NODES ON FACES)
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
    space=None,
    coords_attr=None,
    eps=2.0,            # ALWAYS in vox
    degree_thr=4,
    return_node_ids=False,
):
    validate_box_faces(box)

    space, coords_attr, depth_attr = resolve_space_and_attrs(
        graph,
        space=space,
        coords_attr=coords_attr,
        depth_attr=None,
        require_space=True,
        require_coords=True,
        require_depth=False,
    )

    deg = np.asarray(graph.degree(), dtype=int)
    has_nkind = "nkind" in graph.es.attributes()

    # depth optional (only if present)
    depth_attr = resolve_depth_attr(space=space)
    has_d2s = depth_attr in graph.vs.attributes()

    results = {}

    for face, (axis, key) in FACES_DEF.items():
        value = float(box[key])
        eps_face = resolve_eps(eps, space=space, axis=axis)  # convert only if space="um"
        nodes = bc_nodes_on_face(graph, axis, value, box, coords_attr, eps=eps_face).astype(int)
        n = int(nodes.size)

        deg_counts = Counter(deg[nodes]) if n else Counter()
        high_mask = (deg[nodes] >= int(degree_thr)) if n else np.array([], dtype=bool)
        high_n = int(high_mask.sum()) if n else 0

        out = {
            "count": n,
            "degree_counts": dict(deg_counts),
            "high_degree_count": high_n,
            "high_degree_percent": float(100.0 * high_n / n) if n else 0.0,
        }

        if has_nkind and n:
            labels = [infer_node_type_from_incident_edges(graph, int(v)) for v in nodes]
            tc = Counter(labels)
            out["type_counts"] = dict(tc)
            out["type_percent"] = {k: 100.0 * v / n for k, v in tc.items()}
        else:
            out["type_counts"] = {}
            out["type_percent"] = {}

        out["distance_to_surface_stats"] = (
            distance_to_surface_stats(graph, nodes, space=space) if (has_d2s and n) else None
        )

        if return_node_ids:
            out["nodes"] = nodes
            out["high_degree_nodes"] = nodes[high_mask] if n else np.array([], dtype=int)

        results[face] = out

    return results




# ====================================================================================================================
# PLOTTING: GENERAL
# ====================================================================================================================

def plot_bar_by_category_general(categ, attribute_toplot, label_dict=None,
                        xlabel="Category", ylabel="Value",
                        title="Category statistics ",
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
    G, box,
    coords_attr=None,
    space=None,                 
    eps=2.0,                     # eps ALWAYS specified in VOXELS 
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

    Conventions:
    - `box` must be in the same units as coords_attr.
    - `eps` is ALWAYS specified in VOXELS.
      If space="um", eps is converted to µm per face axis using res_um_per_vox[axis].
    """
    validate_box_faces(box)
    space, coords_attr, _ = resolve_space_and_attrs(
        G,
        space=space,
        coords_attr=coords_attr,
        depth_attr=None,
        require_space=True,
        require_coords=True,
        require_depth=False,
    )
    coords = get_coords(G, coords_attr).astype(float)

    # cube geometry
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

    # --- compute BC nodes per face with eps conversion per axis ---
    face_nodes = {}
    for face, (axis, key) in FACES_DEF.items():
        value = float(box[key])
        eps_face = resolve_eps(eps, space=space, axis=axis)  # <-- key fix
        face_nodes[face] = bc_nodes_on_face(
            G, axis, value, box, coords_attr, eps=eps_face
        ).astype(int)

    def nodes_for_faces(faces_subset):
        if not faces_subset:
            return np.array([], dtype=int)
        ids = np.unique(np.concatenate([face_nodes[f] for f in faces_subset if f in face_nodes]))
        if sample_max is not None and ids.size > sample_max:
            ids = np.random.choice(ids, size=int(sample_max), replace=False)
        return ids

    vessel_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}

    face_colors = {
        "x_min": "tab:orange", "x_max": "tab:orange",
        "y_min": "tab:green",  "y_max": "tab:green",
        "z_min": "tab:purple", "z_max": "tab:purple",
    }

    def draw_panel(ax, faces_subset, title):
        # wireframe
        for a, b in edges:
            ax.plot([corners[a,0], corners[b,0]],
                    [corners[a,1], corners[b,1]],
                    [corners[a,2], corners[b,2]], linewidth=1.0)

        # tinted faces
        polys = [face_polys[f] for f in faces_subset]
        cols  = [face_colors.get(f, "lightgray") for f in faces_subset]
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.6, alpha=face_alpha)
        ax.add_collection3d(pc)

        # scatter BC nodes
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
        unit = "µm" if space == "um" else "vox"
        ax.set_xlabel(f"X ({unit})"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"Boundary nodes on cube faces by nkind | space: {space}", fontsize=16, y=1.02)
    plt.show()
    return




def plot_bc_cube_net(
    res,
    title="BC composition per face (cube net) | space: {space}",
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
def plot_degree_nodes_spatial(
    graph,
    space=None,
    coords_attr=None,
    degree_min=4,
    degree_max=None,
    by_type=True,
    s_all=2,
    s_sel=30,
    alpha_all=0.15,
    alpha_sel=0.95,
    title=None,
):
    """
    3D plot of nodes whose degree is in [degree_min, degree_max].

    Contract:
    - You MUST set space to "vox" or "um" (no default).
    - If coords_attr is None, it is inferred from space (coords_image vs coords_image_R).
    - Units are displayed in axis labels and title.
    - No prints.
    """
    # enforce explicit choice + infer coords_attr only from space
    space, coords_attr, _ = resolve_space_and_attrs(
        graph,
        space=space,
        coords_attr=coords_attr,
        depth_attr=None,
        require_space=True,
        require_coords=True,
        require_depth=False,
    )

    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), dtype=int)

    if degree_max is None:
        sel = np.where(deg >= int(degree_min))[0]
        crit = f"deg ≥ {int(degree_min)}"
    else:
        sel = np.where((deg >= int(degree_min)) & (deg <= int(degree_max)))[0]
        crit = f"{int(degree_min)} ≤ deg ≤ {int(degree_max)}"

    unit = "µm" if space == "um" else "vox"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # background: all nodes
    ax.scatter(
        P[:, 0], P[:, 1], P[:, 2],
        s=s_all, c="lightgray", alpha=alpha_all, depthshade=False
    )

    if sel.size == 0:
        ax.set_title(title or f"No nodes with {crit} [{unit}]")
        ax.set_xlabel(f"X ({unit})"); ax.set_ylabel(f"Y ({unit})"); ax.set_zlabel(f"Z ({unit})")
        plt.tight_layout()
        plt.show()
        return fig, ax

    if not by_type:
        ax.scatter(
            P[sel, 0], P[sel, 1], P[sel, 2],
            s=s_sel, c="black", alpha=alpha_sel, depthshade=False
        )
    else:
        labs = np.array([infer_node_type_from_incident_edges(graph, int(v)) for v in sel], dtype=object)
        col = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}

        for lab in ["arteriole", "venule", "capillary", "unknown"]:
            m = (labs == lab)
            if np.any(m):
                pts = P[sel[m]]
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=s_sel, c=col[lab], alpha=alpha_sel, depthshade=False, label=lab
                )
        ax.legend(loc="best", title=f"{crit} nodes")

    ax.set_xlabel(f"X ({unit})"); ax.set_ylabel(f"Y ({unit})"); ax.set_zlabel(f"Z ({unit})")
    ax.set_title(title or f"Spatial distribution of {crit} [{unit}] | space: {space}")
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
#                                   MICRO-SEGMENTS + VESSEL DENSITY (requires OutGeom-like data dict)
# ====================================================================================================================

def microsegments(data, space="um", nkind_attr="nkind"):
    """
    Returns micro-segments in the requested space.
    space='um' is the ONLY sensible option if you use slab in µm.
    """
    G, x, y, z, L2, r = get_geom_from_data(data, space=space)

    mids, lens, nk, r0s, r1s = [], [], [], [], []

    for eid in range(G.ecount()):
        s, t = get_range_from_data(data, eid)
        if (t - s) < 2:
            continue

        nkind_e = int(get_edge_attr_from_data(data, eid, nkind_attr))

        for i in range(s, t - 1):
            L = float(L2[i])
            if L <= 0:
                continue

            mids.append(((x[i]+x[i+1])*0.5, (y[i]+y[i+1])*0.5, (z[i]+z[i+1])*0.5))
            lens.append(L)
            nk.append(nkind_e)

            if r is not None:
                r0s.append(float(r[i])); r1s.append(float(r[i+1]))
            else:
                r0s.append(np.nan); r1s.append(np.nan)

    return {
        "space": space,
        "midpoints": np.asarray(mids, float),
        "lengths": np.asarray(lens, float),
        "nkind": np.asarray(nk, int),
        "r0": np.asarray(r0s, float),
        "r1": np.asarray(r1s, float),
    }



# Sanity check 
def count_microsegments_by_nkind(ms, label_map=None):
    """Counts micro-segments per nkind."""
    if label_map is None:
        label_map = EDGE_NKIND_TO_LABEL

    nk = ms["nkind"]
    out = {int(k): int(np.sum(nk == k)) for k in np.unique(nk)}

    for k in sorted(out.keys()):
            print(f"  nkind={k} ({label_map.get(k, k)}): {out[k]}")
    print(f"  TOTAL micro-segments: {len(nk)}")
    return out

def vessel_vol_frac_slabs_in_box(ms, box, slab=50.0, axis="z"):
    """
    Volume fraction inside a box, split into slabs along an axis.
    Uses MICRO-SEGMENTS (midpoint inside box).

    Metric:
      vol_frac = sum(pi * r_mean^2 * L) / tissue_vol    (unitless)

    Requires:
      ms["space"] == "um"
      ms["r0"], ms["r1"] finite (atlas radii per point, in µm)
    """
    if ms.get("space") != "um":
        raise ValueError("Expected microsegments in µm (ms['space']=='um').")

    mids = ms["midpoints"]
    L    = ms["lengths"]
    nk   = ms["nkind"]
    r0   = ms["r0"]
    r1   = ms["r1"]

    # must have radii
    if not (np.all(np.isfinite(r0)) and np.all(np.isfinite(r1))):
        raise ValueError("microsegments radii contain NaN/inf. Build ms with atlas radii per point.")

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
    d    = d[inside]
    L    = L[inside]
    nk   = nk[inside]
    r0   = r0[inside]
    r1   = r1[inside]

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

    # volume per micro-segment
    rmean = 0.5 * (r0 + r1)
    amount = np.pi * (rmean ** 2) * L  # µm^3

    rows = []
    kinds = np.unique(nk)

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (d >= lo) & (d < hi) if i < (len(edges) - 2) else (d >= lo) & (d <= hi)

        tissue_vol = A * (hi - lo)  # µm^3
        tot = float(np.sum(amount[m]))

        row = {
            "slab_lo": lo,
            "slab_hi": hi,
            "tissue_vol": tissue_vol,
            "total_vol_frac": (tot / tissue_vol) if tissue_vol > 0 else np.nan
        }

        for k in kinds:
            vv = float(np.sum(amount[m & (nk == k)]))
            row[f"{EDGE_NKIND_TO_LABEL.get(int(k), k)}_vol_frac"] = (vv / tissue_vol) if tissue_vol > 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"\n=== Volume fraction slabs (axis={axis}, slab={slab} µm) ===")
    print(df)

    return df






# ====================================================================================================================
#                                                    REDUNDANCY
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


def _nodes_by_label_in_subgraph(sub, vessel_type_map=EDGE_NKIND_TO_LABEL):            
    labels = [infer_node_type_from_incident_edges(sub, v.index, vessel_type_map=vessel_type_map) for v in sub.vs]
    out = {}
    for lab in ["arteriole", "venule", "capillary", "unknown"]:
        out[lab] = np.where(np.array(labels, dtype=object) == lab)[0]
    return out



def av_paths_in_box(graph, box, k=3,
                           space=None, coords_attr=None,
                           node_eps=0.0):
    """
    Returns up to k shortest A->V paths inside the box.
    Much simpler than maxflow-based extraction.
    """

    validate_box_faces(box)

    space, coords_attr, _ = resolve_space_and_attrs(
        graph, space=space, coords_attr=coords_attr,
        depth_attr=None, require_space=True, require_coords=True, require_depth=False
    )

    sub, sub_to_orig, _ = induced_subgraph_box(
        graph, box, coords_attr=coords_attr,
        node_eps=node_eps, edge_mode="both"
    )

    if sub is None or sub.ecount() == 0:
        return []

    groups = _nodes_by_label_in_subgraph(sub)
    A = np.asarray(groups.get("arteriole", []), dtype=int)
    V = np.asarray(groups.get("venule", []), dtype=int)

    if A.size == 0 or V.size == 0:
        return []

    paths_orig = []

    # take first k A-V combinations
    count = 0
    for a in A:
        for v in V:
            path_sub = sub.get_shortest_paths(a, to=v)[0]
            if len(path_sub) > 1:
                # map to original graph vertex ids
                path_orig = [int(sub_to_orig[p]) for p in path_sub]
                paths_orig.append(path_orig)
                count += 1
                if count >= k:
                    return paths_orig

    return paths_orig


def plot_av_paths_in_box(graph, box, paths_orig,
                      space=None, coords_attr=None,
                      node_eps=0.0,
                      sample_edges=5000):

    space, coords_attr, _ = resolve_space_and_attrs(
        graph, space=space, coords_attr=coords_attr,
        depth_attr=None, require_space=True, require_coords=True, require_depth=False
    )

    P = get_coords(graph, coords_attr)

    # background
    sub, sub_to_orig, _ = induced_subgraph_box(
        graph, box, coords_attr=coords_attr,
        node_eps=node_eps, edge_mode="both"
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if sub is not None:
        bg_pairs = []
        for (u, v) in sub.get_edgelist():
            u0 = sub_to_orig[u]
            v0 = sub_to_orig[v]
            bg_pairs.append((u0, v0))

        bg_pairs = np.array(bg_pairs)
        if len(bg_pairs) > sample_edges:
            bg_pairs = bg_pairs[np.random.choice(len(bg_pairs), sample_edges, replace=False)]

        for (u, v) in bg_pairs:
            ax.plot([P[u,0], P[v,0]],
                    [P[u,1], P[v,1]],
                    [P[u,2], P[v,2]],
                    alpha=0.1, linewidth=0.6)

    # paths in red
    for path in paths_orig:
        for a, b in zip(path[:-1], path[1:]):
            ax.plot([P[a,0], P[b,0]],
                    [P[a,1], P[b,1]],
                    [P[a,2], P[b,2]],
                    linewidth=2.5)

    unit = "µm" if space == "um" else "vox"
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title(f"Simple A→V paths (n={len(paths_orig)}) | space: {space}")
    plt.tight_layout()
    plt.show()


