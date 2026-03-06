"""
Vascular graph analysis module (FORMATTED GAIA GRAPH ONLY).

Expected input
--------------
A formatted igraph.Graph loaded from a pickle:
    G = ig.Graph.Read_Pickle(path)

Required attributes (typical)
-----------------------------
Vertices:
  - vs["coords"] : (x,y,z) tuples, in the unit of the graph (vox OR um)
  - optional:
      vs["is_border"]   : 1 if created as a cut border node
      vs["border_face"] : one of {"x_min","x_max","y_min","y_max","z_min","z_max"}
      vs["distance_to_surface"] or vs["distance_to_surface_R"]

Edges:
  - es["points"]    : list of (x,y,z) tuples (polyline)
  - es["lengths2"]  : list of per-segment lengths
  - es["length"]    : scalar length
  - es["diameters"] : list of per-point diameters
  - es["diameter"]  : scalar diameter
  - es["nkind"]     : 2/3/4 (arteriole/venule/capillary)
  - optional:
      es["length_steps"], es["hd"], es["htt"], es["flow"], ...



Organized by:
1. Constants
2. IO
3. Basic helpers
4. Geometry
5. Topology
6. Density
7. Simulation / BC nodes
8. Resilience
9. Plotting


Author: Ana Barrio
Updated: 4 March 26

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import pickle
import random

from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# ======================================================================
# Constants
# ======================================================================
EDGE_NKIND_TO_LABEL = {2: "arteriole", 3: "venule", 4: "capillary"}

FACES = ("x_min","x_max","y_min","y_max","z_min","z_max")

FACES_DEF = {
    "x_min": (0,"xmin"),
    "x_max": (0,"xmax"),
    "y_min": (1,"ymin"),
    "y_max": (1,"ymax"),
    "z_min": (2,"zmin"),
    "z_max": (2,"zmax"),
}

res_um_per_vox = np.array([1.625,1.625,2.5],dtype=float) # image resolution

VESSEL_COLORS = {
    "arteriole":"#ff2828",
    "venule":"#0072c4",
    "capillary":"#7f7f7f",
}

DEFAULT_DEPTH_BINS_UM = 10

# ======================================================================
# IO helpers
# ======================================================================


def load_graph(path: str) -> ig.Graph: 
    with open(path, "rb") as f: 
        obj = pickle.load(f) 
    if isinstance(obj, ig.Graph): 
        return obj 
    if isinstance(obj, dict) and "graph" in obj and isinstance(obj["graph"], ig.Graph): 
        return obj["graph"] 
    raise TypeError(f"File does not contain an igraph.Graph: {path} (type={type(obj)})")


def save_graph(G: ig.Graph, path: str) -> None:
    G.write_pickle(path)
    print("Saved:", path)


def keep_giant_component(G: ig.Graph) -> ig.Graph: 
    comps = G.components() 
    if len(comps) <= 1: 
        return G 
    keep = np.asarray(comps[np.argmax(comps.sizes())], dtype=int) 
    return G.induced_subgraph(keep)


# ======================================================================
# Basic helpers
# ======================================================================
def get_vessel_color(k):
    k = EDGE_NKIND_TO_LABEL.get(k, "unknown")
    return VESSEL_COLORS.get(k, "black")

def check_attr(graph: ig.Graph, names, where="vs"):
    if isinstance(names, str):
        names = [names]
    where = where.lower()

    if where in ("vs", "vertex"):
        existing = set(graph.vs.attributes())
        missing = [n for n in names if n not in existing]
        if missing:
            raise ValueError(f"Missing vertex attribute(s): {missing}. Available: {graph.vs.attributes()}")
        return

    if where in ("es", "edge"):
        existing = set(graph.es.attributes())
        missing = [n for n in names if n not in existing]
        if missing:
            raise ValueError(f"Missing edge attribute(s): {missing}. Available: {graph.es.attributes()}")
        return

    raise ValueError("where must be 'vs' or 'es'")



def validate_box_faces(box: dict):
    req = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    missing = [k for k in req if k not in box]
    if missing:
        raise ValueError(f"box missing keys: {missing}. Required: {list(req)}")



def get_coords(graph: ig.Graph, coords_attr="coords") -> np.ndarray:
    check_attr(graph, coords_attr, "vs")
    P = np.asarray(graph.vs[coords_attr], dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got {P.shape}.")
    return P



def resolve_eps(eps_vox=2.0, space="um", axis=0, res_um_per_vox=res_um_per_vox) -> float:
    eps = float(eps_vox)
    if space == "um":
        eps *= float(res_um_per_vox[axis])
    return eps



def infer_node_type_from_incident_edges(graph: ig.Graph, node_id: int, vessel_type_map=EDGE_NKIND_TO_LABEL) -> str:
    if "nkind" not in graph.es.attributes():
        return "unknown"
    inc = graph.incident(int(node_id))
    nk = []
    for eid in inc:
        v = graph.es[eid]["nkind"]
        if v is None:
            continue
        try:
            nk.append(int(v))
        except Exception:
            pass
    if not nk:
        return "unknown"
    n_type = Counter(nk).most_common(1)[0][0]
    return vessel_type_map.get(n_type, f"nkind_{n_type}")




# ============================================================================
#                               TOPOLOGY
# ============================================================================

def duplicated_edge_stats(G: ig.Graph) -> dict:
    pairs = [tuple(sorted(e)) for e in G.get_edgelist()]
    c = Counter(pairs)
    n_pairs_duplicated = sum(v > 1 for v in c.values())
    n_extra_edges = sum(v - 1 for v in c.values() if v > 1)
    return {
        "n_pairs_duplicated": int(n_pairs_duplicated),
        "n_extra_edges": int(n_extra_edges),
        "perc_extra_edges": float(100 * n_extra_edges / G.ecount()) if G.ecount() else 0.0,
    }

def loop_edge_stats(G: ig.Graph) -> dict:
    loop_idx = [e.index for e in G.es if e.source == e.target]
    n_loops = len(loop_idx)
    n_edges = G.ecount()
    return {
        "n_loops": int(n_loops),
        "perc_loops": float(100 * n_loops / n_edges) if n_edges else 0.0,
        "loop_indices": loop_idx,
    }



def get_edges_types(graph: ig.Graph, label_dict=EDGE_NKIND_TO_LABEL, return_dict=True):
    check_attr(graph, "nkind", "es")
    edge_types = np.asarray(graph.es["nkind"], dtype=int)
    unique, counts = np.unique(edge_types, return_counts=True)
    total = len(edge_types)

    results = {}
    print("\nEdge types:\n")
    for k, n in zip(unique, counts):
        name = label_dict.get(int(k), str(k)) if label_dict else str(k)
        perc = 100 * n / total
        print(f" - {name} (nkind={k}): {n} edges ({perc:.1f}%)")
        results[int(k)] = {"name": name, "count": int(n), "percentage": float(perc)}

    return results if return_dict else (unique, counts)


def edge_type_counts(nk, label_dict): 
    counts = {"arteriole": 0, "venule": 0, "capillary": 0, "unknown": 0} 
    for k in np.asarray(nk, int): 
        lab = str(label_dict.get(int(k), "unknown")).lower().strip() 
        if "arter" in lab: 
            counts["arteriole"] += 1 
        elif "ven" in lab: 
            counts["venule"] += 1 
        elif "cap" in lab: 
            counts["capillary"] += 1 
        else: 
            counts["unknown"] += 1 
    return counts

def get_degrees(graph: ig.Graph, threshold=4):
    deg = np.asarray(graph.degree(), dtype=int)
    graph.vs["degree"] = deg.tolist()
    mask = deg >= int(threshold)
    graph.vs["high_degree_node"] = np.where(mask, deg, 0).tolist()
    hdn_idx = np.where(mask)[0].astype(int)

    print("Unique degrees:", np.unique(deg))
    print(f"HDN (>= {threshold}): {hdn_idx.size}")
    return np.unique(deg), hdn_idx



def plot_degree_nodes_spatial(
    graph: ig.Graph,
    coords_attr="coords",
    degree_min=4,
    degree_max=None,
    by_type=True,
    s_all=2,
    s_sel=30,
    alpha_all=0.15,
    alpha_sel=0.95,
    title=None,
):
    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), dtype=int)

    if degree_max is None:
        sel = np.where(deg >= int(degree_min))[0]
        crit = f"deg ≥ {int(degree_min)}"
    else:
        sel = np.where((deg >= int(degree_min)) & (deg <= int(degree_max)))[0]
        crit = f"{int(degree_min)} ≤ deg ≤ {int(degree_max)}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s_all, c="lightgray", alpha=alpha_all, depthshade=False)

    if sel.size == 0:
        ax.set_title(title or f"No nodes with {crit}")
        plt.tight_layout()
        plt.show()
        return fig, ax

    if not by_type:
        ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], s=s_sel, c="black", alpha=alpha_sel, depthshade=False)
    else:
        labs = np.array([infer_node_type_from_incident_edges(graph, int(v)) for v in sel], dtype=object)
        col = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}

        for lab in ["arteriole", "venule", "capillary", "unknown"]:
            m = (labs == lab)
            if np.any(m):
                pts = P[sel[m]]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s_sel, c=col[lab], alpha=alpha_sel,
                           depthshade=False, label=lab)
        ax.legend(loc="best", title=f"{crit} nodes")

    ax.set_title(title or f"Spatial distribution of {crit}")
    plt.tight_layout()
    plt.show()
    return fig, ax




def distance_to_surface_stats(
    graph: ig.Graph,
    nodes,
    space="um",  # "vox" or "um"
    depth_attr_vox="distance_to_surface",
    depth_attr_um="distance_to_surface_R",
    depth_bins_um=DEFAULT_DEPTH_BINS_UM,
    res_um_per_vox=res_um_per_vox,
):
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size == 0:
        return None

    if space == "um":
        depth_attr = depth_attr_um if depth_attr_um in graph.vs.attributes() else depth_attr_vox
        check_attr(graph, depth_attr, "vs")
        vals = np.asarray(graph.vs[depth_attr], dtype=float)[nodes]
        bins = depth_bins_um
        range_key = "range_um"
    else:
        depth_attr = depth_attr_vox if depth_attr_vox in graph.vs.attributes() else depth_attr_um
        check_attr(graph, depth_attr, "vs")
        vals = np.asarray(graph.vs[depth_attr], dtype=float)[nodes]

        sx = float(res_um_per_vox[0])
        bins = []
        for label, low_um, high_um in depth_bins_um:
            low_v = float(low_um) / sx
            high_v = float(high_um) / sx if np.isfinite(high_um) else np.inf
            bins.append((label, low_v, high_v))
        range_key = "range_vox"

    out = {
        "space": space,
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
            "proportion": float(c / n) if n else 0.0,
        }

    return out




def analyze_hdn_pattern_in_box(
    graph: ig.Graph,
    box: dict,
    coords_attr="coords",
    space="um",
    degree_thr=4,
    eps_vox=2.0,
    vessel_type_map=EDGE_NKIND_TO_LABEL,
    depth_bins_um=DEFAULT_DEPTH_BINS_UM,
):
    validate_box_faces(box)
    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), int)
    hdn = np.where(deg >= int(degree_thr))[0].astype(int)

    out = {
        "space": space,
        "coords_attr": coords_attr,
        "degree_thr": int(degree_thr),
        "n_nodes": int(graph.vcount()),
        "n_hdn": int(hdn.size),
        "hdn_fraction": float(hdn.size / graph.vcount()) if graph.vcount() else 0.0,
    }

    if hdn.size == 0:
        out["spatial"] = None
        out["face_bias_hdn"] = None
        out["hdn_type_composition"] = None
        out["depth_hdn"] = None
        return out

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

    labels = np.array([infer_node_type_from_incident_edges(graph, int(v), vessel_type_map) for v in hdn], dtype=object)
    uniq, cnt = np.unique(labels, return_counts=True)
    out["hdn_type_composition"] = {str(u): {"count": int(c), "proportion": float(c / labels.size)} for u, c in zip(uniq, cnt)}

    out["depth_hdn"] = None
    # optional depth
    if ("distance_to_surface_R" in graph.vs.attributes()) or ("distance_to_surface" in graph.vs.attributes()):
        out["depth_hdn"] = distance_to_surface_stats(graph, hdn, space=space, depth_bins_um=depth_bins_um)

    eps_x = resolve_eps(eps_vox, space=space, axis=0)
    eps_y = resolve_eps(eps_vox, space=space, axis=1)
    eps_z = resolve_eps(eps_vox, space=space, axis=2)

    out["face_bias_hdn"] = {
        "x_min": float(np.mean(np.abs(Ph[:, 0] - box["xmin"]) <= eps_x)),
        "x_max": float(np.mean(np.abs(Ph[:, 0] - box["xmax"]) <= eps_x)),
        "y_min": float(np.mean(np.abs(Ph[:, 1] - box["ymin"]) <= eps_y)),
        "y_max": float(np.mean(np.abs(Ph[:, 1] - box["ymax"]) <= eps_y)),
        "z_min": float(np.mean(np.abs(Ph[:, 2] - box["zmin"]) <= eps_z)),
        "z_max": float(np.mean(np.abs(Ph[:, 2] - box["zmax"]) <= eps_z)),
    }
    out["face_bias_hdn"]["max_face_bias"] = float(max(out["face_bias_hdn"].values()))
    return out






# ============================================================================
#                               GEOMETRY
# ============================================================================

# -------------------
# Box builder in um 
# -------------------

def make_box_in_um(center_vox, box_size_um, res_um_per_vox=res_um_per_vox) -> dict:
    center_um = np.asarray(center_vox, dtype=float) * np.asarray(res_um_per_vox, dtype=float)
    box_um = np.asarray(box_size_um, dtype=float)
    half = box_um / 2.0
    return {
        "xmin": float(center_um[0] - half[0]), "xmax": float(center_um[0] + half[0]),
        "ymin": float(center_um[1] - half[1]), "ymax": float(center_um[1] + half[1]),
        "zmin": float(center_um[2] - half[2]), "zmax": float(center_um[2] + half[2]),
    }


def get_avg_length_nkind(graph: ig.Graph):
    check_attr(graph, ["length", "nkind"], "es")
    L = np.asarray(graph.es["length"], float)
    nk = np.asarray(graph.es["nkind"], int)

    print("\n- Average length by nkind:\n")
    out = {}
    for k in np.unique(nk):
        m = nk == k
        out[int(k)] = float(np.mean(L[m]))
        print(f"nkind={k} ({EDGE_NKIND_TO_LABEL.get(int(k),k)}): mean length = {out[int(k)]:.6f}")
    return out


def diameter_stats_nkind(
    graph: ig.Graph,
    label_dict=None,
    ranges=None,
    plot=True,
    title_suffix=None,
):
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
    
    print("\n- Average diameter by nkind:\n")
    for k in sorted(stats_dict.keys()):
        s = stats_dict[k]
        
        print(f"{s['name']} (nkind={k}, n={s['n']}):")
        print(f"  mean:   {s['mean']:.2f}")
        print(f"  median: {s['median']:.2f}")
        print(f"  p5–p95: {s['p5']:.2f} – {s['p95']:.2f}")
        if s["perc_in_range"] is not None:
            lo, hi = s["range"]
            print(f"  % in range ({lo}–{hi}): {s['perc_in_range']:.1f}%")
        print()

    if plot:
        base_title = "Diameter distribution by vessel type"
        if title_suffix is not None:
            base_title = f"{base_title} | {title_suffix}"
        plot_violin_box_by_category(
            diam, nkind,
            label_dict=EDGE_NKIND_TO_LABEL,
            xlabel="Vessel type",
            ylabel="Diameter",
            title=base_title
        )

    return stats_dict





# ============================================================================
#                               VESSEL DENSITY
# ============================================================================


def microsegments_from_formatted_graph(G: ig.Graph):
    check_attr(G, ["points", "nkind"], "es")
    has_dpts = ("diameters" in G.es.attributes())
    has_dedge = ("diameter" in G.es.attributes())

    mids, lens, nk, r0s, r1s = [], [], [], [], []

    for e in G.es:
        pts = np.asarray(e["points"], float)
        if pts.shape[0] < 2:
            continue

        # 1) Prefer per-point diameters if present
        d = None
        if has_dpts and e["diameters"] is not None:
            dd = np.asarray(e["diameters"], float)
            if dd.shape[0] == pts.shape[0] and np.all(np.isfinite(dd)) and np.all(dd > 0):
                d = dd

        # 2) Fallback: use scalar edge diameter repeated
        if d is None and has_dedge:
            de = e["diameter"]
            if de is not None and np.isfinite(de) and de > 0:
                d = np.full(pts.shape[0], float(de), dtype=float)

        # If still no diameter info, we cannot compute volume reliably -> skip this edge
        if d is None:
            continue

        nkind_e = int(e["nkind"]) if e["nkind"] is not None else -1

        for i in range(pts.shape[0] - 1):
            p0 = pts[i]; p1 = pts[i + 1]
            L = float(np.linalg.norm(p1 - p0))
            if not np.isfinite(L) or L <= 0:
                continue

            mids.append(((p0 + p1) * 0.5).tolist())
            lens.append(L)
            nk.append(nkind_e)

            r0s.append(0.5 * float(d[i]))
            r1s.append(0.5 * float(d[i + 1]))

    return {
        "midpoints": np.asarray(mids, float),
        "lengths": np.asarray(lens, float),
        "nkind": np.asarray(nk, int),
        "r0": np.asarray(r0s, float),
        "r1": np.asarray(r1s, float),
    }


def generate_boxes(box, box_size=100, stride=50):
    ''' creates boxes of 100 um in each axis but instead of adjaced to each
    other, the centers are shifted 50 um (stride), so there is some overlapping
    between adjacent boxes.  
   '''
    xs = np.arange(box["xmin"], box["xmax"]-box_size+1, stride)
    ys = np.arange(box["ymin"], box["ymax"]-box_size+1, stride)
    zs = np.arange(box["zmin"], box["zmax"]-box_size+1, stride)

    boxes = []

    for x in xs:
        for y in ys:
            for z in zs:
                boxes.append({
                    "xmin":x,
                    "xmax":x+box_size,
                    "ymin":y,
                    "ymax":y+box_size,
                    "zmin":z,
                    "zmax":z+box_size
                })

    return boxes


def vessel_volume_density(ms, box): # mm³/mm³
    ''' 
    calculates volume density. Takes the vessels volume (computed as cylinders in microsegment code)
    and then divides it by the total volume of the box (tissue, 400um³). 
    For compliance with literature units, they values are returned in mm³/mm³
    '''
    mids = ms["midpoints"]
    L = ms["lengths"]
    r0 = ms["r0"]
    r1 = ms["r1"]

    inside = (
        (mids[:,0]>=box["xmin"]) & (mids[:,0]<=box["xmax"]) &
        (mids[:,1]>=box["ymin"]) & (mids[:,1]<=box["ymax"]) &
        (mids[:,2]>=box["zmin"]) & (mids[:,2]<=box["zmax"])
    )

    L = L[inside]
    r0 = r0[inside]
    r1 = r1[inside]

    rmean = 0.5*(r0+r1)

    vessel_vol_um3 = np.sum(np.pi*rmean**2 * L)

    tissue_vol_um3 = (
        (box["xmax"]-box["xmin"])*
        (box["ymax"]-box["ymin"])*
        (box["zmax"]-box["zmin"])
    )

    # convert to mm³
    vessel_vol_mm3 = vessel_vol_um3 * 1e-9
    tissue_vol_mm3 = tissue_vol_um3 * 1e-9

    return vessel_vol_mm3 / tissue_vol_mm3


def vessel_length_density(ms, box):
    '''
    Total length of blood vessels within a volume.
    calculates the vessel length as the total vessel length (in mm)
    divided by the total volume of the tissue (mm³)
    '''

    mids = ms["midpoints"]
    L = ms["lengths"]

    inside = (
        (mids[:,0]>=box["xmin"]) & (mids[:,0]<=box["xmax"]) &
        (mids[:,1]>=box["ymin"]) & (mids[:,1]<=box["ymax"]) &
        (mids[:,2]>=box["zmin"]) & (mids[:,2]<=box["zmax"])
    )

    L = L[inside]

    length_um = np.sum(L)

    tissue_vol_um3 = (
        (box["xmax"]-box["xmin"])*
        (box["ymax"]-box["ymin"])*
        (box["zmax"]-box["zmin"])
    )

    length_mm = length_um * 1e-3
    tissue_vol_mm3 = tissue_vol_um3 * 1e-9

    return length_mm / tissue_vol_mm3


def graph_density_metrics(G, graph_name):
    """
    Devuelve métricas 1-number-per-graph:
      - V, E
      - E_over_V  (edges per node)
      - density_simple_undirected = 2E/(V*(V-1))  (si V>1)
      - mean_degree = 2E/V (para grafo no dirigido)
    """
    V = int(G.vcount())
    E = int(G.ecount())

    graph_density = (2.0 * E / (V * (V - 1))) if V > 1 else np.nan

    return {
        "graph": graph_name,
        "V": V,
        "E": E,
        "graph_density": float(graph_density),
    }


# ======================================================================
#  SIMULATION (BC NODES)
# ======================================================================


# ======================================================================
# BC detection (formatted)
# ======================================================================

def debug_face_plane_counts(G: ig.Graph, box: dict, coords_attr="coords", eps_vox=2.0, space="um"):
    '''How many vertices lie close to each face of the box
    For each face, checks min distance from any node (does not need to be BC) 
    to that plane and how many nodes lie within the tolerance eps
    '''
    validate_box_faces(box)
    P = np.asarray(G.vs[coords_attr], float)
    print("\n=== DEBUG: vertices close to each face plane ===")
    print("coords bounds:", P.min(axis=0), "->", P.max(axis=0))
    print("box:", box)

    for face, (axis, key) in FACES_DEF.items():
        eps = resolve_eps(eps_vox, space=space, axis=axis)
        val = float(box[key])
        dist = np.abs(P[:, axis] - val)
        print(face, "min_dist", float(dist.min()), "n_within_eps", int((dist <= eps).sum()))



def bc_nodes_on_face_plane(graph: ig.Graph, axis: int, value: float, box: dict, coords_attr="coords", eps=5.0):
    C = get_coords(graph, coords_attr).astype(float)
    bounds = np.array([
        [box["xmin"], box["xmax"]],
        [box["ymin"], box["ymax"]],
        [box["zmin"], box["zmax"]],
    ], dtype=float)

    on_plane = np.abs(C[:, axis] - float(value)) <= float(eps)
    other_axes = [i for i in range(3) if i != axis]
    inside = (
        (C[:, other_axes[0]] >= bounds[other_axes[0], 0]) &
        (C[:, other_axes[0]] <= bounds[other_axes[0], 1]) &
        (C[:, other_axes[1]] >= bounds[other_axes[1], 0]) &
        (C[:, other_axes[1]] <= bounds[other_axes[1], 1])
    )
    return np.where(on_plane & inside)[0].astype(int)

def analyze_bc_faces(
    graph: ig.Graph,
    box: dict,
    coords_attr="coords",
    space="um",
    eps_vox=2.0,          # ALWAYS specified in vox, converted if space="um"
    degree_thr=4,
    mode="auto",          # "border" | "plane" | "auto" -> border after cutting 
                          # (is_border/border_face present), plane checks if node is close in eps to the face (when "border" attr not present)
    return_node_ids=False,
):
    validate_box_faces(box)
    deg = np.asarray(graph.degree(), dtype=int)

    has_border = ("is_border" in graph.vs.attributes()) and ("border_face" in graph.vs.attributes())
    if mode == "auto":
        mode = "border" if has_border else "plane"

    out = {}
    for face, (axis, key) in FACES_DEF.items():
        if mode == "border":
            if not has_border:
                raise ValueError("mode='border' requested but graph lacks vs['is_border']/vs['border_face'].")
            isb = np.asarray(graph.vs["is_border"], int) == 1
            bf = np.asarray(graph.vs["border_face"], object)
            nodes = np.where(isb & (bf == face))[0].astype(int)
        else:
            eps = resolve_eps(eps_vox, space=space, axis=axis)
            nodes = bc_nodes_on_face_plane(graph, axis, float(box[key]), box, coords_attr=coords_attr, eps=eps)

        n = int(nodes.size)
        deg_counts = Counter(deg[nodes]) if n else Counter()
        high_mask = (deg[nodes] >= int(degree_thr)) if n else np.array([], dtype=bool)
        high_n = int(high_mask.sum()) if n else 0

        face_res = {
            "mode": mode,
            "count": n,
            "degree_counts": dict(deg_counts),
            "high_degree_count": high_n,
            "high_degree_percent": float(100.0 * high_n / n) if n else 0.0,
        }

        if "nkind" in graph.es.attributes() and n:
            labels = [infer_node_type_from_incident_edges(graph, int(v)) for v in nodes]
            tc = Counter(labels)
            face_res["type_counts"] = dict(tc)
            face_res["type_percent"] = {k: 100.0 * v / n for k, v in tc.items()}
        else:
            face_res["type_counts"] = {}
            face_res["type_percent"] = {}

        # optional depth stats
        if ("distance_to_surface_R" in graph.vs.attributes()) or ("distance_to_surface" in graph.vs.attributes()):
            face_res["distance_to_surface_stats"] = distance_to_surface_stats(graph, nodes, space=space)
        else:
            face_res["distance_to_surface_stats"] = None

        if return_node_ids:
            face_res["nodes"] = nodes
            face_res["high_degree_nodes"] = nodes[high_mask] if n else np.array([], dtype=int)

        out[face] = face_res

    return out

def bc_faces_table(res: dict, box_name="Box") -> pd.DataFrame:
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


# -----------------------------------------
# BC Plots
# -----------------------------------------
BC_FACE_COLORS = {
    "z_max": "#9ECAE1",
    "y_max": "#A1D99B",
    "x_min": "#FDD0A2",
    "x_max": "#FCBBA1",
    "y_min": "#C7C1E3",
    "z_min": "#D9D9D9",
}


def plot_bc_cube_net(
    res: dict,
    title="BC composition per face (cube net)",
    face_alpha=0.65,
    fontsize=10,
    pct_decimals=1,
    show_unknown=False,
    show_high_degree=True,
    face_colors=None,          # <-- NEW (optional override)
):
    layout = {
        "y_max": (1, 2),
        "x_min": (0, 1),
        "z_min": (1, 1),
        "x_max": (2, 1),
        "z_max": (3, 1),
        "y_min": (1, 0),
    }

    if face_colors is None:
        face_colors = BC_FACE_COLORS

    vessel_order = ["arteriole", "venule", "capillary"]
    if show_unknown:
        vessel_order.append("unknown")

    def face_text(face):
        if face not in res:
            return f"{face}\n(no data)"

        n = int(res[face].get("count", 0))
        tp = res[face].get("type_percent", {}) or {}
        tc = res[face].get("type_counts", {}) or {}

        lines = [f"{face}", f"{n} total nodes"]
        for k in vessel_order:
            if k in tp and float(tp[k]) > 0:
                lines.append(f"{k}: {int(tc.get(k,0))} ({float(tp[k]):.{pct_decimals}f}%)")

        if show_high_degree:
            pct_hd = float(res[face].get("high_degree_percent", 0.0))
            n_hd = int(res[face].get("high_degree_count", 0))
            lines.append(f"HD(≥4): {n_hd} ({pct_hd:.{pct_decimals}f}%)")

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


# same as before but summarizes the 3 boxes results, instead of classifying by face and nkind. 
def plot_cube_net_counts_per_face(
    bc_all,
    out_path=None,
    title="BC nodes per face (counts per box)",
    face_col="Face",
    graph_col="graph",
    value_col="BC nodes",
    box_order=("HPC_1", "HPC_2", "HPC_3"),
):
    """
    Draws a single cube-net (like your example) but text is:
      HPC_1: N
      HPC_2: N
      HPC_3: N
    per face.

    bc_all must have columns: Face, graph, BC nodes
    """
    df = bc_all.copy()

    # --- normalize face labels (robust)
    if face_col not in df.columns:
        raise ValueError(f"'{face_col}' not in bc_all. Columns: {df.columns.tolist()}")
    if graph_col not in df.columns:
        raise ValueError(f"'{graph_col}' not in bc_all. Columns: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not in bc_all. Columns: {df.columns.tolist()}")

    df[face_col] = df[face_col].astype(str).str.strip().str.lower()
    df[face_col] = df[face_col].replace({
        "xmin":"x_min","xmax":"x_max",
        "ymin":"y_min","ymax":"y_max",
        "zmin":"z_min","zmax":"z_max",
        "x min":"x_min","x max":"x_max",
        "y min":"y_min","y max":"y_max",
        "z min":"z_min","z max":"z_max",
    })

    df[graph_col] = df[graph_col].astype(str).str.strip()
    df[value_col] = np.asarray(pd.to_numeric(df[value_col], errors="coerce").fillna(0.0), float)

    faces = ["x_min","x_max","y_min","y_max","z_min","z_max"]

    # --- table face x graph
    tab = (
        df.groupby([face_col, graph_col], as_index=False)[value_col].sum()
          .pivot(index=face_col, columns=graph_col, values=value_col)
          .fillna(0.0)
    )

    # guarantee faces + box_order exist
    for f in faces:
        if f not in tab.index:
            tab.loc[f] = 0.0
    tab = tab.reindex(index=faces)

    for g in box_order:
        if g not in tab.columns:
            tab[g] = 0.0
    tab = tab.reindex(columns=list(box_order))

    # --- cube-net layout positions (same as your example)
    # coordinates are in "square units"
    pos = {
        "y_max": (1, 2),
        "z_min": (1, 1),
        "y_min": (1, 0),
        "x_min": (0, 1),
        "x_max": (2, 1),
        "z_max": (3, 1),
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    # draw squares + text
    for face in faces:
        x, y = pos[face]
        # square
        rect = plt.Rectangle(
            (x, y), 1, 1,
            facecolor=BC_FACE_COLORS.get(face, "#dddddd"),
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)

        # text lines
        lines = [face]
        for g in box_order:
            n = int(round(float(tab.loc[face, g])))
            lines.append(f"{g}: {n} nodes")

        ax.text(
            x + 0.5, y + 0.5,
            "\n".join(lines),
            ha="center", va="center",
            fontsize=9
        )

    # set limits
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 3.4)

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print("Saved:", out_path)

    plt.show()




def plot_bc_3_cubes_tinted(
    G: ig.Graph, box: dict,
    coords_attr="coords",
    space="um",
    eps_vox=2.0,
    elev=18, azim=35,
    face_alpha=0.30,
    point_alpha=0.85,
    point_size=8,
    sample_max=20000,
    mode="auto",   # border/plane/auto
    face_colors=None,         # <-- NEW (optional override)
    vessel_colors=None,       # <-- optional override if you want
):
    validate_box_faces(box)
    coords = get_coords(G, coords_attr).astype(float)

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

    has_border = ("is_border" in G.vs.attributes()) and ("border_face" in G.vs.attributes())
    if mode == "auto":
        mode = "border" if has_border else "plane"

    face_nodes = {}
    for face, (axis, key) in FACES_DEF.items():
        if mode == "border":
            isb = np.asarray(G.vs["is_border"], int) == 1
            bf = np.asarray(G.vs["border_face"], object)
            ids = np.where(isb & (bf == face))[0].astype(int)
        else:
            eps = resolve_eps(eps_vox, space=space, axis=axis)
            ids = bc_nodes_on_face_plane(G, axis, float(box[key]), box, coords_attr=coords_attr, eps=eps)

        if sample_max is not None and ids.size > sample_max:
            ids = np.random.choice(ids, size=int(sample_max), replace=False)
        face_nodes[face] = ids

    # default node colors (can override via vessel_colors=...)
    if vessel_colors is None:
        vessel_colors = {"arteriole":"red","venule":"blue","capillary":"gray","unknown":"black"}

    # IMPORTANT: use SAME face colors as cube-net
    if face_colors is None:
        face_colors = BC_FACE_COLORS

    def draw_panel(ax, faces_subset, title_panel):
        for a,b in edges:
            ax.plot([corners[a,0], corners[b,0]],
                    [corners[a,1], corners[b,1]],
                    [corners[a,2], corners[b,2]], linewidth=1.0)

        polys = [face_polys[f] for f in faces_subset]
        cols = [face_colors.get(f, "lightgray") for f in faces_subset]
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.6, alpha=face_alpha)
        ax.add_collection3d(pc)

        ids = np.unique(np.concatenate([face_nodes[f] for f in faces_subset]))
        if ids.size:
            pts = coords[ids]
            labs = np.array([infer_node_type_from_incident_edges(G, int(v)) for v in ids], dtype=object)
            for lab, col in vessel_colors.items():
                m = (labs == lab)
                if np.any(m):
                    ax.scatter(pts[m,0], pts[m,1], pts[m,2],
                               s=point_size, alpha=point_alpha, color=col)

        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
        ax.set_box_aspect((xmax-xmin, ymax-ymin, zmax-zmin))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title_panel)

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    draw_panel(ax1, ("x_min","z_min"), f"View A ({mode})")
    draw_panel(ax2, ("x_max","z_max"), f"View B ({mode})")
    draw_panel(ax3, ("y_min","y_max"), f"View C ({mode})")

    handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=7,
               markerfacecolor=vessel_colors[k], markeredgecolor='none', label=k)
        for k in ["arteriole","venule","capillary","unknown"]
    ]
    ax3.legend(handles=handles, title="Vessel type", loc="upper left")
    plt.tight_layout()
    plt.show()







# ======================================================================
# Resilience
# ======================================================================

def nodes_by_label(graph):
    out = {"arteriole": [], "venule": [], "capillary": [], "unknown": []}

    for v in graph.vs:
        lab = infer_node_type_from_incident_edges(graph, v.index)
        out[lab].append(v.index)

    return {k: np.array(v, dtype=int) for k, v in out.items()}

def _av_sets(graph):
    nodes = nodes_by_label(graph)
    return nodes["arteriole"], nodes["venule"]

def shortest_av_paths(graph):
    A, V = _av_sets(graph)
    paths = []
    # search paths in all AxV combination
    for a in A:
        for v in V:
            p = graph.get_shortest_paths(a, to=v)[0]
            if len(p) > 1:
                paths.append(p)
    return paths


def av_path_stats(graph, graph_name, paths):
    A, V = _av_sets(graph)

    pairs_searched = len(A) * len(V)
    pairs_with_path = len(paths)
    all_connected = pairs_with_path == pairs_searched

    return {
        "graph": graph_name,
        "arterioles": len(A),
        "venules": len(V),
        "pairs_searched": pairs_searched,
        "pairs_with_path": pairs_with_path,
        "all_connected": all_connected
    }

'''
# export a few paths for paraview visualization
def sample_paths(paths, n=20):
    step = max(1, len(paths) // n)
    return paths[::step][:n]
'''
# takes the first n paths generated. Normally they come from the same
# artery, so the path will be very similar. 

def sample_paths(paths, n): 
    if len(paths) <= n: 
        return paths 
    return paths[:n]

import pandas as pd
import vtk

def export_paths_vtp(graph, paths, filename, coords_attr="coords"):
    P = graph.vs[coords_attr]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    pid_array = vtk.vtkIntArray()
    pid_array.SetName("path_id")

    point_id = 0

    for pid, path in enumerate(paths):
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(path))

        for i, v in enumerate(path):
            x,y,z = P[v]
            points.InsertNextPoint(x,y,z)
            polyline.GetPointIds().SetId(i, point_id)

            pid_array.InsertNextValue(pid)

            point_id += 1

        lines.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().AddArray(pid_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def induced_subgraph_box(graph: ig.Graph, box: dict, coords_attr="coords", node_eps=0.0):
    validate_box_faces(box)
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
    sub_to_orig = np.asarray(keep, dtype=int)
    orig_to_sub = {int(o): int(i) for i, o in enumerate(sub_to_orig)}
    return sub, sub_to_orig, orig_to_sub

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# TODO: THIS COULD BE DELETED IF I AM DOING VISUALIZATION IN PARAVIEW
def av_paths_in_box(graph: ig.Graph, box: dict, coords_attr="coords", node_eps=0.0):
    sub, sub_to_orig, _ = induced_subgraph_box(graph, box, coords_attr=coords_attr, node_eps=node_eps)
    if sub is None or sub.ecount() == 0:
        return []

    A_sub, V_sub = _av_sets(sub)
    if A_sub.size == 0 or V_sub.size == 0:
        return []

    paths_sub = shortest_av_paths(sub, A=A_sub, V=V_sub)
    return [[int(sub_to_orig[i]) for i in path] for path in paths_sub]



def plot_av_paths_in_box(
    graph: ig.Graph,
    box: dict,
    paths_orig,
    coords_attr="coords",
    node_eps=0.0,
    sample_edges=5000,
):
    validate_box_faces(box)
    P = get_coords(graph, coords_attr)

    sub, sub_to_orig, _ = induced_subgraph_box(graph, box, coords_attr=coords_attr, node_eps=node_eps)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if sub is not None and sub.ecount() > 0:
        bg_pairs = np.asarray([(sub_to_orig[u], sub_to_orig[v]) for (u, v) in sub.get_edgelist()], dtype=int)
        if bg_pairs.shape[0] > sample_edges:
            idx = np.random.choice(bg_pairs.shape[0], sample_edges, replace=False)
            bg_pairs = bg_pairs[idx]

        for (u, v) in bg_pairs:
            ax.plot([P[u, 0], P[v, 0]],
                    [P[u, 1], P[v, 1]],
                    [P[u, 2], P[v, 2]],
                    alpha=0.1, linewidth=0.6)

    for path in paths_orig:
        for a, b in zip(path[:-1], path[1:]):
            ax.plot([P[a, 0], P[b, 0]],
                    [P[a, 1], P[b, 1]],
                    [P[a, 2], P[b, 2]],
                    linewidth=2.5)

    ax.set_title(f"A→V shortest paths (n={len(paths_orig)})")
    plt.tight_layout()
    plt.show()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def max_edge_disjoint_av(graph: ig.Graph):
    A, V = _av_sets(graph)
    if A.size == 0 or V.size == 0:
        return {"n_edge_disjoint_av": 0, "nA": int(A.size), "nV": int(V.size), "paths": []}

    # directed graph
    D = graph.as_directed(mutual=True) if not graph.is_directed() else graph.copy()

    # source / sink
    s = D.vcount()
    t = D.vcount() + 1
    D.add_vertices(2)

    # connect source→A and V→sink
    extra_edges = [(s, int(a)) for a in A] + [(int(v), t) for v in V]
    D.add_edges(extra_edges)

    BIG = float(max(1, D.ecount()))
    extra_caps = [BIG] * len(extra_edges)

    D.es["cap"] = [1.0] * (D.ecount() - len(extra_caps)) + extra_caps

    mf = D.maxflow(s, t, capacity="cap")

    # -------- reconstruct paths ----------
    flow_edges = [e.tuple for e, f in zip(D.es, mf.flow) if f > 0]

    # remove source/sink edges
    flow_edges = [(u, v) for (u, v) in flow_edges if u != s and v != t]

    # adjacency
    adj = {}
    for u, v in flow_edges:
        adj.setdefault(u, []).append(v)

    paths = []

    Vset = set(V)

    # start from arterioles that have outgoing flow
    for a in A:
        if a not in adj:
            continue

        stack = [(a, [a])]

        while stack:
            node, path = stack.pop()

            if node in Vset:
                paths.append(path)
                break

            for nxt in adj.get(node, []):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))

    return {
        "n_edge_disjoint_av": int(round(mf.value)),
        "nA": int(A.size),
        "nV": int(V.size),
        "paths": paths
    }





# ======================================================================
# Plotting: general
# ======================================================================

def plot_bar_by_category_general(
    categ, attribute_toplot, label_dict=None,
    xlabel="Category", ylabel="Value",
    title="Category statistics",
    show_values=True, value_fmt="{:.2f}"
):
    labels = [label_dict.get(c, c) if label_dict else c for c in categ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, attribute_toplot, edgecolor="black")

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

    if layout == "horizontal":
        fig, axes = plt.subplots(1, len(cats), figsize=(4 * len(cats), 4), sharex=True)
    else:
        fig, axes = plt.subplots(len(cats), 1, figsize=(6, 3 * len(cats)), sharex=True)

    if len(cats) == 1:
        axes = [axes]

    if main_title is None:
        main_title = f"{variable_name} distribution by {category_name}"
    fig.suptitle(main_title)

    for ax, c in zip(axes, cats):
        subset = values[category == c]
        n = int(subset.size)
        pct = (100.0 * n / N_total) if N_total else 0.0

        color = get_vessel_color(c)
        ax.hist(subset, bins=edges, density=density, alpha=0.7, color = color)

        if show_mean and len(subset):
            mean_val = subset.mean()
            ax.axvline(mean_val, linestyle="--", color = color)
            ax.legend([f"Mean = {mean_val:.2f}"])

        name = label_dict.get(c, str(c)) if label_dict else str(c)
        ax.set_title(f"{name} (n={n}: {pct:.1f}%)")
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

    parts = ax.violinplot(data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        name = labels[i].lower()
        color = VESSEL_COLORS.get(name, "lightgray")
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(color="aqua", linewidth=2),
    )

    for i, subset in enumerate(data):
        x_center = positions[i]
        jitter = np.random.normal(x_center, 0.03, size=len(subset))
        ax.scatter(jitter, subset, color="black", s=12, alpha=0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# gives back a set of boxplots for the data stored in the df, for each column
def plot_boxplot_by_graph(df_long, value_col, title, ylabel, graphs_order=("HPC_1", "HPC_2", "HPC_3")):
    groups, labels = [], []
    for g in graphs_order:
        if g not in df_long["graph"].unique():
            continue
        x = df_long.loc[df_long["graph"] == g, value_col].to_numpy(float)
        x = x[np.isfinite(x)]
        groups.append(x)
        labels.append(g)

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()

# =========================================================================
# COMBINED PLOTS
# =========================================================================


def diameter_length_overlay_by_type(dl, bins=40, graphs_order=("HPC_1", "HPC_2", "HPC_3"), box_label=""):
    for t in sorted(dl["type"].unique()):
        sub = dl[dl["type"] == t].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{t} | Diameter & Length distributions | {box_label}", fontsize=12, fontweight="bold")

        ax = axes[0]
        for g in graphs_order:
            x = sub.loc[sub["graph"] == g, "diameter_vox"].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size:
                ax.hist(x, bins=bins, density=True, alpha=0.45, label=g)
        ax.set_title("Diameter")
        ax.set_xlabel("Diameter (vox)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

        ax = axes[1]
        for g in graphs_order:
            x = sub.loc[sub["graph"] == g, "length_um"].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size:
                ax.hist(x, bins=bins, density=True, alpha=0.45, label=g)
        ax.set_title("Length")
        ax.set_xlabel("Length (µm)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

        plt.tight_layout()
        plt.show()


# ---- PLot boxplots per box of ROI
def plot_grouped_boxplot_types_per_graph(
    df_long,
    value_col="diameter_vox",
    type_col="type",
    graphs_order=("HPC_1", "HPC_2", "HPC_3"),
    types_order=("arteriole", "venule", "capillary"),
    type_colors=None,
    title="Diameter by vessel type within each box",
    ylabel="Diameter (vox)"
):
    """
    One figure:
      x-axis = graphs (HPC_1, HPC_2, HPC_3)
      For each graph, 3 boxplots (arteriole/venule/capillary) side-by-side.
    """
    if df_long is None or df_long.empty:
        print("[grouped boxplot] df_long is empty -> nothing to plot.")
        return

    if type_colors is None:
        type_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray"}

    df = df_long.copy()
    df[type_col] = df[type_col].astype(str).str.lower().str.strip()
    df[type_col] = df[type_col].replace({
        "artery": "arteriole",
        "arterial": "arteriole",
        "vein": "venule",
        "venous": "venule",
        "cap": "capillary",
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    base = np.arange(len(graphs_order), dtype=float)

    offsets = {types_order[0]: -0.28, types_order[1]: 0.00, types_order[2]: 0.28}
    width = 0.22

    for t in types_order:
        data, pos = [], []
        for i, g in enumerate(graphs_order):
            x = df.loc[(df["graph"] == g) & (df[type_col] == t), value_col].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            data.append(x)
            pos.append(base[i] + offsets[t])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width,
            patch_artist=True,
            showfliers=False
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(type_colors.get(t, "lightgray"))
            patch.set_alpha(0.70)
        for k in ["whiskers", "caps", "medians"]:
            for line in bp[k]:
                line.set_color("black")

    ax.set_xticks(base)
    ax.set_xticklabels(list(graphs_order))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")

    handles = [Patch(facecolor=type_colors.get(t, "lightgray"), edgecolor="black", label=t) for t in types_order]
    ax.legend(handles=handles, title="Vessel type", frameon=False)

    plt.tight_layout()
    plt.show()


# =========================================================================
# SUMARY TABLES
# =========================================================================

def make_bc_face_table_from_bc_all(bc_all, face_col="Face", graph_col="graph", value_col="BC nodes",
                                   face_order=("x_min","x_max","y_min","y_max","z_min","z_max"),
                                   graph_order=None):
    """
    bc_all: concatenate of all tables
    Returns df_total with index=faces y columns=graphs, valores=BC nodes (int).
    """
    df = bc_all.copy()

    if face_col not in df.columns:
        raise ValueError(f"'{face_col}' not in bc_all.columns. Available: {df.columns.tolist()}")
    if graph_col not in df.columns:
        raise ValueError(f"'{graph_col}' not in bc_all.columns. Available: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not in bc_all.columns. Available: {df.columns.tolist()}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    # Suma por (Face, graph)
    tab = df.groupby([face_col, graph_col], as_index=False)[value_col].sum()

    # Pivot a faces x graphs
    pivot = tab.pivot(index=face_col, columns=graph_col, values=value_col).fillna(0.0)

    # Orden
    if graph_order is not None:
        cols = [g for g in graph_order if g in pivot.columns]
        pivot = pivot.reindex(columns=cols)
    if face_order is not None:
        rows = [f for f in face_order if f in pivot.index]
        pivot = pivot.reindex(index=rows)

    return pivot.round(0).astype(int)
