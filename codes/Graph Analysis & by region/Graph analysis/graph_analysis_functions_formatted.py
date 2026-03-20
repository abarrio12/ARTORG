"""
Vascular graph analysis module for FORMATTED GAIA graphs.

Expected input
--------------
A formatted igraph.Graph loaded from a pickle.

Typical required attributes
---------------------------
Vertices:
  - vs["coords"] : (x, y, z)
  - optional:
      vs["is_border"]
      vs["border_face"]
      vs["distance_to_surface"]
      vs["distance_to_surface_R"]

Edges:
  - es["points"]
  - es["length"]
  - es["diameter"]
  - es["nkind"] : 2 / 3 / 4 = arteriole / venule / capillary
  - optional:
      es["diameters"], es["lengths2"], ...

Main blocks
-----------
1. Constants
2. IO
3. Basic helpers
4. Topology / geometry
5. Density
6. BC nodes
7. A-V paths / resilience
8. Statistical helpers
9. Plotting

Author: Ana Barrio
Cleaned version
"""

import os
import pickle
from collections import Counter, defaultdict
from itertools import combinations

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import ttest_ind, ttest_rel


# ======================================================================
# Constants
# ======================================================================

ARTERY = 2
VEIN = 3
CAPILLARY = 4

EDGE_NKIND_TO_LABEL = {
    ARTERY: "arteriole",
    VEIN: "venule",
    CAPILLARY: "capillary",
}

FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

FACES_DEF = {
    "x_min": (0, "xmin"),
    "x_max": (0, "xmax"),
    "y_min": (1, "ymin"),
    "y_max": (1, "ymax"),
    "z_min": (2, "zmin"),
    "z_max": (2, "zmax"),
}

res_um_per_vox = np.array([1.625, 1.625, 2.5], dtype=float)

VESSEL_COLORS = {
    "arteriole": "#ff2828",
    "venule": "#0072c4",
    "capillary": "#7f7f7f",
    "unknown": "#000000",
}

BC_FACE_COLORS = {
    "z_max": "#9ECAE1",
    "y_max": "#A1D99B",
    "x_min": "#FDD0A2",
    "x_max": "#FCBBA1",
    "y_min": "#C7C1E3",
    "z_min": "#D9D9D9",
}

DEFAULT_DEPTH_BINS_UM = 10


# ======================================================================
# IO
# ======================================================================

def load_graph(path: str) -> ig.Graph:
    """Load a graph pickle. Accepts either Graph or {'graph': Graph}."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, ig.Graph):
        return obj

    if isinstance(obj, dict) and "graph" in obj and isinstance(obj["graph"], ig.Graph):
        return obj["graph"]

    raise TypeError(f"File does not contain an igraph.Graph: {path} (type={type(obj)})")


def save_graph(graph: ig.Graph, path: str) -> None:
    """Save graph as pickle."""
    graph.write_pickle(path)
    print("Saved:", path)


def keep_giant_component(graph: ig.Graph) -> ig.Graph:
    """Keep only the largest connected component."""
    comps = graph.components()
    if len(comps) <= 1:
        return graph
    keep = np.asarray(comps[np.argmax(comps.sizes())], dtype=int)
    return graph.induced_subgraph(keep)


# ======================================================================
# Basic helpers
# ======================================================================

def check_attr(graph: ig.Graph, names, where="vs"):
    """Raise an error if required attributes are missing."""
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
    """Check that a box has xmin/xmax/ymin/ymax/zmin/zmax."""
    req = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    missing = [k for k in req if k not in box]
    if missing:
        raise ValueError(f"box missing keys: {missing}. Required: {list(req)}")


def get_coords(graph: ig.Graph, coords_attr="coords") -> np.ndarray:
    """Return vertex coordinates as an Nx3 array."""
    check_attr(graph, coords_attr, "vs")
    coords = np.asarray(graph.vs[coords_attr], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got {coords.shape}.")
    return coords


def resolve_eps(eps_vox=2.0, space="um", axis=0, res_um_per_vox=res_um_per_vox) -> float:
    """Convert a voxel tolerance to the graph space if needed."""
    eps = float(eps_vox)
    if space == "um":
        eps *= float(res_um_per_vox[axis])
    return eps


def get_vessel_color(k):
    """Map nkind code to display color."""
    label = EDGE_NKIND_TO_LABEL.get(k, "unknown")
    return VESSEL_COLORS.get(label, "black")


def infer_node_type_from_incident_edges(graph: ig.Graph, node_id: int) -> str:
    """Infer node type from the nkind values of incident edges."""
    if "nkind" not in graph.es.attributes():
        return "unknown"

    inc = graph.incident(int(node_id))
    if len(inc) == 0:
        return "unknown"

    nk = set()
    for eid in inc:
        val = graph.es[eid]["nkind"]
        if val is None:
            continue
        try:
            nk.add(int(val))
        except Exception:
            pass

    if ARTERY in nk:
        return "arteriole"
    if VEIN in nk:
        return "venule"
    if CAPILLARY in nk:
        return "capillary"
    return "unknown"


def dataframe_to_table_figure(df, title=None, figsize=(8, 2.5), round_decimals=3):
    """Render a dataframe as a simple matplotlib table."""
    df_show = df.copy()

    for col in df_show.columns:
        if pd.api.types.is_numeric_dtype(df_show[col]):
            df_show[col] = df_show[col].round(round_decimals)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    if title is not None:
        ax.set_title(title, pad=12)

    plt.tight_layout()
    plt.show()
    return fig, ax


def _finite(x):
    """Keep only finite values."""
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _normalize_types(df):
    """Normalize vessel type names in long tables."""
    d = df.copy()
    d["type"] = d["type"].astype(str).str.lower().str.strip()
    d["type"] = d["type"].replace({
        "artery": "arteriole",
        "arterial": "arteriole",
        "vein": "venule",
        "venous": "venule",
        "cap": "capillary",
    })
    return d


def delta_median_pct(medians):
    """Percent range between medians, normalized by the median of medians."""
    medians = _finite(medians)
    if medians.size < 2:
        return np.nan
    m = float(np.median(medians))
    if abs(m) < 1e-12:
        return np.nan
    return 100.0 * (float(np.max(medians)) - float(np.min(medians))) / m


def removed_by_type(graph, path):
    """Return how much of a path length belongs to artery / vein / capillary."""
    eids = path_to_edge_ids(graph, path)
    if len(eids) == 0:
        return {"art_um": np.nan, "ven_um": np.nan, "cap_um": np.nan}

    nk = np.asarray(graph.es[eids]["nkind"], dtype=int)
    L = np.asarray(graph.es[eids]["length"], dtype=float)

    return {
        "art_um": float(np.sum(L[nk == ARTERY])),
        "ven_um": float(np.sum(L[nk == VEIN])),
        "cap_um": float(np.sum(L[nk == CAPILLARY])),
    }


# ======================================================================
# Topology / geometry
# ======================================================================

def duplicated_edge_stats(graph: ig.Graph) -> dict:
    """Count duplicated undirected edges."""
    pairs = [tuple(sorted(e)) for e in graph.get_edgelist()]
    c = Counter(pairs)
    n_pairs_duplicated = sum(v > 1 for v in c.values())
    n_extra_edges = sum(v - 1 for v in c.values() if v > 1)
    return {
        "n_pairs_duplicated": int(n_pairs_duplicated),
        "n_extra_edges": int(n_extra_edges),
        "perc_extra_edges": float(100 * n_extra_edges / graph.ecount()) if graph.ecount() else 0.0,
    }


def loop_edge_stats(graph: ig.Graph) -> dict:
    """Count self-loops."""
    loop_idx = [e.index for e in graph.es if e.source == e.target]
    n_loops = len(loop_idx)
    n_edges = graph.ecount()
    return {
        "n_loops": int(n_loops),
        "perc_loops": float(100 * n_loops / n_edges) if n_edges else 0.0,
        "loop_indices": loop_idx,
    }


def get_edges_types(graph: ig.Graph, label_dict=EDGE_NKIND_TO_LABEL, return_dict=True):
    """Print and return edge type counts."""
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
    """Count arteriole / venule / capillary / unknown labels."""
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
    """Attach degree attributes and return high-degree node indices."""
    deg = np.asarray(graph.degree(), dtype=int)
    graph.vs["degree"] = deg.tolist()
    mask = deg >= int(threshold)
    graph.vs["high_degree_node"] = np.where(mask, deg, 0).tolist()
    hdn_idx = np.where(mask)[0].astype(int)

    print("Unique degrees:", np.unique(deg))
    print(f"HDN (>= {threshold}): {hdn_idx.size}")
    return np.unique(deg), hdn_idx


def make_box_in_um(center_vox, box_size_um, res_um_per_vox=res_um_per_vox) -> dict:
    """Build a box in um from a center given in voxels."""
    center_um = np.asarray(center_vox, dtype=float) * np.asarray(res_um_per_vox, dtype=float)
    box_um = np.asarray(box_size_um, dtype=float)
    half = box_um / 2.0
    return {
        "xmin": float(center_um[0] - half[0]), "xmax": float(center_um[0] + half[0]),
        "ymin": float(center_um[1] - half[1]), "ymax": float(center_um[1] + half[1]),
        "zmin": float(center_um[2] - half[2]), "zmax": float(center_um[2] + half[2]),
    }


def get_avg_length_nkind(graph: ig.Graph):
    """Mean edge length by nkind."""
    check_attr(graph, ["length", "nkind"], "es")
    L = np.asarray(graph.es["length"], float)
    nk = np.asarray(graph.es["nkind"], int)

    print("\n- Average length by nkind:\n")
    out = {}
    for k in np.unique(nk):
        m = nk == k
        out[int(k)] = float(np.mean(L[m]))
        print(f"nkind={k} ({EDGE_NKIND_TO_LABEL.get(int(k), k)}): mean length = {out[int(k)]:.6f}")
    return out


def diameter_stats_nkind(graph: ig.Graph, label_dict=None, ranges=None, plot=True, title_suffix=None):
    """Diameter summary by vessel type."""
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
            diam,
            nkind,
            label_dict=EDGE_NKIND_TO_LABEL,
            xlabel="Vessel type",
            ylabel="Diameter",
            title=base_title,
        )

    return stats_dict


def major_components_from_edge_code(graph, target_code, abs_thresh=10, rel_thresh=0.20):
    """
    Find connected components for a given vessel code and mark 'major' trees.
    """
    g = graph.copy()

    if "orig_eid" not in g.es.attributes():
        g.es["orig_eid"] = list(range(g.ecount()))

    nk = np.asarray(g.es["nkind"], int)
    keep_eids = np.where(nk == int(target_code))[0].tolist()

    if len(keep_eids) == 0:
        comp_df = pd.DataFrame(columns=[
            "component_id", "n_nodes", "n_edges",
            "n_branch_nodes", "edge_threshold", "is_major", "major_tree_id"
        ])
        edge_df = pd.DataFrame(columns=[
            "orig_eid", "component_id", "is_major", "major_tree_id", "target_code"
        ])
        return comp_df, edge_df

    sg = g.subgraph_edges(keep_eids, delete_vertices=True)
    comps = sg.components(mode="weak") if sg.is_directed() else sg.components()

    rows = []
    for cid, vids in enumerate(comps, start=1):
        sub = sg.induced_subgraph(vids)
        n_edges = int(sub.ecount())
        deg = sub.degree()
        n_branch_nodes = int(sum(d >= 3 for d in deg))

        rows.append({
            "component_id": cid,
            "n_nodes": int(sub.vcount()),
            "n_edges": n_edges,
            "n_branch_nodes": n_branch_nodes,
        })

    comp_df = pd.DataFrame(rows)

    if comp_df.empty:
        edge_df = pd.DataFrame(columns=[
            "orig_eid", "component_id", "is_major", "major_tree_id", "target_code"
        ])
        return comp_df, edge_df

    max_edges = int(comp_df["n_edges"].max())
    th = max(abs_thresh, rel_thresh * max_edges)

    comp_df["edge_threshold"] = th
    comp_df["is_major"] = (
        (comp_df["n_edges"] >= th) &
        (comp_df["n_branch_nodes"] >= 1)
    )

    comp_df = comp_df.sort_values("n_edges", ascending=False).reset_index(drop=True)

    major_tree_ids = {}
    next_id = 1
    for _, row in comp_df.iterrows():
        cid = int(row["component_id"])
        if bool(row["is_major"]):
            major_tree_ids[cid] = next_id
            next_id += 1
        else:
            major_tree_ids[cid] = 0

    comp_df["major_tree_id"] = comp_df["component_id"].map(major_tree_ids)

    edge_rows = []
    for _, row in comp_df.iterrows():
        cid = int(row["component_id"])
        is_major = int(bool(row["is_major"]))
        mtid = int(row["major_tree_id"])
        vids = comps[cid - 1]
        sub = sg.induced_subgraph(vids)

        for oeid in sub.es["orig_eid"]:
            edge_rows.append({
                "orig_eid": int(oeid),
                "component_id": cid,
                "is_major": is_major,
                "major_tree_id": mtid,
                "target_code": int(target_code),
            })

    edge_df = pd.DataFrame(edge_rows).sort_values("orig_eid").reset_index(drop=True)
    return comp_df, edge_df


def induced_subgraph_box(graph: ig.Graph, box: dict, coords_attr="coords", node_eps=0.0):
    """Induced subgraph of nodes whose coords lie inside a box."""
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


# ======================================================================
# Density
# ======================================================================

def microsegments_from_formatted_graph(graph: ig.Graph):
    """
    Convert each edge polyline into microsegments.
    Each microsegment stores midpoint, length and local radii.
    """
    check_attr(graph, ["points", "nkind"], "es")
    has_dpts = "diameters" in graph.es.attributes()
    has_dedge = "diameter" in graph.es.attributes()

    mids, lens, nk, r0s, r1s = [], [], [], [], []

    for e in graph.es:
        pts = np.asarray(e["points"], float)
        if pts.shape[0] < 2:
            continue

        d = None
        if has_dpts and e["diameters"] is not None:
            dd = np.asarray(e["diameters"], float)
            if dd.shape[0] == pts.shape[0] and np.all(np.isfinite(dd)) and np.all(dd > 0):
                d = dd

        if d is None and has_dedge:
            de = e["diameter"]
            if de is not None and np.isfinite(de) and de > 0:
                d = np.full(pts.shape[0], float(de), dtype=float)

        if d is None:
            continue

        nkind_e = int(e["nkind"]) if e["nkind"] is not None else -1

        for i in range(pts.shape[0] - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
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
    """Generate overlapping sub-boxes inside a larger box."""
    xs = np.arange(box["xmin"], box["xmax"] - box_size + 1, stride)
    ys = np.arange(box["ymin"], box["ymax"] - box_size + 1, stride)
    zs = np.arange(box["zmin"], box["zmax"] - box_size + 1, stride)

    boxes = []
    for x in xs:
        for y in ys:
            for z in zs:
                boxes.append({
                    "xmin": x,
                    "xmax": x + box_size,
                    "ymin": y,
                    "ymax": y + box_size,
                    "zmin": z,
                    "zmax": z + box_size,
                })
    return boxes


def vessel_volume_density(ms, box):
    """Vessel volume density in mm^3 / mm^3."""
    mids = ms["midpoints"]
    L = ms["lengths"]
    r0 = ms["r0"]
    r1 = ms["r1"]

    inside = (
        (mids[:, 0] >= box["xmin"]) & (mids[:, 0] <= box["xmax"]) &
        (mids[:, 1] >= box["ymin"]) & (mids[:, 1] <= box["ymax"]) &
        (mids[:, 2] >= box["zmin"]) & (mids[:, 2] <= box["zmax"])
    )

    L = L[inside]
    r0 = r0[inside]
    r1 = r1[inside]
    rmean = 0.5 * (r0 + r1)

    vessel_vol_um3 = np.sum(np.pi * rmean ** 2 * L)
    tissue_vol_um3 = (
        (box["xmax"] - box["xmin"]) *
        (box["ymax"] - box["ymin"]) *
        (box["zmax"] - box["zmin"])
    )

    return (vessel_vol_um3 * 1e-9) / (tissue_vol_um3 * 1e-9)


def vessel_length_density(ms, box):
    """Vessel length density in mm / mm^3."""
    mids = ms["midpoints"]
    L = ms["lengths"]

    inside = (
        (mids[:, 0] >= box["xmin"]) & (mids[:, 0] <= box["xmax"]) &
        (mids[:, 1] >= box["ymin"]) & (mids[:, 1] <= box["ymax"]) &
        (mids[:, 2] >= box["zmin"]) & (mids[:, 2] <= box["zmax"])
    )

    length_um = np.sum(L[inside])
    tissue_vol_um3 = (
        (box["xmax"] - box["xmin"]) *
        (box["ymax"] - box["ymin"]) *
        (box["zmax"] - box["zmin"])
    )

    return (length_um * 1e-3) / (tissue_vol_um3 * 1e-9)


def vessel_volume_density_nkind(ms, box, nkind_filter=None):
    """Volume density restricted to a given nkind."""
    mids = ms["midpoints"]
    L = ms["lengths"]
    r0 = ms["r0"]
    r1 = ms["r1"]
    nk = ms["nkind"]

    inside = (
        (mids[:, 0] >= box["xmin"]) & (mids[:, 0] <= box["xmax"]) &
        (mids[:, 1] >= box["ymin"]) & (mids[:, 1] <= box["ymax"]) &
        (mids[:, 2] >= box["zmin"]) & (mids[:, 2] <= box["zmax"])
    )

    if nkind_filter is not None:
        inside &= (nk == int(nkind_filter))

    L = L[inside]
    r0 = r0[inside]
    r1 = r1[inside]
    rmean = 0.5 * (r0 + r1)

    vessel_vol_um3 = np.sum(np.pi * rmean ** 2 * L)
    tissue_vol_um3 = (
        (box["xmax"] - box["xmin"]) *
        (box["ymax"] - box["ymin"]) *
        (box["zmax"] - box["zmin"])
    )

    return (vessel_vol_um3 * 1e-9) / (tissue_vol_um3 * 1e-9)


def vessel_length_density_nkind(ms, box, nkind_filter=None):
    """Length density restricted to a given nkind."""
    mids = ms["midpoints"]
    L = ms["lengths"]
    nk = ms["nkind"]

    inside = (
        (mids[:, 0] >= box["xmin"]) & (mids[:, 0] <= box["xmax"]) &
        (mids[:, 1] >= box["ymin"]) & (mids[:, 1] <= box["ymax"]) &
        (mids[:, 2] >= box["zmin"]) & (mids[:, 2] <= box["zmax"])
    )

    if nkind_filter is not None:
        inside &= (nk == int(nkind_filter))

    length_um = np.sum(L[inside])
    tissue_vol_um3 = (
        (box["xmax"] - box["xmin"]) *
        (box["ymax"] - box["ymin"]) *
        (box["zmax"] - box["zmin"])
    )

    return (length_um * 1e-3) / (tissue_vol_um3 * 1e-9)


def graph_density_metrics(graph, graph_name):
    """Simple graph-level density metrics."""
    V = int(graph.vcount())
    E = int(graph.ecount())
    graph_density = (2.0 * E / (V * (V - 1))) if V > 1 else np.nan

    return {
        "graph": graph_name,
        "V": V,
        "E": E,
        "graph_density": float(graph_density),
    }


# ======================================================================
# High-degree nodes / surface distance
# ======================================================================

def distance_to_surface_stats(
    graph: ig.Graph,
    nodes,
    space="um",
    depth_attr_vox="distance_to_surface",
    depth_attr_um="distance_to_surface_R",
    depth_bins_um=DEFAULT_DEPTH_BINS_UM,
    res_um_per_vox=res_um_per_vox,
):
    """Summarize distance-to-surface for a node set."""
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
    depth_bins_um=DEFAULT_DEPTH_BINS_UM,
):
    """Analyze high-degree nodes inside a box."""
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

    labels = np.array([infer_node_type_from_incident_edges(graph, int(v)) for v in hdn], dtype=object)
    uniq, cnt = np.unique(labels, return_counts=True)
    out["hdn_type_composition"] = {
        str(u): {"count": int(c), "proportion": float(c / labels.size)}
        for u, c in zip(uniq, cnt)
    }

    out["depth_hdn"] = None
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


# ======================================================================
# BC nodes
# ======================================================================

def debug_face_plane_counts(graph: ig.Graph, box: dict, coords_attr="coords", eps_vox=2.0, space="um"):
    """Debug how many nodes lie close to each box face."""
    validate_box_faces(box)
    P = np.asarray(graph.vs[coords_attr], float)
    print("\n=== DEBUG: vertices close to each face plane ===")
    print("coords bounds:", P.min(axis=0), "->", P.max(axis=0))
    print("box:", box)

    for face, (axis, key) in FACES_DEF.items():
        eps = resolve_eps(eps_vox, space=space, axis=axis)
        val = float(box[key])
        dist = np.abs(P[:, axis] - val)
        print(face, "min_dist", float(dist.min()), "n_within_eps", int((dist <= eps).sum()))


def bc_nodes_on_face_plane(graph: ig.Graph, axis: int, value: float, box: dict, coords_attr="coords", eps=5.0):
    """Find nodes close to a given face plane and inside the other two box bounds."""
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


def boundary_node_diameter(graph: ig.Graph, vid: int, diam_attr="diameter"):
    """Diameter proxy for a BC node from incident edges."""
    if diam_attr not in graph.es.attributes():
        return np.nan

    eids = graph.incident(int(vid))
    vals = []
    for eid in eids:
        d = graph.es[eid][diam_attr]
        if d is None:
            continue
        try:
            if np.isnan(d):
                continue
        except TypeError:
            pass
        vals.append(float(d))

    if len(vals) == 0:
        return np.nan
    if len(vals) == 1:
        return vals[0]
    return float(np.mean(vals))


def analyze_bc_faces(
    graph: ig.Graph,
    box: dict,
    coords_attr="coords",
    space="um",
    eps_vox=2.0,
    degree_thr=4,
    mode="auto",
    return_node_ids=False,
    diam_attr="diameter",
    return_diameter_values=False,
):
    """Analyze BC nodes face by face."""
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
            nodes = bc_nodes_on_face_plane(
                graph, axis, float(box[key]), box,
                coords_attr=coords_attr, eps=eps
            )

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

            diam_by_type = defaultdict(list)
            for v, lab in zip(nodes, labels):
                d = boundary_node_diameter(graph, int(v), diam_attr=diam_attr)
                if not np.isnan(d):
                    diam_by_type[lab].append(float(d))

            type_diam_stats = {}
            for lab, vals in diam_by_type.items():
                arr = np.asarray(vals, dtype=float)
                type_diam_stats[lab] = {
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)) if arr.size else np.nan,
                    "median": float(np.median(arr)) if arr.size else np.nan,
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
                }

            face_res["type_diameter_stats"] = type_diam_stats
            if return_diameter_values:
                face_res["type_diameter_values"] = {k: list(map(float, v)) for k, v in diam_by_type.items()}
        else:
            face_res["type_counts"] = {}
            face_res["type_percent"] = {}
            face_res["type_diameter_stats"] = {}
            if return_diameter_values:
                face_res["type_diameter_values"] = {}

        if ("distance_to_surface_R" in graph.vs.attributes()) or ("distance_to_surface" in graph.vs.attributes()):
            face_res["distance_to_surface_stats"] = distance_to_surface_stats(graph, nodes, space=space)
        else:
            face_res["distance_to_surface_stats"] = None

        if return_node_ids:
            face_res["nodes"] = nodes
            face_res["high_degree_nodes"] = nodes[high_mask] if n else np.array([], dtype=int)

        out[face] = face_res

    return out


def bc_diameter_longtable(res: dict, box_name="Box") -> pd.DataFrame:
    """BC diameter long-format table."""
    rows = []
    for face, face_data in res.items():
        vals_dict = face_data.get("type_diameter_values", {}) or {}
        for vessel_type, vals in vals_dict.items():
            for d in vals:
                if pd.notna(d):
                    rows.append({
                        "Box": box_name,
                        "Face": face,
                        "vessel_type": vessel_type,
                        "diameter": float(d),
                    })
    return pd.DataFrame(rows)


def bc_faces_table(res: dict, box_name="Box") -> pd.DataFrame:
    """BC summary table by face."""
    rows = []
    vessel_types = ["arteriole", "venule", "capillary", "unknown"]

    for face, face_data in res.items():
        total = int(face_data.get("count", 0))
        tc = face_data.get("type_counts", {}) or {}
        tp = face_data.get("type_percent", {}) or {}
        tds = face_data.get("type_diameter_stats", {}) or {}

        row = {
            "Box": box_name,
            "Face": face,
            "BC nodes": total,
            "High degree %": float(face_data.get("high_degree_percent", 0.0)),
        }

        for vt in vessel_types:
            row[f"n_{vt}"] = int(tc.get(vt, 0))
            row[f"% {vt.capitalize()}"] = float(tp.get(vt, 0.0))
            stats = tds.get(vt, {})
            row[f"{vt}_diam_mean"] = float(stats.get("mean", np.nan))
            row[f"{vt}_diam_median"] = float(stats.get("median", np.nan))
            row[f"{vt}_diam_std"] = float(stats.get("std", np.nan))

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# A-V paths / resilience
# ======================================================================

def av_sets(graph):
    """Return arteriole and venule node sets inferred from incident edges."""
    A, V = [], []
    for v in graph.vs:
        lab = infer_node_type_from_incident_edges(graph, v.index)
        if lab == "arteriole":
            A.append(v.index)
        elif lab == "venule":
            V.append(v.index)
    return np.array(A, int), np.array(V, int)


def get_ac_frontier_nodes(graph, edge_type_attr="nkind"):
    """Nodes touching both artery and capillary edges."""
    frontier = []
    for v in graph.vs:
        inc = graph.incident(v.index)
        if not inc:
            continue

        nkinds = set()
        for eid in inc:
            try:
                nkinds.add(int(graph.es[eid][edge_type_attr]))
            except Exception:
                pass

        if ARTERY in nkinds and CAPILLARY in nkinds:
            frontier.append(v.index)

    return np.array(sorted(set(frontier)), dtype=int)


def get_venous_nodes(graph, edge_type_attr="nkind"):
    """Nodes touching at least one venous edge."""
    venous = []
    for v in graph.vs:
        inc = graph.incident(v.index)
        if not inc:
            continue

        nkinds = set()
        for eid in inc:
            try:
                nkinds.add(int(graph.es[eid][edge_type_attr]))
            except Exception:
                pass

        if VEIN in nkinds:
            venous.append(v.index)

    return np.array(sorted(set(venous)), dtype=int)


def build_allowed_subgraph(graph, artery_continuity=True, edge_type_attr="nkind"):
    """
    Build the graph used for shortest-path calculation.

    artery_continuity=True  -> keep all edges
    artery_continuity=False -> remove arterial edges
    """
    if artery_continuity:
        keep_eids = list(range(graph.ecount()))
    else:
        keep_eids = [e.index for e in graph.es if int(e[edge_type_attr]) != ARTERY]

    subg = graph.subgraph_edges(keep_eids, delete_vertices=False)
    old_to_new = {int(v.index): int(v.index) for v in subg.vs}
    new_to_old = {int(v.index): int(v.index) for v in subg.vs}

    return subg, old_to_new, new_to_old, keep_eids


def add_penalization_weights(
    graph,
    new_weight_attr="w_cap_prior",
    edge_type_attr="nkind",
    length_attr="length",
    cap_code=CAPILLARY,
    art_code=ARTERY,
    ven_code=VEIN,
    penalty_art=1e5,
    penalty_ven=1e5,
):
    """
    Create artificial weights to strongly discourage artery and vein traversal.
    """
    nk = np.asarray(graph.es[edge_type_attr], dtype=int)
    L = np.asarray(graph.es[length_attr], dtype=float)
    w = np.empty(graph.ecount(), dtype=float)

    for i in range(graph.ecount()):
        if nk[i] == cap_code:
            w[i] = L[i]
        elif nk[i] == art_code:
            w[i] = penalty_art + L[i]
        elif nk[i] == ven_code:
            w[i] = penalty_ven + L[i]
        else:
            w[i] = L[i]

    graph.es[new_weight_attr] = w
    return new_weight_attr


def shortest_av_paths_from_ac_frontier(
    graph,
    artery_continuity=False,
    edge_type_attr="nkind",
    weight_attr="length",
    tie_break_edges=True,
    tol=1e-9,
):
    """
    For each A/C frontier node, find the closest venous node.

    Selection rule:
      1) minimum path distance using weight_attr
      2) optional tie-break by minimum number of edges
    """
    A = get_ac_frontier_nodes(graph, edge_type_attr=edge_type_attr)
    V = get_venous_nodes(graph, edge_type_attr=edge_type_attr)

    subg, old_to_new, new_to_old, _ = build_allowed_subgraph(
        graph,
        artery_continuity=artery_continuity,
        edge_type_attr=edge_type_attr,
    )

    A_valid = [a for a in A if a in old_to_new]
    V_valid = [v for v in V if v in old_to_new]

    A_sub = np.array([old_to_new[a] for a in A_valid], dtype=int)
    V_sub = np.array([old_to_new[v] for v in V_valid], dtype=int)

    paths = []
    distances = []
    path_n_edges = []
    source_frontiers = []
    target_venous = []

    if len(A_sub) == 0 or len(V_sub) == 0:
        return (
            paths,
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            A_valid,
            V_valid,
        )

    for a_old, a_sub in zip(A_valid, A_sub):
        dvec = np.asarray(
            subg.distances(source=a_sub, target=V_sub, weights=weight_attr)[0],
            dtype=float,
        )

        finite = np.isfinite(dvec)
        if not np.any(finite):
            continue

        valid_idx = np.where(finite)[0]
        valid_d = dvec[valid_idx]
        best_dist = np.min(valid_d)

        if tie_break_edges:
            candidate_mask = np.abs(valid_d - best_dist) <= tol
            candidate_idx = valid_idx[candidate_mask]
        else:
            candidate_idx = np.array([valid_idx[np.argmin(valid_d)]], dtype=int)

        best_path_old = None
        best_path_edges = None
        best_v_old = None

        for j in candidate_idx:
            v_sub = V_sub[j]
            p_sub = subg.get_shortest_paths(
                a_sub,
                to=v_sub,
                weights=weight_attr,
                output="vpath",
            )[0]

            if len(p_sub) == 0:
                continue

            p_old = [int(new_to_old[x]) for x in p_sub]
            n_edges = max(len(p_old) - 1, 0)

            if best_path_old is None or n_edges < best_path_edges:
                best_path_old = p_old
                best_path_edges = n_edges
                best_v_old = int(new_to_old[v_sub])

        if best_path_old is None:
            continue

        paths.append(best_path_old)
        distances.append(float(best_dist))
        path_n_edges.append(int(best_path_edges))
        source_frontiers.append(int(a_old))
        target_venous.append(int(best_v_old))

    return (
        paths,
        np.array(distances, dtype=float),
        np.array(path_n_edges, dtype=int),
        np.array(source_frontiers, dtype=int),
        np.array(target_venous, dtype=int),
        A_valid,
        V_valid,
    )


def path_to_edge_ids(graph, path):
    """Convert a node path into edge IDs."""
    eids = []
    for u, v in zip(path[:-1], path[1:]):
        eid = graph.get_eid(u, v, directed=False, error=False)
        if eid == -1:
            return []
        eids.append(eid)
    return eids


def path_edge_type_sequence(graph, path, edge_type_attr="nkind"):
    """Return nkind sequence along a path."""
    eids = path_to_edge_ids(graph, path)
    if len(eids) == 0:
        return np.array([], dtype=int)
    return np.asarray(graph.es[eids][edge_type_attr], dtype=int)


def trimmed_capillary_length(graph, path, edge_type_attr="nkind", length_attr="length"):
    """Capillary-only length inside a path."""
    eids = path_to_edge_ids(graph, path)
    if len(eids) == 0:
        return np.nan

    nk = np.asarray(graph.es[eids][edge_type_attr], dtype=int)
    L = np.asarray(graph.es[eids][length_attr], dtype=float)
    return float(np.sum(L[nk == CAPILLARY]))


def arterial_length_in_path(graph, path, edge_type_attr="nkind", length_attr="length"):
    """Arterial length inside a path."""
    eids = path_to_edge_ids(graph, path)
    if len(eids) == 0:
        return np.nan

    nk = np.asarray(graph.es[eids][edge_type_attr], dtype=int)
    L = np.asarray(graph.es[eids][length_attr], dtype=float)
    return float(np.sum(L[nk == ARTERY]))


def venous_length_in_path(graph, path, edge_type_attr="nkind", length_attr="length"):
    """Venous length inside a path."""
    eids = path_to_edge_ids(graph, path)
    if len(eids) == 0:
        return np.nan

    nk = np.asarray(graph.es[eids][edge_type_attr], dtype=int)
    L = np.asarray(graph.es[eids][length_attr], dtype=float)
    return float(np.sum(L[nk == VEIN]))


def av_path_stats_from_frontier(graph, graph_name, source_frontiers, target_venous, edge_type_attr="nkind"):
    """Basic path coverage stats from the frontier analysis."""
    A = get_ac_frontier_nodes(graph, edge_type_attr=edge_type_attr)
    V = get_venous_nodes(graph, edge_type_attr=edge_type_attr)

    return {
        "graph": graph_name,
        "frontier_nodes": len(A),
        "venous_nodes": len(V),
        "pairs_searched": len(A) * len(V),
        "paths_found": len(source_frontiers),
        "unique_frontiers_with_path": len(np.unique(source_frontiers)),
        "unique_venous_targets_reached": len(np.unique(target_venous)),
    }


def summarize_frontier_paths(
    graph,
    paths,
    distances,
    source_frontiers,
    target_venous,
    artery_continuity,
    edge_type_attr="nkind",
    length_attr="length",
):
    """Convert a list of shortest paths into a detailed row table."""
    rows = []

    for i, path in enumerate(paths):
        seq = path_edge_type_sequence(graph, path, edge_type_attr=edge_type_attr)

        rows.append({
            "path_id": i,
            "source_frontier": int(source_frontiers[i]),
            "target_venous": int(target_venous[i]),
            "n_nodes": len(path),
            "n_edges": max(0, len(path) - 1),
            "total_length": float(distances[i]),
            "capillary_length": trimmed_capillary_length(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "arterial_length": arterial_length_in_path(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "venous_length": venous_length_in_path(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "starts_with_artery": bool(len(seq) > 0 and seq[0] == ARTERY),
            "contains_artery": bool(np.any(seq == ARTERY)),
            "artery_continuity": bool(artery_continuity),
            "path": path,
        })

    return rows


# ----------------------------------------------------------------------
# Comparison 1: with vs without arterial continuity
# ----------------------------------------------------------------------

def compare_frontier_modes_with_without(
    graph,
    edge_type_attr="nkind",
    length_attr="length",
    tie_break_edges=True,
    tol=1e-9,
):
    """Compare shortest A-V paths with and without arterial continuity."""
    paths_yes, dist_yes, n_edges_yes, src_yes, tgt_yes, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        weight_attr=length_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    paths_no, dist_no, n_edges_no, src_no, tgt_no, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        weight_attr=length_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    rows_yes = summarize_frontier_paths(
        graph, paths_yes, dist_yes, src_yes, tgt_yes,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        length_attr=length_attr,
    )

    rows_no = summarize_frontier_paths(
        graph, paths_no, dist_no, src_no, tgt_no,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        length_attr=length_attr,
    )

    by_frontier_yes = {r["source_frontier"]: r for r in rows_yes}
    by_frontier_no = {r["source_frontier"]: r for r in rows_no}

    all_frontiers = sorted(set(by_frontier_yes) | set(by_frontier_no))
    comparison = []

    for f in all_frontiers:
        ry = by_frontier_yes.get(f)
        rn = by_frontier_no.get(f)

        comparison.append({
            "source_frontier": int(f),

            "has_path_with_artery": ry is not None,
            "has_path_without_artery": rn is not None,

            "target_with_artery": None if ry is None else ry["target_venous"],
            "target_without_artery": None if rn is None else rn["target_venous"],

            "path_len_with_artery": None if ry is None else ry["total_length"],
            "path_len_without_artery": None if rn is None else rn["total_length"],

            "capillary_len_with_artery": None if ry is None else ry["capillary_length"],
            "capillary_len_without_artery": None if rn is None else rn["capillary_length"],

            "arterial_len_with_artery": None if ry is None else ry["arterial_length"],
            "arterial_len_without_artery": None if rn is None else rn["arterial_length"],

            "venous_len_with_artery": None if ry is None else ry["venous_length"],
            "venous_len_without_artery": None if rn is None else rn["venous_length"],

            "path_len_with_artery_edges": None if ry is None else ry["n_edges"],
            "path_len_without_artery_edges": None if rn is None else rn["n_edges"],

            "delta_no_minus_yes": None if (ry is None or rn is None) else rn["total_length"] - ry["total_length"],
            "delta_capillary_no_minus_yes": None if (ry is None or rn is None) else rn["capillary_length"] - ry["capillary_length"],
            "delta_arterial_no_minus_yes": None if (ry is None or rn is None) else rn["arterial_length"] - ry["arterial_length"],
            "delta_venous_no_minus_yes": None if (ry is None or rn is None) else rn["venous_length"] - ry["venous_length"],
            "delta_edges_no_minus_yes": None if (ry is None or rn is None) else rn["n_edges"] - ry["n_edges"],

            "same_target_venous": None if (ry is None or rn is None) else bool(ry["target_venous"] == rn["target_venous"]),
            "uses_artery_with_artery": None if ry is None else bool(ry["arterial_length"] > 0),

            "path_with_artery": None if ry is None else ry["path"],
            "path_without_artery": None if rn is None else rn["path"],
        })

    return pd.DataFrame(comparison), pd.DataFrame(rows_yes), pd.DataFrame(rows_no)


def summarize_frontier_comparison_with_without(df_cmp_local, graph_name):
    """Graph-level summary for with/without continuity comparison."""
    ss = df_cmp_local.dropna(subset=["path_len_with_artery", "path_len_without_artery"]).copy()

    if ss.empty:
        return {
            "graph": graph_name,
            "n_pairs": 0,
            "n_target_changed": 0,
            "pct_target_changed": np.nan,
            "n_delta_um_nonzero": 0,
            "pct_delta_um_nonzero": np.nan,
            "n_delta_um_positive": 0,
            "pct_delta_um_positive": np.nan,
            "n_delta_um_negative": 0,
            "pct_delta_um_negative": np.nan,
            "n_delta_edges_nonzero": 0,
            "pct_delta_edges_nonzero": np.nan,
            "n_delta_edges_positive": 0,
            "pct_delta_edges_positive": np.nan,
            "n_delta_edges_negative": 0,
            "pct_delta_edges_negative": np.nan,
            "median_delta_um": np.nan,
            "mean_delta_um": np.nan,
            "median_delta_edges": np.nan,
            "mean_delta_edges": np.nan,
            "n_paths_using_artery_with_continuity": 0,
            "pct_paths_using_artery_with_continuity": np.nan,
            "median_arterial_len_with_continuity_um": np.nan,
            "mean_arterial_len_with_continuity_um": np.nan,
            "n_same_target_and_delta_positive": 0,
            "n_changed_target_and_delta_positive": 0,
        }

    delta_um = ss["delta_no_minus_yes"].to_numpy(float)
    delta_edges = ss["delta_edges_no_minus_yes"].to_numpy(float)

    target_changed = (ss["same_target_venous"] == False).to_numpy()
    target_same = (ss["same_target_venous"] == True).to_numpy()

    delta_um_positive = delta_um > 0
    delta_um_negative = delta_um < 0
    delta_edges_positive = delta_edges > 0
    delta_edges_negative = delta_edges < 0
    uses_artery = (ss["uses_artery_with_artery"] == True).to_numpy()

    return {
        "graph": graph_name,
        "n_pairs": int(len(ss)),
        "n_target_changed": int(np.sum(target_changed)),
        "pct_target_changed": 100.0 * float(np.mean(target_changed)),
        "n_delta_um_nonzero": int(np.sum(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "pct_delta_um_nonzero": 100.0 * float(np.mean(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "n_delta_um_positive": int(np.sum(delta_um_positive)),
        "pct_delta_um_positive": 100.0 * float(np.mean(delta_um_positive)),
        "n_delta_um_negative": int(np.sum(delta_um_negative)),
        "pct_delta_um_negative": 100.0 * float(np.mean(delta_um_negative)),
        "n_delta_edges_nonzero": int(np.sum(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "pct_delta_edges_nonzero": 100.0 * float(np.mean(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "n_delta_edges_positive": int(np.sum(delta_edges_positive)),
        "pct_delta_edges_positive": 100.0 * float(np.mean(delta_edges_positive)),
        "n_delta_edges_negative": int(np.sum(delta_edges_negative)),
        "pct_delta_edges_negative": 100.0 * float(np.mean(delta_edges_negative)),
        "median_delta_um": float(np.nanmedian(delta_um)),
        "mean_delta_um": float(np.nanmean(delta_um)),
        "median_delta_edges": float(np.nanmedian(delta_edges)),
        "mean_delta_edges": float(np.nanmean(delta_edges)),
        "n_paths_using_artery_with_continuity": int(np.sum(uses_artery)),
        "pct_paths_using_artery_with_continuity": 100.0 * float(np.mean(uses_artery)),
        "median_arterial_len_with_continuity_um": float(np.nanmedian(ss["arterial_len_with_artery"])),
        "mean_arterial_len_with_continuity_um": float(np.nanmean(ss["arterial_len_with_artery"])),
        "n_same_target_and_delta_positive": int(np.sum(target_same & delta_um_positive)),
        "n_changed_target_and_delta_positive": int(np.sum(target_changed & delta_um_positive)),
    }


def build_frontier_comparison_summary_table_with_without(df_summary, graphs_order):
    """Pretty summary table for with/without continuity."""
    rows = []

    metric_specs = [
        ("n_pairs", "Frontiers compared"),
        ("n_target_changed", "Target changed"),
        ("n_delta_um_positive", "Path longer without artery"),
        ("median_delta_um", "Median Δ length (um)"),
        ("median_delta_edges", "Median Δ length (edges)"),
        ("n_paths_using_artery_with_continuity", "Paths using artery when allowed"),
        ("median_arterial_len_with_continuity_um", "Median arterial length with continuity (um)"),
        ("n_same_target_and_delta_positive", "Same target + longer path"),
        ("n_changed_target_and_delta_positive", "Changed target + longer path"),
    ]

    for metric_key, metric_label in metric_specs:
        row = {"metric": metric_label}

        for g in graphs_order:
            sub = df_summary[df_summary["graph"] == g]
            if sub.empty:
                row[g] = "NA"
                continue

            r = sub.iloc[0]

            if metric_key == "n_pairs":
                row[g] = f"{int(r['n_pairs'])}"
            elif metric_key == "n_target_changed":
                row[g] = f"{int(r['n_target_changed'])} ({r['pct_target_changed']:.1f}%)"
            elif metric_key == "n_delta_um_positive":
                row[g] = f"{int(r['n_delta_um_positive'])} ({r['pct_delta_um_positive']:.1f}%)"
            elif metric_key == "n_paths_using_artery_with_continuity":
                row[g] = f"{int(r['n_paths_using_artery_with_continuity'])} ({r['pct_paths_using_artery_with_continuity']:.1f}%)"
            elif metric_key in ["median_delta_um", "median_delta_edges", "median_arterial_len_with_continuity_um"]:
                val = r[metric_key]
                row[g] = "NA" if not np.isfinite(val) else f"{val:.2f}"
            else:
                val = r[metric_key]
                row[g] = "NA" if not np.isfinite(val) else f"{int(val)}"

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Comparison 2: restricted vs weighted+continuity
# ----------------------------------------------------------------------

def compare_frontier_restricted_vs_weighted(
    graph,
    edge_type_attr="nkind",
    length_weight_attr="length",
    weighted_attr="w_cap_prior",
    tie_break_edges=True,
    tol=1e-9,
):
    """Compare restricted shortest paths vs weighted full-graph shortest paths."""
    paths_restricted, dist_restricted, n_edges_restricted, src_restricted, tgt_restricted, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        weight_attr=length_weight_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    paths_weighted, dist_weighted, n_edges_weighted, src_weighted, tgt_weighted, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        weight_attr=weighted_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    rows_restricted = summarize_frontier_paths(
        graph, paths_restricted, dist_restricted, src_restricted, tgt_restricted,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        length_attr=length_weight_attr,
    )

    rows_weighted = summarize_frontier_paths(
        graph, paths_weighted, dist_weighted, src_weighted, tgt_weighted,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        length_attr="length",
    )

    by_frontier_restricted = {r["source_frontier"]: r for r in rows_restricted}
    by_frontier_weighted = {r["source_frontier"]: r for r in rows_weighted}

    all_frontiers = sorted(set(by_frontier_restricted) | set(by_frontier_weighted))
    comparison = []

    for f in all_frontiers:
        rr = by_frontier_restricted.get(f)
        rw = by_frontier_weighted.get(f)

        comparison.append({
            "source_frontier": int(f),

            "has_restricted": rr is not None,
            "has_weighted": rw is not None,

            "target_restricted": None if rr is None else rr["target_venous"],
            "target_weighted": None if rw is None else rw["target_venous"],

            "path_len_restricted": None if rr is None else rr["total_length"],
            "path_len_weighted": None if rw is None else rw["total_length"],

            "capillary_len_restricted": None if rr is None else rr["capillary_length"],
            "capillary_len_weighted": None if rw is None else rw["capillary_length"],

            "arterial_len_restricted": None if rr is None else rr["arterial_length"],
            "arterial_len_weighted": None if rw is None else rw["arterial_length"],

            "venous_len_restricted": None if rr is None else rr["venous_length"],
            "venous_len_weighted": None if rw is None else rw["venous_length"],

            "path_len_restricted_edges": None if rr is None else rr["n_edges"],
            "path_len_weighted_edges": None if rw is None else rw["n_edges"],

            "delta_weighted_minus_restricted": None if (rr is None or rw is None) else rw["total_length"] - rr["total_length"],
            "delta_capillary_weighted_minus_restricted": None if (rr is None or rw is None) else rw["capillary_length"] - rr["capillary_length"],
            "delta_arterial_weighted_minus_restricted": None if (rr is None or rw is None) else rw["arterial_length"] - rr["arterial_length"],
            "delta_venous_weighted_minus_restricted": None if (rr is None or rw is None) else rw["venous_length"] - rr["venous_length"],
            "delta_edges_weighted_minus_restricted": None if (rr is None or rw is None) else rw["n_edges"] - rr["n_edges"],

            "same_target_venous": None if (rr is None or rw is None) else bool(rr["target_venous"] == rw["target_venous"]),

            "weighted_uses_artery": None if rw is None else bool(rw["arterial_length"] > 0),
            "weighted_arterial_len_um": None if rw is None else rw["arterial_length"],

            "path_restricted": None if rr is None else rr["path"],
            "path_weighted": None if rw is None else rw["path"],
        })

    return pd.DataFrame(comparison), pd.DataFrame(rows_restricted), pd.DataFrame(rows_weighted)


def summarize_frontier_comparison_restricted_vs_weighted(df_cmp_local, graph_name):
    """Graph-level summary for restricted vs weighted comparison."""
    ss = df_cmp_local.dropna(subset=["path_len_restricted", "path_len_weighted"]).copy()

    if ss.empty:
        return {
            "graph": graph_name,
            "n_pairs": 0,
            "n_target_changed": 0,
            "pct_target_changed": np.nan,
            "n_delta_um_nonzero": 0,
            "pct_delta_um_nonzero": np.nan,
            "n_delta_um_positive": 0,
            "pct_delta_um_positive": np.nan,
            "n_delta_um_negative": 0,
            "pct_delta_um_negative": np.nan,
            "n_delta_edges_nonzero": 0,
            "pct_delta_edges_nonzero": np.nan,
            "n_delta_edges_positive": 0,
            "pct_delta_edges_positive": np.nan,
            "n_delta_edges_negative": 0,
            "pct_delta_edges_negative": np.nan,
            "median_delta_um": np.nan,
            "mean_delta_um": np.nan,
            "median_delta_edges": np.nan,
            "mean_delta_edges": np.nan,
            "n_weighted_paths_using_artery": 0,
            "pct_weighted_paths_using_artery": np.nan,
            "median_weighted_arterial_len_um": np.nan,
            "mean_weighted_arterial_len_um": np.nan,
        }

    delta_um = ss["delta_weighted_minus_restricted"].to_numpy(float)
    delta_edges = ss["delta_edges_weighted_minus_restricted"].to_numpy(float)
    target_changed = (ss["same_target_venous"] == False).to_numpy()

    delta_um_positive = delta_um > 0
    delta_um_negative = delta_um < 0
    delta_edges_positive = delta_edges > 0
    delta_edges_negative = delta_edges < 0
    uses_artery = (ss["weighted_uses_artery"] == True).to_numpy()

    return {
        "graph": graph_name,
        "n_pairs": int(len(ss)),
        "n_target_changed": int(np.sum(target_changed)),
        "pct_target_changed": 100.0 * float(np.mean(target_changed)),
        "n_delta_um_nonzero": int(np.sum(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "pct_delta_um_nonzero": 100.0 * float(np.mean(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "n_delta_um_positive": int(np.sum(delta_um_positive)),
        "pct_delta_um_positive": 100.0 * float(np.mean(delta_um_positive)),
        "n_delta_um_negative": int(np.sum(delta_um_negative)),
        "pct_delta_um_negative": 100.0 * float(np.mean(delta_um_negative)),
        "n_delta_edges_nonzero": int(np.sum(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "pct_delta_edges_nonzero": 100.0 * float(np.mean(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "n_delta_edges_positive": int(np.sum(delta_edges_positive)),
        "pct_delta_edges_positive": 100.0 * float(np.mean(delta_edges_positive)),
        "n_delta_edges_negative": int(np.sum(delta_edges_negative)),
        "pct_delta_edges_negative": 100.0 * float(np.mean(delta_edges_negative)),
        "median_delta_um": float(np.nanmedian(delta_um)),
        "mean_delta_um": float(np.nanmean(delta_um)),
        "median_delta_edges": float(np.nanmedian(delta_edges)),
        "mean_delta_edges": float(np.nanmean(delta_edges)),
        "n_weighted_paths_using_artery": int(np.sum(uses_artery)),
        "pct_weighted_paths_using_artery": 100.0 * float(np.mean(uses_artery)),
        "median_weighted_arterial_len_um": float(np.nanmedian(ss["weighted_arterial_len_um"])),
        "mean_weighted_arterial_len_um": float(np.nanmean(ss["weighted_arterial_len_um"])),
    }


def build_frontier_comparison_summary_table_restricted_vs_weighted(df_summary, graphs_order):
    """Simple sorted summary table for restricted vs weighted."""
    cols = [
        "graph",
        "n_pairs",
        "n_target_changed",
        "pct_target_changed",
        "n_delta_um_nonzero",
        "pct_delta_um_nonzero",
        "n_delta_um_positive",
        "pct_delta_um_positive",
        "n_delta_um_negative",
        "pct_delta_um_negative",
        "median_delta_um",
        "mean_delta_um",
        "n_delta_edges_nonzero",
        "pct_delta_edges_nonzero",
        "n_delta_edges_positive",
        "pct_delta_edges_positive",
        "n_delta_edges_negative",
        "pct_delta_edges_negative",
        "median_delta_edges",
        "mean_delta_edges",
        "n_weighted_paths_using_artery",
        "pct_weighted_paths_using_artery",
        "median_weighted_arterial_len_um",
        "mean_weighted_arterial_len_um",
    ]
    cols = [c for c in cols if c in df_summary.columns]

    out = df_summary[cols].copy()
    out["graph"] = pd.Categorical(out["graph"], categories=graphs_order, ordered=True)
    out = out.sort_values("graph").reset_index(drop=True)
    return out




def max_edge_disjoint_av(graph: ig.Graph):
    """Maximum number of edge-disjoint arteriole-to-venule paths."""
    A, V = av_sets(graph)
    if A.size == 0 or V.size == 0:
        return {"n_edge_disjoint_av": 0, "nA": int(A.size), "nV": int(V.size), "paths": []}

    D = graph.as_directed(mutual=True) if not graph.is_directed() else graph.copy()

    s = D.vcount()
    t = D.vcount() + 1
    D.add_vertices(2)

    extra_edges = [(s, int(a)) for a in A] + [(int(v), t) for v in V]
    D.add_edges(extra_edges)

    BIG = float(max(1, D.ecount()))
    extra_caps = [BIG] * len(extra_edges)
    D.es["cap"] = [1.0] * (D.ecount() - len(extra_caps)) + extra_caps

    mf = D.maxflow(s, t, capacity="cap")
    n_disjoint = int(round(mf.value))

    flow_left = {}
    adj = {}
    for e, f in zip(D.es, mf.flow):
        f = int(round(f))
        if f > 0:
            u, v = e.tuple
            flow_left[(u, v)] = f
            adj.setdefault(u, []).append(v)

    paths = []
    for _ in range(n_disjoint):
        stack = [(s, [s])]
        found_path = None

        while stack:
            node, path = stack.pop()
            if node == t:
                found_path = path
                break

            for nxt in adj.get(node, []):
                if flow_left.get((node, nxt), 0) > 0 and nxt not in path:
                    stack.append((nxt, path + [nxt]))

        if found_path is None:
            print(f"[WARNING] maxflow={n_disjoint} but could only reconstruct {len(paths)} paths")
            break

        for u, v in zip(found_path[:-1], found_path[1:]):
            flow_left[(u, v)] -= 1

        paths.append(found_path[1:-1])

    return {
        "n_edge_disjoint_av": n_disjoint,
        "nA": int(A.size),
        "nV": int(V.size),
        "paths": paths,
    }


def sample_paths(paths, n):
    """Return the first n paths."""
    if len(paths) <= n:
        return paths
    return paths[:n]


def export_paths_vtp(graph, paths, filename, coords_attr="coords"):
    """Export a list of paths as VTP polylines."""
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
            x, y, z = P[v]
            points.InsertNextPoint(x, y, z)
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


def export_labeled_edges_vtp(graph, edge_labels_df, out_path, coords_attr="coords", include_only_major=False):
    """Export labeled original edges to a VTP file."""
    if edge_labels_df is None or edge_labels_df.empty:
        print(f"[export_labeled_edges_vtp] Nothing to export: {out_path}")
        return

    df = edge_labels_df.copy()
    if include_only_major:
        df = df[df["is_major"] == 1].copy()

    if df.empty:
        print(f"[export_labeled_edges_vtp] No major edges to export: {out_path}")
        return

    coords = np.asarray(graph.vs[coords_attr], float)

    points = []
    connectivity = []
    offsets = []

    cell_orig_eid = []
    cell_component_id = []
    cell_is_major = []
    cell_major_tree_id = []
    cell_nkind = []
    cell_length = []
    cell_diameter = []

    pidx = 0
    for _, row in df.iterrows():
        eid = int(row["orig_eid"])
        e = graph.es[eid]
        s, t = e.tuple

        ps = coords[int(s)]
        pt = coords[int(t)]

        points.append([float(ps[0]), float(ps[1]), float(ps[2])])
        points.append([float(pt[0]), float(pt[1]), float(pt[2])])

        connectivity.extend([pidx, pidx + 1])
        pidx += 2
        offsets.append(pidx)

        cell_orig_eid.append(eid)
        cell_component_id.append(int(row["component_id"]))
        cell_is_major.append(int(row["is_major"]))
        cell_major_tree_id.append(int(row["major_tree_id"]))
        cell_nkind.append(int(e["nkind"]) if "nkind" in graph.es.attributes() else -1)
        cell_length.append(float(e["length"]) if "length" in graph.es.attributes() else np.nan)
        cell_diameter.append(float(e["diameter"]) if "diameter" in graph.es.attributes() else np.nan)

    points_arr = np.asarray(points, float).reshape(-1, 3)
    connectivity_arr = np.asarray(connectivity, int)
    offsets_arr = np.asarray(offsets, int)

    def arr_to_ascii(a):
        return " ".join(map(str, np.ravel(a)))

    xml = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{len(points_arr)}" NumberOfLines="{len(cell_orig_eid)}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {arr_to_ascii(points_arr.astype(np.float32))}
        </DataArray>
      </Points>
      <Lines>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {arr_to_ascii(connectivity_arr.astype(np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {arr_to_ascii(offsets_arr.astype(np.int32))}
        </DataArray>
      </Lines>
      <CellData>
        <DataArray type="Int32" Name="orig_eid" format="ascii">
          {arr_to_ascii(np.asarray(cell_orig_eid, dtype=np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="component_id" format="ascii">
          {arr_to_ascii(np.asarray(cell_component_id, dtype=np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="is_major" format="ascii">
          {arr_to_ascii(np.asarray(cell_is_major, dtype=np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="major_tree_id" format="ascii">
          {arr_to_ascii(np.asarray(cell_major_tree_id, dtype=np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="nkind" format="ascii">
          {arr_to_ascii(np.asarray(cell_nkind, dtype=np.int32))}
        </DataArray>
        <DataArray type="Float32" Name="length" format="ascii">
          {arr_to_ascii(np.asarray(cell_length, dtype=np.float32))}
        </DataArray>
        <DataArray type="Float32" Name="diameter" format="ascii">
          {arr_to_ascii(np.asarray(cell_diameter, dtype=np.float32))}
        </DataArray>
      </CellData>
    </Piece>
  </PolyData>
</VTKFile>
'''

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"[export_labeled_edges_vtp] Saved: {out_path}")


# ======================================================================
# Statistical helpers
# ======================================================================

def pairwise_ttests_table(df, value_col, graphs_order, group_col="graph"):
    """Pairwise unpaired t-tests between graphs."""
    rows = []
    present = [g for g in graphs_order if g in set(df[group_col].astype(str))]

    for g1, g2 in combinations(present, 2):
        x1 = _finite(df.loc[df[group_col] == g1, value_col].to_numpy(float))
        x2 = _finite(df.loc[df[group_col] == g2, value_col].to_numpy(float))

        if x1.size < 2 or x2.size < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")

        rows.append({
            "metric": value_col,
            "group1": g1,
            "group2": g2,
            "n1": int(x1.size),
            "n2": int(x2.size),
            "mean1": float(np.mean(x1)) if x1.size else np.nan,
            "mean2": float(np.mean(x2)) if x2.size else np.nan,
            "median1": float(np.median(x1)) if x1.size else np.nan,
            "median2": float(np.median(x2)) if x2.size else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    return pd.DataFrame(rows)


def _p_to_text(p):
    """Compact p-value text."""
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.3g}"


def _p_to_stars(p):
    """Stars for p-values."""
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_pairwise_pvalues(ax, stats_df, positions, data_by_group):
    """Add p-value brackets above boxplots."""
    if stats_df is None or stats_df.empty:
        return

    valid_arrays = [np.asarray(v, float) for v in data_by_group.values() if len(v) > 0]
    if not valid_arrays:
        return

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if yr <= 0:
        yr = 1.0

    max_y = max(np.nanmax(v) for v in valid_arrays)
    n_pairs = len(stats_df)

    base = max_y + 0.05 * yr
    step = 0.10 * yr
    needed_top = base + (n_pairs + 1) * step

    if needed_top > ymax:
        ax.set_ylim(ymin, needed_top)
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin
        base = max_y + 0.05 * yr
        step = 0.08 * yr

    for i, row in enumerate(stats_df.itertuples(index=False)):
        g1, g2 = row.group1, row.group2
        if g1 not in positions or g2 not in positions:
            continue

        x1, x2 = positions[g1], positions[g2]
        if x1 > x2:
            x1, x2 = x2, x1

        y = base + i * step
        h = 0.025 * yr

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
        ax.text(
            (x1 + x2) / 2.0,
            y + h + 0.01 * yr,
            f"{_p_to_stars(row.p_value)} ({_p_to_text(row.p_value)})",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )


def add_jitter_scatter(ax, xpos, y, color="black", jitter=0.08, alpha=0.22, s=10, max_points=2000):
    """Add jittered scatter points on top of a boxplot."""
    y = _finite(y)
    if y.size == 0:
        return

    if y.size > max_points:
        rng = np.random.default_rng(0)
        y = rng.choice(y, size=max_points, replace=False)

    rng = np.random.default_rng(0)
    xj = xpos + rng.uniform(-jitter, jitter, size=y.size)

    ax.scatter(
        xj, y,
        s=s,
        alpha=alpha,
        color=color,
        edgecolors="none",
        zorder=2,
    )


def paired_ttest_with_without(df, graphs_order, group_col="graph"):
    """Paired t-test for with vs without continuity."""
    rows = []
    present = [g for g in graphs_order if g in set(df[group_col].astype(str))]

    for g in present:
        sub = df[df[group_col] == g].copy()
        sub = sub.dropna(subset=["path_len_with_artery", "path_len_without_artery"])

        x = _finite(sub["path_len_with_artery"].to_numpy(float))
        y = _finite(sub["path_len_without_artery"].to_numpy(float))

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if n < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(y, x, nan_policy="omit")

        rows.append({
            "graph": g,
            "n_pairs": int(n),
            "mean_with_artery": float(np.mean(x)) if n else np.nan,
            "mean_without_artery": float(np.mean(y)) if n else np.nan,
            "median_with_artery": float(np.median(x)) if n else np.nan,
            "median_without_artery": float(np.median(y)) if n else np.nan,
            "mean_delta_no_minus_yes": float(np.mean(y - x)) if n else np.nan,
            "median_delta_no_minus_yes": float(np.median(y - x)) if n else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    return pd.DataFrame(rows)


def paired_ttest_restricted_vs_weighted(df, graphs_order, group_col="graph"):
    """Paired t-test for restricted vs weighted+continuity."""
    rows = []
    present = [g for g in graphs_order if g in set(df[group_col].astype(str))]

    for g in present:
        sub = df[df[group_col] == g].copy()
        sub = sub.dropna(subset=["path_len_restricted", "path_len_weighted"])

        x = _finite(sub["path_len_restricted"].to_numpy(float))
        y = _finite(sub["path_len_weighted"].to_numpy(float))

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if n < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(y, x, nan_policy="omit")

        rows.append({
            "graph": g,
            "n_pairs": int(n),
            "mean_restricted": float(np.mean(x)) if n else np.nan,
            "mean_weighted": float(np.mean(y)) if n else np.nan,
            "median_restricted": float(np.median(x)) if n else np.nan,
            "median_weighted": float(np.median(y)) if n else np.nan,
            "mean_delta_weighted_minus_restricted": float(np.mean(y - x)) if n else np.nan,
            "median_delta_weighted_minus_restricted": float(np.median(y - x)) if n else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================

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
    """3D scatter of high-degree nodes."""
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
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=s_sel, c=col[lab], alpha=alpha_sel,
                    depthshade=False, label=lab
                )
        ax.legend(loc="best", title=f"{crit} nodes")

    ax.set_title(title or f"Spatial distribution of {crit}")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_bc_cube_net(
    res: dict,
    title="BC composition per face (cube net)",
    face_alpha=0.65,
    fontsize=10,
    pct_decimals=1,
    show_unknown=False,
    show_high_degree=True,
    face_colors=None,
):
    """2D cube-net summary of BC composition."""
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
                lines.append(f"{k}: {int(tc.get(k, 0))} ({float(tp[k]):.{pct_decimals}f}%)")

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
            alpha=face_alpha,
        )
        ax.add_patch(rect)
        ax.text(gx + 0.5, gy + 0.5, face_text(face), ha="center", va="center", fontsize=fontsize)

    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 3.2)
    plt.tight_layout()
    plt.show()


def plot_bc_3_cubes_tinted(
    G: ig.Graph,
    box: dict,
    coords_attr="coords",
    space="um",
    eps_vox=2.0,
    elev=18,
    azim=35,
    face_alpha=0.30,
    point_alpha=0.85,
    point_size=8,
    sample_max=20000,
    mode="auto",
    face_colors=None,
    vessel_colors=None,
):
    """3 tinted cube views with BC nodes colored by vessel type."""
    validate_box_faces(box)
    coords = get_coords(G, coords_attr).astype(float)

    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

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

    if vessel_colors is None:
        vessel_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}
    if face_colors is None:
        face_colors = BC_FACE_COLORS

    def draw_panel(ax, faces_subset, title_panel):
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                linewidth=1.0
            )

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
                    ax.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=point_size, alpha=point_alpha, color=col)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title_panel)

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    draw_panel(ax1, ("x_min", "z_min"), f"View A ({mode})")
    draw_panel(ax2, ("x_max", "z_max"), f"View B ({mode})")
    draw_panel(ax3, ("y_min", "y_max"), f"View C ({mode})")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=vessel_colors[k], markeredgecolor="none", label=k)
        for k in ["arteriole", "venule", "capillary", "unknown"]
    ]
    ax3.legend(handles=handles, title="Vessel type", loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_bar_by_category_general(
    categ, attribute_toplot, label_dict=None,
    xlabel="Category", ylabel="Value",
    title="Category statistics",
    show_values=True, value_fmt="{:.2f}",
):
    """Simple bar plot by category."""
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
                fontsize=9,
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
    main_title=None,
):
    """Histogram(s) split by category."""
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
        ax.hist(subset, bins=edges, density=density, alpha=0.7, color=color)

        if show_mean and len(subset):
            mean_val = subset.mean()
            ax.axvline(mean_val, linestyle="--", color=color)
            ax.legend([f"Mean = {mean_val:.2f}"])

        name = label_dict.get(c, str(c)) if label_dict else str(c)
        ax.set_title(f"{name} (n={n}: {pct:.1f}%)")
        ax.set_xlabel(variable_name)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_violin_box_by_category(values, category, label_dict=None, xlabel="Category", ylabel="Value", title="Distribution by category"):
    """Violin + box + raw points by category."""
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

    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_boxplot_by_graph(df_long, value_col, title, ylabel, graphs_order=("HPC_1", "HPC_2", "HPC_3")):
    """Simple boxplot grouped by graph name."""
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


def diameter_length_overlay_by_type(dl, bins=40, graphs_order=("HPC_1", "HPC_2", "HPC_3"), box_label=""):
    """Overlay histograms of diameter and length for each vessel type."""
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


def plot_grouped_boxplot_types_per_graph(
    df_long,
    value_col="diameter_vox",
    type_col="type",
    graphs_order=("HPC_1", "HPC_2", "HPC_3"),
    types_order=("arteriole", "venule", "capillary"),
    type_colors=None,
    title="Diameter by vessel type within each box",
    ylabel="Diameter (vox)",
):
    """Grouped boxplots: one graph, multiple vessel types."""
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

        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True, showfliers=False)
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


def plot_simple_type_boxplots_with_stats(
    dl,
    value_col,
    ylabel,
    graphs_order,
    types_order,
    box_colors=None,
    show_scatter=True,
    scatter_jitter=0.08,
    scatter_alpha=0.20,
    scatter_size=10,
    max_scatter_points=2000,
    title=None,
):
    """One panel per vessel type with boxplots and p-values."""
    dl = _normalize_types(dl)

    if box_colors is None:
        box_colors = {
            "HPC_1": "tab:blue",
            "HPC_2": "tab:orange",
            "HPC_3": "tab:green",
        }

    types_use = [t for t in types_order if t in set(dl["type"].unique())]
    fig, axes = plt.subplots(1, len(types_use), figsize=(4.8 * len(types_use), 5.4), sharey=True)
    if len(types_use) == 1:
        axes = [axes]

    stats_out = []

    for ax, t in zip(axes, types_use):
        data, meds, labels = [], [], []
        data_by_group = {}

        sub = dl[dl["type"] == t].copy()

        for g in graphs_order:
            x = sub.loc[sub["graph"] == g, value_col].to_numpy(float)
            x = _finite(x)
            if x.size == 0:
                continue

            data.append(x)
            meds.append(np.median(x))
            labels.append(g)
            data_by_group[g] = x

        if not data:
            ax.set_axis_off()
            continue

        dm = delta_median_pct(np.asarray(meds, float))

        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, lab in zip(bp["boxes"], labels):
            patch.set_facecolor(box_colors.get(lab, "lightgray"))
            patch.set_alpha(0.30)

        if show_scatter:
            for i, g in enumerate(labels, start=1):
                add_jitter_scatter(
                    ax=ax,
                    xpos=i,
                    y=data_by_group[g],
                    color=box_colors.get(g, "black"),
                    jitter=scatter_jitter,
                    alpha=scatter_alpha,
                    s=scatter_size,
                    max_points=max_scatter_points,
                )

        ax.set_title(f"{t}\nΔmedian={dm:.1f}%")
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylabel(ylabel)

        st = pairwise_ttests_table(sub, value_col, labels)
        if not st.empty:
            st["type"] = t
            stats_out.append(st)

        positions = {g: i + 1 for i, g in enumerate(labels)}
        add_pairwise_pvalues(ax, st, positions, data_by_group)

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()

    if stats_out:
        return pd.concat(stats_out, ignore_index=True)
    return pd.DataFrame()


def plot_paired_with_without_continuity_boxplots_from_comparison(
    df_comparison,
    graphs_order,
    ylabel="Shortest path length (um)",
    title=None,
    sharey=True,
):
    """Paired boxplots for with vs without continuity."""
    present = [g for g in graphs_order if g in set(df_comparison["graph"].astype(str))]
    if len(present) == 0:
        print("No graphs found.")
        return pd.DataFrame()

    all_y = []
    for g in present:
        sub = df_comparison[df_comparison["graph"] == g].copy()
        sub = sub.dropna(subset=["path_len_with_artery", "path_len_without_artery"])
        x_with = _finite(sub["path_len_with_artery"].to_numpy(float))
        x_without = _finite(sub["path_len_without_artery"].to_numpy(float))
        if x_with.size:
            all_y.append(x_with)
        if x_without.size:
            all_y.append(x_without)

    if len(all_y) == 0:
        print("No paired values found.")
        return pd.DataFrame()

    all_y = np.concatenate(all_y)
    global_ymin_data = float(np.nanmin(all_y))
    global_ymax_data = float(np.nanmax(all_y))
    yr = global_ymax_data - global_ymin_data
    if not np.isfinite(yr) or yr <= 0:
        yr = 1.0

    common_ymin = global_ymin_data - 0.03 * yr

    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 5.8), sharey=sharey)
    if len(present) == 1:
        axes = [axes]

    stats_out = []

    for ax, g in zip(axes, present):
        sub = df_comparison[df_comparison["graph"] == g].copy()
        sub = sub.dropna(subset=["path_len_with_artery", "path_len_without_artery"])

        x_with = _finite(sub["path_len_with_artery"].to_numpy(float))
        x_without = _finite(sub["path_len_without_artery"].to_numpy(float))

        n = min(len(x_with), len(x_without))
        x_with = x_with[:n]
        x_without = x_without[:n]

        if n == 0:
            ax.set_axis_off()
            continue

        bp = ax.boxplot(
            [x_with, x_without],
            labels=["with artery", "without artery"],
            showfliers=False,
            patch_artist=True,
            widths=0.5,
        )

        bp["boxes"][0].set_facecolor("tab:blue")
        bp["boxes"][0].set_alpha(0.30)
        bp["boxes"][1].set_facecolor("tab:orange")
        bp["boxes"][1].set_alpha(0.30)

        add_jitter_scatter(ax, 1, x_with, color="tab:blue")
        add_jitter_scatter(ax, 2, x_without, color="tab:orange")

        if n < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(x_without, x_with, nan_policy="omit")

        st = pd.DataFrame([{
            "group1": "with artery",
            "group2": "without artery",
            "p_value": p_val,
        }])

        add_pairwise_pvalues(
            ax,
            st,
            positions={"with artery": 1, "without artery": 2},
            data_by_group={"with artery": x_with, "without artery": x_without},
        )

        dm = delta_median_pct(np.array([np.median(x_with), np.median(x_without)], float))
        ax.set_title(f"{g}\nΔmedian={dm:.1f}%", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10, labelleft=True)
        ax.set_xlim(0.3, 2.7)

        stats_out.append({
            "graph": g,
            "n_pairs": int(n),
            "mean_with_artery": float(np.mean(x_with)),
            "mean_without_artery": float(np.mean(x_without)),
            "median_with_artery": float(np.median(x_with)),
            "median_without_artery": float(np.median(x_without)),
            "mean_delta_no_minus_yes": float(np.mean(x_without - x_with)),
            "median_delta_no_minus_yes": float(np.median(x_without - x_with)),
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    ymaxs = [ax.get_ylim()[1] for ax in axes if ax.axison]
    common_ymax = max(ymaxs) if ymaxs else (global_ymax_data + 0.10 * yr)

    for ax in axes:
        if ax.axison:
            ax.set_ylim(common_ymin, common_ymax)

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])
    else:
        plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    plt.show()
    return pd.DataFrame(stats_out)


def comparison_to_df_with_without(comparison):
    """Convert with/without comparison to a typed dataframe."""
    df = pd.DataFrame(comparison).copy()

    num_cols = [
        "source_frontier",
        "target_with_artery",
        "target_without_artery",
        "path_len_with_artery",
        "path_len_without_artery",
        "capillary_len_with_artery",
        "capillary_len_without_artery",
        "arterial_len_with_artery",
        "arterial_len_without_artery",
        "venous_len_with_artery",
        "venous_len_without_artery",
        "path_len_with_artery_edges",
        "path_len_without_artery_edges",
        "delta_no_minus_yes",
        "delta_capillary_no_minus_yes",
        "delta_arterial_no_minus_yes",
        "delta_venous_no_minus_yes",
        "delta_edges_no_minus_yes",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "same_target_venous" in df.columns:
        df["same_target_venous"] = df["same_target_venous"].astype("boolean")

    return df


def comparison_to_df_restricted_vs_weighted(comparison):
    """Convert restricted/weighted comparison to a typed dataframe."""
    df = pd.DataFrame(comparison).copy()

    num_cols = [
        "source_frontier",
        "target_restricted",
        "target_weighted",
        "path_len_restricted",
        "path_len_weighted",
        "capillary_len_restricted",
        "capillary_len_weighted",
        "arterial_len_restricted",
        "arterial_len_weighted",
        "venous_len_restricted",
        "venous_len_weighted",
        "path_len_restricted_edges",
        "path_len_weighted_edges",
        "delta_weighted_minus_restricted",
        "delta_capillary_weighted_minus_restricted",
        "delta_arterial_weighted_minus_restricted",
        "delta_venous_weighted_minus_restricted",
        "delta_edges_weighted_minus_restricted",
        "weighted_arterial_len_um",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "same_target_venous" in df.columns:
        df["same_target_venous"] = df["same_target_venous"].astype("boolean")

    return df


def plot_frontier_path_differences_with_without(
    comparison,
    sort_by="delta_no_minus_yes",
    figsize=(12, 5),
    show_target_change=True,
):
    """Per-frontier plot for with vs without continuity."""
    df = comparison_to_df_with_without(comparison)

    dfp = df[
        df["has_path_with_artery"].fillna(False) &
        df["has_path_without_artery"].fillna(False)
    ].copy()

    if len(dfp) == 0:
        print("No frontier nodes with path in both conditions.")
        return dfp

    if sort_by in dfp.columns:
        dfp = dfp.sort_values(sort_by).reset_index(drop=True)
    else:
        dfp = dfp.sort_values("source_frontier").reset_index(drop=True)

    x = np.arange(len(dfp))

    plt.figure(figsize=figsize)
    plt.plot(x, dfp["path_len_with_artery"].to_numpy(), marker="o", linewidth=1, label="with artery")
    plt.plot(x, dfp["path_len_without_artery"].to_numpy(), marker="o", linewidth=1, label="without artery")
    plt.xlabel("Frontier nodes (sorted)")
    plt.ylabel("Shortest path length")
    plt.title("Shortest A-V path from frontier: with vs without arterial continuity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=figsize)
    y = dfp["delta_no_minus_yes"].to_numpy()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.axhline(0, linewidth=1)

    if show_target_change and "same_target_venous" in dfp.columns:
        changed = dfp["same_target_venous"] == False
        if np.any(changed):
            plt.scatter(x[changed.to_numpy()], y[changed.to_numpy()], marker="x", s=60, label="target venous changed")
            plt.legend()

    plt.xlabel("Frontier nodes (sorted)")
    plt.ylabel("Δ length = without artery - with artery")
    plt.title("Increase in shortest path length when arterial continuity is forbidden")
    plt.tight_layout()
    plt.show()

    return dfp


def plot_frontier_path_differences_restricted_vs_weighted(
    comparison,
    sort_by="delta_weighted_minus_restricted",
    figsize=(12, 5),
    show_target_change=True,
):
    """Per-frontier plot for restricted vs weighted+continuity."""
    df = comparison_to_df_restricted_vs_weighted(comparison)

    dfp = df[
        df["has_restricted"].fillna(False) &
        df["has_weighted"].fillna(False)
    ].copy()

    if len(dfp) == 0:
        print("No frontier nodes with path in both conditions.")
        return dfp

    if sort_by in dfp.columns:
        dfp = dfp.sort_values(sort_by).reset_index(drop=True)
    else:
        dfp = dfp.sort_values("source_frontier").reset_index(drop=True)

    x = np.arange(len(dfp))

    plt.figure(figsize=figsize)
    plt.plot(x, dfp["path_len_restricted"].to_numpy(), marker="o", linewidth=1, label="restricted")
    plt.plot(x, dfp["path_len_weighted"].to_numpy(), marker="o", linewidth=1, label="weighted+continuity")
    plt.xlabel("Frontier nodes (sorted)")
    plt.ylabel("Shortest path length")
    plt.title("Shortest A-V path from frontier: restricted vs weighted+continuity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=figsize)
    y = dfp["delta_weighted_minus_restricted"].to_numpy()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.axhline(0, linewidth=1)

    if show_target_change and "same_target_venous" in dfp.columns:
        changed = dfp["same_target_venous"] == False
        if np.any(changed):
            plt.scatter(x[changed.to_numpy()], y[changed.to_numpy()], marker="x", s=60, label="target venous changed")
            plt.legend()

    plt.xlabel("Frontier nodes (sorted)")
    plt.ylabel("Δ length = weighted - restricted")
    plt.title("Change in shortest path length with weighted+continuity")
    plt.tight_layout()
    plt.show()

    return dfp


# ======================================================================
# Summary tables / exports
# ======================================================================

def make_bc_face_table_from_bc_all(
    bc_all,
    face_col="Face",
    graph_col="graph",
    value_col="BC nodes",
    face_order=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
    graph_order=None,
):
    """Build a face x graph table of BC node counts."""
    df = bc_all.copy()

    if face_col not in df.columns:
        raise ValueError(f"'{face_col}' not in bc_all.columns. Available: {df.columns.tolist()}")
    if graph_col not in df.columns:
        raise ValueError(f"'{graph_col}' not in bc_all.columns. Available: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not in bc_all.columns. Available: {df.columns.tolist()}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    tab = df.groupby([face_col, graph_col], as_index=False)[value_col].sum()
    pivot = tab.pivot(index=face_col, columns=graph_col, values=value_col).fillna(0.0)

    if graph_order is not None:
        cols = [g for g in graph_order if g in pivot.columns]
        pivot = pivot.reindex(columns=cols)
    if face_order is not None:
        rows = [f for f in face_order if f in pivot.index]
        pivot = pivot.reindex(index=rows)

    return pivot.round(0).astype(int)


def save_major_trees_table_png(summary_df, out_path, graph_order=None):
    """Save a clean PNG table summarizing major artery and vein trees."""
    df = summary_df.copy()

    if graph_order is not None:
        df["graph"] = pd.Categorical(df["graph"], categories=graph_order, ordered=True)
        df = df.sort_values("graph")

    table_df = df[[
        "graph",
        "n_major_arteriole_trees",
        "n_arteriole_components",
        "n_major_venule_trees",
        "n_venule_components",
    ]].copy()

    table_df = table_df.rename(columns={
        "graph": "Box",
        "n_major_arteriole_trees": "Major arterioles",
        "n_arteriole_components": "Total arteriole components",
        "n_major_venule_trees": "Major venules",
        "n_venule_components": "Total venule components",
    })

    table_df["Arterioles (major/total)"] = (
        table_df["Major arterioles"].astype(int).astype(str)
        + "/"
        + table_df["Total arteriole components"].astype(int).astype(str)
    )

    table_df["Venules (major/total)"] = (
        table_df["Major venules"].astype(int).astype(str)
        + "/"
        + table_df["Total venule components"].astype(int).astype(str)
    )

    table_df = table_df[[
        "Box",
        "Major arterioles",
        "Total arteriole components",
        "Arterioles (major/total)",
        "Major venules",
        "Total venule components",
        "Venules (major/total)",
    ]]

    nrows, ncols = table_df.shape
    fig_w = max(10, ncols * 2.2)
    fig_h = max(1.8, 0.65 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.6)

    for c in range(ncols):
        cell = tbl[(0, c)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9eaf7")

    for r in range(1, nrows + 1):
        for c in range(ncols):
            cell = tbl[(r, c)]
            cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Saved table image:", out_path)


    # ======================================================================
# EXTRA HELPERS FOR FRONTIER COMPARISON, MIN-CUT AND PAIRED PLOTS
# ======================================================================

from itertools import combinations
from scipy.stats import ttest_ind, ttest_rel


def subset_ms_by_nkind(ms, nkind_code):
    """
    Return a filtered microsegment dictionary for one vessel class.
    """
    m = np.asarray(ms["nkind"], dtype=int) == int(nkind_code)
    return {
        "midpoints": ms["midpoints"][m],
        "lengths": ms["lengths"][m],
        "nkind": ms["nkind"][m],
        "r0": ms["r0"][m],
        "r1": ms["r1"][m],
    }


def _finite(x):
    """
    Keep only finite numeric values.
    """
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def pairwise_ttests_table(df, value_col, graphs_order, group_col="graph"):
    """
    Pairwise Welch t-tests between groups.
    """
    rows = []
    present = [g for g in graphs_order if g in set(df[group_col].astype(str))]

    for g1, g2 in combinations(present, 2):
        x1 = _finite(df.loc[df[group_col] == g1, value_col].to_numpy(float))
        x2 = _finite(df.loc[df[group_col] == g2, value_col].to_numpy(float))

        if x1.size < 2 or x2.size < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")

        rows.append({
            "metric": value_col,
            "group1": g1,
            "group2": g2,
            "n1": int(x1.size),
            "n2": int(x2.size),
            "mean1": float(np.mean(x1)) if x1.size else np.nan,
            "mean2": float(np.mean(x2)) if x2.size else np.nan,
            "median1": float(np.median(x1)) if x1.size else np.nan,
            "median2": float(np.median(x2)) if x2.size else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    return pd.DataFrame(rows)


def _p_to_text(p):
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.3g}"


def _p_to_stars(p):
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_pairwise_pvalues(ax, stats_df, positions, data_by_group):
    """
    Draw p-value brackets on a boxplot.
    """
    if stats_df is None or stats_df.empty:
        return

    valid_arrays = [np.asarray(v, float) for v in data_by_group.values() if len(v) > 0]
    if not valid_arrays:
        return

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if yr <= 0:
        yr = 1.0

    max_y = max(np.nanmax(v) for v in valid_arrays)
    n_pairs = len(stats_df)

    base = max_y + 0.05 * yr
    step = 0.10 * yr
    needed_top = base + (n_pairs + 1) * step

    if needed_top > ymax:
        ax.set_ylim(ymin, needed_top)
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin
        base = max_y + 0.05 * yr
        step = 0.08 * yr

    for i, row in enumerate(stats_df.itertuples(index=False)):
        g1, g2 = row.group1, row.group2
        if g1 not in positions or g2 not in positions:
            continue

        x1, x2 = positions[g1], positions[g2]
        if x1 > x2:
            x1, x2 = x2, x1

        y = base + i * step
        h = 0.025 * yr

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
        ax.text(
            (x1 + x2) / 2.0,
            y + h + 0.01 * yr,
            f"{_p_to_stars(row.p_value)} ({_p_to_text(row.p_value)})",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )


def add_jitter_scatter(ax, xpos, y, color="black", jitter=0.08, alpha=0.22, s=10, max_points=2000):
    """
    Add a light jittered scatter over a boxplot.
    """
    y = _finite(y)
    if y.size == 0:
        return

    if y.size > max_points:
        rng = np.random.default_rng(0)
        y = rng.choice(y, size=max_points, replace=False)

    rng = np.random.default_rng(0)
    xj = xpos + rng.uniform(-jitter, jitter, size=y.size)

    ax.scatter(
        xj, y,
        s=s,
        alpha=alpha,
        color=color,
        edgecolors="none",
        zorder=2
    )


def delta_median_pct(medians):
    """
    Relative spread between min and max medians.
    """
    medians = _finite(medians)
    if medians.size < 2:
        return np.nan
    m = float(np.median(medians))
    return np.nan if abs(m) < 1e-12 else 100.0 * (float(np.max(medians)) - float(np.min(medians))) / m


def summarize_frontier_paths(
    graph,
    paths,
    distances,
    source_frontiers,
    target_venous,
    artery_continuity,
    edge_type_attr="nkind",
    length_attr="length",
):
    """
    Summarize one set of frontier-to-vein shortest paths.
    """
    rows = []

    for i, path in enumerate(paths):
        seq = path_edge_type_sequence(graph, path, edge_type_attr=edge_type_attr)

        rows.append({
            "path_id": i,
            "source_frontier": int(source_frontiers[i]),
            "target_venous": int(target_venous[i]),
            "n_nodes": len(path),
            "n_edges": max(0, len(path) - 1),
            "total_length": float(distances[i]),
            "capillary_length": trimmed_capillary_length(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "arterial_length": arterial_length_in_path(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "venous_length": venous_length_in_path(
                graph, path,
                edge_type_attr=edge_type_attr,
                length_attr=length_attr,
            ),
            "starts_with_artery": bool(len(seq) > 0 and seq[0] == ARTERY),
            "contains_artery": bool(np.any(seq == ARTERY)),
            "artery_continuity": bool(artery_continuity),
            "path": path,
        })

    return pd.DataFrame(rows)


def compare_frontier_restricted_vs_weighted(
    graph,
    edge_type_attr="nkind",
    length_weight_attr="length",
    weighted_attr="w_cap_prior",
    tie_break_edges=True,
    tol=1e-9,
):
    """
    Compare:
      1) restricted shortest path: no arterial continuity, weight=length
      2) weighted shortest path: full continuity, weight=weighted_attr
    """
    paths_r, dist_r, n_edges_r, src_r, tgt_r, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        weight_attr=length_weight_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    paths_w, dist_w, n_edges_w, src_w, tgt_w, _, _ = shortest_av_paths_from_ac_frontier(
        graph,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        weight_attr=weighted_attr,
        tie_break_edges=tie_break_edges,
        tol=tol,
    )

    rows_r = summarize_frontier_paths(
        graph, paths_r, dist_r, src_r, tgt_r,
        artery_continuity=False,
        edge_type_attr=edge_type_attr,
        length_attr="length",
    )

    rows_w = summarize_frontier_paths(
        graph, paths_w, dist_w, src_w, tgt_w,
        artery_continuity=True,
        edge_type_attr=edge_type_attr,
        length_attr="length",
    )

    by_r = {int(r["source_frontier"]): r for _, r in rows_r.iterrows()}
    by_w = {int(r["source_frontier"]): r for _, r in rows_w.iterrows()}

    all_frontiers = sorted(set(by_r) | set(by_w))
    rows = []

    for f in all_frontiers:
        rr = by_r.get(f)
        rw = by_w.get(f)

        rows.append({
            "source_frontier": int(f),

            "has_restricted": rr is not None,
            "has_weighted": rw is not None,

            "target_restricted": np.nan if rr is None else float(rr["target_venous"]),
            "target_weighted": np.nan if rw is None else float(rw["target_venous"]),

            "same_target_venous": (
                np.nan if (rr is None or rw is None)
                else bool(int(rr["target_venous"]) == int(rw["target_venous"]))
            ),

            "path_len_restricted": np.nan if rr is None else float(rr["total_length"]),
            "path_len_weighted": np.nan if rw is None else float(rw["total_length"]),
            "delta_weighted_minus_restricted": (
                np.nan if (rr is None or rw is None)
                else float(rw["total_length"] - rr["total_length"])
            ),

            "path_len_restricted_edges": np.nan if rr is None else float(rr["n_edges"]),
            "path_len_weighted_edges": np.nan if rw is None else float(rw["n_edges"]),
            "delta_edges_weighted_minus_restricted": (
                np.nan if (rr is None or rw is None)
                else float(rw["n_edges"] - rr["n_edges"])
            ),

            "weighted_uses_artery": False if rw is None else bool(rw["arterial_length"] > 0),
            "weighted_arterial_len_um": np.nan if rw is None else float(rw["arterial_length"]),
            "weighted_arterial_edges": (
                0 if rw is None else int(np.sum(path_edge_type_sequence(graph, rw["path"], edge_type_attr=edge_type_attr) == ARTERY))
            ),
        })

    return pd.DataFrame(rows), rows_r, rows_w


def summarize_frontier_comparison_restricted_vs_weighted(df_cmp_local, graph_name):
    """
    Summary table for restricted vs weighted comparison.
    """
    ss = df_cmp_local.dropna(subset=["path_len_restricted", "path_len_weighted"]).copy()

    if ss.empty:
        return {
            "graph": graph_name,
            "n_pairs": 0,
            "n_target_changed": 0,
            "pct_target_changed": np.nan,
            "n_delta_um_nonzero": 0,
            "pct_delta_um_nonzero": np.nan,
            "n_delta_um_positive": 0,
            "pct_delta_um_positive": np.nan,
            "n_delta_um_negative": 0,
            "pct_delta_um_negative": np.nan,
            "n_delta_edges_nonzero": 0,
            "pct_delta_edges_nonzero": np.nan,
            "n_delta_edges_positive": 0,
            "pct_delta_edges_positive": np.nan,
            "n_delta_edges_negative": 0,
            "pct_delta_edges_negative": np.nan,
            "median_delta_um": np.nan,
            "mean_delta_um": np.nan,
            "median_delta_edges": np.nan,
            "mean_delta_edges": np.nan,
            "n_weighted_paths_using_artery": 0,
            "pct_weighted_paths_using_artery": np.nan,
            "median_weighted_arterial_len_um": np.nan,
            "mean_weighted_arterial_len_um": np.nan,
        }

    delta_um = ss["delta_weighted_minus_restricted"].to_numpy(float)
    delta_edges = ss["delta_edges_weighted_minus_restricted"].to_numpy(float)
    target_changed = (ss["same_target_venous"] == False).to_numpy()
    uses_artery = (ss["weighted_uses_artery"] == True).to_numpy()

    return {
        "graph": graph_name,
        "n_pairs": int(len(ss)),
        "n_target_changed": int(np.sum(target_changed)),
        "pct_target_changed": 100.0 * float(np.mean(target_changed)),
        "n_delta_um_nonzero": int(np.sum(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "pct_delta_um_nonzero": 100.0 * float(np.mean(~np.isclose(delta_um, 0.0, atol=1e-9))),
        "n_delta_um_positive": int(np.sum(delta_um > 0)),
        "pct_delta_um_positive": 100.0 * float(np.mean(delta_um > 0)),
        "n_delta_um_negative": int(np.sum(delta_um < 0)),
        "pct_delta_um_negative": 100.0 * float(np.mean(delta_um < 0)),
        "n_delta_edges_nonzero": int(np.sum(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "pct_delta_edges_nonzero": 100.0 * float(np.mean(~np.isclose(delta_edges, 0.0, atol=1e-9))),
        "n_delta_edges_positive": int(np.sum(delta_edges > 0)),
        "pct_delta_edges_positive": 100.0 * float(np.mean(delta_edges > 0)),
        "n_delta_edges_negative": int(np.sum(delta_edges < 0)),
        "pct_delta_edges_negative": 100.0 * float(np.mean(delta_edges < 0)),
        "median_delta_um": float(np.nanmedian(delta_um)),
        "mean_delta_um": float(np.nanmean(delta_um)),
        "median_delta_edges": float(np.nanmedian(delta_edges)),
        "mean_delta_edges": float(np.nanmean(delta_edges)),
        "n_weighted_paths_using_artery": int(np.sum(uses_artery)),
        "pct_weighted_paths_using_artery": 100.0 * float(np.mean(uses_artery)),
        "median_weighted_arterial_len_um": float(np.nanmedian(ss["weighted_arterial_len_um"])),
        "mean_weighted_arterial_len_um": float(np.nanmean(ss["weighted_arterial_len_um"])),
    }


def build_frontier_comparison_summary_table_restricted_vs_weighted(df_summary, graphs_order):
    """
    Compact summary table for the restricted vs weighted comparison.
    """
    metric_specs = [
        ("n_pairs", "Frontiers compared"),
        ("n_target_changed", "Target changed"),
        ("n_delta_um_positive", "Weighted longer than restricted"),
        ("median_delta_um", "Median Δ length (um)"),
        ("median_delta_edges", "Median Δ length (edges)"),
        ("n_weighted_paths_using_artery", "Weighted paths using artery"),
        ("median_weighted_arterial_len_um", "Median arterial length in weighted path (um)"),
    ]

    rows = []
    for metric_key, metric_label in metric_specs:
        row = {"metric": metric_label}

        for g in graphs_order:
            sub = df_summary[df_summary["graph"] == g]
            if sub.empty:
                row[g] = "NA"
                continue

            r = sub.iloc[0]

            if metric_key == "n_pairs":
                row[g] = f"{int(r['n_pairs'])}"

            elif metric_key == "n_target_changed":
                row[g] = f"{int(r['n_target_changed'])} ({r['pct_target_changed']:.1f}%)"

            elif metric_key == "n_delta_um_positive":
                row[g] = f"{int(r['n_delta_um_positive'])} ({r['pct_delta_um_positive']:.1f}%)"

            elif metric_key == "n_weighted_paths_using_artery":
                row[g] = f"{int(r['n_weighted_paths_using_artery'])} ({r['pct_weighted_paths_using_artery']:.1f}%)"

            else:
                val = r[metric_key]
                row[g] = "NA" if not np.isfinite(val) else f"{val:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


def paired_ttest_restricted_vs_weighted(df, graphs_order, group_col="graph"):
    """
    Paired t-test per graph for restricted vs weighted path length.
    """
    rows = []
    present = [g for g in graphs_order if g in set(df[group_col].astype(str))]

    for g in present:
        sub = df[df[group_col] == g].copy()
        sub = sub.dropna(subset=["path_len_restricted", "path_len_weighted"])

        x = _finite(sub["path_len_restricted"].to_numpy(float))
        y = _finite(sub["path_len_weighted"].to_numpy(float))

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if n < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(y, x, nan_policy="omit")

        rows.append({
            "graph": g,
            "n_pairs": int(n),
            "mean_restricted": float(np.mean(x)) if n else np.nan,
            "mean_weighted": float(np.mean(y)) if n else np.nan,
            "median_restricted": float(np.median(x)) if n else np.nan,
            "median_weighted": float(np.median(y)) if n else np.nan,
            "mean_delta_weighted_minus_restricted": float(np.mean(y - x)) if n else np.nan,
            "median_delta_weighted_minus_restricted": float(np.median(y - x)) if n else np.nan,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    return pd.DataFrame(rows)


def plot_restricted_vs_weighted_scatter(df_comparison, graph_name, figsize=(5, 5)):
    """
    Scatter plot: restricted vs weighted path length.
    """
    sub = df_comparison[df_comparison["graph"] == graph_name].copy()
    sub = sub.dropna(subset=["path_len_restricted", "path_len_weighted"])

    x = sub["path_len_restricted"].to_numpy(float)
    y = sub["path_len_weighted"].to_numpy(float)

    if len(sub) == 0:
        print(f"No paired data for {graph_name}")
        return

    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.7)
    mx = max(np.max(x), np.max(y)) if len(x) else 1.0
    plt.plot([0, mx], [0, mx], "--")
    plt.xlabel("Restricted (um)")
    plt.ylabel("Weighted + continuity (um)")
    plt.title(f"Restricted vs weighted | {graph_name}")
    plt.tight_layout()
    plt.show()


def plot_frontier_path_differences_restricted_vs_weighted(df_comparison, graph_name, figsize=(12, 5)):
    """
    Plot per-frontier delta for restricted vs weighted.
    """
    sub = df_comparison[df_comparison["graph"] == graph_name].copy()
    sub = sub.dropna(subset=["delta_weighted_minus_restricted"])
    sub = sub.sort_values("delta_weighted_minus_restricted").reset_index(drop=True)

    if len(sub) == 0:
        print(f"No delta data for {graph_name}")
        return

    x = np.arange(len(sub))
    y = sub["delta_weighted_minus_restricted"].to_numpy(float)
    changed = sub["same_target_venous"] == False

    plt.figure(figsize=figsize)
    plt.plot(x, y, marker="o", linewidth=1)
    plt.axhline(0, linewidth=1)

    if np.any(changed):
        plt.scatter(
            x[changed.to_numpy()],
            y[changed.to_numpy()],
            marker="x",
            s=60,
            label="target venous changed"
        )
        plt.legend()

    plt.xlabel("Frontier nodes (sorted)")
    plt.ylabel("Δ length = weighted - restricted (um)")
    plt.title(f"Per-frontier delta | {graph_name}")
    plt.tight_layout()
    plt.show()


def plot_paired_restricted_vs_weighted_boxplots_from_comparison(
    df_comparison,
    graphs_order,
    ylabel="Shortest path length (um)",
    title=None,
    sharey=True,
):
    """
    Paired boxplots per graph: restricted vs weighted.
    """
    present = [g for g in graphs_order if g in set(df_comparison["graph"].astype(str))]
    if len(present) == 0:
        print("No graphs found.")
        return pd.DataFrame()

    all_y = []
    for g in present:
        sub = df_comparison[df_comparison["graph"] == g].copy()
        sub = sub.dropna(subset=["path_len_restricted", "path_len_weighted"])

        x_r = _finite(sub["path_len_restricted"].to_numpy(float))
        x_w = _finite(sub["path_len_weighted"].to_numpy(float))

        if x_r.size:
            all_y.append(x_r)
        if x_w.size:
            all_y.append(x_w)

    if len(all_y) == 0:
        print("No paired values found.")
        return pd.DataFrame()

    all_y = np.concatenate(all_y)
    global_ymin_data = float(np.nanmin(all_y))
    global_ymax_data = float(np.nanmax(all_y))
    yr = global_ymax_data - global_ymin_data
    if not np.isfinite(yr) or yr <= 0:
        yr = 1.0

    common_ymin = global_ymin_data - 0.03 * yr

    fig, axes = plt.subplots(
        1, len(present),
        figsize=(5.2 * len(present), 5.8),
        sharey=sharey
    )
    if len(present) == 1:
        axes = [axes]

    stats_out = []

    for ax, g in zip(axes, present):
        sub = df_comparison[df_comparison["graph"] == g].copy()
        sub = sub.dropna(subset=["path_len_restricted", "path_len_weighted"])

        x_r = _finite(sub["path_len_restricted"].to_numpy(float))
        x_w = _finite(sub["path_len_weighted"].to_numpy(float))

        n = min(len(x_r), len(x_w))
        x_r = x_r[:n]
        x_w = x_w[:n]

        if n == 0:
            ax.set_axis_off()
            continue

        bp = ax.boxplot(
            [x_r, x_w],
            labels=["restricted", "weighted"],
            showfliers=False,
            patch_artist=True,
            widths=0.5
        )

        bp["boxes"][0].set_facecolor("tab:blue")
        bp["boxes"][0].set_alpha(0.30)
        bp["boxes"][1].set_facecolor("tab:orange")
        bp["boxes"][1].set_alpha(0.30)

        add_jitter_scatter(ax, 1, x_r, color="tab:blue")
        add_jitter_scatter(ax, 2, x_w, color="tab:orange")

        if n < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = ttest_rel(x_w, x_r, nan_policy="omit")

        st = pd.DataFrame([{
            "group1": "restricted",
            "group2": "weighted",
            "p_value": p_val,
        }])

        add_pairwise_pvalues(
            ax,
            st,
            positions={"restricted": 1, "weighted": 2},
            data_by_group={"restricted": x_r, "weighted": x_w}
        )

        dm = delta_median_pct(np.array([np.median(x_r), np.median(x_w)], float))
        ax.set_title(f"{g}\nΔmedian={dm:.1f}%", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10, labelleft=True)
        ax.set_xlim(0.3, 2.7)

        stats_out.append({
            "graph": g,
            "n_pairs": int(n),
            "mean_restricted": float(np.mean(x_r)),
            "mean_weighted": float(np.mean(x_w)),
            "median_restricted": float(np.median(x_r)),
            "median_weighted": float(np.median(x_w)),
            "mean_delta_weighted_minus_restricted": float(np.mean(x_w - x_r)),
            "median_delta_weighted_minus_restricted": float(np.median(x_w - x_r)),
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    ymaxs = []
    for ax in axes:
        if ax.axison:
            ymaxs.append(ax.get_ylim()[1])

    common_ymax = max(ymaxs) if ymaxs else (global_ymax_data + 0.10 * yr)

    for ax in axes:
        if ax.axison:
            ax.set_ylim(common_ymin, common_ymax)

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])
    else:
        plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    plt.show()
    return pd.DataFrame(stats_out)


def av_sets_from_incident_types(graph):
    """
    Define arterial and venous node sets from incident edge types.
    """
    A, V = [], []
    for v in graph.vs:
        lab = infer_node_type_from_incident_edges(graph, v.index)
        if lab == "arteriole":
            A.append(v.index)
        elif lab == "venule":
            V.append(v.index)
    return np.array(A, dtype=int), np.array(V, dtype=int)


def export_edge_ids_vtp(graph, edge_ids, out_path, coords_attr="coords"):
    """
    Export selected original graph edges to VTP for ParaView.
    """
    edge_ids = list(sorted(set(int(eid) for eid in edge_ids)))
    if len(edge_ids) == 0:
        print(f"[export_edge_ids_vtp] No edges to export: {out_path}")
        return

    coords = np.asarray(graph.vs[coords_attr], float)

    points = []
    connectivity = []
    offsets = []

    cell_eid = []
    cell_nkind = []
    cell_length = []

    pidx = 0
    for eid in edge_ids:
        e = graph.es[int(eid)]
        s, t = e.tuple

        ps = coords[int(s)]
        pt = coords[int(t)]

        points.append([float(ps[0]), float(ps[1]), float(ps[2])])
        points.append([float(pt[0]), float(pt[1]), float(pt[2])])

        connectivity.extend([pidx, pidx + 1])
        pidx += 2
        offsets.append(pidx)

        cell_eid.append(int(eid))
        cell_nkind.append(int(e["nkind"]) if "nkind" in graph.es.attributes() else -1)
        cell_length.append(float(e["length"]) if "length" in graph.es.attributes() else np.nan)

    points_arr = np.asarray(points, float).reshape(-1, 3)
    connectivity_arr = np.asarray(connectivity, int)
    offsets_arr = np.asarray(offsets, int)

    def arr_to_ascii(a):
        return " ".join(map(str, np.ravel(a)))

    xml = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{len(points_arr)}" NumberOfLines="{len(cell_eid)}">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {arr_to_ascii(points_arr.astype(np.float32))}
        </DataArray>
      </Points>
      <Lines>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {arr_to_ascii(connectivity_arr.astype(np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {arr_to_ascii(offsets_arr.astype(np.int32))}
        </DataArray>
      </Lines>
      <CellData>
        <DataArray type="Int32" Name="eid" format="ascii">
          {arr_to_ascii(np.asarray(cell_eid, dtype=np.int32))}
        </DataArray>
        <DataArray type="Int32" Name="nkind" format="ascii">
          {arr_to_ascii(np.asarray(cell_nkind, dtype=np.int32))}
        </DataArray>
        <DataArray type="Float32" Name="length" format="ascii">
          {arr_to_ascii(np.asarray(cell_length, dtype=np.float32))}
        </DataArray>
      </CellData>
    </Piece>
  </PolyData>
</VTKFile>
'''

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"[export_edge_ids_vtp] Saved: {out_path}")



# ---------------------------------------------------------------------
# Saturation / interface proxies
# ---------------------------------------------------------------------


def _nodes_of_edge_type(graph, edge_type_attr="nkind", labels=(2,)):
    labels = {int(x) for x in labels}
    nodes = set()
    for e in graph.es:
        if int(e[edge_type_attr]) in labels:
            u, v = e.tuple
            nodes.add(int(u))
            nodes.add(int(v))
    return nodes


def saturation_interface_proxies(graph, edge_type_attr="nkind"):
    arterial_nodes = _nodes_of_edge_type(graph, edge_type_attr=edge_type_attr, labels=(ARTERY,))
    venous_nodes = _nodes_of_edge_type(graph, edge_type_attr=edge_type_attr, labels=(VEIN,))

    art_cap_eids = []
    ven_cap_eids = []

    art_cap_neighbors = set()
    ven_cap_neighbors = set()

    for e in graph.es:
        if int(e[edge_type_attr]) != CAPILLARY:
            continue

        u, v = map(int, e.tuple)

        if u in arterial_nodes or v in arterial_nodes:
            art_cap_eids.append(int(e.index))
            if u in arterial_nodes and v not in arterial_nodes:
                art_cap_neighbors.add(v)
            elif v in arterial_nodes and u not in arterial_nodes:
                art_cap_neighbors.add(u)

        if u in venous_nodes or v in venous_nodes:
            ven_cap_eids.append(int(e.index))
            if u in venous_nodes and v not in venous_nodes:
                ven_cap_neighbors.add(v)
            elif v in venous_nodes and u not in venous_nodes:
                ven_cap_neighbors.add(u)

    try:
        ac_frontier_nodes = get_ac_frontier_nodes(graph, edge_type_attr=edge_type_attr)
        n_ac_frontier = int(len(ac_frontier_nodes))
    except Exception:
        ac_frontier_nodes = sorted(list(art_cap_neighbors))
        n_ac_frontier = int(len(ac_frontier_nodes))

    quick_upper_bound = int(min(len(art_cap_neighbors), len(ven_cap_neighbors)))

    out = {
        "arterial_nodes": int(len(arterial_nodes)),
        "venous_nodes": int(len(venous_nodes)),
        "capillary_edges_attached_to_artery": int(len(art_cap_eids)),
        "capillary_edges_attached_to_vein": int(len(ven_cap_eids)),
        "unique_capillary_nodes_touching_artery": int(len(art_cap_neighbors)),
        "unique_capillary_nodes_touching_vein": int(len(ven_cap_neighbors)),
        "ac_frontier_nodes": int(n_ac_frontier),
        "quick_upper_bound_interface": quick_upper_bound,
    }
    return out
# ---------------------------------------------------------------------
# Min-cut as main connectivity metric
# ---------------------------------------------------------------------
def av_min_cut_metrics(
    graph,
    edge_type_attr="nkind",
    per_edge_capacity=1.0,
):
    arterial_nodes = sorted(_nodes_of_edge_type(graph, edge_type_attr=edge_type_attr, labels=(ARTERY,)))
    venous_nodes = sorted(_nodes_of_edge_type(graph, edge_type_attr=edge_type_attr, labels=(VEIN,)))

    if len(arterial_nodes) == 0 or len(venous_nodes) == 0:
        return {
            "min_cut_value": np.nan,
            "max_flow_value": np.nan,
            "n_arterial_nodes": int(len(arterial_nodes)),
            "n_venous_nodes": int(len(venous_nodes)),
            "cut_edge_ids": [],
            "cut_edge_types": [],
            "n_cut_edges_original_graph": 0,
        }

    g2 = graph.copy()

    orig_n = g2.vcount()
    g2.add_vertices(2)
    super_source = orig_n
    super_sink = orig_n + 1

    big_cap = float(max(1, graph.ecount() + 1))

    extra_edges = []
    extra_caps = []
    extra_orig_eid = []

    if "tmp_capacity_mc" in g2.es.attributes():
        del g2.es["tmp_capacity_mc"]

    g2.es["tmp_capacity_mc"] = [float(per_edge_capacity)] * g2.ecount()
    g2.es["orig_eid_mc"] = list(range(graph.ecount()))

    for a in arterial_nodes:
        extra_edges.append((super_source, int(a)))
        extra_caps.append(big_cap)
        extra_orig_eid.append(-1)

    for v in venous_nodes:
        extra_edges.append((int(v), super_sink))
        extra_caps.append(big_cap)
        extra_orig_eid.append(-1)

    if len(extra_edges) > 0:
        start_e = g2.ecount()
        g2.add_edges(extra_edges)
        new_eids = list(range(start_e, g2.ecount()))

        cap_all = list(g2.es["tmp_capacity_mc"])
        orig_all = list(g2.es["orig_eid_mc"])

        for eid, cap, oeid in zip(new_eids, extra_caps, extra_orig_eid):
            cap_all[eid] = float(cap)
            orig_all[eid] = int(oeid)

        g2.es["tmp_capacity_mc"] = cap_all
        g2.es["orig_eid_mc"] = orig_all

    mf = g2.maxflow(
        super_source,
        super_sink,
        capacity=g2.es["tmp_capacity_mc"]
    )

    cut_eids_aug = list(mf.cut)
    cut_orig_eids = []
    cut_edge_types = []

    for eid in cut_eids_aug:
        oeid = int(g2.es[eid]["orig_eid_mc"])
        if oeid >= 0:
            cut_orig_eids.append(oeid)
            cut_edge_types.append(int(graph.es[oeid][edge_type_attr]))

    return {
        "min_cut_value": float(mf.value),
        "max_flow_value": float(mf.value),
        "n_arterial_nodes": int(len(arterial_nodes)),
        "n_venous_nodes": int(len(venous_nodes)),
        "cut_edge_ids": cut_orig_eids,
        "cut_edge_types": cut_edge_types,
        "n_cut_edges_original_graph": int(len(cut_orig_eids)),
    }


def cut_type_counts(cut_edge_types):
    arr = np.asarray(cut_edge_types, dtype=int)
    return {
        "min_cut_n_arteriole_edges": int(np.sum(arr == ARTERY)),
        "min_cut_n_venule_edges": int(np.sum(arr == VEIN)),
        "min_cut_n_capillary_edges": int(np.sum(arr == CAPILLARY)),
        "min_cut_n_unknown_edges": int(np.sum(~np.isin(arr, [ARTERY, VEIN, CAPILLARY]))),
    }

