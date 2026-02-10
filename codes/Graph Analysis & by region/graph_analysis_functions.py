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

def make_box(
    center_vox,
    box_um,
    res_um_per_vox,
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
    if coords_attr not in graph.vs.attributes():
        raise ValueError(f"Missing vertex attribute '{coords_attr}'. "
                         f"Available: {graph.vs.attributes()}")
    P = np.asarray(graph.vs[coords_attr], dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got {P.shape}.")
    return P


def validate_box(box):
    """Raise if box misses required keys."""
    req = ("xmin","xmax","ymin","ymax","zmin","zmax")
    missing = [k for k in req if k not in box]
    if missing:
        raise ValueError(f"box missing keys: {missing}. Required: {list(req)}")


def _nkind_label_and_color(nkind):
    """
    Requested scheme:
      arteriole = red
      venule   = blue
      capillary= gray
    """
    if nkind == 2:
        return "arteriole", "red"
    if nkind == 3:
        return "venule", "blue"
    if nkind == 4:
        return "capillary", "gray"
    return "other", "purple"


def _incident_edges_for_vertices(graph, vertices):
    """Return unique incident edge IDs for a list of vertices."""
    inc = set()
    for v in vertices:
        inc.update(graph.incident(int(v)))
    return sorted(inc)


def _degree_distribution(graph, nodes):
    deg = np.asarray(graph.degree(), dtype=int)
    nodes = np.asarray(nodes, dtype=int)
    return Counter(deg[nodes]) if len(nodes) else Counter()


def _distance_to_surface_stats(graph, nodes):
    if "distance_to_surface" not in graph.vs.attributes():
        return None
    d = np.asarray(graph.vs["distance_to_surface"], dtype=float)
    nodes = np.asarray(nodes, dtype=int)
    if len(nodes) == 0:
        return None
    vals = d[nodes]
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


# ====================================================================================================================
#                                               CLASSIC GRAPH ANALYSIS
# ====================================================================================================================

def single_connected_component(graph):
    """
    Analyze whether the graph consists of a single connected component.
    Returns (is_single, n_components).
    """
    is_single = graph.is_connected()
    components = graph.components()
    n_components = len(components)

    if is_single:
        print("The graph is a single connected component.")
    else:
        print("The graph has more than one connected component.")
    print("Number of connected components:", n_components)
    return is_single, n_components


def get_edges_types(graph):
    """Count how many edges belong to each nkind type."""
    if "nkind" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'nkind'.")
    edge_types = graph.es["nkind"]
    unique, counts = np.unique(edge_types, return_counts=True)
    print("\nEdge types:\n")
    for i, n in zip(unique, counts):
        print(f" - {vessel_type.get(int(i), i)}, {i}, Count: {n}")
    return unique, counts


def get_diameter_nkind(graph):
    """Compute mean edge diameter per nkind."""
    if "diameter" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'diameter'.")
    if "nkind" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'nkind'.")
    diam = np.asarray(graph.es["diameter"], dtype=float)
    nkind = np.asarray(graph.es["nkind"], dtype=int)

    unique = np.unique(nkind)
    mean_diameters = np.array([diam[nkind == k].mean() for k in unique])

    print("\nAverage diameter by nkind:\n")
    for k, m in zip(unique, mean_diameters):
        print(f"nkind = {k}: average diameter = {m:.6f} µm")
    return unique, mean_diameters


def diameter_stats(graph, label_dict=None, ranges=None, plot=False):
    """
    Compute statistics of edge diameters grouped by nkind, and optionally plot them.
    Returns stats_dict (always).
    """
    if "diameter" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'diameter'.")
    if "nkind" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'nkind'.")

    diam = np.array(graph.es["diameter"], dtype=float)
    nkind = np.array(graph.es["nkind"], dtype=int)

    stats_dict = {}
    print("\n=== Diameter statistics ===\n")

    for k in np.unique(nkind):
        subset = diam[nkind == k]
        vname = label_dict.get(k, str(k)) if label_dict else str(k)

        mean = float(np.mean(subset))
        median = float(np.median(subset))
        p5 = float(np.percentile(subset, 5))
        p95 = float(np.percentile(subset, 95))

        perc_in_range = None
        if ranges and k in ranges:
            low, high = ranges[k]
            perc_in_range = float(np.mean((subset >= low) & (subset <= high)) * 100)
        else:
            low, high = None, None

        stats_dict[k] = {
            "name": vname,
            "mean": mean,
            "median": median,
            "p5": p5,
            "p95": p95,
            "perc_in_range": perc_in_range,
            "range": (low, high)
        }

        print(f"{vname} (nkind={k}):")
        print(f"  Mean diameter:     {mean:.2f} µm")
        print(f"  Median diameter:   {median:.2f} µm")
        print(f"  P5–P95 range:      {p5:.2f} – {p95:.2f} µm")
        if perc_in_range is not None:
            print(f"  % in normal range ({low}–{high} µm): {perc_in_range:.1f}%")
        print()

    if plot:
        plt.figure(figsize=(8, 5))
        nkinds_sorted = sorted(stats_dict.keys())
        names = [stats_dict[k]["name"] for k in nkinds_sorted]
        means = [stats_dict[k]["mean"] for k in nkinds_sorted]
        p5s = [stats_dict[k]["p5"] for k in nkinds_sorted]
        p95s = [stats_dict[k]["p95"] for k in nkinds_sorted]

        x = np.arange(len(nkinds_sorted))
        plt.bar(x, means, color="steelblue", edgecolor="black", label="Mean")
        plt.errorbar(
            x, means,
            yerr=[np.array(means) - np.array(p5s), np.array(p95s) - np.array(means)],
            fmt="none", ecolor="red", capsize=5, label="P5–P95"
        )

        if ranges:
            for i, k in enumerate(nkinds_sorted):
                low, high = ranges.get(k, (None, None))
                if low is not None:
                    plt.fill_between([i - 0.4, i + 0.4], low, high, color="orange", alpha=0.2)

        plt.xticks(x, names)
        plt.xlabel("Vessel type")
        plt.ylabel("Diameter (µm)")
        plt.title("Edge diameter statistics by vessel type")
        plt.legend()
        plt.show()

    return stats_dict


def get_length_nkind(graph):
    """Compute mean edge length per nkind."""
    if "length" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'length'.")
    if "nkind" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'nkind'.")
        
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


def get_degrees(graph, threshold=4):
    """
    Compute node degrees and identify high-degree nodes based on a threshold.
    Adds:
      graph.vs["degree"]
      graph.vs["high_degree_node"] (degree if high, else 0)
    Returns (unique_degrees, high_degree_nodes)
    """
    degrees = np.array(graph.degree(), dtype=int)
    graph.vs["degree"] = degrees

    degree_hd = np.where(degrees >= int(threshold), degrees, 0)
    graph.vs["high_degree_node"] = degree_hd

    high_degree = np.where(degrees >= int(threshold))[0]

    print("\nDegrees of nodes:", np.unique(degrees))
    print(f"High-degree nodes (>= {threshold}): {len(high_degree)}")
    return np.unique(degrees), high_degree


def get_location_degrees(graph, node_list):
    """Print distance_to_surface stats for a list of node indices."""
    if "distance_to_surface" not in graph.vs.attributes():
        raise ValueError("graph.vs['distance_to_surface'] missing.")
    dist = graph.vs["distance_to_surface"]
    node_list = list(map(int, node_list))

    distances = np.array([dist[i] for i in node_list], dtype=float)
    print(f"\nHigh-degree nodes analyzed: {len(node_list)}")

    for i in node_list[:10]:
        print(f"Node {i} -> distance to surface = {dist[i]:.2f} µm")

    print("\nDistance-to-surface statistics (µm):")
    print(f"  Min:   {distances.min():.2f}")
    print(f"  Mean:  {distances.mean():.2f}")
    print(f"  Median:{np.median(distances):.2f}")
    print(f"  Max:   {distances.max():.2f}")

    superficial = np.sum(distances < 20)
    superficial_2 = np.sum((distances >= 20) & (distances < 50))
    middle = np.sum((distances >= 50) & (distances < 200))
    deep = np.sum(distances >= 200)

    print("\nClassification by depth:")
    print(f"  Superficial (<20 µm):        {superficial}  ({100*superficial/len(distances):.1f}%)")
    print(f"  Superficial_2 (20–50 µm):    {superficial_2}  ({100*superficial_2/len(distances):.1f}%)")
    print(f"  Middle (50–200 µm):          {middle}       ({100*middle/len(distances):.1f}%)")
    print(f"  Deep (>200 µm):              {deep}         ({100*deep/len(distances):.1f}%)")

    return distances


def find_duplicated_edges(graph):
    """Detect duplicated edges regardless of direction."""
    edges = graph.get_edgelist()
    sorted_edges = [tuple(sorted(edge)) for edge in edges]
    count = Counter(sorted_edges)
    duplicated = [edge for edge, cnt in count.items() if cnt > 1]
    print("\nNumber of duplicated edges:", len(duplicated))
    return len(duplicated)


def find_loops(graph):
    """Detect self-loops (edges where source == target)."""
    loop_edges = [e.index for e in graph.es if e.source == e.target]
    count = len(loop_edges)
    perc = (count / graph.ecount()) * 100 if graph.ecount() > 0 else 0.0

    print(f"Percentage of self-loop edges: {perc:.4f}%")
    print("Loop edges indices:", loop_edges)
    print("Count of loop edges:", count)

    return loop_edges, count, perc


# ====================================================================================================================
# BC DETECTION (faces-only)
# ====================================================================================================================

def bc_nodes_on_plane(graph, axis, value, box, coords_attr, eps=2.0):
    """
    Returns node indices that are:
      (1) on the plane within eps
      (2) inside the box in the other coordinates (with eps margin)
    """
    C = get_coords(graph, coords_attr)

    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    inside = (
        (C[:, 0] >= xmin - eps) & (C[:, 0] <= xmax + eps) &
        (C[:, 1] >= ymin - eps) & (C[:, 1] <= ymax + eps) &
        (C[:, 2] >= zmin - eps) & (C[:, 2] <= zmax + eps)
    )
    on_plane = np.abs(C[:, axis] - float(value)) <= float(eps)
    return np.where(inside & on_plane)[0]


def bc_nodes_on_box_faces(graph, box, coords_attr, eps=0.5):
    """Return dict face_name -> np.array(node_ids) for 6 faces."""
    required = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    missing = [k for k in required if k not in box]
    if missing:
        raise ValueError(f"box is missing keys: {missing}. Required: {required}")

    return {
        "x_min": bc_nodes_on_plane(graph, 0, box["xmin"], box, coords_attr, eps),
        "x_max": bc_nodes_on_plane(graph, 0, box["xmax"], box, coords_attr, eps),
        "y_min": bc_nodes_on_plane(graph, 1, box["ymin"], box, coords_attr, eps),
        "y_max": bc_nodes_on_plane(graph, 1, box["ymax"], box, coords_attr, eps),
        "z_min": bc_nodes_on_plane(graph, 2, box["zmin"], box, coords_attr, eps),
        "z_max": bc_nodes_on_plane(graph, 2, box["zmax"], box, coords_attr, eps),
    }


def bc_node_common_nkind(graph, node_id):
    """Return most common nkind among incident edges of node."""
    if "nkind" not in graph.es.attributes():
        raise ValueError("Missing edge attribute 'nkind'.")
    inc_edges = graph.incident(int(node_id))
    nk = [graph.es[e]["nkind"] for e in inc_edges]
    nk = [x for x in nk if x is not None]
    if len(nk) == 0:
        return None
    return Counter(nk).most_common(1)[0][0]


def infer_node_type_from_incident_edges(graph, node_id, vessel_type_map=EDGE_NKIND_TO_LABEL):
    nk = bc_node_common_nkind(graph, node_id)
    if nk is None:
        return "unknown"
    return vessel_type_map.get(int(nk), f"nkind_{nk}")


def analyze_bc_faces(graph, box, coords_attr, eps=2.0, degree_thr=4, verbose=True):
    """
    BC ANALYSIS PER FACE
    Returns dict: face -> metrics
    """
    faces = bc_nodes_on_box_faces(graph, box, coords_attr, eps=eps)
    degrees = np.asarray(graph.degree(), dtype=int)
    has_d2s = ("distance_to_surface" in graph.vs.attributes())
    validate_box(box)

    results = {}

    if verbose:
        print(f"\n=== BC ANALYSIS (PER FACE) ===")
        print(f"Graph: {graph.vcount()} vertices, {graph.ecount()} edges")
        print(f"Coords attr: '{coords_attr}' | eps: {eps}")

    for face, nodes in faces.items():
        nodes = np.asarray(nodes, dtype=int)
        total = int(len(nodes))

        labels = [infer_node_type_from_incident_edges(graph, int(v)) for v in nodes]
        type_counts = Counter(labels)

        deg_counts = Counter(degrees[nodes]) if total else Counter()
        high_deg_nodes = nodes[degrees[nodes] >= int(degree_thr)] if total else np.array([], dtype=int)
        high_deg_percent = (100.0 * len(high_deg_nodes) / total) if total else 0.0

        dstat = _distance_to_surface_stats(graph, nodes) if has_d2s else None

        results[face] = {
            "nodes": nodes,
            "count": total,
            "type_counts": dict(type_counts),
            "type_percent": {k: (100.0 * v / total) for k, v in type_counts.items()} if total else {},
            "degree_counts": dict(deg_counts),
            "high_degree_nodes_face": high_deg_nodes,
            "high_degree_percent_face": float(high_deg_percent),
            "distance_to_surface_stats": dstat,
        }

        if verbose:
            print(f"\n--- Face {face} ---")
            print(f"BC nodes: {total}")
            if total:
                for k, v in type_counts.most_common():
                    print(f"  {k}: {v} ({100.0*v/total:.1f}%)")
                deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(deg_counts.items())])
                print(f"Degree distribution (degree:count): {deg_str}")
                print(f"High-degree (>= {degree_thr}): {len(high_deg_nodes)} ({high_deg_percent:.2f}%)")
            if dstat is not None:
                print("distance_to_surface (µm): "
                      f"min={dstat['min']:.2f}, mean={dstat['mean']:.2f}, "
                      f"median={dstat['median']:.2f}, max={dstat['max']:.2f}")

    return results


# ====================================================================================================================
# PLOTTING: GENERAL
# ====================================================================================================================

def plot_bar_by_category(categ, attribute_toplot, label_dict=None,
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


def plot_hist_by_category(attribute_toplot, category, label_dict=None,
                                xlabel="Value", plot_title="Category"):
    unique_cats = np.unique(category)
    global_min = attribute_toplot.min()
    global_max = attribute_toplot.max()
    bins = np.linspace(global_min, global_max, 80)

    plt.figure(figsize=(12, 4 * len(unique_cats)))

    for i, c in enumerate(unique_cats, 1):
        plt.subplot(len(unique_cats), 1, i)
        subset = attribute_toplot[category == c]
        mean_value = subset.mean()

        plt.hist(subset, bins=bins, alpha=0.75, density=True)
        plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5)
        plt.xlim(global_min, global_max)

        name = label_dict.get(c, "Unknown") if label_dict else c
        plt.title(f"{plot_title} {c} ({name})")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend([f"Mean = {mean_value:.2f}"])

    plt.tight_layout()
    plt.show()


# ====================================================================================================================
# PLOTTING: BC CUBES (faces-only)
# ====================================================================================================================
def cube_geometry_from_box(box):
    xmin, xmax = float(box["xmin"]), float(box["xmax"])
    ymin, ymax = float(box["ymin"]), float(box["ymax"])
    zmin, zmax = float(box["zmin"]), float(box["zmax"])

    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=float)

    cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]

    face_polys = {
        "x_min": [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        "x_max": [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
        "y_min": [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        "y_max": [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        "z_min": [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        "z_max": [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
    }

    limits = (xmin, xmax, ymin, ymax, zmin, zmax)
    return {"corners": corners, "edges": cube_edges, "face_polys": face_polys, "limits": limits}



def draw_box_cube(ax, box_geom, faces_to_tint=(), face_colors=None, face_alpha=0.08,
                  elev=18, azim=35, title=None, edge_lw=1.2):
    corners = box_geom["corners"]
    cube_edges = box_geom["edges"]
    face_polys = box_geom["face_polys"]
    xmin, xmax, ymin, ymax, zmin, zmax = box_geom["limits"]

    # edges
    for a, b in cube_edges:
        ax.plot([corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                linewidth=edge_lw)

    # faces
    if faces_to_tint:
        polys = [face_polys[f] for f in faces_to_tint]
        cols = [face_colors.get(f, "lightgray") for f in faces_to_tint] if face_colors else ["lightgray"]*len(polys)
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors="k", linewidths=0.8, alpha=face_alpha)
        ax.add_collection3d(pc)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)


def scatter_nodes_by_label(ax, coords, node_ids, label_fn,
                           label_to_color, point_size=10, alpha=0.85,
                           sample_max=None, clip_box=None, clip_eps=2.0, verbose_clip=False):
    node_ids = np.asarray(node_ids, dtype=int)
    if node_ids.size == 0:
        return {"dropped": 0, "kept": 0}

    if sample_max is not None and node_ids.size > sample_max:
        node_ids = np.random.choice(node_ids, size=int(sample_max), replace=False)

    pts = coords[node_ids]

    # optional clipping to box
    if clip_box is not None:
        xmin, xmax = float(clip_box["xmin"]), float(clip_box["xmax"])
        ymin, ymax = float(clip_box["ymin"]), float(clip_box["ymax"])
        zmin, zmax = float(clip_box["zmin"]), float(clip_box["zmax"])
        m = (
            (pts[:, 0] >= xmin - clip_eps) & (pts[:, 0] <= xmax + clip_eps) &
            (pts[:, 1] >= ymin - clip_eps) & (pts[:, 1] <= ymax + clip_eps) &
            (pts[:, 2] >= zmin - clip_eps) & (pts[:, 2] <= zmax + clip_eps)
        )
        dropped = int(np.sum(~m))
        if verbose_clip and dropped:
            print(f"[clip] Dropped {dropped}/{len(node_ids)} nodes outside box (eps={clip_eps}).")
        node_ids = node_ids[m]
        pts = pts[m]
    else:
        dropped = 0

    if node_ids.size == 0:
        return {"dropped": dropped, "kept": 0}

    labels = np.array([label_fn(v) for v in node_ids], dtype=object)
    for lab, col in label_to_color.items():
        mm = labels == lab
        if np.any(mm):
            ax.scatter(pts[mm, 0], pts[mm, 1], pts[mm, 2], s=point_size, alpha=alpha, color=col)

    return {"dropped": dropped, "kept": int(node_ids.size)}



def plot_bc_3_cubes(
    G, res, box, coords_attr,
    faces_A=("x_min", "z_max"),
    faces_B=("x_max", "z_min"),
    faces_C=("y_min", "y_max"),
    elev=18, azim=35,
    face_alpha=0.08,
    point_alpha=0.85,
    point_size=10,
    sample_max=None,
    clip_eps=2.0,
    verbose_clip=True,
    show_face_legend=True,
    show_vessel_legend=True
):
    coords = get_coords(G, coords_attr)  # <- usa tu función única
    geom = cube_geometry_from_box(box)

    vessel_colors = {"arteriole": "red", "venule": "blue", "capillary": "gray", "unknown": "black"}
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    face_colors = {
        "x_min": cycle[0], "x_max": cycle[0],
        "y_min": cycle[1], "y_max": cycle[1],
        "z_min": cycle[2], "z_max": cycle[2],
    }

    def label_fn(v):  # reuse existing BC logic
        return infer_node_type_from_incident_edges(G, int(v))

    def nodes_for_faces(faces_subset):
        nodes = []
        for f in faces_subset:
            nodes.extend(list(res[f]["nodes"]))
        return np.asarray(nodes, dtype=int)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    draw_box_cube(ax1, geom, faces_to_tint=faces_A, face_colors=face_colors, face_alpha=face_alpha,
                  elev=elev, azim=azim, title=f"View A: {faces_A[0]} + {faces_A[1]}")
    scatter_nodes_by_label(ax1, coords, nodes_for_faces(faces_A), label_fn, vessel_colors,
                           point_size=point_size, alpha=point_alpha, sample_max=sample_max,
                           clip_box=box, clip_eps=clip_eps, verbose_clip=verbose_clip)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_box_cube(ax2, geom, faces_to_tint=faces_B, face_colors=face_colors, face_alpha=face_alpha,
                  elev=elev, azim=azim, title=f"View B: {faces_B[0]} + {faces_B[1]}")
    scatter_nodes_by_label(ax2, coords, nodes_for_faces(faces_B), label_fn, vessel_colors,
                           point_size=point_size, alpha=point_alpha, sample_max=sample_max,
                           clip_box=box, clip_eps=clip_eps, verbose_clip=verbose_clip)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_box_cube(ax3, geom, faces_to_tint=faces_C, face_colors=face_colors, face_alpha=face_alpha,
                  elev=elev, azim=azim, title=f"View C: {faces_C[0]} + {faces_C[1]}")
    scatter_nodes_by_label(ax3, coords, nodes_for_faces(faces_C), label_fn, vessel_colors,
                           point_size=point_size, alpha=point_alpha, sample_max=sample_max,
                           clip_box=box, clip_eps=clip_eps, verbose_clip=verbose_clip)

    # legends (igual que antes)
    if show_vessel_legend:
        vessel_handles = [
            Line2D([0], [0], marker='o', linestyle='', markersize=7,
                   markerfacecolor=vessel_colors[k], markeredgecolor='none', label=k)
            for k in ["arteriole", "venule", "capillary", "unknown"]
        ]
        ax3.legend(handles=vessel_handles, title="Vessel type", loc="upper left")

    if show_face_legend:
        used_faces = list(dict.fromkeys(list(faces_A) + list(faces_B) + list(faces_C)))
        face_handles = [Patch(facecolor=face_colors[f], edgecolor="k", alpha=face_alpha, label=f)
                        for f in used_faces]
        ax1.legend(handles=face_handles, title="Highlighted faces", loc="upper left")

    plt.tight_layout()
    plt.show()
    return fig


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
      res[face]["high_degree_percent_face"] (optional)
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

        if show_high_degree and "high_degree_percent_face" in res[face]:
            pct_hd = float(res[face].get("high_degree_percent_face", 0.0))
            n_hd = len(res[face].get("high_degree_nodes_face", []))
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

def plot_nkinds_spatial(graph, coords_attr="coords", mode="edges", sample_edges=8000,
                       title=None, ax=None, s=6, alpha=0.85, linewidth=0.6):
    """
    Spatial plot with nkind color-coding.
    - mode="edges": color edges by graph.es['nkind'] (recommended)
    - mode="nodes": color nodes by bc_node_common_nkind (slower)
    """
    P = get_coords(graph, coords_attr)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if mode == "nodes":
        labels = [bc_node_common_nkind(graph, v.index) for v in graph.vs]
        cols = [_nkind_label_and_color(k)[1] for k in labels]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=cols, alpha=alpha, depthshade=False)

    elif mode == "edges":
        if "nkind" not in graph.es.attributes():
            raise ValueError("graph edges need attribute 'nkind' for mode='edges'")

        if sample_edges is None:
            edge_ids = range(graph.ecount())
        else:
            if isinstance(sample_edges, int):
                rng = np.random.default_rng(0)
                edge_ids = rng.choice(graph.ecount(), size=min(sample_edges, graph.ecount()), replace=False)
            else:
                edge_ids = sample_edges

        for eid in edge_ids:
            e = graph.es[int(eid)]
            s_idx, t_idx = e.tuple
            k = int(e["nkind"]) if "nkind" in e.attributes() else None
            col = _nkind_label_and_color(k)[1]
            ax.plot([P[s_idx, 0], P[t_idx, 0]],
                    [P[s_idx, 1], P[t_idx, 1]],
                    [P[s_idx, 2], P[t_idx, 2]],
                    color=col, alpha=alpha, linewidth=linewidth)
    else:
        raise ValueError("mode must be 'nodes' or 'edges'")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"Spatial nkind map ({mode})")
    ax.legend(handles=[
        Line2D([0], [0], color="red", lw=2, label="arteriole"),
        Line2D([0], [0], color="blue", lw=2, label="venule"),
        Line2D([0], [0], color="gray", lw=2, label="capillary"),
    ], loc="best")

    return fig, ax


def plot_high_degree_nodes(graph, coords_attr="coords", high_degree_threshold=4,
                           title=None, ax=None, s_all=3, s_hdn=30,
                           c_all="lightgray", c_hdn="black", alpha_all=0.35, alpha_hdn=0.95):
    """Shows where High Degree Nodes (HDN) are in 3D."""
    P = get_coords(graph, coords_attr)
    deg = np.asarray(graph.degree(), dtype=int)
    hdn = np.where(deg >= int(high_degree_threshold))[0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s_all, c=c_all, alpha=alpha_all, depthshade=False)
    if len(hdn) > 0:
        ax.scatter(P[hdn, 0], P[hdn, 1], P[hdn, 2], s=s_hdn, c=c_hdn, alpha=alpha_hdn, depthshade=False)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title or f"High-degree nodes (deg ≥ {high_degree_threshold})")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', label='all nodes', markerfacecolor=c_all, markersize=6),
        Line2D([0], [0], marker='o', color='w', label=f'HDN (≥{high_degree_threshold})',
               markerfacecolor=c_hdn, markersize=8),
    ], loc="best")
    return fig, ax


def plot_diameter_by_nkind(graph, bins=40, show_boxplot=True, title=None, ax=None):
    """Diameter distributions split by nkind (edges)."""
    if "diameter" not in graph.es.attributes():
        raise ValueError("graph edges must have attribute 'diameter'")
    if "nkind" not in graph.es.attributes():
        raise ValueError("graph edges must have attribute 'nkind'")

    diam = np.asarray(graph.es["diameter"], dtype=float)
    nk = np.asarray(graph.es["nkind"], dtype=int)

    groups = {
        "arteriole": diam[nk == 2],
        "venule": diam[nk == 3],
        "capillary": diam[nk == 4],
    }

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    colors = {"arteriole": "red", "venule": "blue", "capillary": "gray"}

    for lab in ["arteriole", "venule", "capillary"]:
        vals = groups[lab]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=bins, alpha=0.35, color=colors[lab], label=f"{lab} (n={len(vals)})")

    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Diameter distribution by nkind")
    ax.legend(loc="best")

    if not show_boxplot:
        return (fig, ax)

    fig2, ax2 = plt.subplots()
    data = [groups["arteriole"], groups["venule"], groups["capillary"]]
    data = [d[np.isfinite(d)] for d in data]
    ax2.boxplot(data, labels=["arteriole", "venule", "capillary"], showfliers=False)
    ax2.set_ylabel("Diameter (µm)")
    ax2.set_title("Diameter (µm) by nkind (boxplot)")

    return (fig, ax), (fig2, ax2)


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
            "High degree %": float(face_data.get("high_degree_percent_face", 0.0)),
        })

    return pd.DataFrame(rows)


# ====================================================================================================================
# GAIA-STYLE MICRO-SEGMENTS + VESSEL DENSITY (requires OutGeom-like data dict)
# ====================================================================================================================

def microsegments_from_outgeom(data, require_attrs=("geom_start", "geom_end", "nkind")):
    """
    Build micro-segments between consecutive polyline points for each igraph edge.
    Expects OutGeom-like:
      data["graph"] = igraph Graph
      data["coords"]["x"],["y"],["z"] arrays for ALL polyline points concatenated
      edge attrs: geom_start, geom_end, nkind
      optional: data["radii_geom"] array per point

    Returns dict arrays:
      - midpoints (Nx3)
      - lengths (N,)
      - nkind (N,)
      - r0, r1 (N,) radii at segment endpoints (NaN if missing)
    """
    G = data["graph"]

    if "coords" not in data or not all(k in data["coords"] for k in ("x","y","z")):
        raise ValueError("data['coords'] must contain x,y,z arrays (OutGeom-like).")

    x = np.asarray(data["coords"]["x"], dtype=float)
    y = np.asarray(data["coords"]["y"], dtype=float)
    z = np.asarray(data["coords"]["z"], dtype=float)

    r = np.asarray(data["radii_geom"], dtype=float) if "radii_geom" in data else None

    for a in require_attrs:
        if a not in G.es.attributes():
            raise ValueError(f"Edge attr '{a}' missing in graph.es; available: {G.es.attributes()}")

    mids, lens, nk, r0s, r1s = [], [], [], [], []

    for e in range(G.ecount()):
        s = int(G.es[e]["geom_start"])
        t = int(G.es[e]["geom_end"])
        if (t - s) < 2:
            continue

        nkind_e = int(G.es[e]["nkind"])

        for i in range(s, t - 1):
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            dz = z[i + 1] - z[i]
            L = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if L <= 0:
                continue

            mids.append([(x[i] + x[i + 1]) * 0.5, (y[i] + y[i + 1]) * 0.5, (z[i] + z[i + 1]) * 0.5])
            lens.append(L)
            nk.append(nkind_e)

            if r is not None:
                r0s.append(float(r[i]))
                r1s.append(float(r[i + 1]))
            else:
                r0s.append(np.nan)
                r1s.append(np.nan)

    return {
        "midpoints": np.asarray(mids, float),
        "lengths": np.asarray(lens, float),
        "nkind": np.asarray(nk, int),
        "r0": np.asarray(r0s, float),
        "r1": np.asarray(r1s, float),
    }


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


def vessel_density_slabs_in_box(ms, box, slab=50.0, axis="z",
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
    kinds = [2, 3, 4]

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
            row[f"{vessel_type.get(k, k)}_{metric}"] = (vv / tissue_vol) if tissue_vol > 0 else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n=== Vessel density slabs (axis={axis}, slab={slab} µm, metric={metric}) ===")
        print(df)

    return df

import numpy as np
import matplotlib.pyplot as plt




def edge_radius_consistency_report_MAX(
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

def check_polyline_radii_match_endnodes(
    data,
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

def check_radii_endpoints_allow_swap(data, tol=1e-3, max_examples=15, use_coords_check=False, coords_tol=1e-6):
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


def _nodes_by_label_in_subgraph(sub, vessel_type_map=EDGE_NKIND_TO_LABEL):
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

def select_largest_component_data(data, verbose=True):
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


def attach_vertex_attrs_from_data(data, verbose=False):
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
    validate_box(box)
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


# ====================================================================================================================
#                                   HDN PATTERN: quantify face bias + type enrichment + d2s
# ====================================================================================================================
import numpy as np

def analyze_hdn_pattern_adapted(data, xBox, yBox, zBox, degree_thr=4, face_eps=2.0, verbose=True):
    """
    Versión adaptada a la estructura de diccionarios:
    - data["graph"]: El objeto igraph
    - data["vertex"]: Coordenadas y anotaciones de nodos
    - xBox, yBox, zBox: Listas [min, max]
    """
    G = data["graph"]
    validate_box(box)
    
    # --- 1) Obtener coordenadas ---
    # Usamos coords_image del diccionario vertex
    P = np.asarray(data["vertex"]["coords_image"], dtype=float)
    deg = np.asarray(G.degree(), int)

    xmin, xmax = float(xBox[0]), float(xBox[1])
    ymin, ymax = float(yBox[0]), float(yBox[1])
    zmin, zmax = float(zBox[0]), float(zBox[1])

    # --- 2) Filtrar nodos dentro del Box ---
    inside = (
        (P[:,0] >= xmin) & (P[:,0] <= xmax) &
        (P[:,1] >= ymin) & (P[:,1] <= ymax) &
        (P[:,2] >= zmin) & (P[:,2] <= zmax)
    )
    nodes_in_indices = np.where(inside)[0]
    
    # High Degree Nodes (HDN)
    hdn_indices = nodes_in_indices[deg[nodes_in_indices] >= int(degree_thr)]

    # --- 3) Manejo de Anotaciones (Node Types) ---
    # Buscamos en vertex_annotation si existe
    def get_labels(indices):
        if "vertex_annotation" in data["vertex"]:
            # Asumimos que los labels están mapeados (ej: 1: arteriole, etc.)
            # Si prefieres usar una función externa como infer_node_type_from_incident_edges, pásala aquí
            return np.asarray(data["vertex"]["vertex_annotation"])[indices]
        return np.array(["unknown"] * len(indices))

    labels_in = get_labels(nodes_in_indices)
    labels_hdn = get_labels(hdn_indices)

    def pct(labels, key_val):
        if len(labels) == 0: return 0.0
        return float(100.0 * np.mean(labels == key_val))

    # --- 4) Sesgo de cara (Face Bias) ---
    # ¿Están los HDN amontonados en las paredes del corte?
    face_bias = {
        "x_min": float(np.mean(np.abs(P[hdn_indices,0] - xmin) <= face_eps)) if len(hdn_indices) else 0.0,
        "x_max": float(np.mean(np.abs(P[hdn_indices,0] - xmax) <= face_eps)) if len(hdn_indices) else 0.0,
        "y_min": float(np.mean(np.abs(P[hdn_indices,1] - ymin) <= face_eps)) if len(hdn_indices) else 0.0,
        "y_max": float(np.mean(np.abs(P[hdn_indices,1] - ymax) <= face_eps)) if len(hdn_indices) else 0.0,
        "z_min": float(np.mean(np.abs(P[hdn_indices,2] - zmin) <= face_eps)) if len(hdn_indices) else 0.0,
        "z_max": float(np.mean(np.abs(P[hdn_indices,2] - zmax) <= face_eps)) if len(hdn_indices) else 0.0,
    }

    # --- 5) Distancia a la superficie ---
    d2s_stats = None
    if "distance_to_surface" in data["vertex"] and len(hdn_indices):
        vals = np.asarray(data["vertex"]["distance_to_surface"], float)[hdn_indices]
        d2s_stats = {
            "min": float(vals.min()), 
            "mean": float(vals.mean()), 
            "median": float(np.median(vals)), 
            "max": float(vals.max())
        }

    # --- Resultado ---
    out = {
        "n_nodes_in_box": int(len(nodes_in_indices)),
        "n_hdn": int(len(hdn_indices)),
        "face_bias_frac_hdn": face_bias,
        "distance_to_surface_stats_hdn": d2s_stats
    }

    if verbose:
        print("\n=== HDN pattern analysis (Box) ===")
        print(f"Nodes in box: {out['n_nodes_in_box']} | HDN (deg≥{degree_thr}): {out['n_hdn']}")
        if d2s_stats:
            print(f"HDN distance_to_surface: Mean={d2s_stats['mean']:.2f}, Median={d2s_stats['median']:.2f}")
        print("Face bias (fraction of HDN on boundaries):")
        for face, val in face_bias.items():
            if val > 0.1: # Resaltar si hay más del 10% en una cara
                print(f"  [!] {face}: {val:.3f}")
            else:
                print(f"      {face}: {val:.3f}")

    return out