"""
BC (Boundary Condition) node analysis for a box cut (axis-aligned).

What it does (simple + robust):
- Detect BC nodes on each of the 6 box faces using coordinates and a tolerance eps
- For each face: count BC nodes + classify vessel type (via edge 'nkind' mode)
- Print percentages per face
- Sanity checks:
  * coords exist and are Nx3
  * eps suggestion + auto-warning if too few/many BC
  * degree distribution of BC nodes per face
  * duplicate counting note (nodes on edges/corners belong to multiple faces)
  * optional distance_to_surface stats for BC nodes

Copy-paste and run.
"""

import pickle
import numpy as np
from collections import Counter


# ============================================================
# Load / Save
# ============================================================

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def dump_graph(graph, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved graph to: {out_path}")


# ============================================================
# Vessel type mapping (edit if needed)
# ============================================================

VESSEL_TYPE = {
    2: "arteriole",
    3: "venule",
    4: "capillary",
}


# ============================================================
# BC detection (minimal, correct for box cut)
# ============================================================

def bc_nodes_on_plane(graph, axis, value, coords_attr, eps=1e-3):
    """
    axis: 0=x, 1=y, 2=z
    value: plane coordinate
    returns: np.array of vertex indices lying on that plane within eps
    """
    if coords_attr not in graph.vs.attributes():
        raise ValueError(
            f"Missing vertex attribute '{coords_attr}'. "
            f"Available vertex attrs: {graph.vs.attributes()}"
        )

    coords = np.asarray(graph.vs[coords_attr], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got shape {coords.shape}.")

    mask = np.abs(coords[:, axis] - float(value)) <= float(eps)
    return np.where(mask)[0]


def bc_nodes_on_box_faces(graph, box, coords_attr, eps=1e-3):
    """
    box: dict with keys xmin,xmax,ymin,ymax,zmin,zmax
    returns dict face_name -> np.array(node_ids)
    """
    required = ["xmin","xmax","ymin","ymax","zmin","zmax"]
    missing = [k for k in required if k not in box]
    if missing:
        raise ValueError(f"box is missing keys: {missing}. Required: {required}")

    faces = {
        "x_min": bc_nodes_on_plane(graph, 0, box["xmin"], coords_attr, eps),
        "x_max": bc_nodes_on_plane(graph, 0, box["xmax"], coords_attr, eps),
        "y_min": bc_nodes_on_plane(graph, 1, box["ymin"], coords_attr, eps),
        "y_max": bc_nodes_on_plane(graph, 1, box["ymax"], coords_attr, eps),
        "z_min": bc_nodes_on_plane(graph, 2, box["zmin"], coords_attr, eps),
        "z_max": bc_nodes_on_plane(graph, 2, box["zmax"], coords_attr, eps),
    }
    return faces


# ============================================================
# Classification (simple)
# ============================================================

def bc_node_nkind(graph, node_id):
    """
    Classify a node by the most common nkind among its incident edges.
    Returns nkind int or None if not available.
    """
    if "nkind" not in graph.es.attributes():
        raise ValueError(
            "Missing edge attribute 'nkind'. "
            f"Available edge attrs: {graph.es.attributes()}"
        )

    inc_edges = graph.incident(int(node_id))
    nk = [graph.es[e]["nkind"] for e in inc_edges]
    nk = [x for x in nk if x is not None]

    if len(nk) == 0:
        return None
    return Counter(nk).most_common(1)[0][0]


def bc_node_type_label(graph, node_id, vessel_type_map=VESSEL_TYPE):
    nk = bc_node_nkind(graph, node_id)
    if nk is None:
        return "unknown"
    return vessel_type_map.get(nk, f"nkind_{nk}")


# ============================================================
# Sanity helpers
# ============================================================

def _degree_distribution(graph, nodes):
    deg = np.asarray(graph.degree(), dtype=int)
    return Counter(deg[nodes]) if len(nodes) else Counter()

def _distance_to_surface_stats(graph, nodes):
    if "distance_to_surface" not in graph.vs.attributes():
        return None
    d = np.asarray(graph.vs["distance_to_surface"], dtype=float)
    if len(nodes) == 0:
        return None
    vals = d[nodes]
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_bc_for_box(graph, box, coords_attr, eps=1e-3,
                       print_examples=False, max_examples=10):
    """
    Prints:
      - counts per face
      - vessel type counts + %
      - degree distribution per face
      - TOTAL unique BC nodes across faces (note: corners counted multiple times per-face)
      - optional distance_to_surface stats

    Returns dict with all computed results.
    """
    faces = bc_nodes_on_box_faces(graph, box, coords_attr=coords_attr, eps=eps)

    results = {}
    all_bc_set = set()

    nV = graph.vcount()
    nE = graph.ecount()
    print(f"\n=== BC ANALYSIS ===")
    print(f"Graph: {nV} vertices, {nE} edges")
    print(f"Coords attr: '{coords_attr}' | eps: {eps}")
    print("NOTE: Nodes on cube edges/corners may appear in multiple faces (expected).")

    # Per face
    for face, nodes in faces.items():
        nodes = np.array(nodes, dtype=int)
        all_bc_set.update(nodes.tolist())

        # type counts
        labels = [bc_node_type_label(graph, v) for v in nodes]
        type_counts = Counter(labels)
        total = len(nodes)

        # degree dist
        deg_counts = _degree_distribution(graph, nodes)

        # distance_to_surface
        dstat = _distance_to_surface_stats(graph, nodes)

        results[face] = {
            "nodes": nodes,
            "count": total,
            "type_counts": dict(type_counts),
            "type_percent": {k: (100.0*v/total) for k, v in type_counts.items()} if total else {},
            "degree_counts": dict(deg_counts),
            "distance_to_surface_stats": dstat,
        }

        # ---- printing ----
        print(f"\n--- Face {face} ---")
        print(f"BC nodes: {total}")

        if total == 0:
            print("WARNING: 0 BC nodes on this face. (eps too small? wrong coords_attr? wrong box units?)")
        elif total > 0.2 * nV:
            print("WARNING: Many nodes on this face (>20% of vertices). (eps too big? box value off?)")

        # type counts
        if total:
            for k, v in type_counts.most_common():
                print(f"  {k}: {v} ({100.0*v/total:.1f}%)")
        else:
            print("  (no types)")

        # degree dist
        if total:
            # print sorted by degree
            deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(deg_counts.items())])
            print(f"Degree distribution (degree:count): {deg_str}")

        # distance_to_surface stats
        if dstat is not None:
            print("distance_to_surface (µm): "
                  f"min={dstat['min']:.2f}, mean={dstat['mean']:.2f}, "
                  f"median={dstat['median']:.2f}, max={dstat['max']:.2f}")

        # examples
        if print_examples and total:
            print(f"Example node IDs (first {min(max_examples,total)}): {nodes[:max_examples].tolist()}")

    # Total unique
    all_bc = np.array(sorted(all_bc_set), dtype=int)
    total_unique = len(all_bc)

    tot_labels = [bc_node_type_label(graph, v) for v in all_bc]
    tot_counts = Counter(tot_labels)
    tot_deg = _degree_distribution(graph, all_bc)
    tot_dstat = _distance_to_surface_stats(graph, all_bc)

    results["TOTAL_unique"] = {
        "nodes": all_bc,
        "count": total_unique,
        "type_counts": dict(tot_counts),
        "type_percent": {k: (100.0*v/total_unique) for k, v in tot_counts.items()} if total_unique else {},
        "degree_counts": dict(tot_deg),
        "distance_to_surface_stats": tot_dstat,
    }

    print(f"\n=== TOTAL UNIQUE BC NODES ===")
    print(f"Unique BC nodes across all faces: {total_unique}")

    if total_unique == 0:
        print("FATAL WARNING: 0 BC nodes total. Almost certainly wrong box/coords_attr/eps.")
    else:
        for k, v in tot_counts.most_common():
            print(f"  {k}: {v} ({100.0*v/total_unique:.1f}%)")

        deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(tot_deg.items())])
        print(f"Total degree distribution (degree:count): {deg_str}")

        if tot_dstat is not None:
            print("TOTAL distance_to_surface (µm): "
                  f"min={tot_dstat['min']:.2f}, mean={tot_dstat['mean']:.2f}, "
                  f"median={tot_dstat['median']:.2f}, max={tot_dstat['max']:.2f}")

    return results


# ============================================================
# Minimal "how to run"
# ============================================================
if __name__ == "__main__":
    # ---- EDIT THESE ----
    pkl_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"

    # Your cut box limits (same units as coords_attr)
    box = dict(
        xmin=1500/1.625, xmax=2500/1.625,
        ymin=1500/1.625, ymax=2500/1.625,
        zmin=1500/2.5, zmax=2500/2.5
    )

    # Choose coordinates:
    # - "coordinates_atlas" (usually µm)
    # - "coordinates" (if your vertices store image/voxel coords)
    coords_attr = "coords_image"

    # Tolerance. If µm and floats from interpolation, try 1e-2 or 1e-1.
    eps = 1e-2

    # ---- RUN ----

    data = pickle.load(open(pkl_path, "rb"))

    G = data["graph"]
    _ = analyze_bc_for_box(G, box, coords_attr=coords_attr, eps=eps, print_examples=False)