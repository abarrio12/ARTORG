# OJO DEBERÍA ELIMINARLO AQUI NO HAGO REESCALADO NI HAGO +1 EN LENGTH
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 
# 
# FROM .MAT TO .PKL

# Notes on Ji graph coordinates and voxel scaling:
# - The graph coordinates in the shared Ji/Kleinfeld vessel graphs are stored in VOXELS.
# - The centerline position was recorded at 1 µm isotropic resolution before any scale correction.
# - Therefore, in the sample graph 1 VOXEL is treated as 1 µm.
# - Calibration measurements in the METHOD section of the paper indicate the brains shrank by 1/1.048 in each dimension after fixation.
# - The third brain, ML20200201, shrank further by 1/1.0926 along each dimension due to extra delipidation.
# - These shrinkages are not included in the shared vessel graphs; the graph remains in raw voxel units.
#
# Ji graph structure:
# - Ji stores node and link connected components at the voxel level.
# - A node is built from one or more node voxels grouped by a connected component.
# - A link is built from one or more link voxels along a vessel segment.
# - Connectivity is defined by voxel labels: node records know which links touch them and link records know which nodes they connect.
# - 26-neighborhood connectivity in 3D is used to build these connected components.
# - A node may contain many voxel elements, while a link may sometimes consist of a single voxel (special case).
# - Radii are stored per voxel in a sparse vector, so every voxel has its own radius value.

import numpy as np
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import igraph as ig
import scipy.sparse
import mat73


# ============================================================
# Debug / inspection
# ============================================================

DEBUG_PRINT_MAT_INFO = True


# In Ji 1 = capillary, 2 = artery, 3 = vein.
# In MVN1 we have 1=cap, 2=art, 3=vein, so we remap.
JI_TO_MVN_NKIND = {
    1: 4,  # capillary
    2: 2,  # artery
    3: 3,  # vein
}


# ============================================================
# IO
# ============================================================

def load_ji_mat(mat_path: str) -> dict:
    """
    Load Ji/Kleinfeld MATLAB spatial graph from MATLAB v7.3 file.
    """
    mat = mat73.loadmat(mat_path)

    if DEBUG_PRINT_MAT_INFO:
        print(mat.keys())

        for k in [
            "num", "node", "link", "endpoint", "isopoint",
            "radius", "label", "info", "stat_data", "isoloop"
        ]:
            if k in mat:
                print(f"Type of mat['{k}']: {type(mat[k])}")
            else:
                print(f"Key '{k}' no existe en mat")

        if "node" in mat and isinstance(mat["node"], dict):
            print("node keys:", mat["node"].keys())
        if "link" in mat and isinstance(mat["link"], dict):
            print("link keys:", mat["link"].keys())

    return mat


# ============================================================
# Basic helpers
# ============================================================

def yxz_to_xyz(points: np.ndarray) -> np.ndarray:
    """Convert coordinates from MATLAB (Y, X, Z) to Python (X, Y, Z) order."""
    points = np.asarray(points)
    return points[:, [1, 0, 2]]


def lin_to_coordinates(idx, mask_size) -> np.ndarray:
    """
    Convert MATLAB linear indices to 3D voxel coordinates (Y, X, Z).

    Parameters
    ----------
    idx : array-like
        MATLAB linear indices (1-based).
    mask_size : array-like of length 3
        Volume shape in MATLAB order.

    Returns
    -------
    coords : (N, 3) ndarray
        Coordinates in zero-based Python indexing, ordered as (Y, X, Z).
    """
    idx = np.asarray(idx).astype(np.int64).ravel()
    mask_size = tuple(np.asarray(mask_size).astype(np.int64).ravel())
    coords = np.array(np.unravel_index(idx - 1, mask_size, order="F")).T
    return coords



def segment_lengths(points: np.ndarray) -> np.ndarray:
    """
    Length of each consecutive segment along an ordered polyline.
    "lengths2" MVN1 style: per-segment lengths.
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return np.array([], dtype=float)

    diffs = np.diff(points, axis=0)
    return np.linalg.norm(diffs, axis=1)


def are_26_neighbors(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if two voxels are 26-connected neighbors in 3D.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    d = np.abs(a - b)
    return np.all(d <= 1) and np.any(d > 0)


def find_touching_node_voxel_link_voxel(node_points: np.ndarray, link_point: np.ndarray) -> np.ndarray:
    """
    Find a node voxel that is a 26-neighbor of the link voxel.
    Useful for the special case where a link connected component has only one voxel.
    """
    node_points = np.asarray(node_points, dtype=int)
    link_point = np.asarray(link_point, dtype=int)

    candidates = [p for p in node_points if are_26_neighbors(p, link_point)]

    if len(candidates) == 0:
        raise ValueError(
            f"No touching node voxel found for link_point={link_point.tolist()}"
        )

    if len(candidates) == 1:
        return np.asarray(candidates[0], dtype=float)

    dists = [
        np.linalg.norm(np.asarray(p, dtype=float) - link_point.astype(float))
        for p in candidates
    ]
    return np.asarray(candidates[int(np.argmin(dists))], dtype=float)


def sparse_vector_to_dict(sparse_vec) -> Dict[int, float]:
    """
    Convert MATLAB sparse column vector into Python dict:
        matlab_linear_index -> value

    Supports:
    - scipy sparse vector/matrix
    - dense vector/array fallback
    """
    if scipy.sparse.issparse(sparse_vec):
        d = {}
        coo = sparse_vec.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            idx = max(r, c) + 1  # back to MATLAB 1-based indexing
            d[int(idx)] = float(v)
        return d

    arr = np.asarray(sparse_vec)

    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
        return {
            int(i + 1): float(v)
            for i, v in enumerate(arr)
            if not np.isnan(v) and v != 0
        }

    raise TypeError(
        f"Unsupported sparse/dense vector format: type={type(sparse_vec)}, "
        f"shape={getattr(arr, 'shape', None)}"
    )


def majority_vote(values: List[int]) -> Optional[int]:
    vals = [int(v) for v in values if v is not None and not np.isnan(v)]
    if len(vals) == 0:
        return None
    return Counter(vals).most_common(1)[0][0]


# ============================================================
# Node extraction
# ============================================================

def build_nodes_from_mat(mat: dict) -> List[dict]:
    """
    Build intermediate node records from Ji's MATLAB graph.
    """
    mask_size = mat["num"]["mask_size"]
    node_cc = np.asarray(mat["node"]["cc_ind"], dtype=object).squeeze()

    radius_map = sparse_vector_to_dict(mat["radius"])

    nodes = []

    for matlab_label, cc in enumerate(node_cc, start=1):
        voxel_indices = np.asarray(cc).astype(np.int64).ravel()
        points_yxz = lin_to_coordinates(voxel_indices, mask_size)
        points = yxz_to_xyz(points_yxz)
        centroid = points.mean(axis=0)

        radii_node = np.array([radius_map.get(int(v), np.nan) for v in voxel_indices], dtype=float)
        radius_node = np.nanmedian(radii_node) if np.any(~np.isnan(radii_node)) else np.nan
        diameter_node = 2.0 * radius_node if not np.isnan(radius_node) else np.nan

        node_rec = {
            "id": matlab_label - 1,
            "matlab_label": matlab_label,
            "voxel_indices": voxel_indices,
            "node_points": points,
            "coords": centroid,
            "n_voxels": len(voxel_indices),
            "radius": radius_node,
            "diameter": diameter_node,
        }
        nodes.append(node_rec)

    return nodes


# ============================================================
# Edge extraction
# ============================================================

def build_edges_from_mat(mat: dict, nodes: List[dict]) -> List[dict]:
    """
    Build intermediate edge records from Ji's MATLAB graph.
    """
    mask_size = mat["num"]["mask_size"]
    link_cc = np.asarray(mat["link"]["cc_ind"], dtype=object).squeeze()
    connected = np.asarray(mat["link"]["connected_node_label"]).astype(int)

    radius_map = sparse_vector_to_dict(mat["radius"])
    label_map = sparse_vector_to_dict(mat["label"]) if "label" in mat else {}
    nodes_by_label = {n["matlab_label"]: n for n in nodes}

    edges = []
    count_single_voxel_links = 0
    count_single_voxel_links_diam_link_equals_node = 0

    for matlab_label, cc in enumerate(link_cc, start=1):
        voxel_indices = np.asarray(cc).astype(np.int64).ravel()

        # Geometry
        points_yxz = lin_to_coordinates(voxel_indices, mask_size)
        points = yxz_to_xyz(points_yxz)
        lengths2 = segment_lengths(points)
        length = float(lengths2.sum())

        # Radii / diameters
        radii = np.array(
            [radius_map.get(int(v), np.nan) for v in voxel_indices],
            dtype=float
        )
        diameters = 2.0 * radii

        radius = np.nanmedian(radii) if np.any(~np.isnan(radii)) else np.nan
        diameter = 2.0 * radius if not np.isnan(radius) else np.nan

        # Vessel types
        voxel_types = np.array(
            [label_map.get(int(v), np.nan) for v in voxel_indices],
            dtype=float
        )
        nkind_ji = majority_vote(voxel_types.tolist()) if len(label_map) > 0 else None
        nkind = JI_TO_MVN_NKIND.get(nkind_ji, None) if nkind_ji is not None else None

        # Topology
        n1, n2 = connected[matlab_label - 1]
        source = int(n1 - 1) if n1 > 0 else None
        target = int(n2 - 1) if n2 > 0 else None

        # Special case: only 1 voxel link
        if len(points) == 1 and n1 > 0 and n2 > 0:
            count_single_voxel_links += 1
            link_point = points[0]

            node1_rec = nodes_by_label.get(int(n1))
            node2_rec = nodes_by_label.get(int(n2))
            if node1_rec is None or node2_rec is None:
                raise KeyError(
                    f"Missing node record for connected node labels: n1={n1}, n2={n2}"
                )

            touch1 = find_touching_node_voxel_link_voxel(node1_rec["node_points"], link_point)
            touch2 = find_touching_node_voxel_link_voxel(node2_rec["node_points"], link_point)

            points = np.vstack([touch1, link_point, touch2])

            diam_link = 2.0 * radii[0] if not np.isnan(radii[0]) else np.nan
            diam1 = node1_rec["diameter"]
            diam2 = node2_rec["diameter"]
            diameters = np.array([diam1, diam_link, diam2], dtype=float)

            if np.isclose(diam_link, diam1) or np.isclose(diam_link, diam2):
                count_single_voxel_links_diam_link_equals_node += 1

            lengths2 = segment_lengths(points)
            length = float(lengths2.sum())

            print(
                "1-voxel link",
                "edge", matlab_label,
                "n1", n1, "n2", n2,
                "link_point", link_point.tolist(),
                "touch1", touch1.tolist(),
                "touch2", touch2.tolist(),
                "points", points.tolist(),
                "radii", radii.tolist(),
                "diameters", diameters.tolist(),
                "node1_diam", diam1,
                "node2_diam", diam2,
                "lengths2", lengths2.tolist(),
                "length", length,
            )

        edge_rec = {
            "id": matlab_label - 1,
            "matlab_label": matlab_label,
            "source": source,
            "target": target,
            "connected_node_labels_matlab": (int(n1), int(n2)),
            "voxel_indices": voxel_indices,
            "points": points,
            "lengths2": lengths2,
            "length": length,
            "radii": radii,
            "diameters": diameters,
            "diameter": diameter,
            "voxel_types": voxel_types,
            "nkind": nkind,
            "n_voxels": len(voxel_indices),
            "radius": radius,
        }
        edges.append(edge_rec)

    print("Count edges with 1 voxel link:", count_single_voxel_links)
    print(
        "Count 1-voxel links where link diameter equals one endpoint node diameter:",
        count_single_voxel_links_diam_link_equals_node,
    )
    return edges


# ============================================================
# Graph assembly
# ============================================================

def build_igraph_mvn1_style(nodes: List[dict], edges: List[dict]) -> ig.Graph:
    """
    Build igraph.Graph in MVN1 / Paris-like format.
    """
    valid_edges = [e for e in edges if e["source"] is not None and e["target"] is not None]
    dropped_edges = [e for e in edges if e["source"] is None or e["target"] is None]

    G = ig.Graph()
    G.add_vertices(len(nodes))
    G.add_edges([(e["source"], e["target"]) for e in valid_edges])

    # Vertex attributes
    G.vs["matlab_label"] = [n["matlab_label"] for n in nodes]
    G.vs["coords"] = [tuple(map(float, n["coords"])) for n in nodes]
    G.vs["node_points"] = [n["node_points"].tolist() for n in nodes]
    G.vs["voxel_indices"] = [n["voxel_indices"].tolist() for n in nodes]
    G.vs["n_voxels"] = [n["n_voxels"] for n in nodes]
    G.vs["radius"] = [n["radius"] for n in nodes]
    G.vs["diameter"] = [n["diameter"] for n in nodes]
    G.vs["index"] = [n["id"] for n in nodes]

    # Edge attributes
    G.es["matlab_label"] = [e["matlab_label"] for e in valid_edges]
    G.es["points"] = [e["points"].tolist() for e in valid_edges]
    G.es["lengths2"] = [e["lengths2"].tolist() for e in valid_edges]
    G.es["length"] = [e["length"] for e in valid_edges]
    G.es["diameters"] = [e["diameters"].tolist() for e in valid_edges]
    G.es["diameter"] = [e["diameter"] for e in valid_edges]
    G.es["radii"] = [e["radii"].tolist() for e in valid_edges]
    G.es["nkind"] = [e["nkind"] for e in valid_edges]
    G.es["voxel_types"] = [e["voxel_types"].tolist() for e in valid_edges]
    G.es["voxel_indices"] = [e["voxel_indices"].tolist() for e in valid_edges]
    G.es["n_voxels"] = [e["n_voxels"] for e in valid_edges]
    G.es["connected_node_labels_matlab"] = [
        e["connected_node_labels_matlab"] for e in valid_edges
    ]
    G.es["radius"] = [e["radius"] for e in valid_edges]
    G.es["connectivity"] = [(e["source"], e["target"]) for e in valid_edges]

    # Graph-level metadata
    G["n_total_edges_in_mat"] = len(edges)
    G["n_valid_edges_added"] = len(valid_edges)
    G["n_dropped_open_links"] = len(dropped_edges)

    return G


# ============================================================
# Convenience wrapper
# ============================================================

def convert_ji_mat_to_mvn1_graph(mat_path: str) -> Tuple[ig.Graph, List[dict], List[dict], dict]:
    """
    Full pipeline:
    Ji MAT -> intermediate nodes/edges -> igraph
    """
    mat = load_ji_mat(mat_path)
    nodes = build_nodes_from_mat(mat)
    edges = build_edges_from_mat(mat, nodes)
    G = build_igraph_mvn1_style(nodes, edges)

    G["mask_size"] = tuple(np.asarray(mat["num"]["mask_size"]).astype(int).ravel().tolist())

    return G, nodes, edges, mat


# ============================================================
# Saving / summary
# ============================================================

def save_graph_pickle(G: ig.Graph, out_path: str) -> None:
    with open(out_path, "wb") as f:
        pickle.dump(G, f)


def print_summary(G: ig.Graph) -> None:
    print("=" * 60)
    print("Graph summary")
    print("=" * 60)
    print(G.summary())
    print("mask_size:", G["mask_size"])
    print("n_total_edges_in_mat:", G["n_total_edges_in_mat"])
    print("n_valid_edges_added:", G["n_valid_edges_added"])
    print("n_dropped_open_links:", G["n_dropped_open_links"])

    if G.ecount() > 0:
        print("\nEdge attributes:", G.es.attributes())
        print("Vertex attributes:", G.vs.attributes())
        print("\nFirst edge:")
        e = G.es[0]
        print("  length:", e["length"])
        print("  diameter:", e["diameter"])
        print("  nkind:", e["nkind"])
        print("  n_points:", len(e["points"]))
        print("  n_lengths2:", len(e["lengths2"]))
        print("  n_diameters:", len(e["diameters"]))


def graph_attributes_MVN(
    G: ig.Graph,
    keep_v: set,
    keep_e: set,
    keep_g: Optional[set] = None,
) -> ig.Graph:
    """
    Return a copy of G keeping only the selected vertex, edge,
    and optionally graph-level attributes.
    """
    H = G.copy()

    for attr in list(H.vs.attributes()):
        if attr not in keep_v:
            del H.vs[attr]

    for attr in list(H.es.attributes()):
        if attr not in keep_e:
            del H.es[attr]

    if keep_g is not None:
        for attr in list(H.attributes()):
            if attr not in keep_g:
                del H[attr]

    return H


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mat_path = "/home/ana/MicroBrain/MicroBrain/ARTORG/XiangJi/WholeBrain_ML_2018_08_15_whole_brain_graph.mat"
    out_path = "/home/ana/MicroBrain/MicroBrain/ARTORG/XiangJi/WholeBrain_ML_2018_08_15_whole_brain_graph.pkl"

    G, nodes, edges, mat = convert_ji_mat_to_mvn1_graph(mat_path)

    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {
        "connectivity", "nkind", "diameter", "diameters",
        "length", "lengths2", "points"
    }
    keep_g = {"mask_size"}

    G_pruned = graph_attributes_MVN(G, keep_v=keep_v, keep_e=keep_e, keep_g=keep_g)
    # print_summary(G_pruned)
    save_graph_pickle(G_pruned, out_path)
    print(f"\nSaved to: {out_path}")
