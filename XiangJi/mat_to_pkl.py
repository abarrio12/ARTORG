# FROM MAT TO PKL 


import scipy.io
import numpy as np

import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import igraph as ig
import scipy.sparse



mat = scipy.io.loadmat(
    r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\XiangJi\ML20180815_240_c5o1_578.mat",
    struct_as_record=False,
    squeeze_me=True # This option simplifies the structure of the loaded data, making it easier to access the fields without needing to use additional indexing -> mat["link"][0,0]["cc_ind"]
)

print(mat.keys())

print("Type of mat['num']: ", type(mat["num"]))
print("Type of mat['node']: ", type(mat["node"]))
print("Type of mat['link']: ", type(mat["link"]))
print("Type of mat['endpoint']: ", type(mat["endpoint"]))
print("Type of mat['isopoint']: ", type(mat["isopoint"]))
print("Type of mat['radius']: ", type(mat["radius"]))
print("Type of mat['label']: ", type(mat["label"]))
print("Type of mat['info']: ", type(mat["info"]))


# In Ji 1 = capillary, 2 = artery, 3 = vein. In MVN1 we have 1=cap, 2=art, 3=vein, so we need to remap the labels.
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
    Load Ji/Kleinfeld MATLAB spatial graph.

    Returns
    -------
    mat : dict
        Dictionary with keys such as:
        'num', 'link', 'node', 'endpoint', 'isopoint', 'radius', 'label', ...
    """
    mat = scipy.io.loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=True,
    )
    return mat


# ============================================================
# Basic helpers
# ============================================================

def yxz_to_xyz(points: np.ndarray) -> np.ndarray:
    '''Convert coordinates from MATLAB (Y, X, Z) to Python (X, Y, Z) order.'''
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
    "lengths2" MVN1 style: per-segment lengths
    
    Parameters
    ----------
    points : (N, 3) ndarray

    Returns
    -------
    lengths2 : (N-1,) ndarray
    """
    points = np.asarray(points, dtype=float)
    # less than 2 points = no segments, return empty array
    if len(points) < 2:
        return np.array([], dtype=float)

    # axis = 0 works vertically between rows, so np.diff(points, axis=0) gives the vector differences between consecutive points 
    diffs = np.diff(points, axis=0)
    # axis = 1 works horizontally between columns, so np.linalg.norm(diffs, axis=1) computes the Euclidean distance 
    # inside each 3D vector (a, b, c)
    return np.linalg.norm(diffs, axis=1)


def sparse_vector_to_dict(sparse_vec) -> Dict[int, float]:
    """
    Convert MATLAB sparse column vector into Python dict:
        matlab_linear_index -> value

    Notes
    -----
    Ji's radius and label are sparse vectors over the whole volume.
    """
    if not scipy.sparse.issparse(sparse_vec):
        raise TypeError("Expected a scipy sparse matrix/vector")

    d = {}
    coo = sparse_vec.tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        idx = max(r, c) + 1  # back to MATLAB 1-based indexing
        d[int(idx)] = float(v)
    return d


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

    Returns
    -------
    nodes : list of dict
        One dict per node CC.
    """
    mask_size = mat["num"].mask_size
    node_cc = np.asarray(mat["node"].cc_ind).squeeze()

    nodes = []

    for matlab_label, cc in enumerate(node_cc, start=1):
        voxel_indices = np.asarray(cc).astype(np.int64).ravel()
        points_yxz = lin_to_coordinates(voxel_indices, mask_size)
        points = yxz_to_xyz(points_yxz)  # convert to (X, Y, Z) order for consistency
        centroid = points.mean(axis=0) # no need to divide cases if n_voxels = 1 OR > 1, since the mean of a single point is the point itself
        
        node_rec = {
            "id": matlab_label - 1,              # Python/igraph vertex id
            "matlab_label": matlab_label,        # original Ji label
            "voxel_indices": voxel_indices,      # MATLAB linear indices
            "node_points": points,               # all node voxel coords
            "coords": centroid,                  # representative position
            "n_voxels": len(voxel_indices),
        }
        nodes.append(node_rec)

    return nodes


# ============================================================
# Edge extraction
# ============================================================

def build_edges_from_mat(mat: dict) -> List[dict]:
    """
    Build intermediate edge records from Ji's MATLAB graph.

    Edge style is adapted to MVN1 / Paris style:
    - points
    - lengths2
    - length
    - diameters
    - diameter
    - nkind
    """
    mask_size = mat["num"].mask_size
    link_cc = np.asarray(mat["link"].cc_ind).squeeze()
    connected = np.asarray(mat["link"].connected_node_label).astype(int)

    radius_map = sparse_vector_to_dict(mat["radius"])
    label_map = sparse_vector_to_dict(mat["label"]) if "label" in mat else {}

    edges = []

    for matlab_label, cc in enumerate(link_cc, start=1):
        voxel_indices = np.asarray(cc).astype(np.int64).ravel()

        # Geometry
        points_yxz = lin_to_coordinates(voxel_indices, mask_size)
        points = yxz_to_xyz(points_yxz)
        lengths2 = segment_lengths(points)
        length = float(lengths2.sum())

        # Per-point diameters from radius
        radii = np.array(
            [radius_map.get(int(v), np.nan) for v in voxel_indices],
            dtype=float
        )
        diameters = 2.0 * radii
        
        # notice in Ji they use median radius for edge radius (in Paris was the max(radii_points))
        radius = np.nanmedian(radii) if np.any(~np.isnan(radii)) else np.nan
        diameter = 2.0 * radius if not np.isnan(radius) else np.nan
        
        # Vessel type aggregation
        # Ji has the nkind stores per voxel in the same way as radius, so we can do majority vote among the voxels of the edge to assign a single nkind per edge.
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

        edge_rec = {
            "id": matlab_label - 1,
            "matlab_label": matlab_label,
            "source": source,
            "target": target,
            "connected_node_labels_matlab": (int(n1), int(n2)),
            "voxel_indices": voxel_indices,
            "points": points,               # per-point coords
            "lengths2": lengths2,           # per-segment lengths
            "length": length,               # full edge length
            "radii": radii,
            "diameters": diameters,         # per-point diameters
            "diameter": diameter,           # edge summary diameter
            "voxel_types": voxel_types,
            "nkind": nkind,
            "n_voxels": len(voxel_indices),
            "radius": radius,
        }
        edges.append(edge_rec)

    return edges


# ============================================================
# Graph assembly
# ============================================================

def build_igraph_mvn1_style(nodes: List[dict], edges: List[dict]) -> ig.Graph:
    """
    Build igraph.Graph in MVN1 / Paris-like format.

    Notes
    -----
    Only edges with two valid node endpoints are added to the graph.
    Open links (with a 0 endpoint in MATLAB) are skipped for now.
    """
    valid_edges = [e for e in edges if e["source"] is not None and e["target"] is not None]
    dropped_edges = [e for e in edges if e["source"] is None or e["target"] is None]

    G = ig.Graph()
    G.add_vertices(len(nodes))
    G.add_edges([(e["source"], e["target"]) for e in valid_edges])

    # -------------------------
    # Vertex attributes
    # -------------------------
    G.vs["matlab_label"] = [n["matlab_label"] for n in nodes]
    G.vs["coords"] = [tuple(map(float, n["coords"])) for n in nodes]
    G.vs["node_points"] = [n["node_points"].tolist() for n in nodes]
    G.vs["voxel_indices"] = [n["voxel_indices"].tolist() for n in nodes]
    G.vs["n_voxels"] = [n["n_voxels"] for n in nodes]

    # TODO: FALTA INDEX, ANNOTATION, DIAMETER, 
    # -------------------------
    # Edge attributes
    # -------------------------
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
    G.es["radius"] = [e["radius"] for e in valid_edges] # TODO: SHOULD I KEEP IT? IT'S REDUNDANT WITH DIAMETER, BUT MAYBE IT'S GOOD TO HAVE IT SEPARATE FOR EASE OF USE
    # TODO: FALTA CONNECTIVITY, GEOM START Y END Y LENGTH STEPS? 

    # -------------------------
    # Graph-level metadata
    # -------------------------
    #G["mask_size"] = tuple(np.asarray(nodes and [0] or []).tolist())  # placeholder, overwrite below if wanted
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
    edges = build_edges_from_mat(mat)
    G = build_igraph_mvn1_style(nodes, edges)

    # optional graph-level metadata
    G["mask_size"] = tuple(np.asarray(mat["num"].mask_size).astype(int).ravel().tolist())

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mat_path = r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\XiangJi\ML20180815_240_c5o1_578.mat"
    out_path = r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\XiangJi\ML20180815_240_c5o1_578_mvn1.pkl"

    G, nodes, edges, mat = convert_ji_mat_to_mvn1_graph(mat_path)

    print_summary(G)
    save_graph_pickle(G, out_path)
    print(f"\nSaved to: {out_path}")
