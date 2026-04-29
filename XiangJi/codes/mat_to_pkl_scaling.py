# FROM .MAT TO .PKL

# Notes on Ji graph coordinates and voxel scaling:
# - The graph coordinates in the shared Ji/Kleinfeld vessel graphs are stored in VOXELS.
# - The centerline position was recorded at 1 µm isotropic resolution before any scale correction.
# - In the shared graph from Ji, coordinates are stored in raw voxel units.
# - For ML20180815, 1 voxel corresponds to 1 µm in the recorded centerline grid,
#   but in vivo lengths should be recovered by multiplying geometric quantities by 1.051
#   because the fixed brain is isotropically shrunken by 1/1.051.
# - Radii/diameters are kept unchanged if they have already been calibrated to in vivo values (radii >= 1.8 um).

# Ji graph structure:
# - Ji stores node and link connected components at the voxel level.
# - A node is built from one or more node voxels grouped by a connected component.
# - A link is built from one or more link voxels along a vessel segment.
# - Connectivity is defined by voxel labels: node records know which links touch them and link records know which nodes they connect.
# - 26-neighborhood connectivity in 3D is used to build these connected components.
# - A node may contain many voxel elements, while a link may sometimes consist of a single voxel.
# - Radii are stored per voxel in a sparse vector, so every voxel has its own radius value.

import numpy as np
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple

import igraph as ig
import scipy.io
import scipy.sparse
import mat73


DEBUG_PRINT_MAT_INFO = True


# In Ji 1 = capillary, 2 = artery, 3 = vein.
# In MVN1 we use 4=capillary, 2=artery, 3=vein.
JI_TO_MVN_NKIND = {
    1: 4,  # capillary
    2: 2,  # artery
    3: 3,  # vein
}


# ============================================================
# IO
# ============================================================

def load_ji_mat(mat_path: str) -> Tuple[dict, str]:
    """
    Load Ji/Kleinfeld MATLAB spatial graph, supporting both:
    - pre-v7.3 MAT files via scipy.io.loadmat
    - v7.3 MAT files via mat73

    Returns
    -------
    mat : dict
    loader : str
        "scipy" or "mat73"
    """
    try:
        mat = scipy.io.loadmat(
            mat_path,
            struct_as_record=False,
            squeeze_me=True,
        )
        loader = "scipy"
    except NotImplementedError:
        mat = mat73.loadmat(mat_path)
        loader = "mat73"
    except ValueError:
        mat = mat73.loadmat(mat_path)
        loader = "mat73"

    if DEBUG_PRINT_MAT_INFO:
        print(f"Loaded with: {loader}")
        print(mat.keys())

        keys_to_check = [
            "num", "node", "link", "endpoint", "isopoint",
            "radius", "label", "info", "stat_data", "isoloop"
        ]
        for k in keys_to_check:
            if k in mat:
                print(f"Type of mat['{k}']: {type(mat[k])}")
            else:
                print(f"Key '{k}' does not exist in mat")

    return mat, loader


# ============================================================
# Access helpers for scipy vs mat73
# ============================================================

def get_mask_size(mat: dict, loader: str):
    if loader == "scipy":
        return mat["num"].mask_size
    elif loader == "mat73":
        return mat["num"]["mask_size"]
    raise ValueError(f"Unknown loader: {loader}")


def get_node_cc(mat: dict, loader: str):
    if loader == "scipy":
        return np.asarray(mat["node"].cc_ind).squeeze()
    elif loader == "mat73":
        return np.asarray(mat["node"]["cc_ind"], dtype=object).squeeze()
    raise ValueError(f"Unknown loader: {loader}")


def get_link_cc(mat: dict, loader: str):
    if loader == "scipy":
        return np.asarray(mat["link"].cc_ind).squeeze()
    elif loader == "mat73":
        return np.asarray(mat["link"]["cc_ind"], dtype=object).squeeze()
    raise ValueError(f"Unknown loader: {loader}")


def get_connected_node_labels(mat: dict, loader: str):
    if loader == "scipy":
        return np.asarray(mat["link"].connected_node_label).astype(int)
    elif loader == "mat73":
        return np.asarray(mat["link"]["connected_node_label"]).astype(int)
    raise ValueError(f"Unknown loader: {loader}")


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


def segment_lengths_and_length(points: np.ndarray, voxel_size: float = 1.0):
    """
    Compute center-to-center segment lengths and connected-component length
    following Ji/Guo convention:
        cc_length = sum(segment lengths) + voxel_size

    If there is only one point:
        cc_length = voxel_size
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.array([], dtype=float), 0.0

    if len(points) == 1:
        return np.array([], dtype=float), float(voxel_size)

    lengths2 = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(lengths2.sum() + voxel_size)
    return lengths2, length


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

    if arr.ndim == 1:
        return {
            int(i + 1): float(v)
            for i, v in enumerate(arr)
            if np.isfinite(v) and v != 0
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

def build_nodes_from_mat(mat: dict, loader: str) -> List[dict]:
    """
    Build intermediate node records from Ji's MATLAB graph.

    Ji nodes are formed from connected components of node voxels.
    Each node CC may contain multiple voxel indices, and we store both the voxel-level
    coordinates and a representative centroid position for the graph vertex.
    """
    mask_size = get_mask_size(mat, loader)
    node_cc = get_node_cc(mat, loader)

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

def build_edges_from_mat(mat: dict, nodes: List[dict], loader: str) -> List[dict]:
    """
    Build intermediate edge records from Ji's MATLAB graph.

    Ji links are formed from connected components of link voxels.
    Each link voxel has its own radius value in the sparse radius map, so the
    per-point radii/diameters are preserved and then summarized for the edge.

    Length follows Ji convention:
    - 1-voxel CC -> length = voxel_size
    - N-voxel CC -> length = sum(center-to-center distances) + voxel_size
    """
    mask_size = get_mask_size(mat, loader)
    link_cc = get_link_cc(mat, loader)
    connected = get_connected_node_labels(mat, loader)

    radius_map = sparse_vector_to_dict(mat["radius"])
    label_map = sparse_vector_to_dict(mat["label"]) if "label" in mat else {}

    VOXEL_SIZE_LENGTH = 1.0
    edges = []

    print("connected_node_label shape:", connected.shape)
    
    for matlab_label, cc in enumerate(link_cc, start=1):
        voxel_indices = np.asarray(cc).astype(np.int64).ravel()
    
        # Geometry from link voxels only (Ji convention)
        points_yxz = lin_to_coordinates(voxel_indices, mask_size)
        points = yxz_to_xyz(points_yxz)

        lengths2, length = segment_lengths_and_length(
            points,
            voxel_size=VOXEL_SIZE_LENGTH,
        )
        
        # Per-point diameters from sparse radius map
        radii = np.array(
            [radius_map.get(int(v), np.nan) for v in voxel_indices],
            dtype=float
        )
        diameters = 2.0 * radii
        
        # Ji edge summary radius: median over edge voxels
        radius = np.nanmedian(radii) if np.any(~np.isnan(radii)) else np.nan
        diameter = 2.0 * radius if not np.isnan(radius) else np.nan

        # Vessel type aggregation from voxel labels
        voxel_types = np.array(
            [label_map.get(int(v), np.nan) for v in voxel_indices],
            dtype=float
        )
        nkind_ji = majority_vote(voxel_types.tolist()) if len(label_map) > 0 else None
        nkind = JI_TO_MVN_NKIND.get(nkind_ji, None) if nkind_ji is not None else None
        
        # Topology
        if connected.shape[1] == 2:
            n1, n2 = connected[matlab_label - 1]
        elif connected.shape[0] == 2:
            n1, n2 = connected[:, matlab_label - 1]
        else:
            raise ValueError(f"Unexpected connected_node_label shape: {connected.shape}")

        source = int(n1 - 1) if n1 > 0 else None
        target = int(n2 - 1) if n2 > 0 else None
    

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

    # Vertex attributes
    G.vs["matlab_label"] = [n["matlab_label"] for n in nodes]
    G.vs["coords"] = [tuple(map(float, n["coords"])) for n in nodes]
    G.vs["node_points"] = [n["node_points"].tolist() for n in nodes]
    G.vs["voxel_indices"] = [n["voxel_indices"].tolist() for n in nodes]
    G.vs["n_voxels"] = [n["n_voxels"] for n in nodes]
    G.vs["radius"] = [n["radius"] for n in nodes]
    G.vs["diameter"] = [n["diameter"] for n in nodes]
    G.vs["index"] = [n["id"] for n in nodes]
    
    # TODO: ANNOTATION MISSING (in Ji they do it in a second step)
    
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
    G.es["radius"] = [e["radius"] for e in valid_edges]  # TODO: SHOULD I KEEP IT? IT'S REDUNDANT WITH DIAMETER, BUT MAYBE IT'S GOOD TO HAVE IT SEPARATE FOR EASY OF USE
    G.es["connectivity"] = [(e["source"], e["target"]) for e in valid_edges] # connectivity simple: endpoints in igraph indexing

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
    mat, loader = load_ji_mat(mat_path)
    nodes = build_nodes_from_mat(mat, loader)
    edges = build_edges_from_mat(mat, nodes, loader)
    G = build_igraph_mvn1_style(nodes, edges)

    G["mask_size"] = tuple(np.asarray(get_mask_size(mat, loader)).astype(int).ravel().tolist())
    G["mat_loader"] = loader

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
    if "mat_loader" in G.attributes():
        print("mat_loader:", G["mat_loader"])
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


def rescale_graph_lengths_inplace(G: ig.Graph, scale: float) -> None:
    if "coords" in G.vs.attributes():
        G.vs["coords"] = [
            (np.asarray(c, dtype=float) * scale).tolist()
            for c in G.vs["coords"]
        ]

    if "node_points" in G.vs.attributes():
        G.vs["node_points"] = [
            (np.asarray(p, dtype=float) * scale).tolist()
            for p in G.vs["node_points"]
        ]

    if "points" in G.es.attributes():
        G.es["points"] = [
            (np.asarray(p, dtype=float) * scale).tolist()
            for p in G.es["points"]
        ]

    if "lengths2" in G.es.attributes():
        G.es["lengths2"] = [
            (np.asarray(l, dtype=float) * scale).tolist()
            for l in G.es["lengths2"]
        ]

    if "length" in G.es.attributes():
        G.es["length"] = [
            float(l) * scale
            for l in G.es["length"]
        ]


def sanity_check_rescaling(G_raw, G_scaled, scale=1.051, n_edges_check=10, n_nodes_check=10):
    print("=== EDGE LENGTH RATIO CHECK ===")
    raw_len = np.asarray(G_raw.es["length"], dtype=float)
    sca_len = np.asarray(G_scaled.es["length"], dtype=float)
    ratios = sca_len / raw_len
    finite = np.isfinite(ratios) & (raw_len > 0)
    print("mean length ratio:", ratios[finite].mean())
    print("expected ratio   :", scale)
    print("ok?", np.allclose(ratios[finite], scale, atol=1e-8, rtol=1e-6))

    print("\n=== LENGTH = SUM(lengths2) + VOXEL_SIZE CHECK ===")
    ok_sum = []
    voxel_size_scaled = scale
    for i in range(min(n_edges_check, G_scaled.ecount())):
        l = float(G_scaled.es[i]["length"])
        l2 = np.asarray(G_scaled.es[i]["lengths2"], dtype=float).sum()
        npts = len(G_scaled.es[i]["points"])

        if npts == 0:
            ok_sum.append(np.isclose(l, 0.0))
        elif npts == 1:
            ok_sum.append(np.isclose(l, voxel_size_scaled))
        else:
            ok_sum.append(np.isclose(l, l2 + voxel_size_scaled))

    print("all checked edges ok?", all(ok_sum))

    print("\n=== POINTS -> LENGTHS2 CONSISTENCY CHECK ===")
    ok_geom = []
    for i in range(min(n_edges_check, G_scaled.ecount())):
        p = np.asarray(G_scaled.es[i]["points"], dtype=float)
        if len(p) < 2:
            ok_geom.append(True)
            continue
        l2_re = np.linalg.norm(np.diff(p, axis=0), axis=1)
        l2_st = np.asarray(G_scaled.es[i]["lengths2"], dtype=float)
        ok_geom.append(np.allclose(l2_re, l2_st))
    print("all checked edges ok?", all(ok_geom))

    print("\n=== DIAMETER UNCHANGED CHECK ===")
    raw_d = np.asarray(G_raw.es["diameter"], dtype=float)
    sca_d = np.asarray(G_scaled.es["diameter"], dtype=float)
    print("diameters unchanged?", np.allclose(raw_d, sca_d, equal_nan=True))

    print("\n=== NODE COORD RATIO CHECK ===")
    ratios_nodes = []
    for i in range(min(n_nodes_check, G_raw.vcount())):
        a = np.asarray(G_raw.vs[i]["coords"], dtype=float)
        b = np.asarray(G_scaled.vs[i]["coords"], dtype=float)
        mask = a != 0
        if np.any(mask):
            ratios_nodes.extend((b[mask] / a[mask]).tolist())
    ratios_nodes = np.asarray(ratios_nodes, dtype=float)
    print("mean node coord ratio:", ratios_nodes.mean())
    print("expected ratio       :", scale)
    print("ok?", np.allclose(ratios_nodes, scale, atol=1e-8, rtol=1e-6))


def check_bad_edges(G, top=30):
    import numpy as np

    rows = []

    for i, e in enumerate(G.es):
        s, t = e.tuple
        c0 = np.asarray(G.vs[s]["coords"], float)
        c1 = np.asarray(G.vs[t]["coords"], float)
        ps = np.asarray(e["points"], float)

        direct = np.linalg.norm(c1 - c0)
        length = float(e["length"])

        d_s_link = min(np.linalg.norm(ps - c0, axis=1))
        d_t_link = min(np.linalg.norm(ps - c1, axis=1))

        rows.append((i, direct, length, direct / length, d_s_link, d_t_link, s, t))

    rows = sorted(rows, key=lambda x: x[3], reverse=True)

    print("edge | direct | length | direct/length | source-link | target-link | s | t")
    for r in rows[:top]:
        print(r)
        
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mat_path = r"C:\Users\Ana\Documents\ARTORG\XiangJi\files\ML20180815_240_c5o1_578.mat"
    out_path = r"C:\Users\Ana\Documents\ARTORG\XiangJi\files\ML20180815_240_c5o1_578_mvn1_scaled.pkl"

    G, nodes, edges, mat = convert_ji_mat_to_mvn1_graph(mat_path)

    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {
        "connectivity", "nkind", "diameter", "diameters",
        "length", "lengths2", "points"
    }
    keep_g = {"mask_size", "mat_loader"}

    # 1.051 is the correction factor to recover in vivo lengths from
    # the fixed brain graph, which is in raw voxel units
    G_raw_full = G.copy()
    G_scaled_full = G_raw_full.copy()
    rescale_graph_lengths_inplace(G_scaled_full, 1.051)
    
    
    #check_bad_edges(G_raw_full)
    
    G_raw = graph_attributes_MVN(G_raw_full, keep_v=keep_v, keep_e=keep_e, keep_g=keep_g)
    G_scaled = graph_attributes_MVN(G_scaled_full, keep_v=keep_v, keep_e=keep_e, keep_g=keep_g)

    sanity_check_rescaling(G_raw, G_scaled, scale=1.051)
    save_graph_pickle(G_scaled, out_path)

    print(f"\nSaved to: {out_path}")