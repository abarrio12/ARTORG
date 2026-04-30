"""
Convert igraph from VOX to UM coordinates and diameters.

This utility converts a MVN-format igraph (as output by CSVtoPKL_tortuous.py)
from voxel units (VOX) to micrometers (UM).

Key transformations:
- Coordinates (vertices and points per edge): multiply by [res_x, res_y, res_z]
- Edge lengths (length, lengths2): multiply by appropriate resolution factor
- Diameters (vertex and per-point): multiply by (res_x + res_y) / 2 (transverse scale) !!
- Edge scalar diameter: MAX of per-point diameters (not mean. This comes from how they reduced the graph in Paris)
https://github.com/ClearAnatomics/ClearMap/blob/71444a5c7456901f15e8d0ceb06fab72b74161df/ClearMap/Scripts/TubeMap.py#L433
- Tortuosity: recalculated if length/straight_distance method used

Returns a new graph with UM coordinates and metadata set.

"""

import numpy as np
import igraph as ig


def graph_to_um(G_vox, res_um_per_vox=(1.625, 1.625, 2.5)):
    """
    Convert igraph object from VOX to UM.
    
    Parameters
    ----------
    G_vox : igraph.Graph
        MVN-format igraph with fullgeom attributes in VOX (voxel) coordinates.
        Expected vertex attributes: coords, degree, index, diameter, annotation
        Expected edge attributes: connectivity, nkind, points, diameters, lengths2,
                                   length, diameter (edge scalar), etc.
    
    res_um_per_vox : tuple
        Resolution in µm/voxel: (sx, sy, sz).
        Default: (1.625, 1.625, 2.5) = Paris Graph image resolution.
    
    Returns
    -------
    G_um : igraph.Graph
        New igraph with UM coordinates and diameters.
        Metadata: G_um["unit"] = "um", G_um["diameter_unit"] = "um"
    
    Notes
    -----
    - Creates a deep copy; original graph (vox) is unmodified.
    - Edge diameter (scalar) = MAX of diameters_per_point (not mean).
    - Diameter scaled by (res_x + res_y) / 2 (transverse scale).
    - lengths2 and length recalculated from points_um.
    - lengths is recalculated as per-point segment length. It is not cumulative distance.
    """
    
    G_um = G_vox.copy()
    
    sx, sy, sz = map(float, res_um_per_vox)
    spacing = np.array([sx, sy, sz], dtype=np.float32)
    scale_diam = (sx + sy) / 2.0

    # =========================================================================
    # VERTEX ATTRIBUTES
    # =========================================================================

    # coords: VOX -> UM
    if "coords" in G_um.vs.attributes():
        coords_vox = np.asarray(G_um.vs["coords"], dtype=np.float32)
        coords_um = coords_vox * spacing[None, :]
        G_um.vs["coords"] = [tuple(map(float, row)) for row in coords_um]

    # diameter (vertex): VOX -> UM
    if "diameter" in G_um.vs.attributes():
        diam_vox = np.asarray(G_um.vs["diameter"], dtype=np.float32)
        G_um.vs["diameter"] = (diam_vox * scale_diam).tolist()

    # =========================================================================
    # EDGE ATTRIBUTES
    # =========================================================================

    # ----- points: VOX -> UM  -----
    points_um_list = []
    if "points" in G_um.es.attributes():
        for pts_vox in G_um.es["points"]:
            if len(pts_vox) == 0:
                points_um_list.append([])
            else:
                pts_array = np.asarray(pts_vox, dtype=np.float32)
                pts_um = pts_array * spacing[None, :]
                points_um_list.append([tuple(map(float, row)) for row in pts_um])
        G_um.es["points"] = points_um_list
    else:
        # If no points -> empty list 
        points_um_list = [[] for _ in range(G_um.ecount())]

    # ----- diameters (per-point): VOX -> UM -----
    if "diameters" in G_um.es.attributes():
        diams_um_list = []
        for diams_vox in G_um.es["diameters"]:
            if len(diams_vox) == 0:
                diams_um_list.append([])
            else:
                diams_array = np.asarray(diams_vox, dtype=np.float32)
                diams_um_list.append((diams_array * scale_diam).tolist())
        G_um.es["diameters"] = diams_um_list
    else:
        diams_um_list = [[] for _ in range(G_um.ecount())]

    # ----- lengths2: recalculated from points_um -----
    # multiplying by mean resolution is an approximation that can be inaccurate for long edges with large z component.
    lengths2_um_list = []
    for pts_um in points_um_list:
        if len(pts_um) < 2:
            lengths2_um_list.append([])
        else:
            pts_array = np.asarray(pts_um, dtype=np.float32)
            diffs = np.diff(pts_array, axis=0)               # vectors between consecutive points
            l2 = np.linalg.norm(diffs, axis=1).astype(np.float32)  # real euclidean distance
            lengths2_um_list.append(l2.tolist())
    G_um.es["lengths2"] = lengths2_um_list

    # ----- lengths (per-point segment length) -----
    # length assigned to each point, with the last point repeating the last segment length.
    lengths_um_list = []
    for l2_list in lengths2_um_list:
        if len(l2_list) == 0:
            lengths_um_list.append([])
        else:
            l2_array = np.asarray(l2_list, dtype=np.float32)
            length_per_point = np.zeros(len(l2_array) + 1, dtype=np.float32)
            length_per_point[:-1] = l2_array
            length_per_point[-1] = l2_array[-1]  # last point repeats the last length
            lengths_um_list.append(length_per_point.tolist())
    G_um.es["lengths"] = lengths_um_list

    # ----- length (scalar per edge): sum lengths2_um -----
    # recompute it from lengths2_um, not scaled from vox
    length_um_list = []
    for l2_list in lengths2_um_list:
        if len(l2_list) == 0:
            length_um_list.append(0.0)
        else:
            length_um_list.append(float(np.sum(l2_list)))
    G_um.es["length"] = length_um_list

    # ----- diameter (scalar per edge): MAX of diameters_um -----
    diam_edge_um = []
    for diams_um in diams_um_list:
        if len(diams_um) == 0:
            diam_edge_um.append(float("nan"))
        else:
            arr = np.asarray(diams_um, dtype=np.float32)
            valid = arr[np.isfinite(arr)]
            diam_edge_um.append(float(valid.max()) if valid.size else float("nan"))
    G_um.es["diameter"] = diam_edge_um

    # ----- tortuosity: recompute it from lengths2_um and points_um -----
    if "tortuosity" in G_um.es.attributes():
        tortuosity_um = []
        for l2_list, pts_um in zip(lengths2_um_list, points_um_list):
            if len(pts_um) >= 2 and len(l2_list) > 0:
                length_curved = float(np.sum(l2_list))
                p_start = np.asarray(pts_um[0], dtype=np.float32)
                p_end = np.asarray(pts_um[-1], dtype=np.float32)
                straight_dist = float(np.linalg.norm(p_end - p_start))
                tortuosity_um.append(length_curved / straight_dist if straight_dist > 0 else 1.0)
            else:
                tortuosity_um.append(1.0)
        G_um.es["tortuosity"] = tortuosity_um

    # =========================================================================
    # METADATA
    # =========================================================================
    G_um["unit"] = "um"
    G_um["diameter_unit"] = "um"
    G_um["resolution_image_um_per_voxel"] = list(map(float, res_um_per_vox))
    G_um["resolution_atlas_um_per_voxel"] = [25.0, 25.0, 25.0]
    G_um["coord_space"] = "image"  # or "atlas", but we keep it as "image" since it's in image space coordinates
    return G_um


def load_and_convert(pkl_path_vox, out_pkl_path_um=None, res_um_per_vox=(1.625, 1.625, 2.5)):
    """
    Convenience function: load VOX igraph from PKL, convert to UM, save to new PKL.
    
    Parameters
    ----------
    pkl_path_vox : str
        Path to input VOX pickle file (igraph).
    out_pkl_path_um : str, optional
        Path to output UM pickle file. If None, auto-generates from input path.
    res_um_per_vox : tuple
        Resolution in µm/voxel: (sx, sy, sz).
    
    Returns
    -------
    G_um : igraph.Graph
        Converted UM igraph.
    """
    import pickle
    import os
    
    # Load VOX graph
    with open(pkl_path_vox, "rb") as f:
        G_vox = pickle.load(f)
    
    # Convert
    G_um = graph_to_um(G_vox, res_um_per_vox=res_um_per_vox)
    
    # Auto-generate output path if needed
    if out_pkl_path_um is None:
        in_dir = os.path.dirname(pkl_path_vox)
        in_base = os.path.splitext(os.path.basename(pkl_path_vox))[0]
        out_base = in_base if in_base.endswith("_um") else f"{in_base}_um"
        out_pkl_path_um = os.path.join(in_dir, f"{out_base}.pkl")
    
    # Save UM graph
    os.makedirs(os.path.dirname(out_pkl_path_um), exist_ok=True)
    with open(out_pkl_path_um, "wb") as f:
        pickle.dump(G_um, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved UM graph to: {out_pkl_path_um}")
    print(f"  Vertices: {G_um.vcount()}")
    print(f"  Edges: {G_um.ecount()}")
    print(f"  Unit: {G_um['unit']}")
    print(f"  Diameter unit: {G_um['diameter_unit']}")
    
    return G_um

 
if __name__ == "__main__":
    graph_number = 18

    pkl_vox = f"/storage/homefs/ab25c720/MicroBrain/halfbrain_{graph_number}_igraph.pkl"
    pkl_um = f"/storage/homefs/ab25c720/MicroBrain/halfbrain_{graph_number}_igraph_um.pkl"

    G_um = load_and_convert(
        pkl_vox,
        out_pkl_path_um=pkl_um,
        res_um_per_vox=(1.625, 1.625, 2.5)
    )
