"""
Convert fullgeom igraph from VOX to UM coordinates and diameters.

This utility converts a Gaia-format igraph (as output by CSVtoPKL_FULLGEOM.py)
from voxel units (VOX) to micrometers (UM).

Key transformations:
- Coordinates (vertices and points per edge): multiply by [res_x, res_y, res_z]
- Edge lengths (length, lengths2): multiply by appropriate resolution factor
- Diameters (vertex and per-point): multiply by (res_x + res_y) / 2 (transverse scale) !!
- Edge scalar diameter: MAX of per-point diameters (not mean)
- Tortuosity: recalculated if length/straight_distance method used

Returns a new graph with UM coordinates and metadata set.

Author: Ana
Updated: 24 Apr 2026
"""

import numpy as np
import igraph as ig


def graph_to_um(G_vox, res_um_per_vox=(1.625, 1.625, 2.5)):
    """
    Convert fullgeom igraph from VOX to UM.
    
    Parameters
    ----------
    G_vox : igraph.Graph
        Gaia-format igraph with fullgeom attributes in VOX (voxel) coordinates.
        Expected vertex attributes: coords, degree, index, diameter, annotation
        Expected edge attributes: connectivity, nkind, points, diameters, lengths2,
                                   length, diameter (edge scalar), etc.
    
    res_um_per_vox : tuple
        Resolution in µm/voxel: (sx, sy, sz).
        Default: (1.625, 1.625, 2.5) = MicroBrain image resolution.
    
    Returns
    -------
    G_um : igraph.Graph
        New igraph with UM coordinates and diameters.
        Metadata: G_um["unit"] = "um", G_um["diameter_unit"] = "um"
    
    Notes
    -----
    - Creates a deep copy; original graph is unmodified.
    - Edge diameter (scalar) = MAX of diameters_per_point (not mean).
    - Diameters scaled by (res_x + res_y) / 2 (transverse scale).
    - Lengths scaled by appropriate resolution factor (typically res_z for Z-dominant
      edges, or average for isotropic scaling).
    """
    
    # Copy the graph
    G_um = G_vox.copy()
    
    sx, sy, sz = map(float, res_um_per_vox)
    spacing = np.array([sx, sy, sz], dtype=np.float32)
    
    # Transverse scale for diameters (XY plane average)
    scale_diam = (sx + sy) / 2.0
    
    # =========================================================================
    # VERTEX ATTRIBUTES
    # =========================================================================
    
    # coords: VOX -> UM
    if "coords" in G_um.vs.attributes():
        coords_vox = np.asarray(G_um.vs["coords"], dtype=np.float32)
        coords_um = (coords_vox * spacing[None, :]).astype(np.float32)
        G_um.vs["coords"] = [tuple(map(float, row)) for row in coords_um]
    
    # diameter (vertex): VOX -> UM
    if "diameter" in G_um.vs.attributes():
        diam_vox = np.asarray(G_um.vs["diameter"], dtype=np.float32)
        diam_um = (diam_vox * scale_diam).astype(np.float32)
        G_um.vs["diameter"] = diam_um.tolist()
    
    # =========================================================================
    # EDGE ATTRIBUTES
    # =========================================================================
    
    # points: list of (x,y,z) tuples in VOX -> UM
    if "points" in G_um.es.attributes():
        points_um_list = []
        for pts_vox in G_um.es["points"]:
            if len(pts_vox) == 0:
                points_um_list.append([])
            else:
                pts_array = np.asarray(pts_vox, dtype=np.float32)
                pts_um = (pts_array * spacing[None, :]).astype(np.float32)
                points_um_list.append([tuple(map(float, row)) for row in pts_um])
        G_um.es["points"] = points_um_list
    
    # diameters (per-point): VOX -> UM
    if "diameters" in G_um.es.attributes():
        diams_um_list = []
        for diams_vox in G_um.es["diameters"]:
            if len(diams_vox) == 0:
                diams_um_list.append([])
            else:
                diams_array = np.asarray(diams_vox, dtype=np.float32)
                diams_um = (diams_array * scale_diam).astype(np.float32)
                diams_um_list.append(diams_um.tolist())
        G_um.es["diameters"] = diams_um_list
    
    # lengths2 (segment lengths): scale by average (or Z if appropriate)
    # Using average of all three dimensions for isotropic scaling
    scale_length = np.mean([sx, sy, sz])
    if "lengths2" in G_um.es.attributes():
        lengths2_um_list = []
        for l2_vox in G_um.es["lengths2"]:
            if len(l2_vox) == 0:
                lengths2_um_list.append([])
            else:
                l2_array = np.asarray(l2_vox, dtype=np.float32)
                l2_um = (l2_array * scale_length).astype(np.float32)
                lengths2_um_list.append(l2_um.tolist())
        G_um.es["lengths2"] = lengths2_um_list
    
    # lengths (per-point cumulative): same scale as lengths2
    if "lengths" in G_um.es.attributes():
        lengths_um_list = []
        for lens_vox in G_um.es["lengths"]:
            if len(lens_vox) == 0:
                lengths_um_list.append([])
            else:
                lens_array = np.asarray(lens_vox, dtype=np.float32)
                lens_um = (lens_array * scale_length).astype(np.float32)
                lengths_um_list.append(lens_um.tolist())
        G_um.es["lengths"] = lengths_um_list
    
    # length (scalar edge): scale similarly
    if "length" in G_um.es.attributes():
        length_vox = np.asarray(G_um.es["length"], dtype=np.float32)
        length_um = (length_vox * scale_length).astype(np.float32)
        G_um.es["length"] = length_um.tolist()
    
    # diameter (edge scalar): MAX of per-point diameters (UM units)
    # Recompute from diameters_um (which are already in UM)
    diam_edge_um = []
    for diams_um in G_um.es["diameters"]:
        if len(diams_um) == 0:
            diam_edge_um.append(float("nan"))
        else:
            arr = np.asarray(diams_um, dtype=np.float32)
            valid = arr[np.isfinite(arr)]
            diam_edge_um.append(float(valid.max()) if valid.size else float("nan"))
    G_um.es["diameter"] = diam_edge_um
    
    # Recalculate tortuosity if present and lengths2 available
    if "tortuosity" in G_um.es.attributes() and "lengths2" in G_um.es.attributes():
        tortuosity_um = []
        for eid, (l2_list, pts_um) in enumerate(zip(G_um.es["lengths2"], G_um.es["points"])):
            if len(pts_um) >= 2 and len(l2_list) > 0:
                l2_array = np.asarray(l2_list, dtype=np.float32)
                length_curved = float(np.sum(l2_array))
                p_start = np.asarray(pts_um[0], dtype=np.float32)
                p_end = np.asarray(pts_um[-1], dtype=np.float32)
                straight_dist = float(np.linalg.norm(p_end - p_start))
                tortu = length_curved / straight_dist if straight_dist > 0 else 1.0
                tortuosity_um.append(float(tortu))
            else:
                tortuosity_um.append(1.0)
        G_um.es["tortuosity"] = tortuosity_um
    
    # =========================================================================
    # METADATA
    # =========================================================================
    G_um["unit"] = "um"
    G_um["diameter_unit"] = "um"
    G_um["resolution_um_per_voxel"] = list(map(float, res_um_per_vox))
    
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
    # Example usage:
    # Load fullgeom VOX igraph and convert to UM
    
    graph_number = 18
    root_path = "/home/admin/Ana/MicroBrain/output"
    
    pkl_vox = f"{root_path}{graph_number}/{graph_number}_igraph.pkl"
    pkl_um = f"{root_path}{graph_number}/{graph_number}_igraph_um.pkl"
    
    G_um = load_and_convert(
        pkl_vox,
        out_pkl_path_um=pkl_um,
        res_um_per_vox=(1.625, 1.625, 2.5)
    )
