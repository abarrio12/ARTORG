"""
Convert OutGeom PKL from voxel space to micrometers (µm), WITHOUT overwriting the original voxel arrays.
The new attributes are stored in separate keys with suffix "_R" (for "real-world" units), while the original voxel-based attributes remain unchanged.

What gets converted:
- vertex["coords_image"] (image voxels)          -> vertex_R["coords_image_R"] in µm
- geom["x","y","z"] (tortuous polyline points)   -> geom_R["x_R","y_R","z_R"] in µm
- geom_R["lengths2_R"]                           -> per-segment lengths in µm (between consecutive polyline points)
- graph edge length_R                            -> per-edge length in µm (sum of lengths2_R within geom_start:geom_end)
- graph tortuosity_R                             -> unitless (length_R / straight_distance)

Important note about radii:
- We convert radii using *radii_atlas* (atlas voxel grid, 25 µm/voxel), because this conversion is well-defined:
    radius_um = radii_atlas_vox * 25  

- We intentionally do NOT convert the original 'radii' (in image voxel space) from edge_geometry_radii.csv here, because its mapping
  to physical units (µm) depends on the exact resampling/scaling used upstream (shape-based resample factor in the Paris/ClearMap pipeline),
  and is not guaranteed to be a simple multiplication by the raw image resolution.

Outputs are stored in:
- data["vertex_R"] and data["geom_R"]  (µm)
while keeping:
- data["vertex"] and data["geom"]      (voxels)
unchanged.

Structure of the output PKL in physical units (µm) will be:

data
 ├── geom                  # original voxel data (unchanged) "non-tortuous graph"
 ├── vertex                # original voxel data (unchanged) "non-tortuous graph"
 │
 ├── geom_R                # physical units (µm)
 │     ├── x_R
 │     ├── y_R
 │     ├── z_R
 │     ├── lengths2_R                  # per-segment lengths (µm), last = 0
 │     ├── radii_atlas_geom_R          # µm (atlas radii × 25)
 │     └── diameters_atlas_geom_R      # µm
 │
 ├── vertex_R              # physical units (µm)
 │     ├── coords_image_R
 │     ├── distance_to_surface_R
 │     └── radii_atlas_R              # µm (atlas radii × 25)
 │
 └── graph
       ├── length_R                   # tortuous arc length (µm)
       └── tortuosity_R               # dimensionless


Author: Ana Barrio
Date: 16 Feb 2026
"""


import pickle
import numpy as np

def convert_outgeom_pkl_to_um(
    in_path,
    out_path,
    res_um_per_vox=(1.625, 1.625, 2.5),           # µm/vox      
    min_straight_dist_um=1.0,
    
):
    
    # Paris conversion factor voxels to micrometers (source resolution)
    sx, sy, sz = map(float, res_um_per_vox)

    # sink image resolution (25,25,25) µm --> 1 voxel = 25 µm
    # radii atlas in voxels of the 25 µm atlas grid
    sink_resolution = 25

    data = pickle.load(open(in_path, "rb"))
    G = data["graph"]

    g = data["geom"]
    v = data["vertex"]

    # make µm copies (do NOT modify originals, careful doing in place operations, changes also original (*=, +=...))
    g_R = {}
    v_R = {}

    # -----------------------------------------
    # Geometry points (Tortuous): x,y,z -> µm
    # -----------------------------------------

    # check all keys are within the geometry, if not problem in build_outgeom_indexed.py
    if not {"x", "y", "z"} <= set(g.keys()):
        raise KeyError("data['geom'] must contain x,y,z. Check your building the graph code")

    x_vox = np.asarray(g["x"], dtype=np.float64)
    y_vox = np.asarray(g["y"], dtype=np.float64)
    z_vox = np.asarray(g["z"], dtype=np.float64)

    x_um = x_vox * sx
    y_um = y_vox * sy
    z_um = z_vox * sz

    g_R["x_R"] = x_um.astype(np.float64, copy=False)
    g_R["y_R"] = y_um.astype(np.float64, copy=False)
    g_R["z_R"] = z_um.astype(np.float64, copy=False)

    # -----------------------------------------
    # Coords image (vertex) -> µm
    # -----------------------------------------

    if "coords_image" in v:
        C_img_vox = np.asarray(v["coords_image"], dtype=np.float64)  # shape (N,3) vox
        v_R["coords_image_R"] = (C_img_vox * np.array([sx, sy, sz])).astype(np.float32)

    if "radii_atlas" in v:
        R_atlas_vox = np.asarray(v["radii_atlas"], dtype=np.float64)  # shape (N,) vox
        v_R["radii_atlas_R"] = (R_atlas_vox * sink_resolution).astype(np.float32)
    
    # --------------------------
    # distance_to_surface: vox -> approx in µm using sx due to isotropic assumption (Paris)
    # --------------------------
    if "distance_to_surface" in v:
        dist_surf_vox = np.asarray(v["distance_to_surface"], dtype=np.float64)
        v_R["distance_to_surface_R"] = (dist_surf_vox * sx).astype(np.float32, copy=False)  # sx = 1.625


    # --------------------------
    # Geometry in µm (per-point)
    # --------------------------

    dist_x = np.diff(x_um)   # same as x2-x1
    dist_y = np.diff(y_um)
    dist_z = np.diff(z_um)

    euclidean_seg_R = np.sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z).astype(np.float32)      # (nP-1,)
    
    lengths2_list_R = np.concatenate([euclidean_seg_R, [0.0]]).astype(np.float32)  # (nP,) 
    
    g_R["lengths2_R"] = lengths2_list_R

    # --------------------------
    # Edge length in µm : sum(lengths2[s:en])
    # --------------------------
    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Edges must have geom_start/geom_end to recompute length.")
    
    if "lengths2_R" not in g_R:
        raise KeyError("geom_R['lengths2'] missing")
    
    lengths2_R = np.asarray(g_R["lengths2_R"], dtype=np.float64)
    edge_length_R = np.zeros(G.ecount(), dtype=np.float64)
    
    start_idx = np.asarray(G.es["geom_start"], dtype=np.int64)
    end_idx = np.asarray(G.es["geom_end"], dtype=np.int64)

    for i in range(G.ecount()):
        s = int(start_idx[i]); 
        en = int(end_idx[i])
        if en - s >= 2:
            edge_length_R[i] = float(np.sum(lengths2_R[s:en]))
    G.es["length_R"] = edge_length_R.astype(np.float32).tolist()

    # --------------------------
    # Radii geometry in µm 
    # --------------------------
    if "radii_atlas_geom" in g:  
        r_atlas_vox = np.asarray(g["radii_atlas_geom"], dtype=np.float64)  # (nP,) in atlas vox
        g_R["radii_atlas_geom_R"] = (r_atlas_vox * sink_resolution).astype(np.float32)  # µm
        g_R["diameters_atlas_geom_R"] = (2.0 * g_R["radii_atlas_geom_R"]).astype(np.float32)


    # --------------------------
    # Tortuosity (adimensional)
    # --------------------------
    if "lengths2_R" not in g_R:
        raise KeyError("Missing g['lengths2_R']. Must be computed before tortuosity")
    if "length_R" not in G.es.attributes():
        raise KeyError("length_R must be computed before tortuosity")

    lt = np.asarray(G.es["length_R"], dtype=np.float64) # edge length
    sd = np.zeros(G.ecount(), dtype=np.float64) # straight distance

    for ei in range(G.ecount()):
        s = int(start_idx[ei])
        en = int(end_idx[ei])
        if en - s >= 2:
            sd[ei] = float(np.sqrt(
                (x_um[en-1] - x_um[s])**2 +
                (y_um[en-1] - y_um[s])**2 +
                (z_um[en-1] - z_um[s])**2
            ))

    # Tortuosity
    tort = np.full(G.ecount(), np.nan, dtype=np.float64)
    m = sd >= float(min_straight_dist_um)  # boolean mask, sanity check, only calculates where sd not too small
    tort[m] = lt[m] / sd[m]

    G.es["tortuosity_R"] = tort.astype(np.float32).tolist()


    # --------------------------
    # Save (without overwriting voxels info)
    # --------------------------
    data["geom_R"] = g_R
    data["vertex_R"] = v_R
    data["graph"] = G

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data


if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3_um.pkl"

    convert_outgeom_pkl_to_um(
        in_path=in_path,
        out_path=out_path,
        res_um_per_vox=(1.625, 1.625, 2.5), 
        min_straight_dist_um=1.0,
    )
