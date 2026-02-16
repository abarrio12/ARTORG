"""
Build vascular graph (from OutGeom pseudo-json) from pkl with voxel data IN MICROMETERS (µm) !!!!
- coords_image (vertex) is in voxels of image and has to be converted to µm
- tortuous geometry (points) x/y/z also in voxels of image and converted to µm
- lengths2 (distance between points of the tortuous) computed in µm (per-point) 
- length (sum of lengths2) computed in µm 
- distance_to_surface converted to µm (using sx, XY resolution) --> isotropic assumed in Paris code 
(https://github.com/ClearAnatomics/ClearMap/blob/v3.1.x/ClearMap/Analysis/graphs/vasc_graph_doc.rst)


Idea: we are using coordinates image (in voxels). We want to analyze in micrometers, so we go from 
one measurement system to the other by rescaling with the image resolution (micrometers/voxels). 
We rescale coords_image first, then compute length2_real in micrometers by applying euclidian 
distance to the new in micrometer coordinates of points and then length_real (of the non tortuous edge)
as the sum(lengths2_real)


data
 ├── geom         (vox)
 ├── vertex       (vox)
 ├── geom_R       (µm only)
 │     ├── x_R
 │     ├── y_R
 │     ├── z_R
 │     ├── lengths2_R
 │     └── diameters_R
 ├── vertex_R     (µm only)
 │     ├── coords_image_R
 │     └── distance_to_surface_R
 └── graph
       ├── length_R
       └── tortuosity_R

---------------------------------------
!!! IMPORTANT !!!  RADII UNITS TBD 
---------------------------------------

Author: Ana Barrio
Date: Feb 2026
"""


import pickle
import numpy as np

def convert_outgeom_pkl_to_um(
    in_path,
    out_path,
    res_um_per_vox=(1.625, 1.625, 2.5),           # µm/vox      
    min_straight_dist_um=1.0,
    
):
    
    # Paris conversion factor voxels to image 
    sx, sy, sz = map(float, res_um_per_vox)

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
    if not {"x", "y", "z"} <= g:
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
        C = np.asarray(v["coords_image"], dtype=np.float64)  # shape (N,3) vox
        v_R["coords_image_R"] = (C * np.array([sx, sy, sz])).astype(np.float32)

    
    # --------------------------
    # distance_to_surface: vox -> approx in µm using sx due to isotropic assumption (Paris)
    # --------------------------
    if "distance_to_surface" in v:
        d_vox = np.asarray(v["distance_to_surface"], dtype=np.float64)
        v_R["distance_to_surface_R"] = (d_vox * sx).astype(np.float32, copy=False)  # sx = 1.625


    # --------------------------
    # geom['lengths2'] in µm (per-point)
    # --------------------------

    dist_x = np.diff(x_um)   # same as x2-x1
    dist_y = np.diff(y_um)
    dist_z = np.diff(z_um)

    euclidean_seg_R = np.sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z).astype(np.float32)      # (nP-1,)
    
    lengths2_list_R = euclidean_seg_R.astype(np.float32)    # (nP-1,)
    
    g_R["lengths2_R"] = lengths2_list_R

    
    if "radii" in g and "diameters" not in g:
        g_R["diameters_R"] = (2.0 * np.asarray(g["radii"], np.float32)).astype(np.float32)   # <<<<<<<<<<<<<<<< CAREFUL RADII IN WHAT UNIT???????


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
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"

    convert_outgeom_pkl_to_um(
        in_path=in_path,
        out_path=out_path,
        res_um_per_vox=(1.625, 1.625, 2.5),
        convert_vertex_coords_attr=("coords_image",), 
        min_straight_dist_um=1.0,
    )
