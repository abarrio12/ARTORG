"""
Build vascular graph (from OutGeom pseudo-json) !!!! BUT STORED IN MICROMETERS (µm) !!!!
- coords_image (vertex) is in voxels of image and has to be converted to µm
- tortuous geometry (points) x/y/z also in voxels of image and converted to µm
- lengths2 (distance between points of the tortuous) computed in µm (per-point, last=0) 
- length (sum of lengths2) computed in µm 
- distance_to_surface converted to µm (using sx, XY resolution) --> isotropic assumed in Paris code 
(https://github.com/ClearAnatomics/ClearMap/blob/v3.1.x/ClearMap/Analysis/graphs/vasc_graph_doc.rst)


Idea: we are using coordinates image (in voxels). We want to analyze in micrometers, so we go from 
one measurement system to the other by rescaling with the image resolution (micrometers/voxels). 
We rescale coords_image first, then compute length2_real in micrometers by applying euclidian 
distance to the new in micrometer coordinates of points and then length_real (of the non tortuous edge)
as the sum(lengths2_real)

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
    radii_already_um=True,
    convert_vertex_coords_attr=("coords_image",),  # can also add 'coords' if needed (atlas)
    convert_distance_to_surface=True,             # vox -> µm (asumes isotropic --> uses sx)
    recompute_geom_lengths2=True,                 # rewrites geom["lengths2"] in µm (per-point, last=0)
    recompute_edge_lengths=True,                  # recalculates G.es["length"] in µm from geom["lengths2"]
    recompute_tortuosity=True,                    # recalculates length_tortuous + tortuosity in µm from geom coords (image)
    min_straight_dist_um=1.0,
):
    
    # Paris conversion factor voxels to image 
    sx, sy, sz = map(float, res_um_per_vox)

    data = pickle.load(open(in_path, "rb"))
    G = data["graph"]

    # -----------------------------------------
    # Geometry points (Tortuous): x,y,z -> µm
    # -----------------------------------------

    g = data["geom"]
    v = data["vertex"]

    # check all keys are within the geometry, if not problem in build_outgeom_indexed.py
    if not {"x", "y", "z"} <= g:
        raise KeyError("data['geom'] must contain x,y,z. Check your building the graph code")

    x_vox = np.asarray(g["x"], dtype=np.float64)
    y_vox = np.asarray(g["y"], dtype=np.float64)
    z_vox = np.asarray(g["z"], dtype=np.float64)

    x_um = x_vox * sx
    y_um = y_vox * sy
    z_um = z_vox * sz

    g["x_R"] = x_um.astype(np.float64, copy=False)
    g["y_R"] = y_um.astype(np.float64, copy=False)
    g["z_R"] = z_um.astype(np.float64, copy=False)


    
    # --------------------------
    # distance_to_surface: vox -> µm (approx using sx due to isotropic
    # --------------------------
    if "distance_to_surface" in v:
        d_vox = np.asarray(v["distance_to_surface"], dtype=np.float64)
        v["distance_to_surface"] = (d_vox * sx).astype(np.float32, copy=False)  # sx = 1.625


    # --------------------------
    # geom['lengths2'] in µm (per-point, last=0)
    # --------------------------
    
    # number of points:
    nP = len(x_um)

    dist_x = np.diff(x_um)
    dist_y = np.diff(y_um)
    dist_z = np.diff(z_um)
    euclidean_seg = np.sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z).astype(np.float32)      # (nP-1,)
    lengths2_list = np.concatenate([euclidean_seg, np.array([0.0], np.float32)])     # (nP,)
    g["lengths2"] = lengths2_list

    # Optional convenience arrays (not required but nice for Gaia-style)
    if "radii" in g and "diameters" not in g:
        g["diameters"] = (2.0 * np.asarray(g["radii"], np.float32)).astype(np.float32)   # <<<<<<<<<<<<<<<< CAREFUL RADII IN WHAT UNIT???????


    # --------------------------
    # Edge length in µm : sum(lengths2[s:en])
    # --------------------------
    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Edges must have geom_start/geom_end to recompute length.")
    
    lengths2 = np.asarray(g["lengths2"], dtype=np.float64)
    edge_length = np.zeros(G.ecount(), dtype=np.float64)
    
    start_idx = np.asarray(G.es["geom_start"], dtype=np.int64)
    end_idx = np.asarray(G.es["geom_end"], dtype=np.int64)

    for i in range(G.ecount()):
        s = int(start_idx[i]); 
        en = int(end_idx[i])
        if en - s >= 2:
            edge_length[i] = float(np.sum(lengths2[s:en]))
    G.es["length"] = edge_length.astype(np.float32).tolist()

    # --------------------------
    # Tortuosity in µm (length_tortuous + tortuosity)
    # --------------------------
    if recompute_tortuosity:
        if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
            raise KeyError("Edges must have geom_start/geom_end to recompute tortuosity.")

        start_idx = np.asarray(G.es["geom_start"], dtype=np.int64)
        end_idx = np.asarray(G.es["geom_end"], dtype=np.int64)

        lt = np.zeros(G.ecount(), dtype=np.float64)
        sd = np.zeros(G.ecount(), dtype=np.float64)

        # We can reuse lengths2 sum as lt (same as arclength) if we already computed it
        if "lengths2" in g:
            lengths2 = np.asarray(g["lengths2"], dtype=np.float64)
            for ei in range(G.ecount()):
                s = int(start_idx[ei]); en = int(end_idx[ei])
                if en - s >= 2:
                    lt[ei] = float(np.sum(lengths2[s:en]))
                    sd[ei] = float(np.sqrt((x_um[en-1]-x_um[s])**2 + (y_um[en-1]-y_um[s])**2 + (z_um[en-1]-z_um[s])**2))
        else:
            # fallback (shouldn't happen if recompute_geom_lengths2=True)
            for ei in range(G.ecount()):
                s = int(start_idx[ei]); en = int(end_idx[ei])
                if en - s >= 2:
                    dist_x = np.diff(x_um[s:en])
                    dist_y = np.diff(y_um[s:en])
                    dist_z = np.diff(z_um[s:en])
                    lt[ei] = float(np.sum(np.sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z)))
                    sd[ei] = float(np.sqrt((x_um[en-1]-x_um[s])**2 + (y_um[en-1]-y_um[s])**2 + (z_um[en-1]-z_um[s])**2))

        tort = np.full(G.ecount(), np.nan, dtype=np.float64)
        m = sd >= float(min_straight_dist_um)
        tort[m] = lt[m] / sd[m]

        G.es["length_tortuous"] = lt.astype(np.float32).tolist()
        G.es["tortuosity"] = tort.astype(np.float32).tolist()

    # --------------------------
    # Save
    # --------------------------
    data["geom"] = g
    data["vertex"] = v
    data["graph"] = G

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved µm-converted PKL:", out_path)
    print("Notes:")
    print(" - geom[x,y,z] are now µm")
    for attr in convert_vertex_coords_attr:
        if attr in v:
            print(f" - vertex[{attr}] is now µm")
    if convert_distance_to_surface and ("distance_to_surface" in v):
        print(" - vertex[distance_to_surface] is now µm")
    if "lengths2" in g:
        print(" - geom[lengths2] is now µm per-point (last=0)")
    if "length" in G.es.attributes():
        print(" - edge[length] is now µm")
    if "length_tortuous" in G.es.attributes():
        print(" - edge[length_tortuous] is now µm")
    if "tortuosity" in G.es.attributes():
        print(" - edge[tortuosity] recomputed (unitless)")


if __name__ == "__main__":
    in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
    out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"

    convert_outgeom_pkl_to_um(
        in_path=in_path,
        out_path=out_path,
        res_um_per_vox=(1.625, 1.625, 2.5),
        radii_already_um=True,
        convert_vertex_coords_attr=("coords_image",),  # añade "coords" si ese también está en vox
        convert_distance_to_surface=True,
        recompute_geom_lengths2=True,
        recompute_edge_lengths=True,
        recompute_tortuosity=True,
        min_straight_dist_um=1.0,
        keep_original_vox_arrays=True
    )
