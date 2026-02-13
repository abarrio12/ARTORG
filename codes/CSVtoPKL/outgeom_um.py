"""
Build vascular graph (from OutGeom pseudo-json) !!!! BUT STORED IN MICROMETERS (µm) !!!!
- coords_image (vertex) is converted to µm
- geom x/y/z converted to µm
- lengths2 (geom) computed in µm (per-point, last=0) (distance between nodes of tortuous)
- edge length + length_tortuous computed in µm from geom
- distance_to_surface converted to µm (using sx, XY resolution) --> isotropic assumed

!!! IMPORTANT !!!  radii are ALREADY in µm (DO NOT rescale) (check gt2CSV from Franca for reference)

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
    keep_original_vox_arrays=True                 # saves copies of *_vox for  debug
):
    sx, sy, sz = map(float, res_um_per_vox)

    data = pickle.load(open(in_path, "rb"))
    G = data["graph"]

    # --------------------------
    # GEOM coords: x,y,z -> µm
    # --------------------------
    g = data.get("geom", {})
    if not {"x", "y", "z"} <= set(g.keys()):
        raise KeyError("data['geom'] must contain x,y,z")

    x_vox = np.asarray(g["x"], dtype=np.float64)
    y_vox = np.asarray(g["y"], dtype=np.float64)
    z_vox = np.asarray(g["z"], dtype=np.float64)

    if keep_original_vox_arrays:
        g["x_vox"] = x_vox.astype(np.float32, copy=False)
        g["y_vox"] = y_vox.astype(np.float32, copy=False)
        g["z_vox"] = z_vox.astype(np.float32, copy=False)

    x_um = x_vox * sx
    y_um = y_vox * sy
    z_um = z_vox * sz

    g["x"] = x_um.astype(np.float64, copy=False)
    g["y"] = y_um.astype(np.float64, copy=False)
    g["z"] = z_um.astype(np.float64, copy=False)

    # --------------------------
    # VERTEX coords: coords_image (and/or coords) -> µm
    # --------------------------
    v = data.get("vertex", {})
    for attr in convert_vertex_coords_attr:
        if attr in v:
            P_vox = np.asarray(v[attr], dtype=np.float64)
            if P_vox.ndim != 2 or P_vox.shape[1] != 3:
                raise ValueError(f"vertex['{attr}'] must be (n,3)")

            if keep_original_vox_arrays:
                v[f"{attr}_vox"] = P_vox.astype(np.float32, copy=False)

            P_um = P_vox * np.array([sx, sy, sz], dtype=np.float64).reshape(1, 3)
            v[attr] = P_um.astype(np.float64, copy=False)

    # --------------------------
    # distance_to_surface: vox -> µm (approx using sx)
    # --------------------------
    if convert_distance_to_surface and ("distance_to_surface" in v):
        d_vox = np.asarray(v["distance_to_surface"], dtype=np.float64)
        if keep_original_vox_arrays:
            v["distance_to_surface_vox"] = d_vox.astype(np.float32, copy=False)
        v["distance_to_surface"] = (d_vox * sx).astype(np.float32, copy=False)


    # --------------------------
    # geom['lengths2'] in µm (per-point, last=0)
    # --------------------------
    nP = len(x_um)
    if recompute_geom_lengths2:
        dx = np.diff(x_um)
        dy = np.diff(y_um)
        dz = np.diff(z_um)
        seg = np.sqrt(dx*dx + dy*dy + dz*dz).astype(np.float32)      # (nP-1,)
        L2p = np.concatenate([seg, np.array([0.0], np.float32)])     # (nP,)
        g["lengths2"] = L2p

    # Optional convenience arrays (not required but nice for Gaia-style)
    if "radii" in g and "diameters" not in g:
        g["diameters"] = (2.0 * np.asarray(g["radii"], np.float32)).astype(np.float32)

    if "lengths2" in g and "lengths" not in g:
        # arclen per-point (0, cumsum)
        seg = np.asarray(g["lengths2"], np.float64)[:-1]
        g["lengths"] = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)

    # --------------------------
    # Edge length in µm : sum(lengths2[s:en])
    # --------------------------
    if recompute_edge_lengths:
        if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
            raise KeyError("Edges must have geom_start/geom_end to recompute length.")
        L2p = np.asarray(g["lengths2"], dtype=np.float64)
        edge_len = np.zeros(G.ecount(), dtype=np.float64)
        gs = np.asarray(G.es["geom_start"], dtype=np.int64)
        ge = np.asarray(G.es["geom_end"], dtype=np.int64)

        for ei in range(G.ecount()):
            s = int(gs[ei]); en = int(ge[ei])
            if en - s >= 2:
                edge_len[ei] = float(np.sum(L2p[s:en]))
        G.es["length"] = edge_len.astype(np.float32).tolist()

    # --------------------------
    # Tortuosity in µm (length_tortuous + tortuosity)
    # --------------------------
    if recompute_tortuosity:
        if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
            raise KeyError("Edges must have geom_start/geom_end to recompute tortuosity.")

        gs = np.asarray(G.es["geom_start"], dtype=np.int64)
        ge = np.asarray(G.es["geom_end"], dtype=np.int64)

        lt = np.zeros(G.ecount(), dtype=np.float64)
        sd = np.zeros(G.ecount(), dtype=np.float64)

        # We can reuse lengths2 sum as lt (same as arclength) if we already computed it
        if "lengths2" in g:
            L2p = np.asarray(g["lengths2"], dtype=np.float64)
            for ei in range(G.ecount()):
                s = int(gs[ei]); en = int(ge[ei])
                if en - s >= 2:
                    lt[ei] = float(np.sum(L2p[s:en]))
                    sd[ei] = float(np.sqrt((x_um[en-1]-x_um[s])**2 + (y_um[en-1]-y_um[s])**2 + (z_um[en-1]-z_um[s])**2))
        else:
            # fallback (shouldn't happen if recompute_geom_lengths2=True)
            for ei in range(G.ecount()):
                s = int(gs[ei]); en = int(ge[ei])
                if en - s >= 2:
                    dx = np.diff(x_um[s:en])
                    dy = np.diff(y_um[s:en])
                    dz = np.diff(z_um[s:en])
                    lt[ei] = float(np.sum(np.sqrt(dx*dx + dy*dy + dz*dz)))
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
