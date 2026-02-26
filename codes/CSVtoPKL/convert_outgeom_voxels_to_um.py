import os
import pickle
import numpy as np

import numpy as np

def quick_unit_sanity_check(data):
    G = data["graph"]
    g = data.get("geom", {})
    v = data.get("vertex", {})
    gR = data.get("geom_R", {})
    vR = data.get("vertex_R", {})

    print("\n==============================")
    print("GEOMETRY RANGE CHECK (VOX)")
    print("==============================")

    if {"x","y","z"} <= set(g.keys()):
        x_vox = np.asarray(g["x"])
        y_vox = np.asarray(g["y"])
        z_vox = np.asarray(g["z"])
        print(f"x_vox: min={x_vox.min():.2f}  max={x_vox.max():.2f}")
        print(f"y_vox: min={y_vox.min():.2f}  max={y_vox.max():.2f}")
        print(f"z_vox: min={z_vox.min():.2f}  max={z_vox.max():.2f}")
    else:
        print("No geom[x,y,z] found.")

    print("\n==============================")
    print("ATLAS COORD RANGE CHECK (VOX atlas grid)")
    print("==============================")

    if "coords" in v:
        coords_atlas = np.asarray(v["coords"])
        print(f"atlas coords shape: {coords_atlas.shape}")
        print(f"atlas min per axis: {coords_atlas.min(axis=0)}")
        print(f"atlas max per axis: {coords_atlas.max(axis=0)}")
    else:
        print("No vertex['coords'] found.")

    print("\n==============================")
    print("DIAMETER_ATLAS_R CHECK (µm)")
    print("==============================")

    if "diameter_atlas_R" in G.es.attributes():
        d = np.asarray(G.es["diameter_atlas_R"], dtype=np.float32)
        d = d[np.isfinite(d)]

        if d.size == 0:
            print("diameter_atlas_R exists but empty/NaN.")
        else:
            print(f"n = {d.size}")
            print(f"min  = {np.min(d):.3f} µm")
            print(f"p5   = {np.percentile(d,5):.3f} µm")
            print(f"med  = {np.median(d):.3f} µm")
            print(f"p95  = {np.percentile(d,95):.3f} µm")
            print(f"max  = {np.max(d):.3f} µm")

            if np.max(d) > 200:
                print("⚠ WARNING: diameters > 200 µm detected → possible scale mixup.")
            elif np.median(d) < 2:
                print("⚠ WARNING: median < 2 µm → possible under-scaling.")
            else:
                print("✓ Diameter range looks biologically plausible (dataset dependent).")

    else:
        print("No edge attribute 'diameter_atlas_R' found.")

    print("\n==============================\n")



def convert_outgeom_pkl_to_um(
    in_path,
    out_path=None,
    res_um_per_vox=(1.625, 1.625, 2.5),   # µm/vox
    min_straight_dist_um=1.0,
):
    
    # Paris conversion factor voxels to micrometers (source resolution)
    sx, sy, sz = map(float, res_um_per_vox)

    # sink image resolution (25,25,25) µm --> 1 voxel = 25 µm
    sink_resolution = 25

    data = pickle.load(open(in_path, "rb"))

    quick_unit_sanity_check(data)


    G = data["graph"]

    g = data["geom"]
    v = data["vertex"]

    # make µm copies (do NOT modify originals)
    g_R = {}
    v_R = {}

    # -----------------------------------------
    # Geometry points (Tortuous): x,y,z -> µm
    # -----------------------------------------
    if not {"x", "y", "z"} <= set(g.keys()):
        raise KeyError("data['geom'] must contain x,y,z. Check your building the graph code")

    x_vox = np.asarray(g["x"], dtype=np.float32)
    y_vox = np.asarray(g["y"], dtype=np.float32)
    z_vox = np.asarray(g["z"], dtype=np.float32)

    x_um = x_vox * sx
    y_um = y_vox * sy
    z_um = z_vox * sz

    g_R["x_R"] = x_um.astype(np.float32, copy=False)
    g_R["y_R"] = y_um.astype(np.float32, copy=False)
    g_R["z_R"] = z_um.astype(np.float32, copy=False)

    # -----------------------------------------
    # Coords image (vertex) -> µm
    # -----------------------------------------
    if "coords_image" in v:
        C_img_vox = np.asarray(v["coords_image"], dtype=np.float32)  # (N,3) vox
        v_R["coords_image_R"] = (C_img_vox * np.array([sx, sy, sz], dtype=np.float32)).astype(np.float32)

    if "radii_atlas" in v:
        R_atlas_vox = np.asarray(v["radii_atlas"], dtype=np.float32)  # (N,) atlas vox
        v_R["radii_atlas_R"] = (R_atlas_vox * sink_resolution).astype(np.float32)

    # distance_to_surface: vox -> approx µm using sx (Paris assumption)
    if "distance_to_surface" in v:
        dist_surf_vox = np.asarray(v["distance_to_surface"], dtype=np.float32)
        v_R["distance_to_surface_R"] = (dist_surf_vox * sx).astype(np.float32, copy=False)

    # -----------------------------------------
    # lengths2_R (per-point segment lengths) in µm
    # -----------------------------------------
    dist_x = np.diff(x_um)
    dist_y = np.diff(y_um)
    dist_z = np.diff(z_um)
    euclidean_seg_R = np.sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z).astype(np.float32)  # (nP-1,)
    g_R["lengths2_R"] = np.concatenate([euclidean_seg_R, [0.0]]).astype(np.float32)  # (nP,)

    # -----------------------------------------
    # Edge length_R in µm : sum(lengths2_R[s:en])
    # -----------------------------------------
    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Edges must have geom_start/geom_end to recompute length_R.")

    lengths2_R = np.asarray(g_R["lengths2_R"], dtype=np.float32)
    start_idx = np.asarray(G.es["geom_start"], dtype=np.int64)
    end_idx = np.asarray(G.es["geom_end"], dtype=np.int64)

    edge_length_R = np.zeros(G.ecount(), dtype=np.float32)
    for i in range(G.ecount()):
        s = int(start_idx[i])
        en = int(end_idx[i])
        if en - s >= 2:
            edge_length_R[i] = float(np.sum(lengths2_R[s:en]))
    G.es["length_R"] = edge_length_R.astype(np.float32).tolist()

    # -----------------------------------------
    # Radii geometry in µm (from atlas grid, 25 µm/vox)
    # -----------------------------------------
    if "radii_atlas_geom" in g:
        r_atlas_vox = np.asarray(g["radii_atlas_geom"], dtype=np.float32)  # (nP,) atlas vox
        g_R["radii_atlas_geom_R"] = (r_atlas_vox * sink_resolution).astype(np.float32)
        g_R["diameters_atlas_geom_R"] = (2.0 * g_R["radii_atlas_geom_R"]).astype(np.float32)

    # -----------------------------------------
    # Tortuosity_R (dimensionless)
    # -----------------------------------------
    lt = np.asarray(G.es["length_R"], dtype=np.float32)
    sd = np.zeros(G.ecount(), dtype=np.float32)
    for ei in range(G.ecount()):
        s = int(start_idx[ei])
        en = int(end_idx[ei])
        if en - s >= 2:
            sd[ei] = float(np.sqrt(
                (x_um[en - 1] - x_um[s]) ** 2 +
                (y_um[en - 1] - y_um[s]) ** 2 +
                (z_um[en - 1] - z_um[s]) ** 2
            ))

    tort = np.full(G.ecount(), np.nan, dtype=np.float32)
    m = sd >= float(min_straight_dist_um)
    tort[m] = lt[m] / sd[m]
    G.es["tortuosity_R"] = tort.astype(np.float32).tolist()

    # diameter_atlas_R (edge) in µm
    diam_edge_vox = G.es["diameter_atlas"] if "diameter_atlas" in G.es.attributes() else None
    if diam_edge_vox is not None:
        G.es["diameter_atlas_R"] = (np.asarray(diam_edge_vox, dtype=np.float32) * sink_resolution).astype(np.float32).tolist()

    # -----------------------------------------
    # Save (without overwriting voxels info)
    # -----------------------------------------
    data["geom_R"] = g_R
    data["vertex_R"] = v_R
    data["graph"] = G
    data["unit"] = {"vox": "voxel", "um": "micrometer"}

    # ---------- automatic out_path ----------
    if out_path is None:
        in_dir = os.path.dirname(in_path)
        base = os.path.splitext(os.path.basename(in_path))[0]  # e.g. graph_18_OutGeom_Hcut1
        # if input already endswith _um, don't double it
        base_um = base if base.endswith("_um") else f"{base}_um"

        # if path contains /vox/ replace with /um/, else create sibling /um/
        if os.path.basename(in_dir) == "vox":
            out_dir = os.path.join(os.path.dirname(in_dir), "um")
        else:
            out_dir = os.path.join(in_dir, "um")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base_um}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path)
    return data


if __name__ == "__main__":
    # just set name + space roots once
    name = "graph_18_OutGeom_Hcut1_vox"  
    root = "/home/admin/Ana/MicroBrain/output"
    in_path = f"{root}/vox/{name}.pkl"

    convert_outgeom_pkl_to_um(
        in_path=in_path,
        out_path=None,  # auto
        res_um_per_vox=(1.625, 1.625, 2.5),
        min_straight_dist_um=1.0,
    )

   