'''
Code that converts from voxels to um. Takes the graph build from build_graph_outgeom_voxels.py
As output, you have the new attributes converted to um and also keeps the old ones, in case 
is necessary in future tasks. 

Transformations used:
 - Image resolution: 1.625x1.625x2.5 (used in length, coordinates_image.... to go from voxels of image to um)
 -  Atlas resolution: 25 um (used to go from voxels at 25 um grid to um ONLY used in radii atlas attr)

 Author: Ana
 Updated: 27 Feb 2026
'''
import os
import pickle
import numpy as np


def quick_unit_sanity_check(data, max_print=5):
    G = data["graph"]
    g = data.get("geom", {})
    v = data.get("vertex", {})

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
    print("EDGE LENGTH ATTRS (VOX)")
    print("==============================")
    for name in ["length_steps", "length"]:
        if name in G.es.attributes():
            arr = np.asarray(G.es[name], dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                print(f"{name}: n={arr.size}  min={arr.min():.3f}  p50={np.median(arr):.3f}  p95={np.percentile(arr,95):.3f}  max={arr.max():.3f}")
            else:
                print(f"{name}: exists but empty/NaN")
        else:
            print(f"{name}: not found")

    print("\n==============================")
    print("ATLAS COORD RANGE CHECK (atlas vox grid)")
    print("==============================")
    if "coords" in v:
        coords_atlas = np.asarray(v["coords"])
        print(f"atlas coords shape: {coords_atlas.shape}")
        print(f"atlas min per axis: {coords_atlas.min(axis=0)}")
        print(f"atlas max per axis: {coords_atlas.max(axis=0)}")
    else:
        print("No vertex['coords'] found.")

    print("\n==============================")
    print("DIAMETER_ATLAS (raw units)")
    print("==============================")
    if "diameter_atlas" in G.es.attributes():
        d = np.asarray(G.es["diameter_atlas"], dtype=np.float32)
        d = d[np.isfinite(d)]
        if d.size:
            print(f"diameter_atlas: n={d.size}  p50={np.median(d):.3f}  p95={np.percentile(d,95):.3f}  max={d.max():.3f}  (RAW atlas units)")
    else:
        print("No edge diameter_atlas found.")

    print("\n==============================\n")


def _auto_out_path(in_path: str) -> str:
    in_dir = os.path.dirname(in_path)
    base = os.path.splitext(os.path.basename(in_path))[0]
    base_um = base if base.endswith("_um") else f"{base}_um"

    # if inside ".../vox/", output sibling ".../um/"
    if os.path.basename(in_dir) == "vox":
        out_dir = os.path.join(os.path.dirname(in_dir), "um")
    else:
        out_dir = os.path.join(in_dir, "um")

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base_um}.pkl")


def convert_outgeom_pkl_to_um(
    in_path,
    out_path=None,
    res_um_per_vox=(1.625, 1.625, 2.5),   # µm/vox (image space)
    atlas_um_per_vox=25.0,                # µm/vox (atlas grid)
    min_straight_dist_um=1.0,
):
    """
    Convert VOX outgeom -> UM outgeom.

    Assumptions (aligned with your new VOX builder):
    - geom[x,y,z] are in IMAGE VOXELS
    - geom lengths2 (if present) is VOX Euclidean, but we recompute in µm anyway
    - G.es["geom_start"], G.es["geom_end"] define per-edge point ranges in the global arrays
    - IMPORTANT: global geometry arrays are concatenated, so we must zero the segment at (ge-1)
      to kill cross-edge jumps.
    """

    sx, sy, sz = map(float, res_um_per_vox)
    spacing = np.array([sx, sy, sz], dtype=np.float32)

    data = pickle.load(open(in_path, "rb"))
    quick_unit_sanity_check(data)

    G = data["graph"]
    g = data["geom"]
    v = data["vertex"]

    # Required
    for k in ("x", "y", "z"):
        if k not in g:
            raise KeyError(f"geom['{k}'] missing in VOX PKL")

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise KeyError("Edges must have geom_start / geom_end.")

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    # -------------------------
    # 1) Convert geometry points to µm
    # -------------------------
    x_vox = np.asarray(g["x"], dtype=np.float32)
    y_vox = np.asarray(g["y"], dtype=np.float32)
    z_vox = np.asarray(g["z"], dtype=np.float32)

    x_um = x_vox * sx
    y_um = y_vox * sy
    z_um = z_vox * sz

    g_R = {
        "x_R": x_um.astype(np.float32, copy=False),
        "y_R": y_um.astype(np.float32, copy=False),
        "z_R": z_um.astype(np.float32, copy=False),
    }

    # -------------------------
    # 2) Vertex coords_image -> µm
    # -------------------------
    v_R = {}
    if "coords_image" in v:
        C_img_vox = np.asarray(v["coords_image"], dtype=np.float32)
        v_R["coords_image_R"] = (C_img_vox * spacing[None, :]).astype(np.float32)

    # distance_to_surface (approx scale by sx)
    if "distance_to_surface" in v:
        dist_surf_vox = np.asarray(v["distance_to_surface"], dtype=np.float32)
        v_R["distance_to_surface_R"] = (dist_surf_vox * sx).astype(np.float32, copy=False)

    # atlas radii (vertex): atlas vox -> µm
    if "radii_atlas" in v:
        R_atlas_vox = np.asarray(v["radii_atlas"], dtype=np.float32)
        v_R["radii_atlas_R"] = (R_atlas_vox * float(atlas_um_per_vox)).astype(np.float32)

    # -------------------------
    # 3) Compute lengths2_R in µm (Euclidean segments)
    # -------------------------
    dx = np.diff(x_um.astype(np.float64))
    dy = np.diff(y_um.astype(np.float64))
    dz = np.diff(z_um.astype(np.float64))

    lengths2_R = np.sqrt(dx*dx + dy*dy + dz*dz).astype(np.float32)
    lengths2_R = np.append(lengths2_R, np.float32(0.0))  # same length as x/y/z

    # sanity check: be sure that i am using the indices i want. 
    # Last point of the edge is [en-1]. This ensures in case there's a lengths2[s:en]
    boundary_idx = (ge - 1).astype(np.int64, copy=False)
    boundary_idx = boundary_idx[(boundary_idx >= 0) & (boundary_idx < lengths2_R.shape[0])]
    lengths2_R[boundary_idx] = np.float32(0.0)

    g_R["lengths2_R"] = lengths2_R

    # -------------------------
    # 4) Compute length_R per edge = sum(lengths2_R[s:en-1])  
    # -------------------------
    seg = lengths2_R[:-1].astype(np.float64, copy=False)       # true segments only
    pref = np.concatenate([[0.0], np.cumsum(seg)])             # pref[k] = sum(seg[0:k])

    length_R = (pref[ge - 1] - pref[gs]).astype(np.float32)
    length_R[(ge - gs) < 2] = 0.0

    G.es["length_R"] = length_R.tolist()
   

    # -------------------------
    # 5) Straight distance in µm + tortuosity_R
    # -------------------------
    dxs = (x_um[ge - 1] - x_um[gs]).astype(np.float32, copy=False)
    dys = (y_um[ge - 1] - y_um[gs]).astype(np.float32, copy=False)
    dzs = (z_um[ge - 1] - z_um[gs]).astype(np.float32, copy=False)
    straight_um = np.sqrt(dxs*dxs + dys*dys + dzs*dzs).astype(np.float32)

    tort = np.full(G.ecount(), np.nan, dtype=np.float32)
    m = straight_um >= float(min_straight_dist_um)
    tort[m] = length_R[m] / straight_um[m]
    G.es["tortuosity_R"] = tort.tolist()


    # -------------------------
    # 6) Atlas geometry radii -> µm (per point)
    # -------------------------
    if "radii_atlas_geom" in g:
        r_atlas_vox = np.asarray(g["radii_atlas_geom"], dtype=np.float32)
        r_atlas_um = (r_atlas_vox * float(atlas_um_per_vox)).astype(np.float32)
        g_R["radii_atlas_geom_R"] = r_atlas_um
        g_R["diam_atlas_geom_R"] = (2.0 * r_atlas_um).astype(np.float32)

    # edge atlas diameter -> µm
    if "diameter_atlas" in G.es.attributes():
        diam_edge_atlas_vox = np.asarray(G.es["diameter_atlas"], dtype=np.float32)
        G.es["diameter_atlas_R"] = (diam_edge_atlas_vox * float(atlas_um_per_vox)).astype(np.float32).tolist()

    # -------------------------
    # 7) Store metadata + save
    # -------------------------
    # keep original VOX units in data; add *_R 
    data["geom_R"] = g_R
    data["vertex_R"] = v_R
    data["graph"] = G

    # metadata to stop future confusion
    data["unit"] = {"vox": "voxel", "um": "micrometer"}
    data["spacing_um_per_voxel"] = list(map(float, res_um_per_vox))
    data["atlas_um_per_voxel"] = float(atlas_um_per_vox)

    # default output
    if out_path is None:
        out_path = _auto_out_path(in_path)

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved:", out_path)
    return data


if __name__ == "__main__":
    name = "graph_18_OutGeom"  # adjust
    root = "/home/ana/MicroBrain/output"
    in_path = f"{root}/{name}.pkl"  # adjust to location

    convert_outgeom_pkl_to_um(
        in_path=in_path,
        out_path=None,
        res_um_per_vox=(1.625, 1.625, 2.5),
        atlas_um_per_vox=25.0,
        min_straight_dist_um=1.0,
    )
