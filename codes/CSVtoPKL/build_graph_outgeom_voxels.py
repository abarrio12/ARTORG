"""
Build vascular graph with indexed geometry (VOXEL SPACE)

This script builds a vascular graph from ClearMap CSV exports and stores it
as a pseudo-JSON PKL structure.

Key idea
--------
- Keep igraph lightweight (topology + scalar edge attributes + geometry indices).
- Store tortuous polylines globally in NumPy arrays under data["geom"].
- Each edge stores only geom_start / geom_end indices into the global arrays.

Units
-----
Everything stays in VOXELS in this script.

Two voxel spaces are present:
- Image voxel space (original microscopy resolution).
- Atlas voxel space (25 µm grid; used for radii_atlas and radii_atlas_geom).

No conversion to micrometers (µm) is performed here.
Conversion to physical units is handled separately (see outgeom_um.py).

Output structure
----------------
data
 ├── graph   (igraph topology)
 │      ├── scalar edge attributes (length, radius, radius_atlas, tortuosity, nkind, ...)
 │      └── geometry indices (geom_start, geom_end)
 │
 ├── vertex  (per-vertex arrays, voxels)
 │      ├── coords_image
 │      ├── annotation
 │      ├── distance_to_surface
 │      ├── radii
 │      └── radii_atlas
 │
 └── geom    (per-geometry-point arrays, voxels)
        ├── x, y, z
        ├── annotation
        ├── radii
        └── radii_atlas_geom

Author: Ana Barrio
Date: 10 Feb 2026 (updated: 16 Feb 2026)
"""

import os
import pickle

import igraph as ig
import numpy as np
import pandas as pd

# =============================================================================
# Parameters
# =============================================================================
FOLDER = "/home/admin/Ana/MicroBrain/CSV/"
OUT_PATH = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
MIN_STRAIGHT_DIST = 1.0  # voxels (image space)


# =============================================================================
# Helpers
# =============================================================================
def data_summary_pseudojson(data, max_show=12):
    """Print a lightweight summary of the pseudo-JSON structure."""
    G = data["graph"]
    print(G.summary())

    if "vertex" in data:
        vkeys = sorted(list(data["vertex"].keys()))
        print("\nvertex attrs (v):", ", ".join(vkeys[:max_show]) + (" ..." if len(vkeys) > max_show else ""))
        for k in vkeys[:max_show]:
            arr = data["vertex"][k]
            print(f"  {k} (v): {arr.shape} {arr.dtype}")

    if "geom" in data:
        pkeys = sorted(list(data["geom"].keys()))
        print("\ngeom attrs (p):", ", ".join(pkeys[:max_show]) + (" ..." if len(pkeys) > max_show else ""))
        for k in pkeys[:max_show]:
            arr = data["geom"][k]
            print(f"  {k} (p): {arr.shape} {arr.dtype}")

    if "geom" in data and {"x", "y", "z"} <= set(data["geom"].keys()):
        nP = int(data["geom"]["x"].shape[0])
        ok = (data["geom"]["y"].shape[0] == nP) and (data["geom"]["z"].shape[0] == nP)
        print(f"\npoints count (p): {nP:,}  xyz aligned: {ok}")

        for k, arr in data["geom"].items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] != nP:
                print(f"geom[{k}] first-dim {arr.shape[0]} != nP {nP}")

    print()


def reorient_edge_geometry_to_vertices(
    edges,
    geom_start,
    geom_end,
    x,
    y,
    z,
    coords_image,
    ann_geom=None,
    r_geom_list=None,
    tol=1e-6,
    verbose=True,
):
    """
    Enforce polyline direction to match vertex order (u -> v).
    Flips x/y/z (and optional per-point arrays) IN-PLACE when needed.
    """
    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)

    P0 = np.column_stack([x[geom_start], y[geom_start], z[geom_start]])
    P1 = np.column_stack([x[geom_end - 1], y[geom_end - 1], z[geom_end - 1]])

    Cu = coords_image[u]
    Cv = coords_image[v]

    d_direct = np.sum((P0 - Cu) ** 2, axis=1) + np.sum((P1 - Cv) ** 2, axis=1)
    d_swap = np.sum((P0 - Cv) ** 2, axis=1) + np.sum((P1 - Cu) ** 2, axis=1)

    flip = d_swap + tol < d_direct

    if verbose:
        print(f"[reorient] flipping {int(np.sum(flip))} / {len(flip)} edges")

    for ei in np.where(flip)[0]:
        s = int(geom_start[ei])
        en = int(geom_end[ei])
        if en - s < 2:
            continue

        x[s:en] = x[s:en][::-1]
        y[s:en] = y[s:en][::-1]
        z[s:en] = z[s:en][::-1]

        if ann_geom is not None:
            ann_geom[s:en] = ann_geom[s:en][::-1]

        if r_geom_list is not None:
            for r in r_geom_list:
                if r is not None:
                    r[s:en] = r[s:en][::-1]

    return flip


# =============================================================================
# Sanity checks
# =============================================================================
def sanity_e_len_equals_sum_lengths2(e_len, gs, ge, x, y, z, n_check=5, seed=0, tol=1e-3):
    """
    Check: e_len[ei] ~= sum(||P[i+1] - P[i]||) along the polyline for n_check edges.
    Returns True only if all tested edges satisfy the condition.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e_len), size=min(n_check, len(e_len)), replace=False)

    all_true = True
    for ei in idx:
        s, en = int(gs[ei]), int(ge[ei])
        if en - s < 2:
            print(f"edge {ei}: SKIP (npts={en-s})")
            continue

        dx = np.diff(x[s:en])
        dy = np.diff(y[s:en])
        dz = np.diff(z[s:en])
        lengths2 = np.sqrt(dx * dx + dy * dy + dz * dz)

        equal = abs(float(e_len[ei]) - float(lengths2.sum())) <= tol
        print(f"edge {ei}: e_len == sum(lengths2) ? {equal}")
        all_true = all_true and equal

    return all_true


def sanity_e_rad_equals_max_r_geom(e_rad, gs, ge, r_geom, n_check=5, seed=0, tol=1e-6):
    """
    Check: e_rad[ei] ~= max(r_geom[s:en]) for n_check edges.
    Returns True only if all tested edges satisfy the condition.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e_rad), size=min(n_check, len(e_rad)), replace=False)

    all_true = True
    for ei in idx:
        s, en = int(gs[ei]), int(ge[ei])
        if en - s < 1:
            print(f"edge {ei}: SKIP (npts={en-s})")
            continue

        equal = abs(float(e_rad[ei]) - float(np.max(r_geom[s:en]))) <= tol
        print(f"edge {ei}: e_rad == max(r_geom) ? {equal}")
        all_true = all_true and equal

    return all_true


def sanity_e_rad_atlas_equals_max_r_atlas_geom(e_rad_atlas, gs, ge, r_atlas_geom, n_check=5, seed=0, tol=1e-6):
    """
    Check: e_rad_atlas[ei] ~= max(r_atlas_geom[s:en]) for n_check edges.
    Returns True only if all tested edges satisfy the condition.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e_rad_atlas), size=min(n_check, len(e_rad_atlas)), replace=False)

    all_true = True
    for ei in idx:
        s, en = int(gs[ei]), int(ge[ei])
        if en - s < 1:
            print(f"edge {ei}: SKIP (npts={en-s})")
            continue

        equal = abs(float(e_rad_atlas[ei]) - float(np.max(r_atlas_geom[s:en]))) <= tol
        print(f"edge {ei}: e_rad_atlas == max(r_atlas_geom) ? {equal}")
        all_true = all_true and equal

    return all_true


# =============================================================================
# Load CSVs (pandas -> NumPy arrays)
# =============================================================================
print("=== START CSV → PKL (OUTGEOM pseudo-json) ===")

# vertices
vid = pd.read_csv(FOLDER + "vertices.csv", header=None, dtype=np.int64).to_numpy().reshape(-1)
nV = int(vid.shape[0])

coords_atlas = pd.read_csv(FOLDER + "coordinates_atlas.csv", header=None, dtype=np.float32).to_numpy()  # (nV,3)
coords_img = pd.read_csv(FOLDER + "coordinates.csv", header=None, dtype=np.float32).to_numpy()  # (nV,3)

v_radii = pd.read_csv(FOLDER + "radii.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
v_ann = pd.read_csv(FOLDER + "annotation.csv", header=None, dtype=np.int32).to_numpy().reshape(-1)
v_dist = pd.read_csv(FOLDER + "distance_to_surface.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

# vertex atlas radii (added explicitly)
v_radii_atlas = pd.read_csv(FOLDER + "radii_atlas.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

# edges
edges = pd.read_csv(FOLDER + "edges.csv", header=None, dtype=np.int64).to_numpy()  # (nE,2)
nE = int(edges.shape[0])

e_len = pd.read_csv(FOLDER + "length.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
e_rad = pd.read_csv(FOLDER + "radii_edge.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

# edge atlas radii (NEW): must exist as per-edge atlas radii
# IMPORTANT: rename the filename if your export uses a different name.
e_rad_atlas = pd.read_csv(FOLDER + "radii_edge_atlas.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

e_vein = pd.read_csv(FOLDER + "vein.csv", header=None, dtype=np.int8).to_numpy().reshape(-1)
e_art = pd.read_csv(FOLDER + "artery.csv", header=None, dtype=np.int8).to_numpy().reshape(-1)

# geometry indices
geom_idx = pd.read_csv(FOLDER + "edge_geometry_indices.csv", header=None, dtype=np.int64).to_numpy()  # (nE,2)
gs = geom_idx[:, 0].astype(np.int64, copy=False)
ge = geom_idx[:, 1].astype(np.int64, copy=False)

# global geometry points
geom_xyz = pd.read_csv(FOLDER + "edge_geometry_coordinates.csv", header=None, dtype=np.float32).to_numpy()  # (nP,3)
x = geom_xyz[:, 0]
y = geom_xyz[:, 1]
z = geom_xyz[:, 2]

# per-point attributes
ann_geom = pd.read_csv(FOLDER + "edge_geometry_annotation.csv", header=None, dtype=np.int32).to_numpy().reshape(-1)
r_geom = pd.read_csv(FOLDER + "edge_geometry_radii.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
r_atlas_geom = pd.read_csv(FOLDER + "edge_geometry_radii_atlas.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

print("CSVs loaded")
print(f"nV={nV:,}  nE={nE:,}  nP={x.shape[0]:,}")


# =============================================================================
# Basic shape sanity checks
# =============================================================================
if coords_atlas.shape != (nV, 3):
    raise ValueError(f"coordinates_atlas shape {coords_atlas.shape} != ({nV}, 3)")
if coords_img.shape != (nV, 3):
    raise ValueError(f"coordinates (image) shape {coords_img.shape} != ({nV}, 3)")
if e_len.shape[0] != nE or e_rad.shape[0] != nE:
    raise ValueError("length/radii_edge mismatch with edges.csv")
if e_rad_atlas.shape[0] != nE:
    raise ValueError("radii_edge_atlas mismatch with edges.csv")
if gs.shape[0] != nE or ge.shape[0] != nE:
    raise ValueError("edge_geometry_indices mismatch with edges.csv")
if ann_geom.shape[0] != x.shape[0]:
    raise ValueError(f"edge_geometry_annotation length ({ann_geom.shape[0]}) != geometry coords length ({x.shape[0]})")
if r_geom.shape[0] != x.shape[0]:
    raise ValueError(f"edge_geometry_radii length ({r_geom.shape[0]}) != geometry coords length ({x.shape[0]})")
if r_atlas_geom.shape[0] != x.shape[0]:
    raise ValueError("edge_geometry_radii_atlas mismatch with geometry coords")


# =============================================================================
# Build igraph (lightweight)
# =============================================================================
# nkind encoding: 2=arteries, 3=veins, 4=capillaries (default)
e_nkind = np.full(nE, 4, dtype=np.int16)
e_nkind[e_art == 1] = 2
e_nkind[e_vein == 1] = 3

G = ig.Graph(n=nV, edges=edges.tolist(), directed=False)
G.vs["id"] = vid.tolist()

# edge attributes (scalars + geometry indices)
G.es["nkind"] = e_nkind.tolist()
G.es["length"] = e_len.tolist()

# store per-edge radii (image and atlas)
G.es["radius"] = e_rad.tolist()
G.es["radius_atlas"] = e_rad_atlas.tolist()
G.es["diameter"] = (2 * e_rad).tolist()
G.es["diameter_atlas"] = (2 * e_rad_atlas).tolist()

G.es["geom_start"] = gs.tolist()
G.es["geom_end"] = ge.tolist()


# =============================================================================
# Tortuosity (dimensionless): tortuosity = e_len / straight_distance
# =============================================================================
straight_dist = np.zeros(nE, dtype=np.float32)
for ei in range(nE):
    s = int(gs[ei])
    en = int(ge[ei])
    if en - s < 2:
        continue

    dx = x[en - 1] - x[s]
    dy = y[en - 1] - y[s]
    dz = z[en - 1] - z[s]
    straight_dist[ei] = np.sqrt(dx * dx + dy * dy + dz * dz)


# =============================================================================
# Reorient polylines to match vertex order (in-place)
# =============================================================================
edgesG = np.asarray(G.get_edgelist(), dtype=np.int64)  # same edge order as G.es

_ = reorient_edge_geometry_to_vertices(
    edges=edgesG,
    geom_start=gs,
    geom_end=ge,
    x=x,
    y=y,
    z=z,
    coords_image=coords_img,
    ann_geom=ann_geom,
    r_geom_list=[r_geom, r_atlas_geom],
    tol=1e-6,
    verbose=True,
)


# =============================================================================
# Sanity checks (length + radii + radii_atlas)
# =============================================================================
ok_len = sanity_e_len_equals_sum_lengths2(e_len, gs, ge, x, y, z, n_check=5, seed=0, tol=1e-3)
print("All 5 length checks OK?", ok_len)

ok_rad = sanity_e_rad_equals_max_r_geom(e_rad, gs, ge, r_geom, n_check=5, seed=0, tol=1e-6)
print("All 5 radius checks OK?", ok_rad)

ok_rad_atlas = sanity_e_rad_atlas_equals_max_r_atlas_geom(e_rad_atlas, gs, ge, r_atlas_geom, n_check=5, seed=0, tol=1e-6)
print("All 5 atlas radius checks OK?", ok_rad_atlas)


# =============================================================================
# Tortuosity attribute
# =============================================================================
tortuosity = np.full(nE, np.nan, dtype=np.float32)
mask = straight_dist >= float(MIN_STRAIGHT_DIST)
tortuosity[mask] = e_len[mask] / straight_dist[mask]
G.es["tortuosity"] = tortuosity.tolist()

print("Tortuosity computed")
print("  NaN:", int(np.sum(~mask)))
print("  Max:", float(np.nanmax(tortuosity)))


# =============================================================================
# Save (pseudo-json)
# =============================================================================
data = {
    "graph": G,
    "vertex": {
        "id": vid,
        "coords": coords_atlas,  # atlas voxel space
        "coords_image": coords_img,  # image voxel space
        "vertex_annotation": v_ann,
        "distance_to_surface": v_dist,
        "radii": v_radii,  # image voxel space
        "radii_atlas": v_radii_atlas,  # atlas voxel space (25 µm grid)
        
    },
    "geom": {
        "x": x,  # image voxel space
        "y": y,  # image voxel space
        "z": z,  # image voxel space
        "annotation": ann_geom,
        "radii": r_geom,  # image voxel space
        "radii_atlas_geom": r_atlas_geom,  # atlas voxel space (25 µm grid)  # !! needs to be extracted !!
    },
}

data_summary_pseudojson(data)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved graph to:", OUT_PATH)
print("=== DONE ===")
