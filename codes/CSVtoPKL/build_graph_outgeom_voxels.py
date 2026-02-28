"""
Build vascular graph with indexed geometry (VOXEL SPACE)

IMPORTANT (based on what we learned)
-----------------------------------
- length.csv is NOT a geometric distance.
- length.csv == number of polyline segments per edge:
      length_steps = geom_end - geom_start - 1
  (diagonals do NOT count extra; every tiny-step counts as 1)

Therefore, in this VOX build file we store:
- G.es["length_steps"]  : step-count length from CSV
- (optionally also keep G.es["length"] = length_steps for compatibility)
- geom["lengths2"]      : per-segment costs (1 inside edges, 0 at boundaries)

Units
-----
Everything stays in VOXELS in this script.

Two voxel spaces are present:
- Image voxel space (original microscopy resolution: 1.625 x 1.625 x 2.5).
- Atlas voxel space (25 µm grid; used for radii_atlas and radii_atlas_geom).

No conversion to micrometers (µm) is performed here. 
Conversion to physical units is handled separately. Check convert_outgeom_voxels_to_um.py for that.

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

Note: length in Paris code is equal to the sum of n segments within the tortuous edge,
not the actual value of the segments. This value is been stored in length_steps. 
The actual value of the edge has been computed as the sum(lengths2) here. 
Be aware of this new attribute diference.

Author: Ana Barrio
Updated: 27 Feb 2026
"""

import os
import pickle
import igraph as ig
import numpy as np
import pandas as pd

# =============================================================================
# Parameters
# =============================================================================
FOLDER = "/home/ana/MicroBrain/CSV/"
OUT_PATH = "/home/ana/MicroBrain/output/graph_18_OutGeom.pkl"
MIN_STRAIGHT_DIST = 1.0  # voxels (image space)

# =============================================================================
# Helpers
# =============================================================================
def data_summary_pseudojson(data, max_show=12):
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
    print()


def reorient_edge_geometry_to_vertices(
    edges,
    geom_start,
    geom_end,
    x, y, z,
    coords_image,
    ann_geom=None,
    r_geom_list=None,
    tol=1e-6,
    verbose=True,
):
    """Flip per-edge polyline direction to match edge (u->v). In-place."""
    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)

    P0 = np.column_stack([x[geom_start], y[geom_start], z[geom_start]])
    P1 = np.column_stack([x[geom_end - 1], y[geom_end - 1], z[geom_end - 1]])

    Cu = coords_image[u]
    Cv = coords_image[v]

    d_direct = np.sum((P0 - Cu) ** 2, axis=1) + np.sum((P1 - Cv) ** 2, axis=1)
    d_swap   = np.sum((P0 - Cv) ** 2, axis=1) + np.sum((P1 - Cu) ** 2, axis=1)

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


def sanity_length_is_segcount_or_raise(e_len_steps, gs, ge, tol=0.0, max_report=10):
    """Check length.csv equals (ge-gs-1) for ALL edges."""
    segcount = (ge - gs - 1).astype(np.float32)
    diff = np.abs(e_len_steps.astype(np.float32) - segcount)
    bad = np.where(diff > tol)[0]
    if bad.size:
        print(f"[ERROR] length.csv != (ge-gs-1) for {bad.size} edges. Showing first {min(max_report, bad.size)}:")
        for ei in bad[:max_report]:
            print(f"  ei={int(ei)}  length={float(e_len_steps[ei])}  segcount={float(segcount[ei])}  "
                  f"s={int(gs[ei])} en={int(ge[ei])}")
        raise ValueError("length.csv sanity check failed.")
    print("[OK] length.csv equals segcount (ge-gs-1) for all edges.")


def sanity_length_equals_sum_lengths2_or_raise(length_edge, gs, ge, lengths2, n_check=10, seed=0, tol=1e-3):
    """Random check: length_edge == sum(lengths2[s:en-1]) for a few edges."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(length_edge), size=min(n_check, len(length_edge)), replace=False)
    for ei in idx:
        s = int(gs[ei]); en = int(ge[ei])
        if en - s < 2:
            continue
        sm = float(np.sum(lengths2[s:en-1]))
        if abs(float(length_edge[ei]) - sm) > tol:
            raise ValueError(f"edge {ei}: length != sum(lengths2) -> {float(length_edge[ei])} vs {sm}")
    print(f"[OK] length equals sum(lengths2) on {len(idx)} sampled edges.")


def sanity_e_rad_equals_max_r_geom(e_rad, gs, ge, r_geom, n_check=5, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e_rad), size=min(n_check, len(e_rad)), replace=False)
    all_true = True
    for ei in idx:
        s, en = int(gs[ei]), int(ge[ei])
        if en - s < 1:
            continue
        equal = abs(float(e_rad[ei]) - float(np.max(r_geom[s:en]))) <= tol
        print(f"edge {ei}: e_rad == max(r_geom) ? {equal}")
        all_true = all_true and equal
    return all_true


def sanity_e_rad_atlas_equals_max_r_atlas_geom(e_rad_atlas, gs, ge, r_atlas_geom, n_check=5, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e_rad_atlas), size=min(n_check, len(e_rad_atlas)), replace=False)
    all_true = True
    for ei in idx:
        s, en = int(gs[ei]), int(ge[ei])
        if en - s < 1:
            continue
        equal = abs(float(e_rad_atlas[ei]) - float(np.max(r_atlas_geom[s:en]))) <= tol
        print(f"edge {ei}: e_rad_atlas == max(r_atlas_geom) ? {equal}")
        all_true = all_true and equal
    return all_true


# =============================================================================
# Load CSVs
# =============================================================================
print("=== START CSV → PKL (OUTGEOM pseudo-json) ===")

vid = pd.read_csv(os.path.join(FOLDER, "vertices.csv"), header=None, dtype=np.int64).to_numpy().reshape(-1)
nV = int(vid.shape[0])

coords_atlas = pd.read_csv(os.path.join(FOLDER, "coordinates_atlas.csv"), header=None, dtype=np.float32).to_numpy()
coords_img   = pd.read_csv(os.path.join(FOLDER, "coordinates.csv"), header=None, dtype=np.float32).to_numpy()

v_radii       = pd.read_csv(os.path.join(FOLDER, "radii.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
v_ann         = pd.read_csv(os.path.join(FOLDER, "annotation.csv"), header=None, dtype=np.int32).to_numpy().reshape(-1)
v_dist        = pd.read_csv(os.path.join(FOLDER, "distance_to_surface.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
v_radii_atlas = pd.read_csv(os.path.join(FOLDER, "radii_atlas.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)

edges = pd.read_csv(os.path.join(FOLDER, "edges.csv"), header=None, dtype=np.int64).to_numpy()
nE = int(edges.shape[0])

e_len_steps = pd.read_csv(os.path.join(FOLDER, "length.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
e_rad       = pd.read_csv(os.path.join(FOLDER, "radii_edge.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
e_rad_atlas = pd.read_csv(os.path.join(FOLDER, "radii_atlas_edge.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
e_vein      = pd.read_csv(os.path.join(FOLDER, "vein.csv"), header=None, dtype=np.int8).to_numpy().reshape(-1)
e_art       = pd.read_csv(os.path.join(FOLDER, "artery.csv"), header=None, dtype=np.int8).to_numpy().reshape(-1)

geom_idx = pd.read_csv(os.path.join(FOLDER, "edge_geometry_indices.csv"), header=None, dtype=np.int64).to_numpy()
gs = geom_idx[:, 0].astype(np.int64, copy=False)
ge = geom_idx[:, 1].astype(np.int64, copy=False)

geom_xyz = pd.read_csv(os.path.join(FOLDER, "edge_geometry_coordinates.csv"), header=None, dtype=np.float32).to_numpy()
ann_geom     = pd.read_csv(os.path.join(FOLDER, "edge_geometry_annotation.csv"), header=None, dtype=np.int32).to_numpy().reshape(-1)
r_geom       = pd.read_csv(os.path.join(FOLDER, "edge_geometry_radii.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)
r_atlas_geom = pd.read_csv(os.path.join(FOLDER, "edge_geometry_radii_atlas.csv"), header=None, dtype=np.float32).to_numpy().reshape(-1)

print("CSVs loaded")

nP = int(geom_xyz.shape[0])
print(f"nV={nV:,}  nE={nE:,}  nP={nP:,}")

# =============================================================================
# Basic shape checks
# =============================================================================
if coords_atlas.shape != (nV, 3): raise ValueError("coordinates_atlas shape mismatch")
if coords_img.shape   != (nV, 3): raise ValueError("coordinates (image) shape mismatch")
if e_len_steps.shape[0] != nE: raise ValueError("length.csv mismatch with edges.csv")
if e_rad.shape[0] != nE or e_rad_atlas.shape[0] != nE: raise ValueError("radii_edge mismatch with edges.csv")
if gs.shape[0] != nE or ge.shape[0] != nE: raise ValueError("edge_geometry_indices mismatch with edges.csv")
if ann_geom.shape[0] != nP: raise ValueError("edge_geometry_annotation length mismatch with geometry coords")
if r_geom.shape[0] != nP: raise ValueError("edge_geometry_radii length mismatch with geometry coords")
if r_atlas_geom.shape[0] != nP: raise ValueError("edge_geometry_radii_atlas length mismatch with geometry coords")

# =============================================================================
# Geometry arrays (mutable copies)
# =============================================================================
x = geom_xyz[:, 0].copy()
y = geom_xyz[:, 1].copy()
z = geom_xyz[:, 2].copy()

# =============================================================================
# Build igraph
# =============================================================================
e_nkind = np.full(nE, 4, dtype=np.int16)  # default capillary
e_nkind[e_art == 1] = 2
e_nkind[e_vein == 1] = 3

G = ig.Graph(n=nV, edges=edges.tolist(), directed=False)
G.vs["id"] = vid.tolist()
G.es["nkind"] = e_nkind.tolist()

# per-edge step count
G.es["length_steps"] = e_len_steps.astype(np.float32).tolist()

# other edge attrs
G.es["radius"] = e_rad.tolist()
G.es["radius_atlas"] = e_rad_atlas.tolist()
G.es["diameter"] = (2 * e_rad).tolist()
G.es["diameter_atlas"] = (2 * e_rad_atlas).tolist()
G.es["geom_start"] = gs.tolist()
G.es["geom_end"] = ge.tolist()

# =============================================================================
# Reorient polylines (in-place)
# =============================================================================
edgesG = np.asarray(G.get_edgelist(), dtype=np.int64)
_ = reorient_edge_geometry_to_vertices(
    edges=edgesG,
    geom_start=gs,
    geom_end=ge,
    x=x, y=y, z=z,
    coords_image=coords_img,
    ann_geom=ann_geom,
    r_geom_list=[r_geom, r_atlas_geom],
    tol=1e-6,
    verbose=True,
)

# derived atlas diameter AFTER flips
diam_atlas_geom = (2.0 * r_atlas_geom).astype(np.float32, copy=False)

# =============================================================================
# lengths2 (global) = Euclidean distances between consecutive geometry points (VOX)
# =============================================================================
dx = np.diff(x.astype(np.float64))
dy = np.diff(y.astype(np.float64))
dz = np.diff(z.astype(np.float64))
lengths2 = np.sqrt(dx*dx + dy*dy + dz*dz).astype(np.float32)
lengths2 = np.append(lengths2, np.float32(0.0))

# kill cross-edge jumps in the global array
boundary_idx = (ge - 1).astype(np.int64, copy=False)
boundary_idx = boundary_idx[(boundary_idx >= 0) & (boundary_idx < lengths2.shape[0])]
lengths2[boundary_idx] = np.float32(0.0)

# =============================================================================
# length (per edge) = sum(lengths2[s:en-1]) 
# =============================================================================
start_idx = gs.astype(np.int64, copy=False)
end_idx   = ge.astype(np.int64, copy=False)

seg = lengths2[:-1].astype(np.float64, copy=False)          # only true segments
pref = np.concatenate([[0.0], np.cumsum(seg)])              # pref[k] = sum(seg[0:k])

length_edge = (pref[end_idx - 1] - pref[start_idx]).astype(np.float32)
length_edge[(end_idx - start_idx) < 2] = 0.0
G.es["length"] = length_edge.tolist()

# =============================================================================
# Sanity checks
# =============================================================================
sanity_length_is_segcount_or_raise(e_len_steps, gs, ge, tol=0.0, max_report=10)
sanity_length_equals_sum_lengths2_or_raise(length_edge, gs, ge, lengths2, n_check=10, seed=0, tol=1e-3)

ok_rad = sanity_e_rad_equals_max_r_geom(e_rad, gs, ge, r_geom, n_check=5, seed=0, tol=1e-6)
print("All 5 radius checks OK?", ok_rad)

ok_rad_atlas = sanity_e_rad_atlas_equals_max_r_atlas_geom(e_rad_atlas, gs, ge, r_atlas_geom, n_check=5, seed=0, tol=1e-6)
print("All 5 atlas radius checks OK?", ok_rad_atlas)

# =============================================================================
# Tortuosity (use REAL tortuous length in vox)
# =============================================================================
straight_dist = np.zeros(nE, dtype=np.float32)
for ei in range(nE):
    s = int(gs[ei]); en = int(ge[ei])
    if en - s < 2:
        continue
    ddx = x[en - 1] - x[s]
    ddy = y[en - 1] - y[s]
    ddz = z[en - 1] - z[s]
    straight_dist[ei] = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)

tortuosity = np.full(nE, np.nan, dtype=np.float32)
mask = straight_dist >= float(MIN_STRAIGHT_DIST)
tortuosity[mask] = length_edge[mask] / straight_dist[mask]
G.es["tortuosity"] = tortuosity.tolist()

# (optional) keep step-based tortuosity too
tortuosity_steps = np.full(nE, np.nan, dtype=np.float32)
tortuosity_steps[mask] = e_len_steps[mask] / straight_dist[mask]
G.es["tortuosity_steps"] = tortuosity_steps.tolist()

# metadata
G["unit"] = "voxels of image (res: 1.625 x 1.625 x 2.5)"

print("Tortuosity computed (arc-length in vox)")
print("  NaN:", int(np.sum(~mask)))
print("  Max:", float(np.nanmax(tortuosity)))

# =============================================================================
# Save (pseudo-json)
# =============================================================================
data = {
    "graph": G,
    "vertex": {
        "id": vid,
        "coords": coords_atlas,
        "coords_image": coords_img,
        "vertex_annotation": v_ann,
        "distance_to_surface": v_dist,
        "radii": v_radii,
        "radii_atlas": v_radii_atlas,
    },
    "geom": {
        "x": x,
        "y": y,
        "z": z,
        "lengths2": lengths2,                 # distances between points
        "annotation": ann_geom,
        "radii": r_geom,
        "radii_atlas_geom": r_atlas_geom,
        "diam_atlas_geom": diam_atlas_geom,
    },
}

data_summary_pseudojson(data)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved graph to:", OUT_PATH)
print("=== DONE ===")
