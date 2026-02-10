"""
Build vascular graph with indexed geometry and robust tortuosity.
Pseudo-JSON layout (topology in igraph, heavy arrays in numpy under data["vertex"]/data["geom"]).
Made by Ana (adapted to be RAM-safe).
"""


import os
import pickle
import numpy as np
import pandas as pd
import igraph as ig

# ============================
# Parameters
# ============================
folder = "/home/admin/Ana/MicroBrain/CSV/"
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
MIN_STRAIGHT_DIST = 1.0  # µm

# Performance knobs
CHUNK_EDGES = 200_000  # reduce (50k) if still heavy; increase if fast



def data_summary_pseudojson(data, max_show=12):
    """
    Prints:
      - igraph summary
      - vertex arrays under data["vertex"] as (v)
      - geometry-point arrays under data["geom"] as (p)
    """
    G = data["graph"]
    print(G.summary())

    def fmt_shape(arr):
        if isinstance(arr, np.ndarray):
            return f"{arr.shape} {arr.dtype}"
        return f"type={type(arr)}"

    # --- vertex (v) ---
    if "vertex" in data:
        vkeys = sorted(list(data["vertex"].keys()))
        print("\nvertex attrs (v):", ", ".join(vkeys[:max_show]) + (" ..." if len(vkeys) > max_show else ""))
        for k in vkeys[:max_show]:
            print(f"  {k} (v): {fmt_shape(data['vertex'][k])}")

    # --- geom points (p) ---
    if "geom" in data:
        pkeys = sorted(list(data["geom"].keys()))
        print("\ngeom attrs (p):", ", ".join(pkeys[:max_show]) + (" ..." if len(pkeys) > max_show else ""))
        for k in pkeys[:max_show]:
            print(f"  {k} (p): {fmt_shape(data['geom'][k])}")

    # Optional sanity: lengths
    if "geom" in data and {"x","y","z"} <= set(data["geom"].keys()):
        nP = int(data["geom"]["x"].shape[0])
        ok = (data["geom"]["y"].shape[0] == nP and data["geom"]["z"].shape[0] == nP)
        print(f"\npoints count (p): {nP:,}  xyz aligned: {ok}")

        # extra per-point keys aligned?
        for k, arr in data["geom"].items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] != nP:
                print(f"geom[{k}] first-dim {arr.shape[0]} != nP {nP}")

    print()

def reorient_edge_geometry_to_vertices(
    edges,
    geom_start,
    geom_end,
    x, y, z,
    coords_image,
    ann_geom=None,
    r_geom=None,
    tol=1e-6,
    verbose=True,
):
    """
    Fuerza que, para cada edge (u,v):
      - geom_start  corresponda al endpoint más cercano a coords_image[u]
      - geom_end-1  corresponda al endpoint más cercano a coords_image[v]

    Si no es así, invierte IN-PLACE la geometría (x,y,z) y los atributos point-wise.

    Parameters
    ----------
    edges : (nE,2) int array
    geom_start, geom_end : (nE,) int arrays
    x,y,z : (nP,) float arrays (global geometry)
    coords_image : (nV,3) float array
    ann_geom : (nP,) int array or None
    r_geom   : (nP,) float array or None
    tol : float
        tolerance (voxels) to decide orientation
    """

    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)

    # endpoints of each polyline
    P0 = np.column_stack([x[geom_start],      y[geom_start],      z[geom_start]])
    P1 = np.column_stack([x[geom_end - 1],    y[geom_end - 1],    z[geom_end - 1]])

    Cu = coords_image[u]
    Cv = coords_image[v]

    # squared distances
    d_direct = np.sum((P0 - Cu)**2, axis=1) + np.sum((P1 - Cv)**2, axis=1)
    d_swap   = np.sum((P0 - Cv)**2, axis=1) + np.sum((P1 - Cu)**2, axis=1)

    flip = d_swap + tol < d_direct

    if verbose:
        print(f"[reorient] flipping {int(np.sum(flip))} / {len(flip)} edges")

    # flip geometry IN-PLACE
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

        if r_geom is not None:
            r_geom[s:en] = r_geom[s:en][::-1]

    return flip

# ============================
# Load CSVs (pandas -> numpy arrays, no giant Python lists/tuples)
# ============================
print("=== START CSV → PKL (OUTGEOM pseudo-json) ===")

# vertices
vid = pd.read_csv(folder + "vertices.csv", header=None, dtype=np.int64).to_numpy().reshape(-1)
nV = int(vid.shape[0])

# vertex coords
coords_atlas = pd.read_csv(folder + "coordinates_atlas.csv", header=None, dtype=np.float32).to_numpy()     # (nV,3)
coords_img   = pd.read_csv(folder + "coordinates.csv", header=None, dtype=np.float32).to_numpy()          # (nV,3)

# vertex scalars
v_radii = pd.read_csv(folder + "radii.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
v_ann   = pd.read_csv(folder + "annotation.csv", header=None, dtype=np.int32).to_numpy().reshape(-1)
v_dist  = pd.read_csv(folder + "distance_to_surface.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

# edges (topology)
edges = pd.read_csv(folder + "edges.csv", header=None, dtype=np.int64).to_numpy()  # (nE,2)
nE = int(edges.shape[0])

# edge scalars
e_len  = pd.read_csv(folder + "length.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
e_rad  = pd.read_csv(folder + "radii_edge.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)
e_vein = pd.read_csv(folder + "vein.csv", header=None, dtype=np.int8).to_numpy().reshape(-1)
e_art  = pd.read_csv(folder + "artery.csv", header=None, dtype=np.int8).to_numpy().reshape(-1)

# geometry indices per edge
geom_idx = pd.read_csv(folder + "edge_geometry_indices.csv", header=None, dtype=np.int64).to_numpy()  # (nE,2)
gs = geom_idx[:, 0].astype(np.int64, copy=False)
ge = geom_idx[:, 1].astype(np.int64, copy=False)

# geometry coords (global)
geom_xyz = pd.read_csv(folder + "edge_geometry_coordinates.csv", header=None, dtype=np.float32).to_numpy()  # (nP,3)
x = geom_xyz[:, 0]
y = geom_xyz[:, 1]
z = geom_xyz[:, 2]

# geometry point-wise annotation + radii
ann_geom = pd.read_csv(folder + "edge_geometry_annotation.csv", header=None, dtype=np.int32).to_numpy().reshape(-1)
r_geom   = pd.read_csv(folder + "edge_geometry_radii.csv", header=None, dtype=np.float32).to_numpy().reshape(-1)

print("CSVs loaded")
print(f"nV={nV:,}  nE={nE:,}  nP={x.shape[0]:,}")

# ============================
# Sanity checks
# ============================
if coords_atlas.shape != (nV, 3):
    raise ValueError(f"coordinates_atlas shape {coords_atlas.shape} != ({nV},3)")
if coords_img.shape != (nV, 3):
    raise ValueError(f"coordinates (image) shape {coords_img.shape} != ({nV},3)")
if e_len.shape[0] != nE or e_rad.shape[0] != nE:
    raise ValueError("length/radii_edge mismatch with edges.csv")
if gs.shape[0] != nE or ge.shape[0] != nE:
    raise ValueError("edge_geometry_indices mismatch with edges.csv")
if ann_geom.shape[0] != x.shape[0]:
    raise ValueError(f"edge_geometry_annotation length ({ann_geom.shape[0]}) != geometry coords length ({x.shape[0]})")
if r_geom.shape[0] != x.shape[0]:
    raise ValueError(f"edge_geometry_radii length ({r_geom.shape[0]}) != geometry coords length ({x.shape[0]})")

# ============================
# Build igraph (keep it LIGHT)
#   - no coords/coords_image stored in G.vs (that is what killed RAM)
# ============================
# nkind vectorized
e_nkind = np.full(nE, 4, dtype=np.int16)
e_nkind[e_art == 1] = 2
e_nkind[e_vein == 1] = 3
e_diam = (2.0 * e_rad).astype(np.float32)

# Build graph
# NOTE: igraph needs edges as list of tuples; unavoidable conversion here
G = ig.Graph(n=nV, edges=edges.tolist(), directed=False)

# Minimal vertex attrs (scalars only)
G.vs["id"] = vid.tolist()

# Edge attrs (scalars + indices)
G.es["nkind"] = e_nkind.tolist()
G.es["radius"] = e_rad.tolist()
G.es["diameter"] = e_diam.tolist()
G.es["length"] = e_len.tolist()
G.es["geom_start"] = gs.tolist()
G.es["geom_end"] = ge.tolist()



# ============================
# Tortuosity (ROBUST) — computed without creating extra huge arrays
# ============================
length_tortuous = np.zeros(nE, dtype=np.float32)
straight_dist   = np.zeros(nE, dtype=np.float32)

# Loop in chunks to keep Python overhead reasonable (memory stays stable)
for i0 in range(0, nE, CHUNK_EDGES):
    i1 = min(i0 + CHUNK_EDGES, nE)
    for ei in range(i0, i1):
        s = int(gs[ei])
        en = int(ge[ei])
        if en - s < 2:
            continue

        dx = np.diff(x[s:en])
        dy = np.diff(y[s:en])
        dz = np.diff(z[s:en])

        lt = np.sum(np.sqrt(dx * dx + dy * dy + dz * dz))
        sd = np.sqrt(
            (x[en - 1] - x[s]) ** 2 +
            (y[en - 1] - y[s]) ** 2 +
            (z[en - 1] - z[s]) ** 2
        )

        length_tortuous[ei] = lt
        straight_dist[ei] = sd


# ============================
# Enforce geometry orientation
# ============================

edgesG = np.asarray(G.get_edgelist(), dtype=np.int64)  # (nE,2) en el orden interno de G

flip = reorient_edge_geometry_to_vertices(
    edges=edgesG,
    geom_start=gs,
    geom_end=ge,
    x=x, y=y, z=z,
    coords_image=coords_img,
    ann_geom=ann_geom,
    r_geom=r_geom,
    tol=1e-6,
    verbose=True,
)


tortuosity = np.full(nE, np.nan, dtype=np.float32)
mask = straight_dist >= float(MIN_STRAIGHT_DIST)
tortuosity[mask] = length_tortuous[mask] / straight_dist[mask]

# Store as edge attrs (list is OK: 1 float per edge)
G.es["length_tortuous"] = length_tortuous.tolist()
G.es["tortuosity"] = tortuosity.tolist()

print("Tortuosity computed")
print("  NaN:", int(np.sum(~mask)))
print("  Max:", float(np.nanmax(tortuosity)))

# ============================
# Save (pseudo-json)
# ============================
data = {
    "graph": G,

    # heavy per-vertex arrays live OUTSIDE igraph
    "vertex": {
        "id": vid,                       # (nV,)
        "coords": coords_atlas,          # (nV,3) atlas/um
        "coords_image": coords_img,      # (nV,3) voxel
        "vertex_annotation": v_ann,      # (nV,)
        "distance_to_surface": v_dist,   # (nV,)
        "radii": v_radii,                # (nV,)
    },

    # heavy per-geometry-point arrays
    "geom": {
        "x": x,                          # (nP,)
        "y": y,                          # (nP,)
        "z": z,                          # (nP,)
        "annotation": ann_geom,          # (nP,)
        "radii": r_geom,                 # (nP,)
    },
}

data_summary_pseudojson(data)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved graph to:", out_path)
print("=== DONE ===")

