import pickle
import numpy as np

# ----------------------------
# Inputs
# ----------------------------
in_pkl = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"

# Box USED for the cut (same coordinate system as x,y,z and coords_image)
# >>> Put your values here (in image voxels) <<<
xBox_um = [1500, 2500]
yBox_um = [1500,  2500]
zBox_um = [1500, 2500]


sx, sy, sz = 1.625, 1.625, 2.5
xBox = [xBox_um[0]/sx, xBox_um[1]/sx]
yBox = [yBox_um[0]/sy, yBox_um[1]/sy]
zBox = [zBox_um[0]/sz, zBox_um[1]/sz]

tol_err = 1e-3     # mismatch tolerance (vox)
eps_plane = 1e-2   # "on boundary plane" tolerance (vox) - adjust if needed

# ----------------------------
# Load
# ----------------------------
data = pickle.load(open(in_pkl, "rb"))
G = data["graph"]
x = np.asarray(data["coords"]["x"], np.float32)
y = np.asarray(data["coords"]["y"], np.float32)
z = np.asarray(data["coords"]["z"], np.float32)
V = np.asarray(G.vs["coords_image"], np.float32)

xmin, xmax = xBox
ymin, ymax = yBox
zmin, zmax = zBox

def on_boundary_plane(p):
    """True if point p lies on any of the 6 box planes (within eps)."""
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    return (
        abs(px - xmin) < eps_plane or abs(px - xmax) < eps_plane or
        abs(py - ymin) < eps_plane or abs(py - ymax) < eps_plane or
        abs(pz - zmin) < eps_plane or abs(pz - zmax) < eps_plane
    )

mismatch_edges = []
boundary_mismatch = []
interior_mismatch = []

for e in range(G.ecount()):
    # geometric endpoints (tortuous)
    s  = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])
    if en - s < 2:
        continue

    #polyline endpoints    
    A = np.array([x[s], y[s], z[s]], np.float32)
    B = np.array([x[en-1], y[en-1], z[en-1]], np.float32)

    # topological endpoints (non tortuous)
    u = G.es[e].source
    v = G.es[e].target
    U = V[u]
    W = V[v]

    # allow reversed orientation
    err1 = np.linalg.norm(U - A) + np.linalg.norm(W - B)
    err2 = np.linalg.norm(U - B) + np.linalg.norm(W - A)
    err = min(err1, err2)

    if err > tol_err:
        mismatch_edges.append((e, err))

        # classify as boundary if A or B lies on a box plane
        if on_boundary_plane(A) or on_boundary_plane(B):
            boundary_mismatch.append((e, err))
        else:
            interior_mismatch.append((e, err))

print("Total edges:", G.ecount())
print("Mismatch edges (err > tol):", len(mismatch_edges))
print("  -> boundary mismatches:", len(boundary_mismatch))
print("  -> interior mismatches:", len(interior_mismatch))

# (optional) show a few examples
print("\nExamples (edge_id, err): boundary", boundary_mismatch[:10])
print("Examples (edge_id, err): interior", interior_mismatch[:10])
