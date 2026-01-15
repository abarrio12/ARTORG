import pickle, numpy as np

in_pkl = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"
data = pickle.load(open(in_pkl, "rb"))
G = data["graph"]
x = np.asarray(data["coords"]["x"], np.float32)
y = np.asarray(data["coords"]["y"], np.float32)
z = np.asarray(data["coords"]["z"], np.float32)
V = np.asarray(G.vs["coords_image"], np.float32)

errs = []
for e in range(G.ecount()):
    s  = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])
    if en - s < 2:
        continue

    A = np.array([x[s], y[s], z[s]], np.float32)
    B = np.array([x[en-1], y[en-1], z[en-1]], np.float32)

    u = G.es[e].source
    v = G.es[e].target
    U = V[u]
    W = V[v]

    # probamos ambas correspondencias (porque la polyline puede estar invertida)
    err1 = np.linalg.norm(U - A) + np.linalg.norm(W - B)
    err2 = np.linalg.norm(U - B) + np.linalg.norm(W - A)
    errs.append(min(err1, err2))

errs = np.asarray(errs)
print("edges checked:", len(errs))
print("median err:", np.median(errs))
print("99% err:", np.quantile(errs, 0.99))
print("max err:", errs.max())
print("count err>1e-3:", int((errs>1e-3).sum()))
