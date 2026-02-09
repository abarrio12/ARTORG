import pickle


p_non = "/home/admin/Ana/MicroBrain/18_igraph.pkl"
p_cut_out = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3.pkl"
out_cut_non = "/home/admin/Ana/MicroBrain/output/nonT_Hcut3.pkl"

# load non
non_obj = pickle.load(open(p_non, "rb"), encoding="latin1")
G_non = non_obj["graph"] if isinstance(non_obj, dict) and "graph" in non_obj else non_obj

# load already-cut outgeom
cut = pickle.load(open(p_cut_out, "rb"), encoding="latin1")
H = cut["graph"]  # cut outgeom graph

# edge keys from cut (by vertex ids)
edge_keys = set()
for e in H.es:
    u, v = e.tuple
    idu = int(H.vs[u]["id"])
    idv = int(H.vs[v]["id"])
    edge_keys.add((idu, idv) if idu < idv else (idv, idu))

# filter non edges by connectivity (already ids)
keep_non_edges = []
for ei, e in enumerate(G_non.es):
    a, b = e["connectivity"]
    a = int(a); b = int(b)
    k = (a, b) if a < b else (b, a)
    if k in edge_keys:
        keep_non_edges.append(ei)

G_non_cut = G_non.subgraph_edges(keep_non_edges, delete_vertices=True)

pickle.dump(G_non_cut, open(out_cut_non, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
print("saved non cut:", out_cut_non, "edges:", G_non_cut.ecount(), "verts:", G_non_cut.vcount())

print("edges in outgeom cut:", len(edge_keys))
print("edges kept in non:", G_non_cut.ecount())

import numpy as np

def mean_diameter_by_nkind(G, name):
    diam = np.asarray(G.es["diameter"], dtype=float)
    nk   = np.asarray(G.es["nkind"], dtype=int)

    ok = np.isfinite(diam)
    diam = diam[ok]
    nk   = nk[ok]

    print(f"\n{name}: mean(diameter) por nkind")
    for k in sorted(set(nk.tolist())):
        d = diam[nk == k]
        if d.size:
            print(f"  nkind {k}: mean_diam = {d.mean():.4g}   (n={d.size})")

# NON (cut)
mean_diameter_by_nkind(G_non_cut, "NON cut")

# TORTUOSO (OutGeom cut)
mean_diameter_by_nkind(H, "TORTUOSO cut")
