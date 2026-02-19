# ==============================================================================
# DESCRIPTION: Cross-referencing and filtering a Non-Tortuous graph based on
#              an existing tortuous with OutGeom spatial cut + sanity checks
#              (length consistency + error stats for diameter/length on common edges)
#
# This is already done in the cutting code (cut_outgeom_roi_VOX and UM and in Cut_the_graph_Gaia)
# This was a separate script for testing and sanity checks, but the main logic is now integrated in the cutting code.
#
# AUTHOR: Ana Barrio
# DATE: 6-02-26
# ==============================================================================

import pickle
import numpy as np

p_non = "/home/admin/Ana/MicroBrain/18_igraph.pkl"
p_cut_out = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3.pkl"
out_cut_non = "/home/admin/Ana/MicroBrain/output/nonT_Hcut3.pkl"

# -------------------------
# Load graphs
# -------------------------
non_obj = pickle.load(open(p_non, "rb"), encoding="latin1")
G_non = non_obj["graph"] if isinstance(non_obj, dict) and "graph" in non_obj else non_obj

cut = pickle.load(open(p_cut_out, "rb"), encoding="latin1")
H = cut["graph"]  # cut outgeom graph

# -------------------------
# Sanity check: IDs
# -------------------------
ids_H = set(int(v["id"]) for v in H.vs)
ids_non = set(int(v["id"]) for v in G_non.vs) if "id" in G_non.vs.attributes() else None

print("H: #ids =", len(ids_H))
print("G_non: #ids =", len(ids_non))
print("IDs de H que faltan en G_non:", len(ids_H - ids_non))

# -------------------------
# Edge keys from H (by vertex ids)
# -------------------------
edge_keys = set()
for e in H.es:
    u, v = e.tuple
    idu = int(H.vs[u]["id"])
    idv = int(H.vs[v]["id"])
    edge_keys.add((idu, idv) if idu < idv else (idv, idu))

# -------------------------
# Filter NON edges by connectivity (already ids)
# -------------------------
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

# =============================================================================
# Helpers: error stats
# =============================================================================

def error_stats(delta, name, tol=None):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        print(f"\n{name}: no finite values")
        return
    mean = delta.mean()
    std = delta.std()
    maxabs = np.max(np.abs(delta))
    rmse = np.sqrt(np.mean(delta**2))
    print(f"\n========== {name} ==========")
    print(f"n={delta.size}")
    print(f"mean={mean:.6g}  std={std:.6g}  rmse={rmse:.6g}  maxabs={maxabs:.6g}")
    if tol is not None:
        frac = np.mean(np.abs(delta) > tol)
        print(f"|Δ| > {tol:g}: {frac*100:.3f}%")


# =============================================================================
# 1) TORTUOUS internal length sanity: length == sum(lengths2)
# =============================================================================

# 'cut' es el diccionario cargado del pickle
# 'H' es el grafo que sacamos de cut["graph"]
geom_data = cut.get("geom", {})

# Verificamos si los datos existen en sus respectivos lugares
if "length" in H.es.attributes() and "lengths2" in geom_data:
    dL = []
    all_lengths2 = geom_data["lengths2"]
    
    # IMPORTANTE: 
    # Si lengths2 en geom es una lista de listas (un array por cada edge), 
    # el acceso es directo:
    for i, e in enumerate(H.es):
        L_grafo = float(e["length"])
        
        # Accedemos a la geometría del i-ésimo arco mediante el índice
        # Suponiendo que el orden en 'geom' coincide con el orden de H.es
        segs = all_lengths2[i] 
        
        segs = np.asarray(segs, dtype=float)
        S = float(segs.sum())
        
        if np.isfinite(L_grafo) and np.isfinite(S):
            dL.append(L_grafo - S)
            
    error_stats(dL, "H tortuous: length - sum(lengths2)", tol=1e-6)
else:
    print("\n[H] 'length' no está en el Grafo o 'lengths2' no está en el dict Geom.")
# =============================================================================
# 2) TORTUOUS vs NON on COMMON EDGES (by connectivity key)
# =============================================================================
# Build map from H: (idA,idB)->(diam,length)
map_H = {}
for e in H.es:
    u, v = e.tuple
    a = int(H.vs[u]["id"]); b = int(H.vs[v]["id"])
    k = (a, b) if a < b else (b, a)
    diam = float(e["diameter"]) if "diameter" in H.es.attributes() else np.nan
    L    = float(e["length"])   if "length" in H.es.attributes() else np.nan
    map_H[k] = (diam, L)

d_diam = []
d_len  = []

for e in G_non_cut.es:
    a, b = e["connectivity"]
    a = int(a); b = int(b)
    k = (a, b) if a < b else (b, a)
    if k not in map_H:
        continue

    diam_H, len_H = map_H[k]
    diam_N = float(e["diameter"]) if "diameter" in G_non_cut.es.attributes() else np.nan
    len_N  = float(e["length"])   if "length" in G_non_cut.es.attributes() else np.nan

    if np.isfinite(diam_N) and np.isfinite(diam_H):
        d_diam.append(diam_N - diam_H)

    if np.isfinite(len_N) and np.isfinite(len_H):
        d_len.append(len_N - len_H)

error_stats(d_diam, "Δdiameter = NON - TORT (common edges)", tol=1e-6)
error_stats(d_len,  "Δlength   = NON - TORT (common edges)", tol=1e-6)

# =============================================================================
# Optional: your original mean per nkind (kept)
# =============================================================================
def mean_by_nkind(G, name, attr):
    x = np.asarray(G.es[attr], dtype=float)
    nk = np.asarray(G.es["nkind"], dtype=int)
    ok = np.isfinite(x)
    x = x[ok]; nk = nk[ok]
    print(f"\n{name}: mean({attr}) por nkind")
    for k in sorted(set(nk.tolist())):
        v = x[nk == k]
        if v.size:
            print(f"  nkind {k}: mean_{attr} = {v.mean():.4g}   (n={v.size})")

mean_by_nkind(G_non_cut, "NON cut", "diameter")
mean_by_nkind(H,         "TORT cut", "diameter")
mean_by_nkind(G_non_cut, "NON cut", "length")   if "length" in G_non_cut.es.attributes() else None
mean_by_nkind(H,         "TORT cut", "length")   if "length" in H.es.attributes() else None
