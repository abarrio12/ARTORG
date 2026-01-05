import numpy as np
import igraph as ig
import pickle

# ============================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================

def is_inside_box(p, box):
    return (
        box[0][0] <= p[0] <= box[0][1] and
        box[1][0] <= p[1] <= box[1][1] and
        box[2][0] <= p[2] <= box[2][1]
    )


def intersect_with_plane(p0, p1, axis, value):
    t = (value - p0[axis]) / (p1[axis] - p0[axis])
    return p0 + t * (p1 - p0)


def compute_length(geom):
    if len(geom) < 2:
        return 0.0
    return np.sum(np.linalg.norm(np.diff(geom, axis=0), axis=1))


# ============================================================
# CLASIFICACIÓN DE EDGES
# ============================================================

def classify_edges(G, box):
    inside = []
    across = []
    outside = []

    for eid, e in enumerate(G.es):
        geom = e["geometry"]
        flags = np.array([is_inside_box(p, box) for p in geom])

        if flags.all():
            inside.append(eid)
        elif flags.any():
            across.append(eid)
        else:
            outside.append(eid)

    return (
        np.array(inside, dtype=int),
        np.array(across, dtype=int),
        np.array(outside, dtype=int),
    )


# ============================================================
# CORTE DE UN EDGE QUE CRUZA
# ============================================================

def split_edge_geometry(geom, box):
    inside = np.array([is_inside_box(p, box) for p in geom])
    idx = np.where(inside[:-1] != inside[1:])[0]

    if len(idx) == 0:
        return None

    i = idx[0]
    p0, p1 = geom[i], geom[i+1]

    for axis, (mn, mx) in enumerate(box):
        if p0[axis] < mn or p0[axis] > mx:
            value = mn if p0[axis] < mn else mx
            ip = intersect_with_plane(p0, p1, axis, value)
            break
    else:
        return None

    if inside[i]:
        geom_inside = np.vstack([geom[:i+1], ip])
    else:
        geom_inside = np.vstack([ip, geom[i+1:]])

    return geom_inside, ip


# ============================================================
# FUNCIÓN PRINCIPAL (GAIA → FULLGEOM)
# ============================================================

def cut_graph_fullgeom(G, box):
    """
    Devuelve:
      - subgrafo cortado
      - nodos frontera
    """

    G = G.copy()
    inside, across, outside = classify_edges(G, box)

    border_vertices = []

    for eid in across:
        e = G.es[eid]
        geom = e["geometry"]
        v0, v1 = e.tuple

        result = split_edge_geometry(geom, box)
        if result is None:
            continue

        geom_new, ip = result

        # crear nodo frontera
        new_vid = G.vcount()
        G.add_vertices(1)
        G.vs[new_vid]["coords"] = tuple(ip)
        G.vs[new_vid]["radii"] = G.vs[v0]["radii"]
        G.vs[new_vid]["annotation"] = G.vs[v0]["annotation"]
        G.vs[new_vid]["distance_to_surface"] = G.vs[v0]["distance_to_surface"]

        border_vertices.append(new_vid)

        # crear nuevo edge
        G.add_edge(new_vid, v1)
        ne = G.es[-1]

        ne["geometry"] = geom_new
        ne["radius"] = e["radius"]
        ne["diameter"] = e["diameter"]
        ne["nkind"] = e["nkind"]

        L = compute_length(geom_new)
        ne["length_tortuous"] = L
        ne["tortuosity"] = L / np.linalg.norm(geom_new[-1] - geom_new[0])

    # borrar edges fuera
    edges_to_delete = list(outside)
    G.delete_edges(edges_to_delete)

    # borrar nodos desconectados
    isolated = [v.index for v in G.vs if G.degree(v) == 0]
    G.delete_vertices(isolated)

    return G, np.unique(border_vertices)


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    with open("output/graph_18_FULLGEOM_SUB.pkl", "rb") as f:
        G = pickle.load(f)

    box = [
        (1000, 2000),   # x
        (500, 1500),   # y
        (1000, 2000)      # z
    ]

    G_cut, bc_nodes = cut_graph_fullgeom(G, box)

    print("Subgraph:")
    print(G_cut.summary())
    print("Boundary nodes:", len(bc_nodes))

    with open("graph_18_FULLGEOM_SUB_CUT.pkl", "wb") as f:
        pickle.dump(G_cut, f, protocol=pickle.HIGHEST_PROTOCOL)
