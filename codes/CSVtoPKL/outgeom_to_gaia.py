import numpy as np
import igraph as ig

def outgeom_to_igraph_materialized(data, space="um"):
    G = data["graph"]
    G2 = G.copy()

    Vvox = data.get("vertex", {})
    Vum  = data.get("vertex_R", {})
    nV = G.vcount()
    copy_vertex_attrs = ( # vertex attributes to keep 
        "distance_to_surface_R",
        "vertex_annotation", "id", "coords",
        "coords_image", "coords_image_R",
        "radii_atlas", "radii_atlas_R", "radii",
    )
    copy_edge_attrs = ( # edge attributes to keep 
        "nkind",
        "length", "length_R",
        "radius", "radius_atlas",
        "diameter", "diameter_atlas",
        "diameter_atlas", "diameter_atlas_R",
        "tortuosity", "tortuosity_R",
        "geom_start", "geom_end",
    )

    # copy vertex attrs: graph.vs -> vertex_R -> vertex 
    # some data is only in vertex_R, some only in vertex, and some in both. 
    # Idea: keep everything
    for a in copy_vertex_attrs:
        if a in G.vs.attributes() and len(G.vs[a]) == nV:
            G2.vs[a] = list(G.vs[a])
        elif a in Vum and len(Vum[a]) == nV:
            G2.vs[a] = np.asarray(Vum[a]).tolist()
        elif a in Vvox and len(Vvox[a]) == nV:
            G2.vs[a] = np.asarray(Vvox[a]).tolist()

    # geometry (points/lengths2/diameters per point)
    if space == "um":
        g = data["geom_R"]
        x = np.asarray(g["x_R"], float); y = np.asarray(g["y_R"], float); z = np.asarray(g["z_R"], float)
        L2 = np.asarray(g["lengths2_R"], float)
        r  = np.asarray(g["radii_atlas_geom_R"], float) if "radii_atlas_geom_R" in g else None
        coords_attr = "coords_image_R"
    else:
        g = data["geom"]
        x = np.asarray(g["x"], float); y = np.asarray(g["y"], float); z = np.asarray(g["z"], float)
        L2 = np.asarray(g["lengths2"], float)
        r  = np.asarray(g["radii_atlas_geom"], float) if "radii_atlas_geom" in g else None
        coords_attr = "coords_image"

    gs = np.asarray(G.es["geom_start"], np.int64)
    ge = np.asarray(G.es["geom_end"], np.int64)

    #  copy requested edge attrs (if present) <<<<< Gaia has preasurre and pBC values
    for a in copy_edge_attrs:
        if a in G.es.attributes():
            G2.es[a] = list(G.es[a])

    # Gaia like 
    G2.es["connectivity"] = [tuple(map(int, e.tuple)) for e in G2.es]

    # per-edge lists 
    points, lengths2, diam_atlas = [], [], []

    for eid in range(G2.ecount()):
        s = int(gs[eid]); t = int(ge[eid])

        pts = np.stack([x[s:t], y[s:t], z[s:t]], axis=1)
        points.append([tuple(map(float, row)) for row in pts])

        lengths2.append([float(v) for v in L2[s:t-1]])  # n_points-1

        if r is not None:
            diam_atlas.append([float(v) for v in (2.0 * r[s:t])])  # n_points
        else:
            diam_atlas.append([np.nan] * (t - s))

    G2.es["points"] = points
    G2.es["lengths2"] = lengths2
    G2.es["diameter_p_atlas"] = diam_atlas

    return G2
