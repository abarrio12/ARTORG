'''
This code reformats the attributes into MVN structure and data is stored at edge/vertex level.

Be aware of the space you are working with. Depending on the requested output space, different
geometry attributes are used (VOX: geom; UM: geom_R).

Diameter source:
- We use radii from image space (data["geom"]["radii"] / data["vertex"]["radii"]).
  NOT atlas-derived 

Note:
- From Paris we assume radii are defined in the image XY plane. When sx == sy (like here), using sx is
  consistent. If sx != sy, we use sqrt(sx*sy). This is an approximation !!
  Usually diameter_unit is only been use in vox 

Author: Ana
Updated: 27 Feb 2026 (revised for separate diameter_unit)
'''

import numpy as np

def outgeom_to_igraph_gaia_minimal(data, space="auto", diameter_unit="vox", verbose=True):
    G = data["graph"]
    G2 = G.copy()

    nV = G.vcount()
    nE = G.ecount()

    # -------------------------
    # SPACE
    # -------------------------
    if space == "auto":
        space = "um" if ("geom_R" in data and "vertex_R" in data) else "vox"

    if space == "um":
        geom_pts = data["geom_R"]
        Vsrc = data["vertex_R"]
        coords_attr = "coords_image_R"
        xk, yk, zk, l2k = "x_R", "y_R", "z_R", "lengths2_R"
        unit_str = "um"
    else:
        geom_pts = data["geom"]
        Vsrc = data["vertex"]
        coords_attr = "coords_image"
        xk, yk, zk, l2k = "x", "y", "z", "lengths2"
        unit_str = "vox"

    # -------------------------
    # RADII → DIAMETERS 
    # -------------------------
    r_vox = np.asarray(data["geom"]["radii"], dtype=float)

    if diameter_unit == "um":
        sx, sy, _ = data.get("spacing_um_per_voxel", (1.625, 1.625, 2.5))
        scale = sx if abs(sx - sy) < 1e-9 else np.sqrt(sx * sy)
        d_p = 2.0 * r_vox * scale # this is an approximation, so usually we are using vox
    else:
        d_p = 2.0 * r_vox 

    # -------------------------
    # VERTICES
    # -------------------------
    coords = np.asarray(Vsrc[coords_attr], dtype=float)

    G2.vs["coords"] = [tuple(c) for c in coords]
    G2.vs["index"] = list(range(nV))

    # annotation
    if "vertex_annotation" in data["vertex"]:
        G2.vs["annotation"] = data["vertex"]["vertex_annotation"]
    else:
        G2.vs["annotation"] = [None] * nV

    # vertex diameter
    if "radii" in data["vertex"]:
        vr = np.asarray(data["vertex"]["radii"], dtype=float)
        if diameter_unit == "um":
            G2.vs["diameter"] = (2.0 * vr * scale).tolist()
        else:
            G2.vs["diameter"] = (2.0 * vr).tolist()
    else:
        G2.vs["diameter"] = [float("nan")] * nV

    # -------------------------
    # EDGES
    # -------------------------
    G2.es["connectivity"] = [tuple(e.tuple) for e in G2.es]
    G2.es["nkind"] = list(G.es["nkind"]) if "nkind" in G.es.attributes() else [None]*nE

    gs = np.asarray(G.es["geom_start"])
    ge = np.asarray(G.es["geom_end"])

    x = np.asarray(geom_pts[xk])
    y = np.asarray(geom_pts[yk])
    z = np.asarray(geom_pts[zk])
    L2 = np.asarray(geom_pts[l2k])

    points, lengths2_list, diameters_list = [], [], []

    for i in range(nE):
        s, t = gs[i], ge[i]

        if t <= s:
            points.append([])
            lengths2_list.append([])
            diameters_list.append([])
            continue

        pts = np.stack([x[s:t], y[s:t], z[s:t]], axis=1)

        points.append([tuple(p) for p in pts])
        lengths2_list.append(L2[s:t-1].tolist() if (t - s) >= 2 else [])
        diameters_list.append(d_p[s:t].tolist())

    G2.es["points"] = points
    G2.es["lengths2"] = lengths2_list
    G2.es["diameters"] = diameters_list

    # length
    G2.es["length"] = [
        float(np.sum(l2)) if len(l2) else float("nan")
        for l2 in lengths2_list
    ]

    # diameter (edge)
    G2.es["diameter"] = [
        float(np.mean(d)) if len(d) else float("nan")
        for d in diameters_list
    ]

    # metadata
    G2["unit"] = unit_str
    G2["diameter_unit"] = diameter_unit

    # -------------------------
    # KEEP ONLY MVN ATTR
    # -------------------------
    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {"connectivity", "nkind", "diameter", "diameters",
              "length", "lengths2", "points"}

    for a in list(G2.vs.attributes()):
        if a not in keep_v:
            del G2.vs[a]

    for a in list(G2.es.attributes()):
        if a not in keep_e:
            del G2.es[a]

    # -------------------------
    # DEBUG
    # -------------------------
    if verbose:
        print("V attrs:", G2.vs.attributes())
        print("E attrs:", G2.es.attributes())

    return G2