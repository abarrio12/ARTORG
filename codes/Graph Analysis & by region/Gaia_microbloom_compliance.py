

# ======================================================================================================================================================
#                                                               GAIA COMPLIANCE
# ======================================================================================================================================================



def edge_radius_consistency_report_MAX(                                 # NOTA: EXTREMADAMENTE COMPLICADO, LARGO, NO NECESITO TANTA INFO, HACERLO PARA MUCHOS MENOS EDGES
    data,
    sample=10000,             # Reducido un poco para que el loop de búsqueda sea rápido
    seed=0,
    bins=100,
    tol_abs_list=(0.0, 1e-3, 1e-2, 5e-2, 1e-1),
    use_abs_delta=True
):
    """
    Compara el radio del eje (r_edge) contra el MÁXIMO de su geometría:
        r_ref = max(geom['radii'][start : end])
    """

    G = data["graph"]
    rg = np.asarray(data["geom"]["radii"], dtype=np.float32)

    # 1. Obtener radios de los ejes
    if "radius" in G.es.attributes():
        r_edge = np.asarray(G.es["radius"], dtype=np.float32)
    else:
        raise ValueError("No se encontró G.es['radius'].")

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    nE = G.ecount()
    # Muestreo
    if sample is None or sample >= nE:
        idx = np.arange(nE, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(nE, size=int(sample), replace=False)

    # 2. CÁLCULO DE LA REFERENCIA POR MÁXIMO (Aquí está el cambio clave)
    r_ref = np.zeros(len(idx), dtype=np.float32)
    
    print(f"Calculando el máximo de geometría para {len(idx)} ejes...")
    for i, edge_idx in enumerate(idx):
        start = gs[edge_idx]
        end = ge[edge_idx]
        if end > start:
            # Buscamos el máximo en el segmento real de la geometría
            r_ref[i] = np.max(rg[start:end])
        else:
            r_ref[i] = rg[start] # Caso de un solo punto

    r_edge_i = r_edge[idx]

    # 3. Diferencias
    delta = (r_edge_i - r_ref).astype(np.float64)
    delta_abs = np.abs(delta)

    # Estadísticas básicas
    print(f"\n--- REPORTE DE CONSISTENCIA (POLÍTICA: MÁXIMO) ---")
    print(f"Δr stats: mean={np.mean(delta):.6g} | std={np.std(delta):.6g}")
    print(f"|Δr| <= 0.0 (Coincidencia exacta): {100.0 * np.mean(delta_abs <= 1e-7):.2f}%")

    # Porcentajes de error
    print("\n% dentro de tolerancia ABSOLUTA:")
    for t in tol_abs_list:
        pct = 100.0 * np.mean(delta_abs <= t)
        print(f"  |Δr| <= {t:g}: {pct:.2f}%")

    # Histograma
    plt.figure(figsize=(10,6))
    plt.hist(delta, bins=bins, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.xlabel("Error (Radio_Eje - Max_Geometría)")
    plt.ylabel("Número de ejes")
    plt.title("Distribución del error respecto al MÁXIMO")
    plt.show()

    return delta




# ====================================================================================================================
# RADII SANITY CHECKS (for microbloom / consistency)
# ====================================================================================================================

def check_polyline_radii_match_endnodes(                                             # ADAPTAR A ATRIBUTOS QUE TENGO AHORA, NO HACE FALTA TANTA PARAFERNALIA 
    data,                                                                           # SACAR LOS PUNTOS CON LOS INDICES Y COMPARAR, OJO CON LOS ACCESOS A RADII
    node_r_attr_candidates=("radii", "radius", "radius_point"),
    tol=1e-3,
    verbose=True
):
    """
    Check that polyline endpoint radii match radii at end nodes A/B.
    Needs:
      data["graph"] with edge attrs geom_start/geom_end
      data["radii_geom"] radii per polyline point
      graph.vs has a node radii attribute (one of candidates)
    """
    G = data["graph"]
    if "radii_geom" not in data:
        raise ValueError("data['radii_geom'] missing.")
    r_geom = np.asarray(data["radii_geom"], float)

    node_attr = None
    for cand in node_r_attr_candidates:
        if cand in G.vs.attributes():
            node_attr = cand
            break
    if node_attr is None:
        raise ValueError(f"No node radius attr found in graph.vs. Tried: {node_r_attr_candidates}")

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end' to check polyline endpoints.")

    r_node = np.asarray(G.vs[node_attr], float)

    bad = 0
    max_diff = 0.0
    examples = []

    for e in range(G.ecount()):
        s = int(G.es[e]["geom_start"])
        t = int(G.es[e]["geom_end"])
        if (t - s) < 2:
            continue

        a, b = G.es[e].tuple
        rA = float(r_node[a])
        rB = float(r_node[b])

        r0 = float(r_geom[s])
        r1 = float(r_geom[t - 1])

        d0 = abs(r0 - rA)
        d1 = abs(r1 - rB)
        md = max(d0, d1)

        if md > tol:
            bad += 1
            if len(examples) < 10:
                examples.append((e, md, rA, r0, rB, r1))
        max_diff = max(max_diff, md)

    if verbose:
        print("\n=== Radii check: polyline endpoints vs node radii ===")
        print(f"Node radius attr used: '{node_attr}' | tol={tol}")
        print(f"Mismatching edges: {bad} / {G.ecount()} | max diff: {max_diff:.6g}")
        if examples:
            print("Examples (edge, maxdiff, rA, r_poly0, rB, r_polyEnd):")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": bad, "max_diff": max_diff, "examples": examples, "node_attr": node_attr}

# ============================================================================================================================================0



#  LOS TRES SIGUIENTES CODIGOS PODRÍA ELIMINARLOS, NO TIENEN MUCHO SENTIDO NI TP ME DAN APENAS INFO DE ANALISIS DE BOX 

def check_radii_endpoints_allow_swap(data, tol=1e-3, max_examples=15, use_coords_check=False, coords_tol=1e-6):   # QUITARLO, METER COMPROBACIÓN DE ORDEN COMO CHECK EN CODIGO ANTERIOR 
    """
    Compara radii de vértices vs radii de geom en endpoints de cada edge.
    Permite geometría invertida (swap start/end) y reporta cuántos swaps hay.

    data format:
      data["graph"]  -> igraph Graph (edges have geom_start/geom_end)
      data["vertex"]["radii"] -> (nV,)
      data["geom"]["radii"]   -> (nP,)
      data["geom"]["x/y/z"]   -> (nP,)

    Si use_coords_check=True, además intenta decidir orientación comparando P[0] con coords_image[u].
    """
    G = data["graph"]

    # --- vertex radii ---
    if "vertex" not in data or "radii" not in data["vertex"]:
        raise KeyError("Falta data['vertex']['radii']")
    vr = np.asarray(data["vertex"]["radii"], dtype=np.float32)

    if len(vr) != G.vcount():
        raise ValueError(f"vertex['radii'] len={len(vr)} != G.vcount()={G.vcount()}")

    # --- geom radii ---
    if "geom" not in data or "radii" not in data["geom"]:
        raise KeyError("Falta data['geom']['radii']")
    gr = np.asarray(data["geom"]["radii"], dtype=np.float32)

    # optional geom coords for sanity / orientation check
    x = np.asarray(data["geom"]["x"], dtype=np.float64) if "x" in data["geom"] else None
    y = np.asarray(data["geom"]["y"], dtype=np.float64) if "y" in data["geom"] else None
    z = np.asarray(data["geom"]["z"], dtype=np.float64) if "z" in data["geom"] else None
    if x is not None and len(x) != len(gr):
        raise ValueError(f"geom['x'] len={len(x)} != geom['radii'] len={len(gr)}")

    coords_v = None
    if use_coords_check:
        if "coords_image" not in data["vertex"]:
            raise KeyError("use_coords_check=True pero falta data['vertex']['coords_image']")
        coords_v = np.asarray(data["vertex"]["coords_image"], dtype=np.float64)
        if coords_v.shape[0] != G.vcount() or coords_v.shape[1] != 3:
            raise ValueError("vertex['coords_image'] debe ser (nV,3)")

        if x is None or y is None or z is None:
            raise KeyError("use_coords_check=True requiere geom['x','y','z']")

    bad = 0
    swapped = 0
    max_diff = 0.0
    examples = []

    for ei in range(G.ecount()):
        s = int(G.es[ei]["geom_start"])
        en = int(G.es[ei]["geom_end"])

        if en - s < 2:
            continue
        if s < 0 or en > len(gr) or en <= s:
            raise ValueError(f"Edge {ei}: geom_start/end fuera de rango: start={s}, end={en}, nP={len(gr)}")

        u = int(G.es[ei].source)
        v = int(G.es[ei].target)

        # --- direct match (u->start, v->end) ---
        du = float(abs(vr[u] - gr[s]))
        dv = float(abs(vr[v] - gr[en - 1]))
        d_direct = max(du, dv)

        # --- swapped match (u->end, v->start) ---
        du2 = float(abs(vr[u] - gr[en - 1]))
        dv2 = float(abs(vr[v] - gr[s]))
        d_swap = max(du2, dv2)

        # decide best
        d_best = d_direct
        mode = "DIRECT"
        if d_swap < d_direct:
            d_best = d_swap
            mode = "SWAP"
            swapped += 1

        # optional: compare to coordinates to see if geometry order matches u
        coord_mode = None
        if use_coords_check:
            P0 = np.array([x[s], y[s], z[s]], dtype=np.float64)
            Pend = np.array([x[en-1], y[en-1], z[en-1]], dtype=np.float64)
            cu = coords_v[u]
            # if P0 ~ cu => direct, if Pend ~ cu => swapped
            if np.allclose(P0, cu, atol=coords_tol):
                coord_mode = "DIRECT"
            elif np.allclose(Pend, cu, atol=coords_tol):
                coord_mode = "SWAP"
            else:
                coord_mode = "UNKNOWN"

        max_diff = max(max_diff, d_best)

        if d_best > tol:
            bad += 1
            if len(examples) < max_examples:
                row = (
                    ei, d_best,
                    u, float(vr[u]), float(gr[s]),
                    v, float(vr[v]), float(gr[en - 1]),
                    mode
                )
                if use_coords_check:
                    row = row + (coord_mode,)
                examples.append(row)

    print("=== Radii endpoint check (vertex vs geom endpoints, allow swap) ===")
    msg = f"tol={tol} | bad_edges={bad}/{G.ecount()} | swapped_edges={swapped}/{G.ecount()} | max_diff={max_diff:.6g}"
    if use_coords_check:
        msg += f" | coords_tol={coords_tol}"
    print(msg)

    if examples:
        if use_coords_check:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end], best_mode, coord_mode)")
        else:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end], best_mode)")
        for ex in examples:
            print(" ", ex)

    return {"bad": bad, "swapped": swapped, "max_diff": max_diff, "examples": examples}




def check_polyline_radii_variation(data, tol_rel=0.05, verbose=True):
    """
    For each edge polyline, checks relative variation of radii along points:
      rel = (max-min)/mean
    Flags edges above tol_rel (e.g. 0.05 = 5%)
    """
    G = data["graph"]
    data["radii_geom"] = data["geom"]["radii"]

    if "radii_geom" not in data:
        raise ValueError("data['radii_geom'] missing.")
    r = np.asarray(data["radii_geom"], float)

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end' to check polyline radii variation.")

    bad = 0
    worst = 0.0
    examples = []

    for e in range(G.ecount()):
        s = int(G.es[e]["geom_start"])
        t = int(G.es[e]["geom_end"])
        if (t - s) < 2:
            continue

        rr = r[s:t]
        if len(rr) < 2:
            continue

        m = float(np.nanmean(rr))
        if not np.isfinite(m) or m == 0:
            continue

        rel = float((np.nanmax(rr) - np.nanmin(rr)) / m)
        if rel > tol_rel:
            bad += 1
            if len(examples) < 10:
                examples.append((e, rel, float(np.nanmin(rr)), float(np.nanmax(rr)), m))
        worst = max(worst, rel)

    if verbose:
        print("\n=== Radii variation along polyline ===")
        print(f"Bad edges (rel var > {tol_rel}): {bad} / {G.ecount()} | worst={worst:.3f}")
        if examples:
            print("Examples (edge, rel_var, rmin, rmax, rmean):")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": bad, "worst_rel": worst, "examples": examples}



def debug_edge(data, ei, k=3):
    G = data["graph"]
    cv = np.asarray(data["vertex"]["coords_image"], float)
    vr = np.asarray(data["vertex"]["radii"], float)
    x = np.asarray(data["geom"]["x"], float)
    y = np.asarray(data["geom"]["y"], float)
    z = np.asarray(data["geom"]["z"], float)
    gr = np.asarray(data["geom"]["radii"], float)

    e = G.es[ei]
    u, v = int(e.source), int(e.target)
    s, en = int(e["geom_start"]), int(e["geom_end"])
    P0 = np.array([x[s], y[s], z[s]])
    P1 = np.array([x[en-1], y[en-1], z[en-1]])

    du0 = np.linalg.norm(P0 - cv[u]); dv0 = np.linalg.norm(P0 - cv[v])
    du1 = np.linalg.norm(P1 - cv[u]); dv1 = np.linalg.norm(P1 - cv[v])

    print(f"EDGE {ei} | u={u}, v={v} | npts={en-s}")
    print("coords distances:")
    print(f"  |Pstart-u|={du0:.6g}  |Pstart-v|={dv0:.6g}")
    print(f"  |Pend-u|  ={du1:.6g}  |Pend-v|  ={dv1:.6g}")

    print("endpoint radii:")
    print(f"  vr[u]={vr[u]:.6g}  gr[start]={gr[s]:.6g}  gr[end]={gr[en-1]:.6g}")
    print(f"  vr[v]={vr[v]:.6g}  gr[start]={gr[s]:.6g}  gr[end]={gr[en-1]:.6g}")

    # mira algunos puntos cercanos al start/end por si el nodo no coincide exactamente con s/en-1
    print("\nnear start radii:")
    for j in range(s, min(en, s+k)):
        print(f"  j={j-s:2d}  gr={gr[j]:.6g}  P=({x[j]:.3f},{y[j]:.3f},{z[j]:.3f})")
    print("near end radii:")
    for j in range(max(s, en-k), en):
        print(f"  j={j-(en-k):2d}  gr={gr[j]:.6g}  P=({x[j]:.3f},{y[j]:.3f},{z[j]:.3f})")

     

def classify_edge_endpoint_coords(data, tol=1e-6):
    G = data["graph"]
    cv = np.asarray(data["vertex"]["coords_image"], float)

    x = np.asarray(data["geom"]["x"], float)
    y = np.asarray(data["geom"]["y"], float)
    z = np.asarray(data["geom"]["z"], float)

    counts = {"OK_DIRECT":0, "OK_SWAP":0, "BAD_BOTH":0, "BAD_AMBIG":0}
    examples = {"OK_SWAP":[], "BAD_BOTH":[]}

    for ei in range(G.ecount()):
        e = G.es[ei]
        u, v = int(e.source), int(e.target)
        s, en = int(e["geom_start"]), int(e["geom_end"])
        if en - s < 2:
            continue

        P0 = np.array([x[s], y[s], z[s]])
        P1 = np.array([x[en-1], y[en-1], z[en-1]])

        mu0 = np.allclose(P0, cv[u], atol=tol)
        mv0 = np.allclose(P0, cv[v], atol=tol)
        mu1 = np.allclose(P1, cv[u], atol=tol)
        mv1 = np.allclose(P1, cv[v], atol=tol)

        direct = mu0 and mv1
        swap   = mv0 and mu1

        if direct and not swap:
            counts["OK_DIRECT"] += 1
        elif swap and not direct:
            counts["OK_SWAP"] += 1
            if len(examples["OK_SWAP"]) < 10:
                examples["OK_SWAP"].append((ei,u,v,s,en))
        elif direct and swap:
            counts["BAD_AMBIG"] += 1
        else:
            counts["BAD_BOTH"] += 1
            if len(examples["BAD_BOTH"]) < 10:
                # guarda distancias para debug
                du0 = np.linalg.norm(P0-cv[u]); dv0=np.linalg.norm(P0-cv[v])
                du1 = np.linalg.norm(P1-cv[u]); dv1=np.linalg.norm(P1-cv[v])
                examples["BAD_BOTH"].append((ei,u,v,du0,dv0,du1,dv1,s,en))

    print(counts)
    print("Examples OK_SWAP (ei,u,v,s,en):", examples["OK_SWAP"])
    print("Examples BAD_BOTH (ei,u,v,du0,dv0,du1,dv1,s,en):", examples["BAD_BOTH"])
    return counts, examples



# ESTOS CHECKS QUE INFO ME DAN?????? OSEA ME INTERESA ALGO MÁS ALLÁ DE HDN ? QUE YA LO HE COMPROBADO
# ====================================================================================================================
# DEGREE CHECKS: distribution + by type + spatial mapping for any degree band
# ====================================================================================================================

def degree_summary(graph, max_degree_to_print=None):
    deg = np.asarray(graph.degree(), dtype=int)
    c = Counter(deg.tolist())
    print("\n=== Degree distribution (all nodes) ===")
    for d in sorted(c.keys()):
        if max_degree_to_print is not None and d > max_degree_to_print:
            continue
        print(f"  degree {d}: {c[d]}")
    print(f"  max degree: {int(deg.max()) if deg.size else 0}")
    return c


def degree_summary_by_type(graph):
    deg = np.asarray(graph.degree(), dtype=int)
    labels = np.array([infer_node_type_from_incident_edges(graph, v.index) for v in graph.vs], dtype=object)

    out = {}
    print("\n=== Degree distribution by vessel-type (node label) ===")
    for lab in ["arteriole", "venule", "capillary", "unknown"]:
        m = labels == lab
        c = Counter(deg[m].tolist())
        out[lab] = c
        total = int(np.sum(m))
        print(f"\n  [{lab}] n={total}")
        for d in sorted(c.keys()):
            print(f"    degree {d}: {c[d]}")
    return out





# ====================================================================================================================
#                                  RADII CHECK: vertex endpoints vs geom endpoints
# ====================================================================================================================


                                                # TIENE SENTIDO OTRO CODIGO A PARTE? SOLICITADO POR GAIA, PERO PODRÍA METERLO DONDE COMPRUEBO EL MAX Y YA NO? 

def check_endpoint_radii_vertex_vs_geom(data, tol=1e-3, verbose=True, max_print=10):
    """
    NEW-PKL ONLY.
    Checks: geom radii at first/last polyline point == vertex radii at nodes A/B.

    Needs:
      data["vertex"]["radii"] (nV,)
      data["geom"]["radii"]   (nP,)
      edges have geom_start/geom_end
    """
    if not (isinstance(data, dict) and "graph" in data and "vertex" in data and "geom" in data):
        raise ValueError("Expected NEW-PKL dict with keys: 'graph', 'vertex', 'geom'.")

    G = data["graph"]

    if "radii" not in data["vertex"]:
        raise ValueError("Missing data['vertex']['radii'] for endpoint radii check.")
    if "radii" not in data["geom"]:
        raise ValueError("Missing data['geom']['radii'] for endpoint radii check.")

    vr = np.asarray(data["vertex"]["radii"], float)
    gr = np.asarray(data["geom"]["radii"], float)

    if "geom_start" not in G.es.attributes() or "geom_end" not in G.es.attributes():
        raise ValueError("Edges must have 'geom_start' and 'geom_end'.")

    bad = 0
    maxdiff = 0.0
    examples = []

    for ei in range(G.ecount()):
        e = G.es[ei]
        s = int(e["geom_start"]); en = int(e["geom_end"])
        if en - s < 2:
            continue

        u, v = e.tuple

        d0 = abs(float(gr[s])    - float(vr[u]))
        d1 = abs(float(gr[en-1]) - float(vr[v]))
        md = max(d0, d1)

        if md > tol:
            bad += 1
            if len(examples) < max_print:
                examples.append((ei, md, u, float(vr[u]), float(gr[s]), v, float(vr[v]), float(gr[en-1])))
        maxdiff = max(maxdiff, md)

    if verbose:
        print("\n=== Radii endpoint check (vertex vs geom endpoints) ===")
        print(f"tol={tol} | bad_edges={bad}/{G.ecount()} | max_diff={maxdiff:.6g}")
        if examples:
            print("Examples: (edge, maxdiff, u, vr[u], gr[start], v, vr[v], gr[end])")
            for ex in examples:
                print(" ", ex)

    return {"bad_edges": int(bad), "max_diff": float(maxdiff), "examples": examples}




























