def summarize_cube(G, cube_id, box=None, coords_attr=None, eps=2):
    summary = {"cube_id": cube_id}

    # --- basic size ---
    summary["nV"] = G.vcount()
    summary["nE"] = G.ecount()

    # --- components ---
    is_single, n_comp, comps = G.is_connected(), len(G.components()), G.components()
    summary["n_components"] = n_comp
    if n_comp > 0:
        giant = max((len(c) for c in comps), default=0)
        summary["giant_component_frac"] = giant / summary["nV"] if summary["nV"] else 0.0
    else:
        summary["giant_component_frac"] = 0.0

    # --- duplicated edges / loops ---
    summary["duplicated_edges"] = find_duplicated_edges(G)
    _, loop_count, loop_perc = find_loops(G)
    summary["self_loop_count"] = loop_count
    summary["self_loop_perc"]  = loop_perc

    # --- nkind edge type counts ---
    nk = np.asarray(G.es["nkind"], dtype=int) if "nkind" in G.es.attributes() else None
    if nk is not None:
        unique, counts = np.unique(nk, return_counts=True)
        for k, c in zip(unique, counts):
            summary[f"edge_count_nkind_{k}"] = int(c)
            summary[f"edge_frac_nkind_{k}"]  = float(c) / summary["nE"] if summary["nE"] else 0.0

    # --- diameters ---
    if "diameter" in G.es.attributes() and nk is not None:
        diam = np.asarray(G.es["diameter"], dtype=float)
        for k in np.unique(nk):
            sub = diam[nk == k]
            summary[f"diam_mean_nkind_{k}"] = float(np.mean(sub)) if len(sub) else np.nan
            summary[f"diam_p50_nkind_{k}"]  = float(np.median(sub)) if len(sub) else np.nan
            summary[f"diam_p95_nkind_{k}"]  = float(np.percentile(sub, 95)) if len(sub) else np.nan

    # --- lengths ---
    if "length" in G.es.attributes() and nk is not None:
        L = np.asarray(G.es["length"], dtype=float)
        summary["length_mean"] = float(np.mean(L)) if len(L) else np.nan
        summary["length_sum"]  = float(np.sum(L)) if len(L) else 0.0
        for k in np.unique(nk):
            sub = L[nk == k]
            summary[f"length_mean_nkind_{k}"] = float(np.mean(sub)) if len(sub) else np.nan
            summary[f"length_sum_nkind_{k}"]  = float(np.sum(sub)) if len(sub) else 0.0

    # --- BC analysis (optional) ---
    if box is not None and coords_attr is not None:
        res_bc = analyze_bc_for_box(G, box, coords_attr, eps=eps)
        bc_unique = res_bc["TOTAL_unique"]["count"]
        summary["bc_unique_count"] = int(bc_unique)
        summary["bc_unique_frac"]  = float(bc_unique) / summary["nV"] if summary["nV"] else 0.0

        # face counts
        for face in ["x_min","x_max","y_min","y_max","z_min","z_max"]:
            summary[f"bc_count_{face}"] = int(res_bc[face]["count"])

        # composition of unique BC nodes
        tp = res_bc["TOTAL_unique"]["type_percent"]
        summary["bc_pct_arteriole"] = float(tp.get("arteriole", 0.0))
        summary["bc_pct_venule"]    = float(tp.get("venule", 0.0))
        summary["bc_pct_capillary"] = float(tp.get("capillary", 0.0))

    return summary
