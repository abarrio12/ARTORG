import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig

# ---------------------------------------------------------------------
# formatted analysis toolbox
# ---------------------------------------------------------------------
# OPTION A (recommended): keep a separate module for formatted graphs
sys.path.insert(0, "/home/ana/MicroBrain/codes/Graph Analysis & by region/Graph analysis")
from graph_analysis_functions import *
# OPTION B: if you overwrote the old file name, then use:
# from graph_analysis_functions import *

# ---------------------------------------------------------------------
# paths / params
# ---------------------------------------------------------------------
out_root = "/home/admin/Ana/MicroBrain/output/HPC_FULL_ANALYSIS_FORMATTED"
os.makedirs(out_root, exist_ok=True)

# IMPORTANT: these must now be FORMATTED graph pickles (igraph.Graph),
# not OutGeom dict pickles.
PATHS = {
    "HPC_1": "/home/admin/Ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted_Hcut1_gaia_like.pkl",
    "HPC_2": "/home/admin/Ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted_Hcut2_gaia_like.pkl",
    "HPC_3": "/home/admin/Ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted_Hcut3_gaia_like.pkl",
}

HIPPO_CENTERS = [
    [1200, 3400, 1500],
    [1400, 3900,  950],
    [2100, 4100,  750],
]

box_size_um = (400, 400, 400)

degree_thr = 4
eps_vox    = 2.0

# Density settings
slab_um_cutbox = 10.0
slab_axis      = "z"

# plotting controls
plot_paths_cap = 15
bins_hist = 40

BOX_ORDER = list(PATHS.keys())

V_COL = {
    "arteriole": "#d62728",
    "venule":    "#1f77b4",
    "capillary": "#7f7f7f",
    "unknown":   "#c7c7c7",
}
BOX_COL = {
    "HPC_1": "tab:blue",
    "HPC_2": "tab:orange",
    "HPC_3": "tab:green",
}

FACE_GRID = [
    ["y_max", "z_max", "y_min"],
    ["x_min", "z_min", "x_max"],
]

# ---------------------------------------------------------------------
# helpers (same as yours)
# ---------------------------------------------------------------------
def safe_arr(x, dtype=float):
    return np.asarray(x, dtype=dtype)

def safe_finite(x):
    x = safe_arr(x, float)
    return x[np.isfinite(x)]

def ecdf_xy(x):
    x = safe_finite(x)
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def plot_boxplot_by_graph(df_long, value_col, title, ylabel, graphs_order=("HPC_1","HPC_2","HPC_3")):
    groups, labels = [], []
    for g in graphs_order:
        if g not in df_long["graph"].unique():
            continue
        x = df_long.loc[df_long["graph"] == g, value_col].to_numpy(float)
        x = x[np.isfinite(x)]
        groups.append(x)
        labels.append(g)

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()

def plot_hist_overlay(df_long, value_col, title, xlabel, bins=40, graphs_order=("HPC_1","HPC_2","HPC_3")):
    plt.figure(figsize=(8, 5))
    for g in graphs_order:
        if g not in df_long["graph"].unique():
            continue
        x = df_long.loc[df_long["graph"] == g, value_col].to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size:
            plt.hist(x, bins=bins, density=True, alpha=0.45, label=g)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

def diameter_length_overlay_by_type(dl, bins=40, graphs_order=("HPC_1","HPC_2","HPC_3"), box_label=""):
    for t in sorted(dl["type"].unique()):
        sub = dl[dl["type"] == t].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{t} | Diameter & Length distributions | {box_label}", fontsize=12, fontweight="bold")

        ax = axes[0]
        for g in graphs_order:
            x = sub.loc[sub["graph"] == g, "diameter_um"].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size:
                ax.hist(x, bins=bins, density=True, alpha=0.45, label=g)
        ax.set_title("Diameter")
        ax.set_xlabel("µm")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

        ax = axes[1]
        for g in graphs_order:
            x = sub.loc[sub["graph"] == g, "length_um"].to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size:
                ax.hist(x, bins=bins, density=True, alpha=0.45, label=g)
        ax.set_title("Length")
        ax.set_xlabel("µm")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

        plt.tight_layout()
        plt.show()

def load_formatted_graph(path: str) -> ig.Graph:
    """
    Robust loader:
    - if file contains an igraph.Graph -> returns it
    - if file contains dict with key 'graph' -> returns dict['graph']
      (helps when you accidentally point to an OutGeom pkl)
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, ig.Graph):
        return obj
    if isinstance(obj, dict) and "graph" in obj and isinstance(obj["graph"], ig.Graph):
        return obj["graph"]
    raise TypeError(f"File does not contain an igraph.Graph: {path} (type={type(obj)})")

def keep_giant_component(G: ig.Graph) -> ig.Graph:
    comps = G.components()
    if len(comps) <= 1:
        return G
    keep = np.asarray(comps[np.argmax(comps.sizes())], dtype=int)
    return G.induced_subgraph(keep)

# ---------------------------------------------------------------------
# COLLECTORS
# ---------------------------------------------------------------------
all_summaries   = []
diam_long_parts = []
path_len_rows   = []
bc_rows         = []
hdn_nodes_rows  = []
red_rows        = []

density_box_values = {}
density_box_slabs  = {}
density_long_rows  = []

# ---------------------------------------------------------------------
# analysis loop (CUT BOXES) — FORMATTED GRAPHS
# ---------------------------------------------------------------------
for (name, path), center in zip(PATHS.items(), HIPPO_CENTERS):
    print("\n======================")
    print("Analyzing:", name)
    print("  ", path)

    G = load_formatted_graph(path)
    G = keep_giant_component(G)

    # CutBox in um (same convention you used for cutting)
    cut_box = make_box_in_um(center, box_size_um, res_um_per_vox=res_um_per_vox)
    validate_box_faces(cut_box)

    # --- Basic stats
    dup = duplicated_edge_stats(G)
    loops = loop_edge_stats(G)

    # --- Lengths / diameters by type (formatted: length & diameter are direct edge attrs)
    avg_len_dict = get_avg_length_nkind(G)  # {2: mean, 3: mean, 4: mean}
    avg_len_by_type = {EDGE_NKIND_TO_LABEL.get(int(k), str(k)): float(v) for k, v in avg_len_dict.items()}

    diam_stats = diameter_stats_nkind(G, label_dict=EDGE_NKIND_TO_LABEL, plot=True, title_suffix=name)

    # --- per-edge diameter/length (for global overlays)
    diam = safe_arr(G.es["diameter"], float)
    leng = safe_arr(G.es["length"], float)
    nk   = safe_arr(G.es["nkind"], int)

    for k in np.unique(nk):
        lab = EDGE_NKIND_TO_LABEL.get(int(k), str(k))
        m = (nk == k)
        if np.any(m):
            diam_long_parts.append(pd.DataFrame({
                "graph": name,
                "type": str(lab),
                "diameter_um": diam[m],
                "length_um": leng[m],
            }))

    # per-box length distribution by type (kept)
    plot_hist_by_category_general(
        values=safe_arr(G.es["length"], float),
        category=safe_arr(G.es["nkind"], int),
        label_dict=EDGE_NKIND_TO_LABEL,
        bins=bins_hist,
        layout="horizontal",
        density=True,
        show_mean=True,
        variable_name="Edge length (µm)",
        category_name="Vessel type",
        main_title=f"Length distribution by vessel type | {name}"
    )

    # --- HDN analysis (formatted coords = 'coords')
    get_degrees(G, threshold=degree_thr)

    hdn = analyze_hdn_pattern_in_box(
        G,
        box=cut_box,
        coords_attr="coords",
        space="um",
        degree_thr=degree_thr,
        eps_vox=eps_vox
    )

    deg = safe_arr(G.degree(), int)
    hdn_nodes = np.where(deg >= degree_thr)[0]
    coords = np.asarray(G.vs["coords"], float)

    for v in hdn_nodes:
        t = infer_node_type_from_incident_edges(G, int(v))
        x, y, z = coords[int(v)]
        hdn_nodes_rows.append({
            "graph": name,
            "v": int(v),
            "x": float(x), "y": float(y), "z": float(z),
            "degree": int(deg[int(v)]),
            "type": t
        })

    plot_degree_nodes_spatial(
        G,
        coords_attr="coords",
        degree_min=degree_thr,
        degree_max=None,
        by_type=True,
        title=f"High-degree nodes (deg ≥ {degree_thr}) | {name}"
    )

    # --- BC faces (formatted TRUE BC) -> use border mode
    bc = analyze_bc_faces(
        G,
        cut_box,
        coords_attr="coords",
        space="um",
        eps_vox=eps_vox,
        degree_thr=degree_thr,
        mode="border"  # <- IMPORTANT for cut graphs
    )

    debug_face_plane_counts(G, cut_box, coords_attr="coords", eps_vox=eps_vox, space="um")

    bc_df = bc_faces_table(bc, box_name=name).copy()
    bc_df["graph"] = name

    # Convert % -> counts per type (for global cube-net stacked bars)
    for vname, pcol in {
        "arteriole": "% Arteriole",
        "venule": "% Venule",
        "capillary": "% Capillary",
        "unknown": "% Unknown",
    }.items():
        bc_df[f"n_{vname}"] = (
            pd.to_numeric(bc_df["BC nodes"], errors="coerce").fillna(0.0) *
            pd.to_numeric(bc_df[pcol], errors="coerce").fillna(0.0) / 100.0
        )

    bc_rows.append(bc_df)

    plot_bc_cube_net(bc, title=f"BC composition per face (cube net) | {name}")
    plot_bc_3_cubes_tinted(G, cut_box, coords_attr="coords", space="um", eps_vox=eps_vox, mode="border")

    # --- Redundancy: ALL shortest A->V paths (count ALL; plot limited)
    shortest_paths = shortest_av_paths(G)  # list[list[int]] in THIS graph ids

    for p in shortest_paths:
        path_len_rows.append({"graph": name, "path_len_edges": int(len(p) - 1)})

    if shortest_paths:
        plot_av_paths_in_box(
            G,
            cut_box,
            shortest_paths[:plot_paths_cap],
            coords_attr="coords",
            node_eps=0.0
        )

    # --- Vessel density in CutBox (formatted microsegments)
    density_mean_cutbox = np.nan
    try:
        ms_cut = microsegments_from_formatted_graph(G)
        df_box = vessel_vol_frac_slabs_in_box(ms_cut, cut_box, slab=slab_um_cutbox, axis=slab_axis)
        density_box_slabs[name] = df_box.copy()

        y_box = df_box["total_vol_frac"].to_numpy(float) * 100.0
        density_box_values[name] = y_box
        density_mean_cutbox = float(np.nanmean(y_box))

        for v in y_box:
            density_long_rows.append({"graph": name, "slab_vol_frac_pct": float(v)})

        df_box.to_csv(os.path.join(out_root, f"{name}_CutBox_density_slabs_{int(slab_um_cutbox)}um.csv"), index=False)

    except Exception as e:
        print(f"[{name}] cut-box density failed:", e)

    # --- Distance to surface (optional)
    d2s_mean = np.nan
    d2s_median = np.nan
    if ("distance_to_surface_R" in G.vs.attributes()) or ("distance_to_surface" in G.vs.attributes()):
        nodes = np.arange(G.vcount())
        d2s = distance_to_surface_stats(G, nodes, space="um")
        d2s_mean = float(d2s["mean"])
        d2s_median = float(d2s["median"])

    # --- Edge-disjoint paths (maxflow)
    red = max_edge_disjoint_av(G)
    n_disjoint = int(red["n_edge_disjoint_av"])
    nA = int(red["nA"])
    nV = int(red["nV"])

    red_rows.append({
        "graph": name,
        "edge_disjoint_AV": n_disjoint,
        "A_nodes": nA,
        "V_nodes": nV,
        "shortest_paths_n": int(len(shortest_paths)),
        "shortest_path_len_median": float(np.median([len(p)-1 for p in shortest_paths])) if shortest_paths else np.nan,
        "shortest_path_len_p90": float(np.percentile([len(p)-1 for p in shortest_paths], 90)) if shortest_paths else np.nan,
    })

    # --- Summary row
    summary = {
        "graph": name,
        "V": int(G.vcount()),
        "E": int(G.ecount()),
        "dup_%": float(dup["perc_extra_edges"]),
        "loops_%": float(loops["perc_loops"]),
        "HDN_n": int(hdn.get("n_hdn", 0)),
        "HDN_frac": float(hdn.get("hdn_fraction", 0.0)),
        "shortest_paths_n": int(len(shortest_paths)),
        "edge_disjoint_AV": int(n_disjoint),
        "A_nodes": int(nA),
        "V_nodes": int(nV),
        "density_mean_cutbox_%": float(density_mean_cutbox) if np.isfinite(density_mean_cutbox) else np.nan,
        "d2s_mean": float(d2s_mean) if np.isfinite(d2s_mean) else np.nan,
        "d2s_median": float(d2s_median) if np.isfinite(d2s_median) else np.nan,
    }

    # avg length by type
    for nm, val in avg_len_by_type.items():
        summary[f"avg_len_{nm}_um"] = float(val)

    # diameter stats by type
    for _, st in diam_stats.items():
        nm = st["name"]
        summary[f"diam_mean_{nm}_um"] = float(st["mean"])
        summary[f"diam_median_{nm}_um"] = float(st["median"])
        summary[f"diam_p5_{nm}_um"] = float(st["p5"])
        summary[f"diam_p95_{nm}_um"] = float(st["p95"])

    all_summaries.append(summary)

    # save per-box CSVs
    pd.DataFrame([summary]).to_csv(os.path.join(out_root, f"{name}_summary.csv"), index=False)
    bc_df.to_csv(os.path.join(out_root, f"{name}_bc_faces.csv"), index=False)

# ---------------------------------------------------------------------
# Store summary
# ---------------------------------------------------------------------
summary_df = pd.DataFrame(all_summaries)
summary_csv = os.path.join(out_root, "HPC_summary_COMPARISON.csv")
summary_df.to_csv(summary_csv, index=False)
print("\nSaved:", summary_csv)

# =====================================================================
# GLOBAL FIGURES (kept)
# =====================================================================
BOX_LABEL = " vs ".join(BOX_ORDER)

# Diameter/Length distributions per type
if diam_long_parts:
    dl = pd.concat(diam_long_parts, ignore_index=True)
    dl["type"] = dl["type"].astype(str)

    diameter_length_overlay_by_type(dl, bins=bins_hist, box_label=BOX_LABEL)

    for t in sorted(dl["type"].unique()):
        sub = dl[dl["type"] == t].copy()
        plot_boxplot_by_graph(sub, "diameter_um", title=f"Diameter boxplot | type={t}", ylabel="Diameter (µm)")
        plot_boxplot_by_graph(sub, "length_um", title=f"Length boxplot | type={t}", ylabel="Length (µm)")

# CutBox density heterogeneity (10 µm) -> boxplot per CutBox
if density_box_values:
    order = [g for g in BOX_ORDER if g in density_box_values]
    groups = [safe_finite(density_box_values[g]) for g in order]

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=order, showfliers=False)
    plt.title(f"CutBox vessel density heterogeneity | slab={int(slab_um_cutbox)} µm | {BOX_LABEL}")
    plt.ylabel("Vessel volume fraction per slab (%)")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()

    pd.DataFrame(density_long_rows).to_csv(
        os.path.join(out_root, f"CutBox_density_long_{int(slab_um_cutbox)}um.csv"),
        index=False
    )

# Shortest path length distribution (#edges): overlay hist + boxplot
if path_len_rows:
    pldf = pd.DataFrame(path_len_rows)

    plot_hist_overlay(
        pldf, "path_len_edges",
        title=f"Shortest A→V path length (#edges) | {BOX_LABEL}",
        xlabel="Path length (#edges)",
        bins=30,
    )
    plot_boxplot_by_graph(
        pldf, "path_len_edges",
        title=f"Shortest A→V path length (#edges) | {BOX_LABEL}",
        ylabel="Path length (#edges)",
    )

print("\nGLOBAL figures generated in:", out_root)

# ---------------------------------------------------------------------
# BC cube-net compare (per face, per box, stacked by vessel type)
# ---------------------------------------------------------------------
if len(bc_rows) == 0:
    print("[BC] bc_rows is empty -> nothing to plot.")
else:
    bc_all = pd.concat(bc_rows, ignore_index=True)

    face_col = "Face" if "Face" in bc_all.columns else None
    if face_col is None:
        print("[BC] Could not find face column ('Face') in bc_all.")
    else:
        COLMAP = {
            "arteriole": "n_arteriole",
            "venule": "n_venule",
            "capillary": "n_capillary",
            "unknown": "n_unknown",
        }
        total_col = "BC nodes" if "BC nodes" in bc_all.columns else None

        for col in list(COLMAP.values()) + ([total_col] if total_col else []):
            if col not in bc_all.columns:
                bc_all[col] = 0.0
            bc_all[col] = pd.to_numeric(bc_all[col], errors="coerce").fillna(0.0)

        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        fig.suptitle(f"Boundary conditions | cube-net compare | {BOX_LABEL}", fontsize=14)

        for r in range(2):
            for c in range(3):
                face = FACE_GRID[r][c]
                ax = axes[r, c]
                sub = bc_all.loc[bc_all[face_col] == face].copy()

                if sub.empty:
                    ax.set_title(face)
                    ax.text(0.5, 0.5, "no data", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                group_cols = list(COLMAP.values()) + ([total_col] if total_col else [])
                sub = sub.groupby("graph", as_index=False)[group_cols].sum()

                x = np.arange(len(BOX_ORDER))
                bottom = np.zeros(len(BOX_ORDER), float)

                for vname in ["arteriole", "venule", "capillary", "unknown"]:
                    col = COLMAP[vname]
                    vals = np.array([
                        float(sub.loc[sub["graph"] == g, col].iloc[0]) if (sub["graph"] == g).any() else 0.0
                        for g in BOX_ORDER
                    ], dtype=float)

                    ax.bar(x, vals, bottom=bottom, label=vname, color=V_COL.get(vname, "lightgrey"))
                    bottom += vals

                if total_col and total_col in sub.columns:
                    tot = np.array([
                        float(sub.loc[sub["graph"] == g, total_col].iloc[0]) if (sub["graph"] == g).any() else 0.0
                        for g in BOX_ORDER
                    ], dtype=float)
                    ax.plot(x, tot, "k.", markersize=8, label="total" if (r == 0 and c == 0) else None)

                ax.set_title(face)
                ax.set_xticks(x)
                ax.set_xticklabels(BOX_ORDER)
                ax.grid(alpha=0.25, axis="y")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.show()

# ---------------------------------------------------------------------
# Redundancy: shortest path length ECDF + medians
# ---------------------------------------------------------------------
if path_len_rows:
    sp = pd.DataFrame(path_len_rows)

    plt.figure(figsize=(7.5, 5))
    for g in BOX_ORDER:
        x = sp.loc[sp["graph"] == g, "path_len_edges"].to_numpy(float)
        x = x[np.isfinite(x)]
        if not x.size:
            continue
        xs, ys = ecdf_xy(x)
        plt.plot(xs, ys, label=g, color=BOX_COL.get(g, None))
        plt.axvline(float(np.median(xs)), color=BOX_COL.get(g, None), linestyle="--", alpha=0.8)

    plt.title(f"Shortest A→V path length (ECDF) | {BOX_LABEL}")
    plt.xlabel("Path length (#edges)")
    plt.ylabel("ECDF")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Edge-disjoint (maxflow) as bars
# ---------------------------------------------------------------------
if red_rows:
    ed = pd.DataFrame(red_rows).set_index("graph").reindex(BOX_ORDER)

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(BOX_ORDER, ed["edge_disjoint_AV"].to_numpy(float))
    plt.title(f"Edge-disjoint A→V paths (maxflow) | {BOX_LABEL}")
    plt.ylabel("# edge-disjoint paths")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.show()
