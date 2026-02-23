"""
HPC 3 cut graphs analysis (space = um)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/home/admin/Ana/MicroBrain/codes/Graph Analysis & by region/Graph analysis')

from graph_analysis_functions import *

out_root = "/home/admin/Ana/MicroBrain/output/HPC_FULL_ANALYSIS"
os.makedirs(out_root, exist_ok=True)

PATHS = {
    "HPC_1": "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut1_um.pkl",
    "HPC_2": "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut2_um.pkl",
    "HPC_3": "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_Hcut3_um.pkl",
}

HIPPO_CENTERS = [
    [1200,3400,1500],
    [1400,3900,950],
    [2100,4100,750],
]

box_size_um = (400,400,400)

all_summaries = []

for (name, path), center in zip(PATHS.items(), HIPPO_CENTERS):

    print("\n======================")
    print("Analyzing:", name)

    data = load_data(path)
    sync_vertex_attributes_safe(data, space="um")
    _, _, data = single_connected_component(data)
    G = data["graph"]

    # ROI cube
    box = make_box_in_um(center, box_size_um, res_um_per_vox=res_um_per_vox)
    validate_box_faces(box)

    # --- Basic stats
    dup = duplicated_edge_stats(G)
    loops = loop_edge_stats(G)

    # --- Edge types
    edge_types = get_edges_types(G, return_dict=True)

    # --- Lengths (avg per nkind)
    nk_u, avg_len = get_avg_length_nkind(G, space="um")

    # plot lengths by type
    length_attr = resolve_length_attr(space="um")
    plot_hist_by_category_general(
        values=np.asarray(G.es[length_attr], float),
        category=np.asarray(G.es["nkind"], int),
        label_dict=EDGE_NKIND_TO_LABEL,
        bins=40,
        layout="horizontal",
        density=True,
        show_mean=True,
        variable_name="Edge length (µm)",
        category_name="Vessel type",
        main_title=f"{name} | Length distribution by vessel type"
    )

    # --- Diameters
    diam_stats = diameter_stats_nkind(G, plot=True)

    # --- HDN
    get_degrees(G, threshold=4)
    hdn = analyze_hdn_pattern_in_box(
        G, space="um",
        coords_attr="coords_image_R",
        degree_thr=4,
        box=box
    )

    # plot HDN spatial
    plot_degree_nodes_spatial(
        G,
        space="um",
        coords_attr="coords_image_R",
        degree_min=4,
        degree_max=None,
        by_type=True,
        title=f"{name} | High-degree nodes (deg ≥ 4)"
    )

    # --- BC faces
    bc = analyze_bc_faces(G, box, space="um", coords_attr="coords_image_R")
    bc_df = bc_faces_table(bc, box_name=name)

    # plot BC
    plot_bc_cube_net(bc, title=f"{name} | BC composition per face (cube net) | space=um")
    plot_bc_3_cubes_tinted(G, box, space="um", coords_attr="coords_image_R", eps=2.0)

    # --- Redundancy (shortest paths)
    groups = nodes_by_label(G)
    A = groups.get("arteriole", [])
    V = groups.get("venule", [])

    shortest_paths = []
    for a in A:
        for v in V:
            p = G.get_shortest_paths(a, to=v)[0]
            if len(p) > 1:
                shortest_paths.append(p)

    # plot a few shortest paths
    if len(shortest_paths) > 0:
        plot_av_paths_in_box(
            G, box,
            shortest_paths[:15],
            space="um",
            coords_attr="coords_image_R",
            node_eps=0.0
        )

    # --- Density
    density_mean = None
    density_df = None
    try:
        ms = microsegments(data, space="um")
        density_df = vessel_vol_frac_slabs_in_box(ms, box, slab=50, axis="z")
        density_mean = np.nanmean(density_df["total_vol_frac"].to_numpy())
        density_df.to_csv(os.path.join(out_root, f"{name}_density.csv"), index=False)

        # plot density slabs (simple)
        mid = 0.5 * (density_df["slab_lo"].to_numpy() + density_df["slab_hi"].to_numpy()    )
        plt.figure()
        plt.plot(mid, density_df["total_vol_frac"].to_numpy(), marker="o")
        plt.xlabel("Depth (µm)")
        plt.ylabel("Volume fraction")
        plt.title(f"{name} | Vessel volume fraction vs depth (slabs)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Density failed:", e)

    # --- Distance to surface
    d2s_mean = None
    if "distance_to_surface_R" in G.vs.attributes():
        nodes = np.arange(G.vcount())
        d2s = distance_to_surface_stats(G, nodes, space="um")
        d2s_mean = d2s["mean"]

    # --- Edge-disjoint paths (maxflow)
    red = max_edge_disjoint_av(G)

    n_disjoint = red["n_edge_disjoint_av"]
    nA = red["nA"]
    nV = red["nV"]

    # --- Summary row
    summary = {
        "graph": name,
        "V": G.vcount(),
        "E": G.ecount(),
        "dup_%": dup["perc_extra_edges"],
        "loops_%": loops["perc_loops"],
        "HDN_n": hdn.get("n_hdn", 0),
        "HDN_frac": hdn.get("hdn_fraction", 0),
        "shortest_paths_n": len(shortest_paths),
        "density_mean": density_mean,
        "d2s_mean": d2s_mean,
        "edge_disjoint_AV": n_disjoint,
        "A_nodes": nA,
        "V_nodes": nV
    }

    all_summaries.append(summary)

    pd.DataFrame([summary]).to_csv(os.path.join(out_root, f"{name}_summary.csv"), index=False)
    bc_df.to_csv(os.path.join(out_root, f"{name}_bc_faces.csv"), index=False)


summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(out_root, "HPC_summary_COMPARISON.csv"), index=False)

print("\nDone.")