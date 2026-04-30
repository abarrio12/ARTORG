# Vascular Graph Analysis — Reference

Module: `graph_analysis_functions_formatted.py`  
Main notebooks: `vascular_graph_analysis.ipynb` · `region_comparison.ipynb`

---

## Constants & units

```python
ARTERY = 2 · VEIN = 3 · CAPILLARY = 4
res_um_per_vox = [1.625, 1.625, 2.5]   # µm/voxel
```

Coordinates and lengths → **µm** (`coords`, `length`). Diameters → **voxels** (`diameter`).

---

## Setup

```python
g = load_graph(path)
G = keep_giant_component(g)                          # keep largest component

box = make_box_in_um(center_vox, box_size_um=(400,400,400))
# center_vox from ParaView; all box values in µm
validate_box_faces(box)

add_penalization_weights(G, new_weight_attr="w_cap_prior",
                         penalty_art=1e5, penalty_ven=1e5)
```

---

## Per-box analysis loop (`vascular_graph_analysis.ipynb`)

### 3.1 — Basic stats (sanity checks)

```python
duplicated_edge_stats(G)   # duplicated edges = segmentation artifact
loop_edge_stats(G)         # self-loops = reconstruction error
get_edges_types(G)         # % per nkind; capillaries should dominate (~80–90%)
```

---

### 3.2 — Lengths & diameters

Diameter controls flow resistance (R ∝ 1/r⁴). Capillary diameter (~4–8 µm) is near the size of a red blood cell and determines exchange capacity.

```python
get_avg_length_nkind(G)                          # mean segment length per vessel type (µm)
diameter_stats_nkind(G, label_dict=EDGE_NKIND_TO_LABEL,
                     plot=True, title_suffix=name)
# mean/median/p5–p95 per type + violin plot

# collect per edge for cross-box comparison
for k in np.unique(nk):
    diam_length_rows.append(pd.DataFrame({
        "graph": name, "type": lab,
        "diameter_vox": e_diam[m], "length_um": e_leng[m]
    }))

plot_hist_by_category_general(values=e_leng, category=nk, ...)
plot_pies_with_bars_edge_types(...)   # bar + pie of edge composition
```

---

### 3.3 — Degrees & high-degree nodes (HDN)

Degree-3 nodes are standard bifurcations (Murray's law). Nodes with degree ≥ 4 are branching hubs of the arterial/venous trees — their depth and type reveal supply/drainage architecture.

```python
get_degrees(G, threshold=4)          # attaches vs["degree"]

analyze_hdn_pattern_in_box(G, box=cut_box, coords_attr="coords",
                            space="um", degree_thr=4, eps_vox=2.0)
# → spatial centroid shift, type composition (arterial/venous/capillary hubs),
#   depth distribution (µm from surface), face bias (boundary artifact check)

plot_degree_nodes_spatial(G, coords_attr="coords", degree_min=4, by_type=True)
```

---

### 3.4 — Boundary conditions (BC nodes)

Vessels truncated at the box faces are the inlets/outlets. Their count, type, and diameter matter for flow simulations and reveal how the tissue is supplied — an imbalance across faces (e.g. all arteries on z_max) reflects the local anatomy of feeding vessels.

```python
bc_res = analyze_bc_faces(G, box=cut_box, coords_attr="coords", space="um",
                           eps_vox=2.0, degree_thr=4, return_diameter_values=True)
# per face: count, type %, diameter stats

df_bc      = bc_faces_table(bc_res, box_name=name)
df_bc_long = bc_diameter_longtable(bc_res, box_name=name)

plot_bc_cube_net(bc_res)             # 2D unfolded cube — composition per face
plot_bc_3_cubes_tinted(G, box=cut_box, coords_attr="coords", space="um")
```

---

### 3.5–3.6 — Shortest A→V paths & frontier comparison

The A/C frontier = nodes at the arterio-capillary transition (where blood enters the capillary bed). Shortest paths from there to the nearest venous node proxy capillary transit distance. Longer / more heterogeneous distances → less efficient oxygen extraction.

Two routing modes compared per frontier node:
- **Restricted** (`artery_continuity=False`): capillary + venous edges only. Physiologically correct route.
- **Weighted** (`w_cap_prior`): all edges allowed but artery/vein heavily penalized. A path that still uses an artery → arterio-venous shortcut (blood bypasses capillary bed).

```python
paths, dist, n_edges, src, tgt, _, _ = shortest_av_paths_from_ac_frontier(
    G, artery_continuity=False, weight_attr="length")

df_cmp, df_restricted, df_weighted = compare_frontier_restricted_vs_weighted(G)
# delta = weighted_length - restricted_length
# delta > 0: no shortcut (expected)
# delta < 0: arterial shortcut exists

summ = summarize_frontier_comparison_restricted_vs_weighted(df_cmp, graph_name=name)
build_frontier_comparison_summary_table_restricted_vs_weighted(df_cmp_summary, BOX_ORDER)
```

---

### 3.7 — Min-cut / max-flow

Min-cut = minimum edges to remove to disconnect all arterioles from all venules = maximum number of parallel, non-overlapping A→V routes. With `per_edge_capacity=1.0` this equals the number of independent paths directly. Measures **vascular redundancy** — how many vessel segments can be blocked before perfusion fails. Cut edges are usually capillaries (the anatomical bottleneck).

```python
mc     = av_min_cut_metrics(G, per_edge_capacity=1.0)
counts = cut_type_counts(mc["cut_edge_types"])
# → min_cut_value, cut_edge_ids, n_arteriole/venule/capillary edges in cut

mincut_rows.append({**mc, **counts, "graph": name})
```

---

### 3.8 — Edge-disjoint paths

```python
result = max_edge_disjoint_av(G)
# Same quantity as min-cut but returns actual path lists for export/visualization.
# → n_edge_disjoint_av, nA, nV, paths
```

---

### 3.9 — Saturation

How efficiently the independent paths use the available arterio-capillary (A-C) and veno-capillary (V-C) interface. Low saturation → many transition points unused, capillary bed poorly connected between trees.

```python
sat = saturation_interface_proxies(G, source_frontiers=src,
                                   target_venous=tgt,
                                   n_disjoint=mc["min_cut_value"])
# → saturation_ac_edges, saturation_vc_edges
#   ac_frontier_nodes_used/total, vc_nodes_reached/total
```

---

### 3.10 — Vessel density

- **Volume density** (mm³/mm³): fraction of tissue occupied by vessel lumen → blood volume / O₂ capacity.
- **Length density** (mm/mm³): total vessel length per tissue volume → average diffusion distance to nearest vessel. Capillary length density is the most relevant for oxygen delivery.

```python
ms = microsegments_from_formatted_graph(G)
# decomposes edge polylines into small segments with local radius

vvd = vessel_volume_density(ms, box)
vld = vessel_length_density(ms, box)
vvd_k = vessel_volume_density_nkind(ms, box, nkind_filter=CAPILLARY)
vld_k = vessel_length_density_nkind(ms, box, nkind_filter=CAPILLARY)

sub_boxes = generate_boxes(box, box_size=100, stride=50)  # spatial variability
```

---

### 3.12 — Major vessel trees

Arterial and venous networks appear fragmented in a cut volume — "major trees" are the large components with real branching (≥1 branch node). Tells you how many distinct trees supply the volume and what fraction of the network they cover.

```python
comp_df, edge_df = major_components_from_edge_code(G, target_code=ARTERY)
# comp_df: n_edges, n_branch_nodes, is_major, pct_edges, pct_nodes per component

save_major_trees_table_png(summary_df, out_path, graph_order=BOX_ORDER)
```

---

## Global section & outputs

```python
# build global dataframes from collectors
summary_df     = pd.DataFrame(all_summaries)
df_diam_length = pd.concat(diam_length_rows)
df_bc_global   = pd.concat(bc_rows)
df_bc_long     = pd.concat(bc_long_rows)
df_comparison  = pd.concat(comparison_detail_rows)
df_mincut      = pd.DataFrame(mincut_rows)
df_saturation  = pd.DataFrame(saturation_rows)
df_density_all = pd.DataFrame(density_boxes_rows)

# summary tables → PNG
save_major_trees_table_png(summary_df, ...)
build_frontier_comparison_summary_table_restricted_vs_weighted(df_cmp_summary, BOX_ORDER)
dataframe_to_table_figure(mincut_table, title="Min-cut summary")
```

**CSVs saved:** `{REGION}_summary` · `_diameter_length` · `_bc_faces` · `_bc_diameter_long` · `_density_subboxes` · `_frontier_comparison_detail/summary` · `_mincut` · `_saturation` · `_redundancy` · `_hdn_nodes`

---

## Section 5 — Plots (independent cells, re-runnable)

```python
# 5.0 Composition
plot_pies_with_bars_edge_types(summary_df, box_order=BOX_ORDER)

# 5.1 Connectivity
plot_paired_restricted_vs_weighted_boxplots_from_comparison(df_comparison, BOX_ORDER)
plot_restricted_vs_weighted_scatter(df_comparison, graph_name)

# 5.2 Saturation — printed table only (plots commented out)

# 5.3 Shortest paths
plot_boxplot_by_graph(df_av_path_sizes, "path_len_um", ...)

# 5.4 Vessel density
plot_boxplot_by_graph(df_density_total, "vessel_volume_density", ...)
plot_simple_type_boxplots_with_stats(sub, value_col="vessel_volume_density", ...)
pairwise_ttests_table(sub, col, BOX_ORDER)   # Welch t-tests between boxes

# 5.5 Diameter & length
diameter_length_overlay_by_type(dl, bins=40, box_label=BOX_LABEL)
plot_boxplot_by_graph(sub, "diameter_vox", ...)
plot_grouped_boxplot_types_per_graph(dl, value_col="diameter_vox", ...)

# 5.6 Boundary conditions
plot_simple_type_boxplots_with_stats(bc_plot, value_col="diameter", ...)
```

---

## `region_comparison.ipynb`

Loads the CSVs produced above for HPC and SMC and runs HPC vs SMC statistical comparisons. Run `vascular_graph_analysis.ipynb` for both regions first.

```python
REGION_ORDER = ("HPC", "SMC")
FILE_MAP = {
    "summary":     "{region}_summary.csv",
    "density":     "{region}_density_subboxes.csv",
    "diam_length": "{region}_diameter_length.csv",
    "bc_pooled":   "{region}_bc_pooled.csv",
    "mincut":      "{region}_mincut.csv",
    "saturation":  "{region}_saturation.csv",
    ...
}
summary, density_all, dl_all, bc_long, mincut, saturation = load_both(...)
```

**Core comparison function:** `welch_row(x_hpc, x_smc, label)` — returns n, mean, median, t-stat, p-value.  
**Core plot function:** `_draw_violin_box(ax, data_hpc, data_smc)` — violin (light fill) + narrow box (dark fill, white median line).

### Sections

| Section | What is compared | Key functions |
|---------|-----------------|---------------|
| 3 — Vessel density | Volume + length density per vessel type | `plot_simple_type_boxplots_with_stats`, `welch_row` |
| 4 — Diameter & length | Diameter (vox) and length (µm) per type | `_draw_violin_box`, `welch_row` |
| 5 — Boundary conditions | BC node diameter per face and type | `plot_simple_type_boxplots_with_stats` |
| 6 — Connectivity | Min-cut value, n_edge_disjoint | bar plots, `welch_row` |
| 7 — Saturation | saturation_ac/vc_edges | printed table, `welch_row` |
| 8 — Major trees | n_major trees, pct_edges covered | `save_major_trees_table_png` |
| 9 — Save stats | All Welch results concatenated | `pd.concat` → CSV |

---

## Export to ParaView

```python
export_paths_vtp(G, paths, filename)           # A→V paths as polylines
export_labeled_edges_vtp(G, edge_df, out_path, include_only_major=True)  # tree labels
export_edge_ids_vtp(G, mc["cut_edge_ids"], out_path)   # min-cut bottleneck edges
```