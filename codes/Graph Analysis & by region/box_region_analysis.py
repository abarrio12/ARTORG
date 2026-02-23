"""
Box-based Regional Analysis Tool

Simple tool to:
1. Extract vessel metrics from boxes in brain regions
2. Compare regions (hippocampal vs somatomotor)
3. Visualize differences with plots

Author: Ana Barrio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

sys.path.insert(0, '/home/admin/Ana/MicroBrain/codes/Graph Analysis & by region/Graph analysis')
from graph_analysis_functions import (
    load_data, make_box_in_vox, induced_subgraph_box,
    EDGE_NKIND_TO_LABEL, resolve_length_attr,
    analyze_hdn_pattern_in_box, distance_to_surface_stats,
    analyze_bc_faces, microsegments, vessel_vol_frac_slabs_in_box
)


class BoxAnalyzer:
    """Simple box-based vascular analysis."""
    
    def __init__(self, graph, box_size_um=400, space="um", data=None):
        self.graph = graph
        self.box_size_um = (box_size_um, box_size_um, box_size_um)
        self.space = space
        self.data = data
        self.results = {}
    
    def extract_box(self, center_vox, box_id=None):
        """Extract all metrics from a box."""
        
        box = make_box_in_vox(center_vox, self.box_size_um, as_float=True)
        sub, sub_to_orig, _ = induced_subgraph_box(
            self.graph, box, coords_attr="coords_image", edge_mode="both"
        )
        
        if sub is None or sub.ecount() == 0:
            return None
        
        # Get edge indices
        edge_indices = []
        for edge in sub.es:
            v1 = sub_to_orig[edge.source]
            v2 = sub_to_orig[edge.target]
            eid = self.graph.get_eid(v1, v2)
            edge_indices.append(eid)
        
        # Basic metrics
        diam = np.array([self.graph.es[eid]["diameter"] for eid in edge_indices])
        length = np.array([self.graph.es[eid][resolve_length_attr(self.space)] for eid in edge_indices])
        nkind = np.array([self.graph.es[eid]["nkind"] for eid in edge_indices])
        
        # Degree statistics
        degrees = np.array(sub.degree())
        node_indices = np.array(sub_to_orig, dtype=int)
        
        result = {
            "box_id": box_id,
            "n_edges": len(edge_indices),
            "n_nodes": sub.vcount(),
            "diameter": diam,
            "length": length,
            "nkind": nkind,
            "degree": degrees,
            "_node_indices": node_indices,
            "_box": box,
        }
        
        # HDN pattern
        try:
            hdn_info = analyze_hdn_pattern_in_box(
                self.graph, space=self.space, coords_attr="coords_image",
                depth_attr=None, degree_thr=4, box=box
            )
            result["hdn"] = hdn_info
        except:
            result["hdn"] = None
        
        # Distance to surface
        try:
            d2s = distance_to_surface_stats(self.graph, node_indices, space=self.space)
            result["d2s"] = d2s
        except:
            result["d2s"] = None
        
        # Vessel density (if data provided)
        if self.data:
            try:
                ms = microsegments(self.data, space=self.space, nkind_attr="nkind")
                vf = vessel_vol_frac_slabs_in_box(ms, box, slab=50.0, axis="z")
                result["vessel_density"] = len(ms["lengths"])
                result["vessel_frac"] = vf
            except:
                result["vessel_density"] = None
        
        # Boundary conditions
        try:
            bc = analyze_bc_faces(self.graph, box, space=self.space, 
                                 coords_attr="coords_image", degree_thr=4)
            result["bc"] = bc
        except:
            result["bc"] = None
        
        return result
    
    def analyze_region(self, region_name, box_centers):
        """Analyze multiple boxes in a region."""
        
        boxes = []
        for i, center in enumerate(box_centers):
            box_data = self.extract_box(center, f"{region_name}_box_{i+1}")
            if box_data:
                boxes.append(box_data)
        
        if not boxes:
            return None
        
        # Combine metrics
        all_diam = np.concatenate([b["diameter"] for b in boxes])
        all_length = np.concatenate([b["length"] for b in boxes])
        all_degrees = np.concatenate([b["degree"] for b in boxes])
        all_nkind = np.concatenate([b["nkind"] for b in boxes])
        
        # HDN aggregation
        hdn_total = sum([b.get("hdn", {}).get("count", 0) for b in boxes])
        hdn_art = sum([b.get("hdn", {}).get("arteriole", {}).get("count", 0) for b in boxes])
        hdn_ven = sum([b.get("hdn", {}).get("venule", {}).get("count", 0) for b in boxes])
        hdn_cap = sum([b.get("hdn", {}).get("capillary", {}).get("count", 0) for b in boxes])
        
        # D2S aggregation
        d2s_data = [b.get("d2s", {}) for b in boxes if b.get("d2s")]
        d2s_mean = np.mean([d.get("mean", 0) for d in d2s_data]) if d2s_data else None
        
        # Vessel density
        vd = sum([b.get("vessel_density", 0) for b in boxes])
        
        result = {
            "region": region_name,
            "n_boxes": len(boxes),
            "boxes": boxes,
            
            "diameter": {
                "mean": np.mean(all_diam),
                "std": np.std(all_diam),
                "median": np.median(all_diam),
                "p5_p95": (np.percentile(all_diam, 5), np.percentile(all_diam, 95)),
            },
            
            "length": {
                "mean": np.mean(all_length),
                "std": np.std(all_length),
                "median": np.median(all_length),
                "total": np.sum(all_length),
            },
            
            "degree": {
                "mean": np.mean(all_degrees),
                "hdn_count": np.sum(all_degrees >= 4),
                "hdn_pct": 100.0 * np.sum(all_degrees >= 4) / len(all_degrees),
            },
            
            "hdn": {
                "total": hdn_total,
                "arteriole": hdn_art,
                "venule": hdn_ven,
                "capillary": hdn_cap,
            },
            
            "d2s": {
                "mean": d2s_mean,
            },
            
            "vessel_density": vd,
            
            "vessel_types": self._count_types(all_nkind),
            
            "_diameter": all_diam,
            "_length": all_length,
            "_degree": all_degrees,
        }
        
        self.results[region_name] = result
        return result
    
    def _count_types(self, nkinds):
        """Count vessel types."""
        unique, counts = np.unique(nkinds, return_counts=True)
        return {
            EDGE_NKIND_TO_LABEL.get(int(k), f"type_{k}"): {
                "count": int(c),
                "pct": float(100 * c / len(nkinds))
            }
            for k, c in zip(unique, counts)
        }
    
    def compare(self, region1, region2):
        """Compare two regions statistically."""
        
        r1 = self.results.get(region1)
        r2 = self.results.get(region2)
        
        if not r1 or not r2:
            print(f"Both regions must be analyzed first")
            return None
        
        # T-tests
        t_diam, p_diam = stats.ttest_ind(r1["_diameter"], r2["_diameter"])
        t_length, p_length = stats.ttest_ind(r1["_length"], r2["_length"])
        
        # Mann-Whitney (non-parametric)
        u_diam, p_u_diam = stats.mannwhitneyu(r1["_diameter"], r2["_diameter"])
        u_length, p_u_length = stats.mannwhitneyu(r1["_length"], r2["_length"])
        
        return {
            "region1": region1,
            "region2": region2,
            "diameter": {
                "mean1": r1["diameter"]["mean"],
                "mean2": r2["diameter"]["mean"],
                "diff": r1["diameter"]["mean"] - r2["diameter"]["mean"],
                "t_test_p": p_diam,
                "mw_p": p_u_diam,
            },
            "length": {
                "mean1": r1["length"]["mean"],
                "mean2": r2["length"]["mean"],
                "diff": r1["length"]["mean"] - r2["length"]["mean"],
                "t_test_p": p_length,
                "mw_p": p_u_length,
            },
            "degree": {
                "mean1": r1["degree"]["mean"],
                "mean2": r2["degree"]["mean"],
                "hdn_pct1": r1["degree"]["hdn_pct"],
                "hdn_pct2": r2["degree"]["hdn_pct"],
            },
            "_r1": r1,
            "_r2": r2,
        }
    
    def plot_region(self, region_name, figsize=(16, 5)):
        """Plot all metrics for a region."""
        
        result = self.results[region_name]
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(f"{region_name}", fontsize=14, fontweight="bold")
        
        # Diameter
        axes[0].hist(result["_diameter"], bins=40, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0].axvline(result["diameter"]["mean"], color="red", linestyle="--", linewidth=2)
        axes[0].axvline(result["diameter"]["median"], color="orange", linestyle="--", linewidth=2)
        axes[0].fill_betweenx([0, axes[0].get_ylim()[1]], result["diameter"]["p5_p95"][0], 
                              result["diameter"]["p5_p95"][1], alpha=0.2, color="green")
        axes[0].set_xlabel("Diameter (µm)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Diameter\nμ={result['diameter']['mean']:.2f}±{result['diameter']['std']:.2f}")
        axes[0].grid(alpha=0.3)
        
        # Length
        axes[1].hist(result["_length"], bins=40, color="seagreen", alpha=0.7, edgecolor="black")
        axes[1].axvline(result["length"]["mean"], color="red", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Length (µm)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Length\nμ={result['length']['mean']:.2f}±{result['length']['std']:.2f}")
        axes[1].grid(alpha=0.3)
        
        # Degree distribution
        deg_unique, deg_counts = np.unique(result["_degree"], return_counts=True)
        axes[2].bar(deg_unique, deg_counts, color="coral", alpha=0.7, edgecolor="black")
        axes[2].axvline(4, color="red", linestyle="--", linewidth=2, alpha=0.7)
        axes[2].set_xlabel("Node Degree")
        axes[2].set_ylabel("Count")
        hdn_pct = result["degree"]["hdn_pct"]
        axes[2].set_title(f"Degree\nHDN: {hdn_pct:.1f}%")
        axes[2].grid(alpha=0.3, axis="y")
        
        # HDN patterns
        hdn = result["hdn"]
        if hdn and hdn["total"] > 0:
            hdn_types = ["Arteriole", "Venule", "Capillary"]
            hdn_counts = [hdn["arteriole"], hdn["venule"], hdn["capillary"]]
            colors_hdn = ["red", "blue", "purple"]
            axes[3].bar(hdn_types, hdn_counts, color=colors_hdn, alpha=0.7, edgecolor="black")
            axes[3].set_ylabel("Count")
            axes[3].set_title(f"HDN Breakdown\nTotal: {hdn['total']}")
            axes[3].grid(alpha=0.3, axis="y")
        else:
            axes[3].text(0.5, 0.5, "No HDN data", ha="center", va="center", transform=axes[3].transAxes)
            axes[3].set_title("HDN Breakdown")
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, region1, region2, figsize=(18, 5)):
        """Plot comparison between two regions."""
        
        comp = self.compare(region1, region2)
        r1 = comp["_r1"]
        r2 = comp["_r2"]
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(f"{region1} vs {region2}", fontsize=14, fontweight="bold")
        
        # Diameter
        axes[0].hist(r1["_diameter"], bins=35, alpha=0.6, label=region1, color="steelblue", edgecolor="black")
        axes[0].hist(r2["_diameter"], bins=35, alpha=0.6, label=region2, color="coral", edgecolor="black")
        axes[0].set_xlabel("Diameter (µm)")
        axes[0].set_ylabel("Frequency")
        p_diam = comp['diameter']['mw_p']
        axes[0].set_title(f"Diameter\np={p_diam:.2e}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Length
        axes[1].hist(r1["_length"], bins=35, alpha=0.6, label=region1, color="seagreen", edgecolor="black")
        axes[1].hist(r2["_length"], bins=35, alpha=0.6, label=region2, color="gold", edgecolor="black")
        axes[1].set_xlabel("Length (µm)")
        axes[1].set_ylabel("Frequency")
        p_len = comp['length']['mw_p']
        axes[1].set_title(f"Length\np={p_len:.2e}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Degree comparison
        deg1, deg2 = r1["degree"]["mean"], r2["degree"]["mean"]
        axes[2].bar([region1, region2], [deg1, deg2], color=["steelblue", "coral"], alpha=0.7, edgecolor="black")
        axes[2].set_ylabel("Mean Degree")
        axes[2].set_title("Mean Connectivity")
        axes[2].grid(alpha=0.3, axis="y")
        
        # HDN comparison
        hdn1_total = r1["hdn"]["total"]
        hdn2_total = r2["hdn"]["total"]
        hdn_data = [
            [r1["hdn"]["arteriole"], r1["hdn"]["venule"], r1["hdn"]["capillary"]],
            [r2["hdn"]["arteriole"], r2["hdn"]["venule"], r2["hdn"]["capillary"]]
        ]
        
        x = np.arange(3)
        width = 0.35
        axes[3].bar(x - width/2, hdn_data[0], width, label=region1, color="steelblue", alpha=0.7)
        axes[3].bar(x + width/2, hdn_data[1], width, label=region2, color="coral", alpha=0.7)
        axes[3].set_ylabel("HDN Count")
        axes[3].set_title("HDN by Type")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(["Arteriole", "Venule", "Capillary"])
        axes[3].legend()
        axes[3].grid(alpha=0.3, axis="y")
        
        plt.tight_layout()
        return fig
    
    def summary_table(self, regions=None):
        """Print summary table."""
        
        if regions is None:
            regions = list(self.results.keys())
        
        data = []
        for region in regions:
            r = self.results[region]
            data.append({
                "Region": region,
                "Boxes": r["n_boxes"],
                "Edges": r["n_boxes"] * np.mean([b["n_edges"] for b in r["boxes"]]),
                "Diam (µm)": f"{r['diameter']['mean']:.2f}±{r['diameter']['std']:.2f}",
                "Length (µm)": f"{r['length']['mean']:.2f}±{r['length']['std']:.2f}",
                "Deg": f"{r['degree']['mean']:.2f}",
                "HDN": r['hdn']['total'],
                "HDN%": f"{r['degree']['hdn_pct']:.1f}",
                "D2S": f"{r['d2s']['mean']:.1f}" if r['d2s']['mean'] else "N/A",
                "MicroSeg": r['vessel_density'],
            })
        
        df = pd.DataFrame(data)
        print("\n" + "="*120)
        print(df.to_string(index=False))
        print("="*120 + "\n")


# Example usage
if __name__ == "__main__":
    
    print("Loading graph...")
    data = load_data("/home/admin/Ana/MicroBrain/whole brain/18_vessels_graph.gt")
    graph = data["graph"]
    
    analyzer = BoxAnalyzer(graph, box_size_um=400, data=data)  # Pass data for advanced metrics
    
    # Define box centers (in voxels) - CHANGE THESE TO YOUR ACTUAL COORDINATES
    hippo_centers = [
        [150, 200, 100],
        [200, 250, 120],
        [180, 220, 110],
    ]
    
    somato_centers = [
        [250, 300, 150],
        [280, 320, 160],
        [270, 310, 155],
    ]
    
    print("Analyzing hippocampus...")
    analyzer.analyze_region("hippocampus", hippo_centers)
    
    print("Analyzing somatomotor...")
    analyzer.analyze_region("somatomotor", somato_centers)
    
    # Print summary
    analyzer.summary_table()
    
    # Plot each region
    fig1 = analyzer.plot_region("hippocampus")
    fig1.savefig("/home/admin/Ana/MicroBrain/output/hippocampus_analysis.png", dpi=150, bbox_inches="tight")
    
    fig2 = analyzer.plot_region("somatomotor")
    fig2.savefig("/home/admin/Ana/MicroBrain/output/somatomotor_analysis.png", dpi=150, bbox_inches="tight")
    
    # Comparison plot
    fig3 = analyzer.plot_comparison("hippocampus", "somatomotor")
    fig3.savefig("/home/admin/Ana/MicroBrain/output/comparison.png", dpi=150, bbox_inches="tight")
    
    print("\nPlots saved to /home/admin/Ana/MicroBrain/output/")
    
    plt.show()
