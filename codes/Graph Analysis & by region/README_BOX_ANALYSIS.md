# Box-Based Regional Analysis Tool

## Overview

This toolset analyzes vessel properties across different brain regions by sampling and comparing multiple boxes of interest. It enables:

1. **Regional Analysis**: Analyze vessel properties (diameter, length, etc.) within defined boxes
2. **Regional Summary**: Compute statistics and normal ranges for each region
3. **Statistical Comparison**: Compare regions using t-tests and Mann-Whitney U tests
4. **Visualization**: Generate comparison plots and diagnostic visualizations

## Files

- **box_region_analysis.py** - Standalone Python script with `BoxRegionAnalyzer` class
- **box_region_analysis.ipynb** - Interactive Jupyter notebook (recommended for exploration)
- **README.md** - This file

## Quick Start

### Option 1: Using the Jupyter Notebook (Recommended)

```bash
cd /home/admin/Ana/MicroBrain/codes/Graph\ Analysis\ \&\ by\ region
jupyter notebook box_region_analysis.ipynb
```

Then follow the cells sequentially:

1. **Import libraries and load data**
2. **Define box centers** for your regions (hippocampal and somatomotor)
3. **Analyze each region** independently
4. **Compare regions** with statistical tests
5. **Generate report** with findings

### Option 2: Using the Python Script

```python
from box_region_analysis import BoxRegionAnalyzer
from graph_analysis_functions import load_data, make_box_in_vox

# Load graph
data = load_data("/home/admin/Ana/MicroBrain/whole brain/18_vessels_graph.gt")
graph = data["graph"]

# Create analyzer
analyzer = BoxRegionAnalyzer(graph, box_size_um=(400, 400, 400), space="um")

# Add boxes
analyzer.add_box_to_region("hippocampal", "hippo_box_1", [150, 200, 100])
analyzer.add_box_to_region("hippocampal", "hippo_box_2", [200, 250, 120])
analyzer.add_box_to_region("hippocampal", "hippo_box_3", [180, 220, 110])

analyzer.add_box_to_region("somatomotor", "somato_box_1", [250, 300, 150])
analyzer.add_box_to_region("somatomotor", "somato_box_2", [280, 320, 160])
analyzer.add_box_to_region("somatomotor", "somato_box_3", [270, 310, 155])

# Analyze
analyzer.analyze_region("hippocampal")
analyzer.analyze_region("somatomotor")

# Compare
comparison = analyzer.compare_regions("hippocampal", "somatomotor")

# Plot
fig = analyzer.plot_comparison("hippocampal", "somatomotor")
analyzer.save_results("output/results.pkl")
```

## Step 1: Defining Box Centers

**Important**: Replace placeholder coordinates with actual voxel coordinates for your regions of interest.

### How to get coordinates from ParaView:

1. Open your vascular graph in ParaView
2. Use the pointer/crosshair tool to hover over regions of interest
3. Note the coordinates displayed in ParaView status bar
4. Use these coordinates as box centers

### Example box definition:
```python
hippocampal_centers = [
    [150, 200, 100],  # Box 1: center at (150, 200, 100) voxels
    [200, 250, 120],  # Box 2: center at (200, 250, 120) voxels
    [180, 220, 110],  # Box 3: center at (180, 220, 110) voxels
]
```

**Box size**: Default is 400×400×400 µm³. Adjust as needed using the `box_size_um` parameter.

## Step 2: Understanding the Analysis

### Per-Box Analysis
For each box, the tool extracts:
- Diameter statistics (mean, median, std, p5, p95)
- Length statistics
- Vessel type breakdown (arteriole, venule, capillary)
- Number of edges and nodes

### Per-Region Summary
Aggregates all boxes in a region:
- Combined diameter range (normal range as p5-p95)
- Combined length range
- Vessel type percentages

### Regional Comparison
Compares two regions using:
- **Independent t-test**: parametric test for mean differences
- **Mann-Whitney U test**: non-parametric alternative
- Effect size: difference in means
- p-values for statistical significance

## Step 3: Interpreting Results

### Key Outputs

1. **Diameter Summary Table**
   - Shows mean, median, and range for each region
   - Use p5-p95 range as "normal" for quality control

2. **Length Summary Table**
   - Mean segment length per region
   - Total vascular length in sampled boxes

3. **Statistical Comparison**
   - p-values indicate whether differences are significant (typically α=0.05)
   - p < 0.05 = statistically significant difference
   - p ≥ 0.05 = no significant difference

4. **Visualizations**
   - Histograms: Show full distributions
   - Box plots: Display quartiles and outliers
   - Violin plots: Show detailed distribution shapes

### Example Interpretation

If comparison shows:
- **Diameter p-value = 0.0003**: Hippocampal vessels are significantly smaller (p < 0.05)
- **Length p-value = 0.4521**: No significant length difference (p > 0.05)

## API Reference

### BoxRegionAnalyzer Class

#### Methods

**`add_box_to_region(region_name, box_id, center_vox)`**
- Add a box to a region
- `region_name`: "hippocampal" or "somatomotor"
- `box_id`: unique identifier (e.g., "hippo_box_1")
- `center_vox`: [x, y, z] in voxels

**`analyze_region(region_name)`**
- Analyze all boxes in a region
- Returns: summary dictionary with statistics

**`print_region_summary(region_name)`**
- Print formatted summary to console

**`compare_regions(region_1, region_2)`**
- Compare two regions statistically
- Returns: comparison dictionary with test results

**`plot_comparison(region_1, region_2)`**
- Create comparison plots
- Returns: matplotlib figure

**`save_results(output_path)`**
- Save all analysis results to pickle file

## Output Files

When running the analysis, the following files are generated in `/home/admin/Ana/MicroBrain/output/`:

1. **box_analysis_results.pkl** - Complete analysis data (pickle format)
2. **diameter_summary.csv** - Diameter statistics by region
3. **length_summary.csv** - Length statistics by region
4. **analysis_report.txt** - Formatted text report
5. **region_comparison.png** - Comparison plots
6. **region_comparison_violin.png** - Violin plot comparison

## Advanced Usage

### Custom Box Size

```python
analyzer = BoxRegionAnalyzer(graph, box_size_um=(500, 500, 500), space="um")
```

### Different Coordinate Systems

```python
# Use micrometer coordinates instead of voxels
analyzer = BoxRegionAnalyzer(graph, coords_attr="coords_image_R", space="um")
```

### Accessing Raw Data

```python
# Get all diameter values for a region
hippo_diameters = analyzer.region_summary["hippocampal"]["all_diameters"]

# Perform custom statistical tests
from scipy.stats import mannwhitneyu
u_stat, p_val = mannwhitneyu(hippo_diameters, somato_diameters)
```

## Troubleshooting

### "No edges found in box"
- Box center coordinates may be outside the vascular network
- Verify coordinates in ParaView
- Check that box size encompasses vessel data

### TypeError with coordinates
- Ensure box centers are in voxels (not micrometers)
- Check that coordinates are arrays/lists with 3 elements

### Memory issues with large box
- Reduce box size
- Process fewer boxes at once

## References

The analysis uses vessel properties from the graph:
- **diameter**: Maximum diameter across each edge
- **edge length**: Total path length of vascular segment
- **nkind**: Vessel type classification
  - 2 = arteriole
  - 3 = venule
  - 4 = capillary

## Citation

If using this analysis tool in published work, cite:
- The graph analysis functions library: [graph_analysis_functions.py]
- The regional analysis framework: [box_region_analysis.py]

---

**Author**: Ana Barrio  
**Date**: February 2026  
**Contact**: [Your contact info]
