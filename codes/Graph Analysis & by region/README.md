# Graph Analysis: Measure Vessel Networks

## Overview
This module computes comprehensive **topological, geometric, and spatial metrics** on vascular graphs. It supports regional analysis, enabling comparison across different brain areas and identification of structural features.

## Goal
Extract quantitative properties of vascular networks to:
1. Characterize vessel architecture (diameter distributions, branching patterns)
2. Detect structural boundaries and connectivity anomalies
3. Analyze redundancy via shortest path analysis between artery-vein pairs
4. Compute vascular density by region and vessel type
5. Compare metrics across brain regions (Hippocampal vs. Somatomotor, etc.)

## Folder Structure

```
Graph Analysis & by region/
├── Gaia_microbloom_compliance.py                    # GAIA dataset compliance
├── check_connectivity_nonTort&Tort.ipynb            # Connectivity validation
├── README.md
│
├── Graph analysis/
│   ├── graph_analysis_functions.py                  # Core analysis library
│   ├── Graph_analysis.ipynb                         # General analysis template
│   └── test_analysis.ipynb                          # Testing notebook
│
├── ROI Comparison/
│   ├── HippocampalArea_graphAnalysis.ipynb          # Hippocampus analysis
│   ├── SomatomotorArea_graphAnalysis.ipynb          # Somatomotor cortex analysis
│   └── Somatomotor_VS_Hippocampal_graphAnalysis.ipynb # Comparative study
│
└── SelectBrainRegion/
    ├── SelectBrainRegion_fromJSON.py                # Region selector from JSON
    └── SelectionBrainParaview_SOFIA.py              # ParaView integration
```

## Organization by Function

### Core Analysis Library
- **`Graph analysis/graph_analysis_functions.py`** - Main computational functions

### Analysis Workflows
- **`Graph analysis/Graph_analysis.ipynb`** - Template for general analysis
- **`Graph analysis/test_analysis.ipynb`** - Testing and validation

### Regional Analysis Notebooks
Located in **`ROI Comparison/`** for organized regional studies:
- `HippocampalArea_graphAnalysis.ipynb` - Hippocampal region
- `SomatomotorArea_graphAnalysis.ipynb` - Somatomotor cortex
- `Somatomotor_VS_Hippocampal_graphAnalysis.ipynb` - Comparative analysis

### Utility Tools
- **`Gaia_microbloom_compliance.py`** - GAIA dataset processing
- **`check_connectivity_nonTort&Tort.ipynb`** - Validation of tortuous/non-tortuous

### Region Selection Tools
Located in **`SelectBrainRegion/`**:
- `SelectBrainRegion_fromJSON.py` - Load anatomical regions from JSON
- `SelectionBrainParaview_SOFIA.py` - ParaView region integration

## Core Module: graph_analysis_functions.py

Located in `Graph analysis/`, this is the main computational library with functions grouped by analysis type:

### 1. Classic Graph Metrics
- **Edge length:** Straightforward edge length ("non tortuous edge" length)
- **Diameter:** Vessel cross-sectional diameter (from radii). Depending on the radii used (radii or radii_atlas), diameter will be atlas scaled or not.
- **Degree distribution:** Branching statistics (nodes with k connections)
- **Connectivity metrics:** How interconnected the network is
- **Loop detection:** Identify cycles in graph (edges that come back to the same place)

### 2. Boundary Condition (BC) Detection
- **Face-specific analysis:** Focus on specific cube faces (not whole box)
- **3D boundary identification:** Find surface vessels connecting inside/outside
- **Cube-net representation:** Planar unfolding of boundary structure
- **Boundary classification:** Arteries vs. veins at boundaries

### 3. Spatial Analysis
- **Vessel type maps:** Artery/vein spatial distributions
- **High-Degree Node (HDN) analysis:** Major hubs and junctions. Is there a pattern on HDN location? more in the borders or inside? what nkinds are usually HDN?
- **Diameter distribution:** Vessel size variations across space


### 4. Vessel Density 
- **Micro-segment based computation:** Divide edges in microsegments (point to point) and treat them as new independent edges
- **Volume fraction:** Fraction of space occupied by vessels
- **Slab-based analysis:** Density in horizontal layers
- **Regional aggregation:** Averaging across defined regions

### 5. Redundancy & Robustness Metrics
- **Shortest paths analysis:** Multiple independent paths between artery-vein pairs
- **Path-based connectivity:** Number of alternative routes from arteries to veins
- **Network redundancy:** Quantifies alternative pathways for blood circulation

## Analysis Parameters

### Space Convention
```python
space="vox"  # Use voxel coordinates (coords_image)
space="um"   # Use micrometer coordinates(coords_image_R)
```

### Distance Threshold (eps)
- **Always in voxels** (even if `space="um"`)
- Default: `eps=2` voxels ≈ 3.25 µm (using default 1.625 µm/vox)
- Used for defining neighborhood connectivity

### Resolution
```python
res_um_per_vox = (1.625, 1.625, 2.5)  # µm/voxel (source resolution)

atlas_resolution = 25 # µm/voxel (sink resolution)
```

**Note:** atlas attributes are also coords_image but scaled to the atlas resolution. In case um are wanted, simply multiply atlas attribute * 25. 

## Key Jupyter Notebooks

### General Analysis Templates
Located in **`Graph analysis/`**:

1. **Graph_analysis.ipynb** - General Template
   - Step-by-step graph loading and metric computation
   - Suitable as starting point for custom analysis

2. **test_analysis.ipynb** - Testing & Development
   - Experimental and validation workflows

### Regional Analysis Studies
Located in **`ROI Comparison/`**:

1. **HippocampalArea_graphAnalysis.ipynb** - Hippocampus Analysis
   - Region-specific analysis for hippocampal vasculature
   - Includes regional boundaries and subdivisions

2. **SomatomotorArea_graphAnalysis.ipynb** - Somatomotor Cortex
   - Motor/somatosensory cortex vascular properties
   - Comparison with baseline networks

3. **Somatomotor_VS_Hippocampal_graphAnalysis.ipynb** - Comparative Study
   - Side-by-side comparison of two regions
   - Highlights regional differences in:
     - Vascular density
     - Vessel size distribution
     - Branching complexity
     - Topological features

### Validation & Comparison
- **check_connectivity_nonTort&Tort.ipynb** - Tortuous vs non-tortuous connectivity validation
- **nonT_T_diff.ipynb** - Dataset compliance check
- **Check_full_TnonT.ipynb** - Data integrity validation


## Data Format

Your PKL file has this structure:

```python
data = {
    "graph": igraph.Graph,        # network topology
    
    "vertex": {
        "coords": array,          # vertex positions (atlas voxels)
        "coords_image": array,    # vertex positions (image voxels)
        "radii_atlas": array,     # radius in atlas space
        "annotation": array,      # region ID
    },
    
    "geom": {
        "x", "y", "z": arrays,    # polyline points (voxels)
        "lengths2": array,        # distance between consecutive points
        "radii_atlas_geom": array,# radius at each point
        "annotation": array,      # region ID at each point
    }
}
```

## Author
Ana Barrio - Feb 2026
