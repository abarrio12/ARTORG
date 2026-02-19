# MicroBrain Codes - Vascular Graph Analysis Toolkit

## Overview

This repository contains a comprehensive toolkit for analyzing and processing **vascular graph networks** from brain imaging data. The project focuses on converting, analyzing, and visualizing blood vessel networks (arteries, veins, capillaries) from raw neuroimaging data into computational graph formats for detailed structural and topological analysis.

---

## 📁 Folder Structure

### 1. **CSVtoPKL/** — CSV to Pickle Conversion
Converts CSV data files into Python pickle format containing igraph network objects and geometric data.

📖 **[→ Detailed Guide](CSVtoPKL/README.md)** - Complete documentation with workflow explanation

**Key Scripts:**
- **`CSV2Pickle_SOFIA.py`** - Main script to convert raw CSV files into igraph pickle format
  - Reads vertex and edge data with attributes (coordinates, radii, vessel type labels)
  - Supports both atlas coordinates and image coordinates
  - Creates igraph objects with vertex/edge attributes
  
- **`CSVtoPKL_FULLGEOM.py`** - Converts CSVs into pickle with full geometry preservation
  - Stores complete edge geometry polylines
  - Maintains coordinate sequences for curved vessel segments
  
- **`CSVtoPKL_Tortuous_OutGeom.py`** - Handles tortuous (curved) vessel geometry
  - Optimized for sinuous vessel paths
  - Preserves geometry indices for efficiency
  
- **`build_graph_outgeom_voxels.py`** - Creates indexed geometry representation with outer geometry
  - Builds graph with igraph topology + indexed numpy arrays
  - Computes robust tortuosity metrics for curved vessels
  
- **`convert_outgeom_voxels_to_um.py`** - Converts voxel space to micrometers (µm)
  - Non-destructive unit conversion
  - Stores both voxel and physical-unit data

---

### 2. **cutting/** — Graph ROI Extraction & Clipping
Graph segmentation tools to extract subgraphs based on spatial regions (bounding boxes, anatomical areas).

📖 **[→ Detailed Guide](cutting/README.md)** - Complete documentation with clipping algorithm explanation

**Key Scripts:**
- **`cut_outgeom_roi_UM.py`** - Main cutting algorithm in micrometers (recommended)
  - Intersects vessel networks with 3D bounding boxes
  - Handles edge intersections and clips at box boundaries
  - Creates new nodes at intersections and outputs clipped subgraphs
  
- **`cut_outgeom_roi_VOX.py`** - Variant in voxel space
  - Same functionality as UM version but uses voxel coordinates
  
- **`Cut_The_Graph_GAIA.py`** - GAIA-specific region extraction
  - GAIA dataset methodology
  
- **`Cut_The_Graph_MVN.py`** - MVN (Multi-Vascular Network) variant
  - Region-specific graph extraction for MVN data
  
- **`cut_box.py`** - Simple bounding box cutting utility
  - Basic rectangular region extraction
  
- **`cut_out.py`** - Complementary cutting
  - Extract everything outside a region
  
- **`equivalent_non — Graph-Tool to CSV Conversion
Converts graph-tool format files to CSV format for data interchange and analysis.

📖 **[→ Detailed Guide](GTtoCSV/README.md)** - Complete documentation with format specifications

**Key Scripts:**
- **`gt2CSV_SOFIA.py`** - Main converter: graph-tool → CSV
  - Exports vertices, edges, and attributes to separate CSV files
  - Adapted from Renier dataset format
  - Includes edges, coordinates, radii, annotations
  
- **`create_graph_from_GT.py`** - Creates igraph objects from graph-tool files
  - Helper utilities for graph loading
  
- **`Franca_Extract_ — Pickle to ParaView Visualization
Converts pickle graph objects to VTP (VTK PolyData) format for visualization in ParaView.

📖 **[→ Detailed Guide](PKLtoVTP/README.md)** - Complete documentation with VTP format and ParaView usage guide

**Key Scripts:**
- **`pkl2vtp_SOFIA.py`** - Main converter: pickle → VTP (recommended)
  - Exports 3D vessel networks as VTK polydata
  - Handles vertex and edge attributes as scalars
  - Supports subgraph export and color-mapping
  
- **`PKLtoVTP_tortuous_OUTGEOM.py`** - Tortuous path export (natural curvature)
  - Full polyline geometry preservation
  - Best for anatomical visualization
  
- **`PKLtoVTP_nonT_basic.py`** - Simplified non-tortuous variant
  - Straight-line representation (vertices only)
  - Fast rendering, topology emphasis
  
- **`PKLtoVTP_tortu_nontortu.py`** - Exports both variants
  - Comparative visualization
  
- **`Pkl2vtp_MVN_SOFIA.py`** - MVN-specific VTP export
  - MVN dataset methodology
  
- **`pkltovtp_ANA.py`** - Ana's varian — Vascular Network Metrics & Analysis
Comprehensive tools for analyzing vascular network properties and spatial statistics.

📖 **[→ Detailed Guide](Graph%20Analysis%20%26%20by%20region/README.md)** - Complete documentation with analysis parameters and workflow

**Key Notebooks:**
- **`Graph_analysis.ipynb`** - Main analysis template
  - Graph diameter, length, and degree statistics
  - Vessel type classification (arteriole, venule, capillary)
  - General workflow example for custom analysis
  
- **`HippocampalArea_graphAnalysis.ipynb`** - Hippocampus region-specific analysis
  - Hippocampal vasculature properties
  - Regional boundary definition and statistics
  
- **`SomatomotorArea_graphAnalysis.ipynb`** - Somatomotor cortex analysis
  - Motor cortex vascular characterization
  
- **`Somatomotor_VS_Hippocampal_graphAnalysis.ipynb`** - Comparative regional study
  - Side-by-side comparison of two brain regions
  - Highlights structural differences
  
- **`nonT_T_diff.ipynb`** - Tortuous vs non-tortuous comparison
  - Structural distinctions between vessel types
  
- **`Check_full_TnonT.ipynb`** - Full dataset validation
  - Data integrity checks and summary statistics

**Key Scripts:**
- **`graph_analysis_functions.py`** (1964+ lines) - Core analysis library
  - Classic metrics: diameter, length, degree distribution
  - Boundary condition (BC) detection per anatomical face
  - Vessel density (Gaia-style micro-segment calculation)
  - Redundancy analysis (edge-disjoint paths / maxflow)
  - Depth stratification (superficial vs deep)
  - Spatial filtering and regional aggregation
  - Comprehensive visualization functions
  
- **`cube_analysis.py`** - Cubic region analysis utilities
  - Statistics for cubic subvolumes
  - Spatial hotspot identification
  
- **`SelectBrainRegion_fromJSON.py`** - JSON-based region selection
  - Anatomical region def — Experimental & Development
Experimental and testing code for prototyping and validation.

📖 **[→ Detailed Guide](my%20test%20codes/README.md)** - Status of experimental code and promotion guidelines

⚠️ **Note:** Code in this folder is **NOT production-ready**. Use for experimentation only.

- **`bound_AB.py`** - Boundary detection experimentation
  - Test implementation of interface detection
  
- **`CSVtoPickle_Tortuous.py`** - Alternative tortuous conversion
  - Experimental approach to tortuous CSV→pickle
  - Source Data (Paris/ClearMap)
    ↓
[GTtoCSV] (optional: graph-tool → CSV)
    ↓
CSV Files (../CSV/)
    ↓
[CSVtoPKL] → Build graph
    ├─ build_graph_outgeom_voxels.py     (voxel space)
    └─ convert_outgeom_voxels_to_um.py   (→ micrometers)
    ↓
PKL Files (../output/)
    ├─ graph_*.pkl (complete vessel network)
    │
    ├─ [Optional: cutting/]  (extract region of interest)
    │   └─→ cut_outgeom_roi_UM.py
    │       ↓
    │   data_cut (subgraph)
    │
    ├─→ [Graph Analysis & by region/]
    │   ├─ graph_analysis_functions.py  (compute metrics)
    │   └─ *.ipynb                       (regional analysis)
    │   ↓
    │   Results (tables, figures, statistics)
    │
    └─→ [PKLtoVTP/]       (visualization)
        ├─ pkl2vtp_SOFIA.py
        └─ *.vtp files
        ↓
        [ParaView] → 3D interactive visualizationputation
  - Depth-based stratification (superficial vs deep regions)
  - Radii consistency sanity checks
  - Comprehensive visualization functions (3D plots, heatmaps, degree distributions)
  
- **`cube_analysis.py`** - Boxed region analysis utilities
  - Extracts statistics for cubic subvolumes
  
- **`SelectBrainRegion_fromJSON.py`** - Region selection from JSON atlas data
  - Anatomically-guided subgraph extraction
  
- **`SelectionBrainParaview_SOFIA.py`** - ParaView-compatible region selection
  
- **`Untitled1.ipynb`** - Development/experimental notebook

---

### 6. **my test codes/**
Experimental and testing code for development purposes.

- **`bound_AB.py`** - Boundary/AB testing utilities
- **`CSVtoPickle_Tortuous.py`** - Experimental tortuous CSV→pickle conversion
- **`Graph_analysis.ipynb`** - Local analysis testing

---

## 🔄 Data Conversion Pipeline

```
Raw CSV Data
    ↓
CSVtoPKL/ (CSV → igraph pickle)
    ↓
    └─→ Graph Analysis & by region/
    └─→ cutting/ (extract subregions)
    └─→ PKLtoVTP/ (pickle → VTP for visualization)
```

---

## 📊 Data Attributes

### Vertex Properties:
- Coordinates (atlas space and image space)
- Radii (vessel diameter)
- Brain region annotation

### Edge Properties:
- Length (vessel segment length in µm)
- Radii (vessel radii along edge)
- Topology (arteriole, venule, capillary classification)
- Tortuosity metrics (straight distance vs actual path length)
- Geometry (full polyline coordinates or indexed references)

---

## 🔬 Analysis Capabilities

1. **Topological Analysis**
   - Network diameter and characteristic path length
   - Degree distribution and centrality measures
   - Clustering coefficients

2. **Spatial Analysis**
   - Vessel density (local density of vessel segments)
   - Depth stratification (superficial vs deep classification)
   - Spatial distribution maps by vessel type
� Getting Started

1. **Read the module-specific README first** - Each subfolder has a detailed guide:
   - [CSVtoPKL/README.md](CSVtoPKL/README.md)
   - [cutting/README.md](cutting/README.md)
   - [Graph Analysis & by region/README.md](Graph%20Analysis%20%26%20by%20region/README.md)
   - [GTtoCSV/README.md](GTtoCSV/README.md)
   - [PKLtoVTP/README.md](PKLtoVTP/README.md)

2. **Choose your workflow** - Start from input data type:
   - Have **CSV files**? → [CSVtoPKL](CSVtoPKL/README.md)
   - Have **PKL files**? → [Graph Analysis](Graph%20Analysis%20%26%20by%20region/README.md) or [cutting](cutting/README.md)
   - Want **visualization**? → [PKLtoVTP](PKLtoVTP/README.md)
   - Have **graph-tool files**? → [GTtoCSV](GTtoCSV/README.md)

3. **Check dependencies** - Ensure you have: `igraph`, `numpy`, `pandas`, `matplotlib`, `lxml`

---

## 📌 Key Authors & Variants

- **SOFIA** - Main data processing and conversion pipeline
- **GAIA** - Graph cutting methodology and region extraction (cutting/ module)
- **Ana** - Memory-efficient geometry indexing, micrometers conversion, analysis functions
- **MVN** - Multi-vascular network dataset variants
- **Franca** - Original dataset processing framework (GTtoCSV)
   - Arteriole, Venule, Capillary identification
   - Arteriovenous connection mapping

---

## 🛠️ Main Dependencies

- **igraph** - Graph structure and algorithms
- **graph-tool** - Advanced graph analysis
- **pandas** - Data manipulation (CSV handling)
- **numpy** - Numerical computations
- **matplotlib** - Visualization
- **lxml** - VTP/XML file generation
- **ParaView** - 3D visualization (external)

---

## 📝 Usage Examples

### Convert CSV to pickled igraph:
```python
from CSVtoPKL.CSV2Pickle_SOFIA import ReadWriteGraph
# Creates igraph with vessel network from CSV files
```

### Cut graph by region:
```python
from cutting.Cut_The_Graph_GAIA import cut_by_bounding_box
# Extracts subgraph within spatial bounds
```

### Analyze graph properties:
```python
from Graph_Analysis import graph_analysis_functions
# Computes network metrics, visualizations, regional statistics
```

### Export to ParaView:
```python
from PKLtoVTP.pkl2vtp_SOFIA import ExportToVTP
# Converts igraph pickle to VTP for 3D visualization
```

---

## 📌 Key Authors & Variants

- **SOFIA** - Main data processing and conversion pipeline
- **GAIA** - Graph cutting methodology and region extraction
- **Ana** - Memory-efficient geometry indexing and optimization
- **MVN** - Multi-vascular network variants
- **Franca** - Original dataset processing framework

---

## 📂 Related Directories

- **`../CSV/`** - Input CSV data files (vertices, edges, attributes)
- **`../output/`** - Generated VTP files and processed graphs
- **`../blood flow solver/`** - Hemodynamic simulations on networks

---

## 🔍 Notes

- All coordinates are in **micrometer (µm)** unless otherwise specified
- Graph files typically represent **Graph 18 (full brain hemisphere)**
- Both **atlas coordinates** and **image coordinates** are maintained for alignment
- **Tortuosity** is quantified as the ratio of actual path length to Euclidean distance
- **Vessel density** uses micro-segment calculations for accurate local measurements

---

*Last updated: February 2026*
*Project: MicroBrain Analysis Toolkit* 