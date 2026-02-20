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
  - Computes robust tortuosity metrics
  - Optimized sanity checks (lengths2 computed once, reused for validation)
  
- **`convert_outgeom_voxels_to_um.py`** - Converts voxel space to micrometers (µm)
  - Non-destructive unit conversion
  - Stores both voxel and physical-unit data

---

### 2. **cutting/** — Graph ROI Extraction & Clipp
ing
Graph segmentation tools to extract subgraphs based on spatial regions (bounding boxes, anatomical areas).

📖 **[→ Detailed Guide](cutting/README.md)** - Complete documentation with clipping algorithm explanation

**Key Scripts:**
- **`cut_outgeom_roi_UM.py`** - Main cutting algorithm in micrometers (recommended)
  - Intersects vessel networks with 3D bounding boxes
  - Handles edge intersections and clips at box boundaries
  - Creates new nodes at intersections and outputs clipped subgraphs
  
- **`cut_outgeom_roi_VOX.py`** - Variant in voxel space
  - Same functionality as UM version but uses voxel coordinates
  
- **`Cut_The_Graph_GAIA.py`** - GAIA dataset region extraction
  - Two variants: Full edge clipping and classification-only approaches
  - Handles complex edge-boundary intersections
  
- **`Cut_The_Graph_MVN.py`** - MVN (Multi-Vascular Network) variant
  - Region-specific graph extraction for MVN data
  
- **`cut_box.py`** - Simple bounding box cutting utility
  - Basic rectangular region extraction
  
- **`cut_out.py`** - Complementary cutting
  - Extract everything outside a region

---

### 3. **GTtoCSV/** — Graph-Tool to CSV Conversion
Converts graph-tool format files to CSV format for data interchange and analysis.

📖 **[→ Detailed Guide](GTtoCSV/README.md)** - Complete documentation with format specifications

**Key Scripts:**
- **`gt2CSV_SOFIA.py`** - Main converter: graph-tool → CSV
  - Exports vertices, edges, and attributes to separate CSV files
  - Includes edges, coordinates, radii, annotations
  
- **`create_graph_from_GT.py`** - Creates igraph objects from graph-tool files
  - Helper utilities for graph loading
  
- **`Franca_Extract_graph_info.py`** - Extract metadata from graph-tool format

---

### 4. **PKLtoVTP/** — Pickle to ParaView Visualization
Converts pickle graph objects to VTP (VTK PolyData) format for visualization in ParaView.

📖 **[→ Detailed Guide](PKLtoVTP/README.md)** - Complete documentation with VTP format and ParaView usage guide

**Key Scripts:**
- **`pkl2vtp_SOFIA.py`** - Main converter: pickle → VTP (recommended)
  - Exports 3D vessel networks as VTK polydata
  - Handles vertex and edge attributes as scalars
  - Supports subgraph export and color-mapping
  
- **`PKLtoVTP_Tortuous_OUTGEOM.py`** - Tortuous path export with full polyline geometry
  - Preserves complete curved vessel paths
  - Exports PointData: annotation, radii_p_atlas, diameter_p_atlas, lengths2
  - Exports CellData: radius_atlas, diameter_atlas, length, tortuosity, nkind
  - Best for anatomical visualization
  
- **`PKLtoVTP_nonT_basic.py`** - Simplified non-tortuous variant (vertices only)
  - Straight-line representation between vertices
  - Fast rendering, topology emphasis
  
- **`PKLtoVTP_tortu_nontortu.py`** - Exports both variants
  - Comparative visualization
  
- **`Pkl2vtp_MVN_SOFIA.py`** - MVN-specific VTP export
  - MVN dataset methodology

---

### 5. **Graph Analysis & by region/** — Vascular Network Metrics & Analysis
Comprehensive tools for analyzing vascular network properties and spatial statistics.

📖 **[→ Detailed Guide](Graph%20Analysis%20%26%20by%20region/README.md)** - Complete documentation with analysis parameters and workflow

**Key Notebooks:**
- **`Graph_analysis.ipynb`** - Main analysis template
  - Graph diameter, length, and degree statistics
  - Vessel type classification
  - Workflow example for custom analysis
  
- **`HippocampalArea_graphAnalysis.ipynb`** - Hippocampus region analysis
  - Regional boundary definition and statistics
  
- **`SomatomotorArea_graphAnalysis.ipynb`** - Somatomotor cortex analysis
  - Motor cortex vascular characterization
  
- **`Somatomotor_VS_Hippocampal_graphAnalysis.ipynb`** - Compare two regions
  - Side-by-side regional comparison
  - Highlights structural differences
  
- **`nonT_T_diff.ipynb`** - Compliance check between datasets
  - Data integrity and format validation
  
- **`Check_full_TnonT.ipynb`** - Data integrity check
  - Full dataset validation and summary

**Key Scripts:**
- **`graph_analysis_functions.py`** (1964+ lines) - Core analysis library
  - Classic metrics: diameter, length, degree distribution
  - Boundary condition (BC) detection per anatomical face
  - Vessel density (Gaia-style micro-segment calculation)
  - Redundancy analysis (edge-disjoint paths / maxflow)
  - Depth stratification (superficial vs deep)
  - Spatial filtering and regional aggregation
  - Comprehensive visualization functions
  
- **`SelectBrainRegion_fromJSON.py`** - Region selection from JSON atlas data
  - Anatomically-guided subgraph extraction
  
- **`SelectionBrainParaview_SOFIA.py`** - ParaView-compatible region selection
  
---

## Pipeline

```
Source Data (Paris/ClearMap)
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
    │   └─→ cut_outgeom_roi_VOX/UM.py
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
        ├─ pkl2vtp_Tortuous_OUTGEOM.py
        └─ *.vtp files
        ↓
        [ParaView] → 3D interactive visualization
```

---

## 📊 Data Attributes

### Vertex Properties:
- **Coordinates** - Position in atlas space and image space
- **Radii** - Vessel radius (atlas) at vertex 
- **Annotation** - Brain region label

### Edge Properties:
- **Length** - Vessel segment length (sum(lengths2))
- **Radii** - Vessel segment radii (max(radii points))
- **Tortuosity** - Curvature metric (straight distance vs actual path length)
- **Nkind** - Arteriole, venule, or capillary type
- **Geometry indices** - References to polyline coordinates (`geom_start`, `geom_end`)

### Geometry Data:
- **Polyline coordinates** - Full 3D path of each vessel (x, y, z points)
- **Per-point radii** - Vessel radius varies along the path (`radii_atlas_geom`)
- **Per-point distances** - Distance between consecutive points (`lengths2`)



---

## 🔍 Getting Started

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

## 🛠️ Main Dependencies

- **igraph** - Graph structure and algorithms
- **graph-tool** - Advanced graph analysis
- **pandas** - Data manipulation (CSV handling)
- **numpy** - Numerical computations
- **matplotlib** - Visualization
- **lxml** - VTP/XML file generation
- **ParaView** - 3D visualization (external)

---

## 📌 Key Authors & Variants

- **Sofia** - Main data processing and region extraction
- **Gaia** - Graph cutting methodology
- **Franca** - Original dataset processing gt
- **Ana** - Conversion pipeline and building tortuous graph

---

## 🔍 Notes

- All coordinates are in **voxels (vox)** unless otherwise specified
- Graph files typically represent **Graph 18 (full brain hemisphere)**
- Both **atlas coordinates** and **image coordinates** are maintained for alignment
- **Tortuosity** is quantified as the ratio of actual path length to Euclidean distance
- **Vessel density** uses micro-segment calculations for accurate local measurements

---

*Last updated: February 2026*
