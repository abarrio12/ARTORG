# MicroBrain Codes - Vascular Graph Analysis Toolkit

## Overview

This repository contains a comprehensive toolkit for analyzing and processing **vascular graph networks** from brain imaging data. The project focuses on converting, analyzing, and visualizing blood vessel networks (arteries, veins, capillaries) from raw neuroimaging data into computational graph formats for detailed structural and topological analysis.

---

## 📁 Folder Structure

### 1. **CSVtoPKL/**
Converts CSV data files into Python pickle format containing igraph network objects and geometric data.

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
  
- **`build_outgeom_indexed.py`** - Creates indexed geometry representation (memory-efficient)
  - Builds pseudo-JSON layout with igraph topology + indexed numpy arrays
  - Computes robust tortuosity metrics for curved vessels
  - Chunked processing for large datasets
  
- **`outgeom_um.py`** - Geometry utilities for micrometer-scale vessel measurements

---

### 2. **cutting/**
Graph segmentation tools to extract subgraphs based on spatial regions (bounding boxes, anatomical areas).

**Key Scripts:**
- **`Cut_The_Graph_GAIA.py`** - Main cutting algorithm using GAIA methodology
  - Intersects vessel networks with 3D bounding boxes
  - Handles edge intersections and creates new nodes at box boundaries
  - Outputs subgraphs within specified regions
  
- **`Cut_The_Graph_MVN.py`** - Variant for MVN (Multi-Vascular Network) data
  - Region-specific graph extraction
  
- **`cut_box.py`** - Simple bounding box cutting utility
  - Extracts vessels within coordinate ranges
  
- **`cut_out.py`** - Outflow/region-specific cutting
  
- **`cut_outgeom_gaia_like.py`** - Geometry-aware GAIA-like cutting
  - Preserves edge geometries when cutting
  
- **`graph_cut_SOFIA.py`** - SOFIA's graph cutting implementation
  
- **`equivalent_non.py`** - Computes equivalent non-tortuous representations

---

### 3. **GTtoCSV/**
Converts graph-tool format files to CSV format for data interchange and analysis.

**Key Scripts:**
- **`gt2CSV_SOFIA.py`** - Main converter: graph-tool → CSV
  - Exports vertices, edges, and attributes to separate CSV files
  - Adapted from Renier dataset format
  
- **`create_graph_from_GT.py`** - Creates igraph objects from graph-tool files
  
- **`Franca_Extract_graph_info.py`** - Extracts metadata and properties from graph-tool objects

---

### 4. **PKLtoVTP/**
Converts pickle graph objects to VTP (VTK PolyData) format for visualization in ParaView.

**Key Scripts:**
- **`pkl2vtp_SOFIA.py`** - Main converter: pickle → VTP
  - Exports 3D vessel networks as VTK polydata
  - Handles vertex and edge attributes as scalars
  - Supports subgraph export
  
- **`Pkl2vtp_MVN_SOFIA.py`** - MVN-specific VTP export
  
- **`pkltovtp_ANA.py`** - Ana's variant with custom processing
  
- **`PKLtoVTP_nonT_basic.py`** - Non-tortuous variant (simplified geometries)
  
- **`PKLtoVTP_tortu_nontortu.py`** - Exports both tortuous and simplified versions
  
- **`PKLtoVTP_tortuous_FULLGEOM.py`** - Full geometry tortuous export
  
- **`PKLtoVTP_tortuous_OUTGEOM.py`** - Indexed geometry tortuous export
  
- **`test_edges_pairing.py`** - Utility for validating edge connectivity

---

### 5. **Graph Analysis & by region/**
Comprehensive tools for analyzing vascular network properties and spatial statistics.

**Key Notebooks:**
- **`Graph_analysis.ipynb`** - Main analysis notebook
  - Graph diameter, length, and degree statistics
  - Vessel type classification (arteriole, venule, capillary)
  
- **`HippocampalArea_graphAnalysis.ipynb`** - Analysis specific to hippocampal region
  - Region-specific statistics and visualizations
  
- **`SomatomotorArea_graphAnalysis.ipynb`** - Somatomotor cortex analysis
  
- **`Somatomotor_VS_Hippocampal_graphAnalysis.ipynb`** - Comparative analysis between regions
  
- **`Check_full_TnonT.ipynb`** - Compares tortuous vs non-tortuous variants
  
- **`nonT_T_diff.ipynb`** - Difference metrics between tortuous and non-tortuous networks
  
- **`test_analysis.ipynb`** - Testing and validation notebook

**Key Scripts:**
- **`graph_analysis_functions.py`** (2600+ lines) - Core analysis library
  - Classic graph metrics (diameter, length, degree distribution)
  - Bifurcation complexity (BC) detection and spatial analysis per anatomical face
  - Gaia-style vessel density calculations using micro-segments
  - Redundancy analysis via edge-disjoint path computation
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

3. **Structural Analysis**
   - Bifurcation complexity (BC) detection by anatomical boundary
   - Tortuosity metrics (vessel curvature quantification)
   - Redundancy analysis (alternative routes between vessel types)

4. **Vessel Classification**
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