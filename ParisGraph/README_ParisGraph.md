# Vascular Graph Analysis Toolkit

## Overview

This is a toolkit for processing and analysing vascular graph networks from brain imaging data.

The repository contains workflows to:

- convert raw graph data into Python analysis formats (`.pkl`);
- preserve tortuous vessel geometry;
- convert between voxel/image coordinates and physical units;
- extract regional subgraphs from 3D boxes;
- analyse vascular properties by vessel type or brain region;
- export graph data to VTP for visualization in ParaView.

The main focus is vascular graph analysis across whole-brain, especifically Hippocampal Area and Somatomotor Cortex Region.

---

## Repository structure

```text
ParisGraph/
├── codes/
├── outputs/
└── whole brain annotation/
    └── Extract Area Annotated from graph Brain/
```

---

## Main folders

## `codes/` — General vascular graph toolkit

This is the main reusable codebase for graph conversion, cutting, analysis, and visualization.

```text
codes/
├── CSVtoPKL/
├── GTtoCSV/
├── Graph Analysis & by region/
├── PKLtoVTP/
└── cutting/
```

### `codes/CSVtoPKL/`

Converts CSV vessel tables into Python graph objects.

Typical outputs are `.pkl` files containing `igraph` graphs and/or geometry dictionaries.

Main tasks:

- read vertices, edges, coordinates, radii, annotations, and vessel labels;
- build graph topology;
- preserve tortuous vessel geometry;
- store edge and vertex attributes;
- convert from voxel/image units to micrometers when needed.

Common scripts include:

```text
CSVtoPKL_tortuous.py
vox_to_um.py
save_dict_format.py
```

---

### `codes/GTtoCSV/`

Converts graph-tool data into CSV files.

Use this when the input data is in graph-tool format and you want spreadsheet-style files for later conversion.

---

### `codes/cutting/`

Tools for extracting vessels inside a rectangular 3D region of interest.

The cutting workflow can:

- keep vessels fully inside the selected box;
- detect vessels crossing the box boundary;
- optionally trim crossing vessels at the box surface;
- save a new smaller graph for downstream analysis.


---

### `codes/PKLtoVTP/`

Converts graph `.pkl` files into `.vtp` files for ParaView.

Use this folder when you want to visualize vessels in 3D.

Decide the type of export you want: 
    - If tortuous, on write_vtp(...., True)
    - If straight/non-tortuous, on write_vtp(..., False)


---

### `codes/Graph Analysis & by region/`

Analysis notebooks and helper functions for vascular metrics.

Typical analyses:

- vessel diameter distributions;
- vessel length distributions;
- node degree and topology;
- vessel density;
- artery/vein/capillary comparisons;
- regional analysis by annotation;
- redundancy and connectivity analysis;
- comparison between brain regions such as hippocampus and somatomotor cortex.

Important files include:

```text
graph_analysis_functions_formatted.py
vascular_graph_analysis.ipynb (used for intra-regional analysis)
region_comparison.ipynb (used for inter-regional analysis)
```

---

## `Paris Calculations/` — from ClearMap GitHub
It includes documentation and calculations related to the Paris/ClearMap graph format, including unit handling and diameter/length definitions.


## `whole brain annotation/` 
- **`Extract Area Annotated from graph Brain/`** - Information about brain regions
  - `ABA_annotation_last.json` - Maps vessels to standard brain areas (Allen Brain Atlas)





## Workflow

```text
Raw graph data
    ↓
GTtoCSV/ or CSV files
    ↓
CSVtoPKL/
    ↓
igraph / PKL graph
    ↓
unit conversion to µm
    ↓
cutting / ROI extraction
    ↓
graph analysis or VTP export
    ↓
ParaView visualization / statistics / figures
```

---

## Units and coordinate systems

This project uses several coordinate systems. Always check which one your file uses.

### Image coordinates

Image coordinates come from the original imaging grid.

For ParisGraph:

```text
X = 1.625 µm/voxel
Y = 1.625 µm/voxel
Z = 2.5 µm/voxel
```

### Atlas coordinates

Atlas coordinates refer to the atlas grid.

Typical atlas resolution:

```text
25 µm/voxel
```

### Micrometers

Physical units are used after explicit conversion.

For ParisGraph, this means:

```text
coords_um = coords_vox * [1.625, 1.625, 2.5]
points_um = points_vox * [1.625, 1.625, 2.5]
```

Lengths should be recomputed from scaled points rather than multiplied by an average factor, because voxel spacing is anisotropic.

---

## ParaView warning

ParaView does not know whether coordinates are voxels, atlas units, or micrometers.

It only displays the numbers stored in the file.

Before defining a box, cutting a graph, or comparing two datasets, check:

```text
1. coordinate space
2. unit
3. resolution
4. whether the graph is local crop space or global whole-brain space (may have to add an offset to have same origin coordinates)
```

Note that the graph obtained from CSVtoPKL_tortuous.py, stores metadata of the units!

---

## Author

Ana Barrio

---

Last updated: April 2026
