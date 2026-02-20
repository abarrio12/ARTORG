# CSVtoPKL: CSV to Pickle Conversion

## Overview
This module converts raw CSV data into structured **Pickle (.pkl) files** containing the complete vascular graph with geometric information, vertex attributes and edge properties.

## Goal
Transform tabular CSV data (from the `CSV/` folder) into Python-serialized pickle objects that preserve:
- Graph topology (nodes and edges)
- Geometric information (coordinates, vessel paths)
- Physical properties (radii, length, annotation)
- Both voxel-space and micrometer-space representations

## Folder Structure

```
CSVtoPKL/
├── build_graph_outgeom_voxels.py        # Build graph with outer geometry (voxel space)
├── convert_outgeom_voxels_to_um.py      # Convert voxel coordinates to micrometers (.pkl from previous)
├── CSV2Pickle_SOFIA.py                  # Basic CSV → PKL conversion (SOFIA variant)
├── CSVtoPKL_FULLGEOM.py                 # Full geometry variant (stores info in edges/vertex)
└── README.md
```

## Processing Pipeline

### 1. **CSV2Pickle_SOFIA.py** - Basic Conversion
- Reads edges and vertices from CSV files
- Creates basic graph structure
- Stores minimal vertex/edge attributes

### 2. **build_graph_outgeom_voxels.py** - Graph with Outer Geometry (Voxel Space)

Builds an igraph.Graph from CSV data with complete tortuous geometry information. The output is a Python dict (pickle-serialized) containing the network topology and all geometric information.

**Output structure (voxel space):**
```
data
├── graph                       # igraph.Graph object
│   ├── vertices (N total)
│   ├── edges (E total)
│   └── edge attributes:
│       ├── geom_start         # Index into geom polyline (start point)
│       ├── geom_end           # Index into geom polyline (end point)
│       ├── length             # Arc length in voxels
│       └── ... (other edge properties)
│
├── vertex                      # Vertex-level attributes (N entries each)
│   ├── coords_image            # Coordinates (N × 3) array in voxels
│   ├── radii_atlas             # Atlas-space radii (N,) array
│   ├── distance_to_surface     # Distance to surface (N,) array
│   └── ... (other vertex attributes)
│
└── geom                        # Geometry polyline points (all N_points total)
    ├── x                       # X coordinates of all polyline points (M,) array
    ├── y                       # Y coordinates of all polyline points (M,) array
    ├── z                       # Z coordinates of all polyline points (M,) array
    ├── lengths2                # Distance to next point; last = 0 (M,) array
    ├── radii_atlas_geom        # Radii at each polyline point (M,) array
    ├── annotation              # Vessel type at each point (M,) array
    └── ... (other per-point properties)
```

**Key insight:** 
- `graph.es["geom_start"]` and `graph.es["geom_end"]` are **indices** into the `geom` arrays
- To get all polyline points for edge `i`: `geom["x"][geom_start[i]:geom_end[i]]`
- Each polyline can have many points (M >> E)
- This preserves the true tortuous (curved) vessel paths

### 3. **convert_outgeom_voxels_to_um.py** - Voxel to Micrometer Conversion
- Converts voxel-space coordinates to physical units (micrometers)
- **Keeps original voxel data intact** (non-destructive)
- Stores converted data with suffix `_R` (for "real-world" units)
- Computes per-segment lengths2 and length as per-edge (sum(lengths2)) in µm
- Calculates tortuosity (dimensionless: length / straight_distance)
- Uses conversion factors:
  - Source resolution: (1.625, 1.625, 2.5) µm/voxel # (image resolution)
  - Radii atlas: 25 µm/voxel                        # (atlas resolution)

**Output structure:**
```
data
├── geom              # original voxel data (unchanged)
├── vertex            # original voxel data (unchanged)
├── geom_R            # physical units (µm)
│   ├── x_R, y_R, z_R
│   ├── lengths2_R    # per-segment lengths
│   ├── radii_atlas_geom_R
│   └── diameters_atlas_geom_R
├── vertex_R          # physical units (µm)
│   ├── coords_image_R
│   ├── distance_to_surface_R
│   └── radii_atlas_R
└── graph            # igraph.Graph object
    ├── length_R     # per-edge length (µm)
    └── tortuosity_R # dimensionless
```

### 4. **CSVtoPKL_FULLGEOM.py** - Full Geometry Variant
- Alternative implementation for complete geometric information
- In voxels 
- Stores information in edges/vertex. If the graph is large, processing may use excessive memory (this is why outer geometry was implemented)


## Workflow

```
CSV Files (in ../CSV/)
    ↓
[build_graph_outgeom_voxels.py]  → graph_*.pkl (voxel space)
    ↓
[convert_outgeom_voxels_to_um.py] → graph_*_um.pkl (physical units)
    ↓
Ready for: cutting, analysis, visualization
```

## Input Requirements

Original CSV files in `../CSV/`:
- `vertices.csv` - Vertex coordinates
- `edges.csv` - Edge connectivity
- `radii_atlas.csv` - Vessel radii in atlas space
- `radii_edge.csv` - Edge radii
- `annotation.csv` - Vessel type annotations
- `coordinates.csv` - Vertex coordinates
- `edge_geometry_*.csv` - Per-point geometric information

## Output Files

Generated pickle files in `output/`:
- `graph_*.pkl` - Basic voxel-space graph
- `graph_*_um.pkl` - Physical units (micrometers) with full attributes

## Usage Example

**Step 1: Build graph from CSV in voxels**

Edit the hardcoded paths in `build_graph_outgeom_voxels.py`:
```python
FOLDER = "/home/admin/Ana/MicroBrain/CSV/"
OUT_PATH = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
```

Then run:
```bash
python build_graph_outgeom_voxels.py
```

This creates `graph_18_OutGeom.pkl` (voxel space).

**Step 2: Convert to micrometers**

Edit the paths in `convert_outgeom_voxels_to_um.py`:
```python
in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
out_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl"
```

Then run:
```bash
python convert_outgeom_voxels_to_um.py
```

This creates `graph_18_OutGeom_um.pkl` (micrometers).

**Optional: Load and verify the output in Python**

```python
import pickle
import numpy as np

# Load voxel-space graph
data_vox = pickle.load(open("/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl", "rb"))
print(f"Vertices: {data_vox['graph'].vcount()}")
print(f"Edges: {data_vox['graph'].ecount()}")
print(f"Geometry points: {len(data_vox['geom']['x'])}")

# Load micrometer-space graph
data_um = pickle.load(open("/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_um.pkl", "rb"))
print(f"\nAfter conversion to µm:")
print(f"Vertices: {data_um['graph'].vcount()}")
print(f"Edge lengths (µm): {data_um['graph'].es['length_R'][:5]}...")
```

## Important Notes

- **Radii conversion:** Only atlas radii are converted (atlas grid = 25 µm/voxel). Original image-space radii are NOT converted here due to scaling ambiguities (not sure is as trivial as just multiplying by image resolution, giving it is isotropic).
- **Non-destructive:** Original voxel arrays are preserved, allowing flexibility.
- **Tortuosity:** Calculated only for edges where straight-line distance ≥ min_straight_dist_um (sanity check).

## Author
Ana Barrio - Feb 2026
