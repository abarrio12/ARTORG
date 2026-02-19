# CSVtoPKL: CSV to Pickle Conversion

## Overview
This module converts raw CSV data into structured **Pickle (.pkl) files** containing the complete vascular graph with geometric information, vertex attributes, and edge properties.

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
├── convert_outgeom_voxels_to_um.py      # Convert voxel coordinates to micrometers
├── CSV2Pickle_SOFIA.py                  # Basic CSV → PKL conversion (SOFIA variant)
├── CSVtoPKL_FULLGEOM.py                 # Full geometry variant
├── CSVtoPKL_Tortuous_OutGeom.py         # Tortuous path handling
└── README.md
```

## Processing Pipeline

### 1. **CSV2Pickle_SOFIA.py** - Basic Conversion
- Reads edges and vertices from CSV files
- Creates basic graph structure
- Stores minimal vertex/edge attributes

### 2. **build_graph_outgeom_voxels.py** - Graph with Outer Geometry
- Builds an igraph.Graph from CSV data
- Includes "outer geometry" (tortuous polyline paths for each edge)
- Stores polyline points (x, y, z coordinates) for each vessel
- Creates `data["geom"]` dictionary with per-point attributes
- Maintains vessel annotations and connectivity

**Key attributes stored:**
- Vertices: `coords_image`, `radii_atlas`, `distance_to_surface`
- Geometry: `x, y, z` (polyline points), `radii_atlas_geom`, `annotation`
- Edges: `geom_start`, `geom_end` (indices into polyline), `length`

### 3. **convert_outgeom_voxels_to_um.py** - Voxel to Micrometer Conversion
- Converts voxel-space coordinates to physical units (micrometers)
- **Keeps original voxel data intact** (non-destructive)
- Stores converted data with suffix `_R` (for "real-world" units)
- Computes per-segment lengths in µm
- Calculates tortuosity (dimensionless: arc_length / straight_distance)
- Uses conversion factors:
  - Source resolution: (1.625, 1.625, 2.5) µm/voxel
  - Radii atlas: 25 µm/voxel

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
    ├── length_R     # per-edge arc length (µm)
    └── tortuosity_R # dimensionless
```

### 4. **CSVtoPKL_FULLGEOM.py** - Full Geometry Variant
- Alternative implementation for complete geometric information
- May include additional spatial information

### 5. **CSVtoPKL_Tortuous_OutGeom.py** - Tortuous Path Handling
- Specialized version handling tortuous (curved) vessel paths
- Preserves natural vessel geometry (not straight-line approximation)

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

```python
# Step 1: Build graph from CSV
from build_graph_outgeom_voxels import build_graph_outgeom_indexed
data = build_graph_outgeom_indexed(...)

# Step 2: Convert to micrometers
from convert_outgeom_voxels_to_um import convert_outgeom_pkl_to_um
data_um = convert_outgeom_pkl_to_um(
    in_path="output/graph_18_OutGeom_Hcut3.pkl",
    out_path="output/graph_18_OutGeom_Hcut3_um.pkl",
    res_um_per_vox=(1.625, 1.625, 2.5),
    min_straight_dist_um=1.0
)
```

## Important Notes

- **Radii conversion:** Only atlas radii are converted (atlas grid = 25 µm/voxel). Original image-space radii are NOT converted here due to upstream scaling ambiguities.
- **Non-destructive:** Original voxel arrays are preserved, allowing downstream flexibility.
- **Tortuosity:** Calculated only for edges where straight-line distance ≥ min_straight_dist_um (sanity check).

## Author
Ana Barrio - Feb 2026
