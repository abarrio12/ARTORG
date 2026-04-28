# CSVtoPKL: CSV to Pickle Conversion

## Overview
This module converts raw CSV data into structured **Pickle (.pkl) files** containing the complete vascular graph with geometric information, vertex attributes and edge properties.

## Folder Structure

```
CSVtoPKL/
├── build_graph_outgeom_voxels.py        # Build graph with outer geometry (voxel space)
├── convert_outgeom_voxels_to_um.py      # Convert voxel coordinates to micrometers (.pkl from previous)
├── CSV2Pickle_SOFIA.py                  # CSV → PKL conversion (non tortuous graph only)
├── CSVtoPKL_FULLGEOM.py                 # Full geometry variant (stores info in edges/vertex)
├── materialize_to_mvn.py                    # Transforms outgeom format in mvn format
├── materialize_to_mvn_outputDict.py         # Transforms outgeom format in mvn format and outputs in dictionaries
└── README.md
```


## Workflow

```
CSV Files (in ../CSV/)
    ↓
[build_graph_outgeom_voxels.py] (voxel space)
    ↓
[convert_outgeom_voxels_to_um.py]  (physical units)
    ↓
[materialize_to_mvn.py]  → transforms dict structure into mvn format
[materialize_to_mvn_outputDict.py]  → same as before, but output in dict format (vertices, edges, graph)

Ready for: cutting, analysis, visualization
```


## Processing Pipeline
### 1. **build_graph_outgeom_voxels.py** - Graph with Outer Geometry (Voxel Space)

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

### 2. **convert_outgeom_voxels_to_um.py** - Voxel to Micrometer Conversion
- Converts voxel-space coordinates to physical units (micrometers)
- **Keeps original voxel data intact** (non-destructive)
- Stores converted data with suffix `_R` (for "real-world" units)
- Computes per-segment lengths2 and length as per-edge (sum(lengths2)) in µm
- Uses conversion factors:
  - Source resolution: (1.625, 1.625, 2.5) µm/voxel # (image resolution)
  - Radii atlas: 25 µm/voxel                        # (atlas resolution)

  Be aware that diameter was kept in voxels from image space, since the conversion is not as trivial as multiplying by the image resolution.

### 3. **CSVtoPKL_FULLGEOM.py** - Full Geometry Variant
- In voxels 
- Stores information in edges/vertex (G.es/.vs). If the graph is large, processing may use excessive memory (this is why outer geometry format was implemented)
- Available to run it in supercomputer only (~250 GB)


### 3. **materialize_to_mvn.py** 
- This code materializes the .pkl into a mvn format:
    keep_v = {"coords", "index", "annotation", "diameter"}
    keep_e = {"connectivity", "nkind", "diameter", "diameters",
              "length", "lengths2", "points"}

## Author
Ana Barrio - Feb 2026
