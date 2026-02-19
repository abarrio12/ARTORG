# Cutting: Graph ROI Extraction & Clipping

## Overview
This module **extracts and clips vessel graphs** based on **Region of Interest (ROI)** definitions (hippocampal and somatomotor region in this case). It takes a complete vascular graph and returns a subgraph containing only vessels within specified 3D spatial regions.

## Goal
Enable selective analysis of brain vasculature by:
1. Defining 3D box ROIs (in micrometers or voxels)
2. Selecting vertices inside ROI boundaries
3. Clipping vessel paths that cross box boundaries (creating new border vertices)
4. Rebuilding consistent graph structure with recomputed geometric attributes
5. Preserving all metadata (annotations, radii, properties)

## Folder Structure

```
cutting/
├── cut_box.py                         # Cut graph inside rectangular box
├── cut_out.py                         # Cut graph outside region
├── cut_outgeom_roi_UM.py              # Cut in micrometers (main utility)
├── cut_outgeom_roi_VOX.py             # Cut in voxels
├── Cut_The_Graph_GAIA.py              # GAIA dataset-specific cutting
├── Cut_The_Graph_MVN.py               # MVN dataset-specific cutting
├── equivalent_non.py                  # Non-tortuous equivalent generation
├── graph_cut_SOFIA.py                 # SOFIA-specific implementation
└── README.md
```

## Core Concepts

### Geometric Clipping Algorithm

When a vessel edge geometry (polyline) intersects a box boundary:

```
Original polyline:  •———•———•———•———•  (goes outside box)
                        ↓
Result:             •———•———X   X———•  (clipped at boundaries)
                             ↑ border vertices
```

**Process:**
1. Identify polyline segments that cross the box boundary
2. Compute intersection point (exact position on box surface)
3. Create new "border vertex" at intersection
4. Retain only polyline points inside the box
5. Connect border vertices with appropriate edge properties

### Key Data Structures

Input (`data` dict):
```python
data = {
    "graph": igraph.Graph,
    "vertex_R": {
        "coords_image_R": array(N, 3),      # vertex coordinates in µm
        ...
    },
    "geom_R": {
        "x_R", "y_R", "z_R": arrays(M,),    # polyline points in µm
        "lengths2_R": array(M,),             # per-segment distances
        "diameters_atlas_geom_R": array(M,),
        "annotation": array(M,),
    }
}
```

Output: Same structure, filtered to ROI

## Main Scripts

### 1. **cut_outgeom_roi_UM.py** - Primary Utility (Micrometers)

Cuts graph inside a 3D rectangular box in **micrometers**.

**Inputs:**
- `data`: PKL dict with `_R` (micrometer) attributes
- `x_box`: `(x_min, x_max)` in µm
- `y_box`: `(y_min, y_max)` in µm
- `z_box`: `(z_min, z_max)` in µm

**Output:**
- New `data` dict with filtered graph and clipped geometry

**Key operations:**
- Vertex filtering (keep vertices inside box)
- Polyline clipping (cut at box boundaries)
- Geometry recomputation:
  - `lengths2_R`: distances between consecutive polyline points
  - `length_R`: per-edge arc length (sum of segment lengths)
  - `tortuosity_R`: `length_R / straight_distance`

### 2. **cut_outgeom_roi_VOX.py** - Voxel Space Variant

Same as above but operates in voxel coordinates (uses `data["vertex"]`, `data["geom"]`).

### 3. **Cut_The_Graph_GAIA.py** & **Cut_The_Graph_MVN.py**

Dataset-specific wrappers that:
- Define standard ROI boxes for brain regions (GAIA or MVN naming conventions)
- Automate batch cutting of multiple regions
- Handle dataset-specific coordinate systems

### 4. **cut_box.py** & **cut_out.py**

Simpler rectangular box utilities (may be used internally).

### 5. **equivalent_non.py**

Generates a "non-tortuous" equivalent of a cut graph:
- Simplifies polylines to straight lines between vertices
- Useful for comparison or simplified modeling

## Workflow Example

```python
from cut_outgeom_roi_UM import cut_graph_inside_box

# Define ROI in micrometers
roi_box = {
    "x_box": (1000, 3000),   # µm
    "y_box": (800, 2800),
    "z_box": (500, 2500),
}

# Load PKL
data = pickle.load(open("graph_18_OutGeom_Hcut3_um.pkl", "rb"))

# Cut graph
data_cut = cut_graph_inside_box(
    data,
    x_box=roi_box["x_box"],
    y_box=roi_box["y_box"],
    z_box=roi_box["z_box"],
)

# Result contains only vessels within box, with clipped geometry
print(f"Original: {data['graph'].vcount()} vertices")
print(f"Cut: {data_cut['graph'].vcount()} vertices")
```

## Typical Use Cases

1. **Brain region analysis:** Extract vasculature from specific anatomical regions (hippocampus, somatomotor cortex, etc.)
2. **Comparative studies:** Cut identical ROIs across different datasets/samples
3. **Boundary condition studies:** Focus on vessels at specific interfaces
4. **Vascular density mapping:** Compute properties within uniform cubic divisions

## Related Workflow

```
complete graph (PKL)
        ↓
[cut_outgeom_roi_UM.py]
        ↓
cut graph (PKL) → analysis, visualization, statistics
```

## Coordinate Systems

- **Voxels:** Indices into the original imaging array. Resolution varies (default: 1.625 × 1.625 × 2.5 µm/voxel)
- **Micrometers:** Physical units. Use this for anatomically meaningful ROI definitions.

**Conversion:** Use `convert_outgeom_voxels_to_um.py` from CSVtoPKL module.

## Author
Ana Barrio - Feb 2026
