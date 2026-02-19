# PKLtoVTP: Pickle to VTP Visualization Format

## Overview
This module converts **pickle graph data** (.pkl) into **VTP (VTK PolyData)** format, enabling 3D visualization in **ParaView** for visualization.

## Goal
Create visualization files that:
1. Preserve vascular network geometry and vessel properties
2. Enable interactive 3D exploration in ParaView
3. Support color-mapping vessel attributes (diameter, type, annotation)
4. Allow measurement and region analysis in ParaView
5. Export renderings and measurements

## Folder Structure

```
PKLtoVTP/
├── pkl2vtp_SOFIA.py                     # Main converter (SOFIA implementation)
├── Pkl2vtp_MVN_SOFIA.py                 # MVN dataset variant
├── PKLtoVTP_Tortuous_OUTGEOM.py         # Tortuous outer geometry
├── PKLtoVTP_tortuous_FULLGEOM.py        # Full geometry variant
├── PKLtoVTP_nonT_basic.py               # Non-tortuous simplified variant
├── PKLtoVTP_nonTort_basic_export.py     # Alternative non-tortuous export
├── PKLtoVTP_nonTort_outgeom.py          # Non-tortuous with outer geometry
└── README.md
```

## Core Concepts

### VTP File Format
VTP is an XML-based **VTK PolyData** format containing:
- **Points:** 3D vertex coordinates
- **Lines/Cells:** Edge connectivity (vessel segments)
- **PointData:** Properties at each point (diameter, annotation)
- **CellData:** Properties per edge/cell

### Mapping from PKL to VTP

```
PKL data structure              →  VTP structure
────────────────────────────────────────────────────
vertex (N nodes)                →  Points (N vertices)
    coords_image_R              →  XYZ coordinates

geom (M polyline points)        →  Points + CellData
    x_R, y_R, z_R              →  Point coordinates
    annotation                  →  PointData array (color/label)
    diameters_atlas_geom_R     →  PointData array (tube radius)

edges (connectivity)            →  Lines (cells)
    geom_start, geom_end       →  Line connectivity (point indices)
    vessel_type                →  CellData array
```

## Main Scripts

### 1. **pkl2vtp_SOFIA.py** - Primary Converter

Core conversion implementation with key functions:

```python
def WriteOnFileVTP(
    filename, 
    vertices_array, 
    connectivity_array,
    point_data,         # dict: {"field_name": array}
    cell_data,         # dict: {"field_name": array}
    subgraph=False
)
```

**Converts PKL to VTP:**
```python
import pickle
from pkl2vtp_SOFIA import *

# Load PKL
data = pickle.load(open("graph_18_OutGeom_Hcut3_um.pkl", "rb"))

# Extract geometry
vertices = data["vertex_R"]["coords_image_R"]  # (N, 3)
geom = data["geom_R"]
x_um, y_um, z_um = geom["x_R"], geom["y_R"], geom["z_R"]
polyline_points = np.column_stack([x_um, y_um, z_um])

# Define connectivity (edges → polyline segments)
connectivity = []
for edge in data["graph"].es:
    start = edge["geom_start"]
    end = edge["geom_end"]
    for i in range(start, end - 1):
        connectivity.append([i, i + 1])

# Add properties
point_data = {
    "diameter": geom["diameters_atlas_geom_R"],
    "annotation": geom["annotation"],
}

# Write VTP
WriteOnFileVTP(
    filename="vessel_graph.vtp",
    vertices_array=polyline_points,
    connectivity_array=np.array(connectivity),
    point_data=point_data,
    cell_data={},
)
```

### 2. **PKLtoVTP_Tortuous_OUTGEOM.py** - Tortuous Path Export

Specialized converter preserving **tortuous (curved) geometry**:
- Exports full polyline representation
- Each polyline point becomes a VTP point
- Maintains natural vessel curvature
- Preserves per-point radii variation

**Advantages:**
- ✓ Accurate geometric representation
- ✓ Smooth vessel appearance in ParaView
- ✓ Natural curvature visualization

### 3. **PKLtoVTP_tortuous_FULLGEOM.py** - Full Geometry Tortuous Export

Alternative tortuous implementation with full geometry preservation.

### 4. **PKLtoVTP_nonT_basic.py** - Simplified Non-Tortuous Representation

Creates simplified VTP using only vertex coordinates:
- Vertices act as nodes
- Edges connect vertices directly (straight lines)
- No intermediate polyline points
- ~10-100× fewer points/cells

**Use cases:**
- Quick preview/overview
- Network topology emphasis
- Fast rendering of large graphs

### 5. **PKLtoVTP_nonTort_basic_export.py** & **PKLtoVTP_nonTort_outgeom.py** - Non-Tortuous Variants

Alternative non-tortuous export implementations with different geometry handling approaches.

### 6. **Pkl2vtp_MVN_SOFIA.py** - Dataset-Specific Variant

MVN (Mesonetwork Visualization Network) format:
- Applies MVN-specific coordinate transformations
- Uses MVN anatomical region definitions
- Color-codes by region/structure


## Conversion Pipeline

```
PKL graph (voxels/micrometers)
        ↓
[pkl2vtp_SOFIA.py]
        ↓
VTP file (XML, text-readable)
        ↓
[ParaView]  →  Interactive 3D visualization
        ↓
Export: PNG, PDF, measurements, regions
```

## Output Files

Generated VTP files:
- `graph_18_igraph.vtp` - Full tortuous geometry
- `graph_18_igraph_nHcut3.vtp` - Cut region variant
- `hippo.vtp` - Hippocampus-specific
- `nonT_Hcut3.vtp` - Non-tortuous simplified version

## Coordinate Systems in VTP

⚠️ **Important:** VTP **stores** coordinate values but does NOT preserve coordinate system metadata.

- VTP contains raw XYZ coordinates (whatever you input)
- **No metadata** about what those coordinates represent (voxels? µm? atlas space?)
- ParaView displays them as-is, but has no knowledge of the coordinate system
- Measurements in ParaView assume input coordinates are in the originally specified units

**To ensure correct interpretation:**
1. Always ensure input PKL has `_R` (micrometer) attributes via [CSVtoPKL/convert_outgeom_voxels_to_um.py](../CSVtoPKL/README.md)
2. Document externally that your VTP file contains **micrometers (µm)** coordinates
3. When sharing VTP files, specify the coordinate system separately
4. Use consistent axis orientation if combining with other datasets

**Example:** If you export voxel coordinates by mistake, VTP will still display them but ParaView won't know they're voxels, leading to incorrect measurements.

## Troubleshooting

**Issue: VTP file won't open in ParaView**
- Check XML validity: `xmllint vessel_graph.vtp`
- Verify points/lines arrays match connectivity format
- Ensure correct data types (Float64, UInt64)

**Issue: Vessels appear as points/dots**
- Apply Tube filter: Filters → Geometric → Tube
- Check if diameter data is present: PointData["diameter"]
- Adjust tube radius scale in Tube filter properties

**Issue: Colors don't map**
- Verify PointData array exists: use `h5dump` or ParaView Attributes panel
- Check value ranges (ensure not all NaN)
- Use ParaView histogram editor to adjust scale

## Related Modules

- **CSVtoPKL:** Generates PKL from CSV
- **cutting:** Extracts PKL subregions before visualization
- **Graph Analysis:** Computes properties to visualize

## Supported Property Types

Commonly visualized:
- `diameter` / `radii` - Vessel cross-section size
- `annotation` - Vessel type (artery/vein/capillary)
- `length` - Edge length
- `tortuosity` - Curved vs. straight
- `density` - Local vascular density
- Custom properties from analysis

## Author
Sofia Renier dataset adaptation
Ana Barrio - Feb 2026
