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

Use the function from each script, depending on what you need:

**For realistic visualization (recommended):**
```python
from PKLtoVTP_Tortuous_OUTGEOM import pkl2vtp_tortuous_outgeom
vtp_file = pkl2vtp_tortuous_outgeom("graph.pkl", "output/", radii_mode='atlas')
```
Creates curved vessels exactly as stored in your graph.

**For quick network preview:**
```python
from PKLtoVTP_nonT_basic import pkl2vtp_nonT
vtp_file = pkl2vtp_nonT("graph.pkl", "output/")
```
Simple straight lines between vessels, no detailed geometry.

**Other options:**
- `PKLtoVTP_tortuous_FULLGEOM.py` - Different geometry preservation approach
- `PKLtoVTP_nonTort_outgeom.py` - Outer geometry without curves
- `Pkl2vtp_MVN_SOFIA.py` - MVN format data
- `pkl2vtp_SOFIA.py` - General converter with `WriteOnFileVTP()` utility function


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

## What VTP Files Are Generated

Each output filename tells you what it contains:
- `graph_18_igraph.vtp` - Full brain vessel network
- `graph_18_igraph_nHcut3.vtp` - Network cut by region (nH = region version, cut3 = step 3)
- `nonT_Hcut3.vtp` - Same region but simplified view (nonT = straight lines only)
- `hippo.vtp` - Hippocampus region extracted

**Naming convention:** `[dataset]_[region][cut_level].vtp`

You can open any .vtp file directly in ParaView to visualize.

## Coordinate Systems & Units

**⚠️ Important:** VTP stores coordinates but doesn't know what units they represent.

When you open a VTP file in ParaView:
- You see 3D coordinates as numbers
- ParaView doesn't know if they're voxels, micrometers, or millimeters
- Measurements in ParaView assume your input units

**How to ensure correctness:**
1. Make sure your PKL file has `_R` (micrometer) attributes - these have coordinates scaled to micrometers
   - Check using [CSVtoPKL/convert_outgeom_voxels_to_um.py](../CSVtoPKL/README.md)
   - Example: `data["vertex_R"]["coords_image_R"]` uses _R suffix for µm coordinates
2. In your code, use the `_R` versions of arrays:
   ```python
   x_um, y_um, z_um = geom["x_R"], geom["y_R"], geom["z_R"]  # ← Use _R suffix
   coords = data["vertex_R"]["coords_image_R"]               # ← Use _R suffix
   ```
3. When you save the VTP file, document that it contains **µm (micrometers)** coordinates
4. If combining multiple VTP files, ensure they all use the same coordinate system

**Example:** If you accidentally use voxel coordinates (without _R), VTP will display them but ParaView won't know they're voxels, leading to incorrect measurements.


## Workflow Overview

```
Your PKL graph file
         ↓
  [Choose script]
  ├─ Tortuous (curved) → Realistic appearance
  └─ Basic (straight)  → Quick preview
         ↓
   VTP file created
         ↓
  Open in ParaView → Explore, measure, color by property
         ↓
  Export → Publication images, videos, analysis
```

## Related Modules

- **[CSVtoPKL](../CSVtoPKL/README.md)** - Converts your raw CSV data to PKL format
- **[cutting](../cutting/README.md)** - Extract vessel regions before visualizing
- **[Graph Analysis](../Graph%20Analysis%20&%20by%20region/README.md)** - Compute properties (tortuosity, density) to color-code in ParaView

## Properties You Can Visualize

- `diameter` or `radii_atlas` - Vessel thickness (colored tubes)
- `annotation` - Vessel type (artery vs vein)
- `tortuosity` - How curved each vessel is (straight = 1.0)
- `length` - Distance along vessel
- Custom properties from graph analysis

## Quick Start Example

```python
# Load your graph

import pickle
data = pickle.load(open("my_graph.pkl", "rb"))

# Convert to VTP

from PKLtoVTP_Tortuous_OUTGEOM import pkl2vtp_tortuous_outgeom
output_file = pkl2vtp_tortuous_outgeom("my_graph.pkl", "./output/")
print(f"Created: {output_file}")

# Now open in ParaView
# File → Open → select the .vtp file
# Then color by "annotation" or "tortuosity" to visualize properties
```

## Common Tasks in ParaView

| Task | How |
|------|-----|
| Change vessel colors | Color by → annotation / tortuosity / your property |
| Measure distance | Tools → Measure (hold Shift + click two points) |
| Export image | File → Export Screenshot |
| Adjust tube thickness | Representation → Style → Line Width |
| Select region | Use box/sphere selection tool to isolate vessels |

---

## Author & Version

Sox dataset adaptation - Feb 2026
Ana Barrio
