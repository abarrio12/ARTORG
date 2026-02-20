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

### What Gets Exported to VTP

**Points:** All geometry points (x, y, z) stored in global coordinate space

**PointData** (properties at each point):
- `annotation` - Vessel type/region label
- `radii_p_atlas` - Radius at this point (atlas-scaled)
- `diameter_p_atlas` - Diameter at this point (2 × radius)
- `lengths2` - Distance to next point along geometry

**CellData** (properties per edge/vessel segment):
- `nkind` - Vessel type code
- `radius_atlas` - Edge radius (max of all points in segment)
- `diameter_atlas` - Edge diameter (2 × radius)
- `length` - Total arc length along edge
- `tortuosity` - Straightness ratio (actual/Euclidean distance)

## Main Scripts

Use the function from each script, depending on what you need:

**For realistic curved vessel visualization (recommended):**
```python
# Direct execution - fills in paths, loads graph, exports
python PKLtoVTP_Tortuous_OUTGEOM.py
```
Or use as module:
```python
from PKLtoVTP_Tortuous_OUTGEOM import *  # loads and processes directly
```
→ Creates curved vessels with full polyline geometry, ideal for visualization

**For quick straight-line network preview:**
```python
from PKLtoVTP_nonT_basic import pkl2vtp_nonT
vtp_file = pkl2vtp_nonT("graph.pkl", "output/")
```
→ Simple vertex-to-vertex connections, fast rendering

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

You can open any .vtp file directly in ParaView to visualize.

**Output naming:** `[dataset]_[region].vtp` or `[dataset]_[region]_cut[level].vtp` for cut regions

## Coordinate Systems & Units

**⚠️ Important:** VTP stores coordinates but doesn't know what units they represent.

When you open a VTP file in ParaView:
- You see 3D coordinates as numbers
- ParaView doesn't know if they're voxels, micrometers, or millimeters
- Measurements in ParaView assume your input units

**How to ensure correctness:**
1. If you want um, make sure your PKL file has `_R` attributes - these have coordinates scaled to micrometers
   - Check using [CSVtoPKL/convert_outgeom_voxels_to_um.py](../CSVtoPKL/README.md)
   - Example: `data["vertex_R"]["coords_image_R"]` uses _R suffix for µm coordinates
2. In your code, use the `_R` versions of arrays:
   ```python
   x_um, y_um, z_um = geom["x_R"], geom["y_R"], geom["z_R"]  # ← Use _R suffix
   coords = data["vertex_R"]["coords_image_R"]               # ← Use _R suffix
   ```
3. When you save the VTP file, document that it contains **µm (micrometers)** coordinates
4. If combining multiple VTP files, ensure they all use the same coordinate system

**Note:** If you accidentally use voxel coordinates (without _R), VTP will display them but ParaView won't know they're voxels, leading to incorrect measurements.


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

## What to Color By in ParaView

**Point-level properties:**
- `annotation` - Shows anatomical regions
- `radii_p_atlas` - Vessel diameter at each point (for tube scaling)
- `diameter_p_atlas` - Same as above (diameter = 2 × radius)

**Edge-level properties:**
- `radius_atlas` / `diameter_atlas` - Vessel thickness for entire segment
- `tortuosity` - Red = straight (1.0), Blue = curved (>1.0)
- `length` - Segment length
- `nkind` - Vessel classification code

## Related Modules

- **[CSVtoPKL](../CSVtoPKL/README.md)** - Converts raw CSV data to PKL format
- **[cutting](../cutting/README.md)** - Extract vessel regions before visualizing
- **[Graph Analysis](../Graph%20Analysis%20&%20by%20region/README.md)** - Compute additional properties to visualize

---

## Authors

- **Sofia** - Core pkl2vtp_SOFIA.py conversion engine, MVN variant
- **Ana Barrio** - PKLtoVTP_Tortuous_OUTGEOM improvements, geometry optimization - Feb 2026

