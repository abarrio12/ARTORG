# PKLtoVTP: Pickle to VTP Visualization Format

## Overview
This module converts **pickle graph data** (.pkl) into **VTP (VTK PolyData)** format, enabling 3D visualization in **ParaView** for visualization.


## Folder Structure

```
PKLtoVTP/
├── pkl2vtp.py # Basic converter: MVN format → tortuous / non-tortuous VTP
├── pkl2vtp_ana.py # Extended converter: handles MVN + outgeom and checks attribute existence
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


## Tortuous vs Non-Tortuous Export

### `tortuous=False` (straight graph)

Each vessel is exported as a straight segment between nodes:
node ───────── node


Uses:
- `G.vs["coords"]`
- graph topology (`source`, `target`)

Does **not** use edge geometry.

---

### `tortuous=True` (full geometry)

Each vessel is exported using all intermediate points:


node ── point ── point ── point ── node


Uses:
- `G.es["points"]`
- `G.es["diameters"]`
- `G.es["lengths2"]`

---

## Conversion Pipeline


```
PKL graph (voxels/micrometers)
        ↓
[pkl2vtp_ana.py] -> choose tortuous = True/False
        ↓
VTP file (XML, text-readable)
        ↓
[ParaView]  →  Interactive 3D visualization
```

## Coordinate Systems & Units

**Important:** ParaView does not know the units of your data.

- If coordinates are in voxels → ParaView shows voxels
- If coordinates are in micrometers → ParaView shows micrometers

Make sure your graph is already scaled correctly before exporting.

Check units:

```python
print(G["unit"])
```



## Authors

- **Sofia** - pkl2vtp.py
- **Ana** - pkl2vtp_ana - Feb 2026

