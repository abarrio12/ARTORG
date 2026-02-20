# Cutting: Graph Spatial Clipping by 3D Box

## What This Does

Takes a complete blood vessel map and cuts out just the vessels inside a 3D box you define. Everything outside gets removed.

**Quick steps:**
1. Pick a brain region using [SelectBrainRegion_fromJSON.py](../Graph%20Analysis%20%26%20by%20region/SelectBrainRegion/SelectBrainRegion_fromJSON.py)
2. View it in ParaView
3. Place a 3D box around your region
4. Note the box center coordinates
5. Run the script matching your data type (voxels or micrometers)

## Files

```
cutting/
├── cut_outgeom_roi_VOX.py              # Cut voxel-space data
├── cut_outgeom_roi_UM.py               # Cut micrometer data  
├── Cut_The_Graph_GAIA.py               # Advanced clipping (two methods)
├── equivalent_non.py                   # Make straight-line version
├── graph_cut_SOFIA.py                  # Alternative method
└── README.md
```

## How It Works

The tool:
1. Keeps all vessels completely inside the box
2. Cuts vessels that cross the box edge (creates new connection points)
3. Removes vessels completely outside
4. Recalculates lengths and properties for cut pieces

## Which Script to Use?

| Your Data Type | Use This Script | Box In |
|---|---|---|
| Voxels | `cut_outgeom_roi_VOX.py` | Voxels |
| Micrometers | `cut_outgeom_roi_UM.py` | Micrometers |

## How to Use

**Key:** ParaView doesn't know units—it just shows numbers. YOU need to remember whether your data is voxels or micrometers.

### If your data is in VOXELS

```python
from cut_outgeom_roi_VOX import cut_graph_inside_box
import pickle

data = pickle.load(open("graph_voxels.pkl", "rb"))

# From ParaView
center = (1500, 1200, 800)  # voxels
size_um = 500               # your box size in micrometers

# Convert µm to voxels
res = [1.625, 1.625, 2.5]  # µm/voxel (x, y, z)
size_vox_xy = int(size_um / res[0])  # 500/1.625 ≈ 308 voxels
size_vox_z = int(size_um / res[2])   # 500/2.5 = 200 voxels

# Make box
x_box = (center[0] - size_vox_xy, center[0] + size_vox_xy)
y_box = (center[1] - size_vox_xy, center[1] + size_vox_xy)
z_box = (center[2] - size_vox_z, center[2] + size_vox_z)

# Cut
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)

# Save
with open("cut.pkl", "wb") as f:
    pickle.dump(data_cut, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### If your data is in MICROMETERS

```python
from cut_outgeom_roi_UM import cut_graph_inside_box
import pickle

data = pickle.load(open("graph_micrometers.pkl", "rb"))

# From ParaView
center = (2437, 1950, 2000)  # micrometers
size = 500                   # micrometers

# Make box
x_box = (center[0] - size, center[0] + size)
y_box = (center[1] - size, center[1] + size)
z_box = (center[2] - size, center[2] + size)

# Cut
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)

# Save
with open("cut.pkl", "wb") as f:
    pickle.dump(data_cut, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### Special: You saw VOXELS in ParaView but data is in MICROMETERS?

```python
from cut_outgeom_roi_UM import cut_graph_inside_box
import pickle

data = pickle.load(open("graph_micrometers.pkl", "rb"))

# From ParaView (voxels)
center_vox = (1500, 1200, 800)

# Convert to micrometers
res = [1.625, 1.625, 2.5]  # µm/voxel
center_um = (
    center_vox[0] * res[0],  # 1500 × 1.625
    center_vox[1] * res[1],  # 1200 × 1.625
    center_vox[2] * res[2]   # 800 × 2.5
)

size = 500

# Make box
x_box = (center_um[0] - size, center_um[0] + size)
y_box = (center_um[1] - size, center_um[1] + size)
z_box = (center_um[2] - size, center_um[2] + size)

# Cut
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)

# Save
with open("cut.pkl", "wb") as f:
    pickle.dump(data_cut, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## Advanced: Cut_The_Graph_GAIA.py

Has two clipping methods:
- **Full clipping:** Cuts vessels at box edge with exact coordinates
- **Classification only:** Just finds which vessels cross the edge (doesn't cut them)

Use the main scripts (VOX/UM) unless you need these specialized methods.

## Author
Ana Barrio - Feb 2026
