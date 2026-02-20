# Cutting: Extract Vessels from a 3D Box Region

## What it does
Takes your complete vessel graph and extracts only the vessels inside a rectangular 3D box (like selecting a region of interest). Vessels that cross the box boundary are trimmed at the edge.

## How it works
1. You draw a 3D box in ParaView and note the center and size
2. This tool keeps only vessels (and vessel parts) inside that box
3. If a vessel crosses the boundary, it's cut at that point
4. Returns a new graph with just the vessels in the box

## Folder Structure

```
cutting/
├── cut_outgeom_roi_VOX.py              # Cut in voxels
├── cut_outgeom_roi_UM.py               # Cut in micrometers
├── Cut_The_Graph_GAIA.py               # Advanced tool with two options
├── equivalent_non.py                   # Simplified non-curved versions
├── graph_cut_SOFIA.py                  # Alternative implementation
└── README.md
```


## Main Scripts

### 1. **cut_outgeom_roi_VOX.py** - Primary Utility (Voxel Space)

Cuts graph inside a 3D rectangular box in **voxel coordinates** (image space).

**Inputs:**
- `data`: PKL dict with voxel coordinates
- `center box`: in voxels
- `size box`: in um (need conversion to voxels)
- `x_box`: `(x_min, x_max)` in voxels
- `y_box`: `(y_min, y_max)` in voxels
- `z_box`: `(z_min, z_max)` in voxels

**Output:**
- New `data` dict with filtered graph and clipped geometry

**Key operations:**
- Vertex filtering (keep vertices inside box)
- Polyline clipping (cut at box boundaries)
- Geometry recomputation (lengths2, length...)

### 2. **cut_outgeom_roi_UM.py** - Micrometer Variant

Same as above but operates in micrometer coordinates (`_R` attributes).

**Inputs:**
- `data`: PKL dict with `_R` (micrometer) attributes
- `center box`: in voxels (need conversion to um)
- `size box`: in um 
- `x_box`: `(x_min, x_max)` in µm
- `y_box`: `(y_min, y_max)` in µm
- `z_box`: `(z_min, z_max)` in µm

### 3. **Cut_The_Graph_GAIA.py** - Advanced Geometric Clipping with Two Variants

Implements two different approaches to graph clipping, with the ability to choose between them. Both handle edge classification and clipping.

**Two main functions:**

1. **`get_edges_in_boundingBox_vertex_based()`** - Full clipping with interpolation
   - Classifies edges as **in-box**, **across-border**, or **out-box**
   - For edges crossing the boundary (in-out pairs):
     - Computes exact intersection point using linear interpolation
     - Creates new boundary vertices at box surface
     - Trims polyline points keeping only those inside box
     - Computes edge attributes for clipped geometry
   - For out-out pairs: marks for deletion
   - Returns: edges_in_box, edges_across_border, edges_outside_box, border_vertices, new_edges_on_border

2. **`get_edges_in_boundingBox_vertex_based_2()`** - Simplified classification (no interpolation)
   - Classifies edges into three categories:
     - **edges_in_box:** Both tortuous vertices inside
     - **edges_across_border:** One vertex in, one vertex out
     - **edges_outside_box:** Both vertices outside
   - Identifies border_vertices (the outside node in crossing edges)
   - Does NOT perform clipping/interpolation — used for analysis/filtering (these edges cross boundary and these are the boundary vertex, these edges are full outside and these full in)
   - Returns: edges_in_box, edges_across_border, edges_outside_box, border_vertices

**Typical workflow:**
1. Use `get_edges_in_boundingBox_vertex_based_2()` to classify edges (fast analysis)
2. Use `get_edges_in_boundingBox_vertex_based()` to perform actual clipping (detailed geometry)

### 4. **equivalent_non.py** - Non-Tortuous Equivalent Generation

Generates a "non-tortuous" equivalent of a tortuous cutted graph:
- Simplifies polylines to straight lines between vertices
- Useful for comparison or simplified modeling

**Note**: Codes 1,2,3 also do this directly. This code was implemented only for testing
without having to run all the code. 

### 5. **graph_cut_SOFIA.py** - SOFIA-Specific Implementation

Alternative graph cutting implementation using SOFIA methodology.

## Quick Start

**Important:** ParaView doesn't understand units—it just displays the numbers from your file. You need to know what system your graph is in.

### If your graph is in VOXELS:

```python
from cut_outgeom_roi_VOX import cut_graph_inside_box
import pickle

# Load your voxel graph
data = pickle.load(open("graph_voxels.pkl", "rb"))

# From ParaView, read the box center
center = (1500, 1200, 800)  # these are voxel coordinates from ParaView

# Your desired box size (in micrometers) - convert to voxels
size_um = 500  # micrometers
res = [1.625, 1.625, 2.5]  # µm per voxel (x, y, z)
size_vox = int(size_um / res[0])  # 500 / 1.625 = 308 voxels

# Define the box
x_box = (center[0] - size_vox, center[0] + size_vox)
y_box = (center[1] - size_vox, center[1] + size_vox)
z_box = (center[2] - size_vox, center[2] + size_vox)

# Cut and save
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)
pickle.dump(data_cut, open("graph_cut.pkl", "wb"))
```

### If your graph is in MICROMETERS:

```python
from cut_outgeom_roi_UM import cut_graph_inside_box
import pickle

# Load your micrometer graph
data = pickle.load(open("graph_um.pkl", "rb"))

# From ParaView, read the box
center = (2437, 1950, 2000)  # these are micrometer values from ParaView
size = 500  # micrometers

# Define the box
x_box = (center[0] - size, center[0] + size)
y_box = (center[1] - size, center[1] + size)
z_box = (center[2] - size, center[2] + size)

# Cut and save
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)
pickle.dump(data_cut, open("graph_cut.pkl", "wb"))
```

### If you're viewing a VOXEL graph but want to cut in MICROMETERS:

```python
from cut_outgeom_roi_UM import cut_graph_inside_box
import pickle

# Load your micrometer graph (the one you're actually cutting)
data = pickle.load(open("graph_um.pkl", "rb"))

# From ParaView (which is showing your voxel graph), read the box center
center_vox = (1500, 1200, 800)  # voxel coordinates from ParaView
size_um = 500  # micrometers

# Convert voxel center to micrometers
res = [1.625, 1.625, 2.5]  # µm per voxel
center_um = (
    center_vox[0] * res[0],  # 1500 * 1.625 = 2437.5
    center_vox[1] * res[1],
    center_vox[2] * res[2]
)

# Define the box in micrometers
x_box = (center_um[0] - size_um, center_um[0] + size_um)
y_box = (center_um[1] - size_um, center_um[1] + size_um)
z_box = (center_um[2] - size_um, center_um[2] + size_um)

# Cut and save
data_cut = cut_graph_inside_box(data, x_box=x_box, y_box=y_box, z_box=z_box)
pickle.dump(data_cut, open("graph_cut.pkl", "wb"))
```

## Author
Ana Barrio - Feb 2026
