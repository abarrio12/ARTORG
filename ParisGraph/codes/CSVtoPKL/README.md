# ParisGraph 
## CSV → MVN igraph → µm conversion / dict export

This folder contains the scripts used to convert the Paris/Renier CSV graph files into an `igraph` object with MVN/Gaia-style attributes, optionally convert the graph from voxel coordinates to physical units (µm), and optionally save the graph attributes as separate dictionaries.

## Workflow

Everything can be done directly from `CSVtoPKL_tortuous.py` by changing the optional flags at the end of the script.

```
CSV files
   ↓
CSVtoPKL_tortuous.py
   ↓
halfbrain_18_igraph.pkl                     # MVN/Gaia igraph in voxel/image units
   ↓ if DO_CONVERT_TO_UM = True
halfbrain_18_igraph_um.pkl                  # MVN/Gaia igraph in µm
   ↓ if DO_SAVE_DICT_FORMAT = True
verticesDict.pkl / edgesDict.pkl / graphDict.pkl
```

## Scripts

### 1. `CSVtoPKL_tortuous.py`

Main conversion script.

It reads the CSV files exported from the Paris/Renier graph and creates an `igraph.Graph` object with full tortuous geometry per edge.


### Step 1 — Load CSV files

The script reads node, edge, label, radius, annotation and geometry information from CSV files.

There are two coordinate sources:

```python
coordinates_atlas.csv  # atlas-space node coordinates
coordinates.csv        # image-space node coordinates
```

For the final MVN-style graph, the script uses:

```python
G.vs["coords"] = G.vs["coords_image"]
```

So the final graph coordinates are in **image space**, not atlas space.

### Step 2 — Build the basic graph

The script creates an empty `igraph.Graph`, adds vertices, and then adds edges using the source and target indices from `edges.csv`.

Each edge receives a vessel type:

```text
nkind = 2 → arteriole
nkind = 3 → venule
nkind = 4 → capillary / unlabeled vessel
```


### Step 3 — Add tortuous geometry

The script reads full edge geometry from:

```text
edge_geometry_indices.csv
edge_geometry_coordinates.csv
edge_geometry_radii.csv
```

For each edge, it extracts the corresponding polyline points:

```python
pts = np.column_stack((x[s:e], y[s:e], z[s:e]))
```

It also extracts the per-point radii and converts them to diameters:

```python
diams = 2.0 * r_pts
```

If needed, the edge geometry is reversed so that the first point of the polyline matches the source node.


### Step 4 — Convert to MVN/Gaia-like format

The script keeps a reduced set of attributes compatible with the MVN/Gaia VTP exporter.

Final vertex attributes:

```text
coords       → image-space node coordinates
degree       → graph degree of each node
index        → vertex index
diameter     → node diameter = 2 * node radius
annotation   → atlas/ABA annotation
```

Final edge attributes:

```text
connectivity → tuple(source, target)
nkind        → vessel class
points       → tortuous edge geometry
diameters    → per-point diameters
lengths2     → segment lengths between consecutive points
lengths      → per-point segment length convention used by MVN/Gaia export
length       → total edge length = sum(lengths2)
diameter     → scalar edge diameter
```


Note: `tortuosity` is calculated (tortuous length / straight end-to-end distance), but in the current `keep_e` it is removed unless explicitly added.

To keep tortuosity in the final graph, use:

```python
keep_e = {
    "connectivity", "diameter", "diameters", "length", "lengths",
    "lengths2", "nkind", "points", "tortuosity"
}
```

The scalar edge diameter is computed as the maximum diameter along the edge:

```python
diam_edge.append(float(np.max(diams)))
```

This follows the Paris/ClearMap reduction logic where the edge-level diameter is taken as the maximum of the per-point diameters. Check [Paris calculations](Paris%20calculations.md) for more.

## Units in the raw output

The raw output graph is saved as:

```text
halfbrain_18_igraph.pkl
```

Metadata:

```python
G["unit"] = "vox"
G["coord_space"] = "image"
G["diameter_unit"] = "vox"
G["resolution_image_um_per_voxel"] = [1.625, 1.625, 2.5]
G["resolution_atlas_um_per_voxel"] = [25.0, 25.0, 25.0]
```


# 2. `vox_to_um.py`

This script converts an MVN/Gaia-style `igraph.Graph` from voxel units to micrometers.

It can be used either at the end of the **main script**:
```python
DO_CONVERT_TO_UM = True
```

If this flag is `True`, the script imports:

```python
from vox_to_um import load_and_convert
```

and creates:

```text
halfbrain_18_igraph_um.pkl
```

using:

```python
res_um_per_vox = (1.625, 1.625, 2.5)
```

or directly from terminal:

```bash
python vox_to_um.py
```

## What `vox_to_um.py` does

### Coordinates

Vertex coordinates and edge polyline points are multiplied by the image voxel spacing:

```python
spacing = np.array([1.625, 1.625, 2.5])
```

So:

```python
coords_um = coords_vox * spacing
points_um = points_vox * spacing
```

This correctly handles anisotropic voxel size.

### Lengths

The script does **not** simply multiply edge lengths by an average scale factor.

Instead, it recalculates physical segment lengths from the scaled points:

```python
diffs = np.diff(pts_um, axis=0)
lengths2_um = np.linalg.norm(diffs, axis=1)
length_um = np.sum(lengths2_um)
```

This is important because the voxel spacing is anisotropic:

```text
x = 1.625 µm/voxel
y = 1.625 µm/voxel
z = 2.5 µm/voxel
```

Using an average scaling factor would be inaccurate for edges with a strong Z component.

### Diameters

Diameters are scaled using the transverse XY scale:

```python
scale_diam = (sx + sy) / 2.0
```

For this dataset:

```text
sx = 1.625
sy = 1.625
scale_diam = 1.625
```

So:

```python
diameter_um = diameter_vox * 1.625
```

This is an **APPROXIMATION**. In the analysis, G["diameter_unit"] = "vox" was used. 

### Scalar edge diameter

After converting per-point diameters to µm, the scalar edge diameter is recomputed as:

```text
edge diameter = max(per-point diameters)
```

This is consistent with the Paris pipeline.


## Output metadata

The converted graph receives:

```python
G_um["unit"] = "um"
G_um["diameter_unit"] = "um"
G_um["coord_space"] = "image"
G_um["resolution_image_um_per_voxel"] = [1.625, 1.625, 2.5]
G_um["resolution_atlas_um_per_voxel"] = [25.0, 25.0, 25.0]
```

Output file:

```text
halfbrain_18_igraph_um.pkl
```

# 3. `save_dict_format.py`

This script converts a final MVN-style `igraph.Graph` into three separate dictionaries:

```text
vertices_dict
edges_dict
graph_dict
```

This is useful if another pipeline expects the graph attributes as dictionaries instead of an `igraph.Graph`.

## Optional dictionary export in the main script

In the main script, dictionary export is controlled by:

```python
DO_SAVE_DICT_FORMAT = False
```

If set to `True`, it runs:

```python
from save_dict_format import save_mvn_dicts_from_igraph

save_mvn_dicts_from_igraph(
    G,
    out_dir=dict_out_dir,
    base_name=base_name,
    verbose=True,
)
```

This saves the **VOX version** of the graph as dictionaries.

If you also want to save the **µm version** as dictionaries, the main script should store the returned `G_um` from `load_and_convert()` and call `save_mvn_dicts_from_igraph(G_um, ...)`.

# Important notes

## Image space vs atlas space

Although the script loads both `coordinates_atlas.csv` and `coordinates.csv`, the final graph uses:

```python
G.vs["coords"] = G.vs["coords_image"]
```

Therefore the final graph is in image space.

Atlas coordinates are not kept in the final reduced MVN/Gaia graph unless explicitly added to `keep_v`.
In Paris they mainly use them for the annotation.
