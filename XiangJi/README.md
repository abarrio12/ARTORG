# Ji / Kleinfeld vascular graphs:
This folder contains utilities for working with two related vascular graph datasets from the Ji/Kleinfeld mouse brain vasculature work:

1. **Whole-brain microvascular graph**  
   The brain-wide connectome from Ji et al. 2021 (*Neuron*), reconstructed from whole mouse brain vascular imaging. This is the global reference graph/volume.

2. **MVN / crop graph from the whole brain**  
   A local cortical subvolume extracted from the whole-brain vascular connectome. This is the graph used for the network-flow analysis in Ji et al. 2026 (*PNAS*). In our files this corresponds to the `ML20180815_240_c5o1_578.mat` crop.

The key point is that the crop graph is stored in **local crop coordinates**, whereas the whole-brain volume is in **global whole-brain coordinates**. This matters for visualization in ParaView.

---

## 1. Dataset overview

### Whole-brain graph

The whole-brain graph comes from the Ji et al. 2021 mouse brain microvasculature connectome. The original study reconstructed a complete mouse brain vascular network at sub-micrometer imaging resolution and then represented it as a spatial graph.

Reported whole-brain statistics:

| Quantity | Value |
|---|---:|
| Reconstruction scale | Whole brain |
| Imaging voxel size | `0.30 × 0.30 × 1.0 µm³` |
| Segmentation resolution | `1.0 µm` |
| Radius estimation resolution | `0.25 µm` |
| Number of nodes | `4,132,583` |
| Number of branches / edges | `6,320,303` |
| Branches in largest component | `6,238,701` |
| Brain volume | `443 mm³` |
| Total vessel length | `384 m` |
| Average vessel radius | `2.7 µm` |

For the whole-brain dataset, coordinates are already in the global whole-brain coordinate system. Therefore, no crop offset should be applied.

---

### MVN / crop graph

The MVN graph is a local crop extracted from the whole-brain graph. The crop used here is:

```text
ML20180815_240_c5o1_578.mat
```

The keys are:
```
MAT keys:
dict_keys(['__header__', '__version__', '__globals__', 'num', 'link', 'endpoint', 'isopoint', 'node', 'radius', 'label', 'info'])
```

| Attribute | Value |
|---|---:|
| Node CC count | `11732` |
| Link CC count | `18973` |

---

## 2. Relevant scripts / notebooks

### `mat_to_pkl_scaling.py`

Main conversion script:

```text
MAT graph → igraph object → scaled PKL
```

This script:

1. loads the Ji/Kleinfeld `.mat` graph;
2. extracts node connected components;
3. extracts link / edge connected components;
4. converts MATLAB linear voxel indices into 3D coordinates;
5. converts coordinate order from MATLAB-style `(Y, X, Z)` to Python/VTK-style `(X, Y, Z)`;
6. builds an `igraph.Graph` in MVN format;
7. stores geometry, connectivity, vessel type, length, and diameter attributes;
8. applies the `1.051` shrinkage correction to geometry and lengths;
9. keeps radii/diameters unchanged;
10. saves the final graph as a `.pkl`.

Typical output:

```text
ML20180815_240_c5o1_578_mvn1_scaled.pkl
```

This output is scaled for physical length units but, for a crop, is still in **local crop coordinates**.

---

### `mvn_offset_correction.ipynb`

Post-processing notebook for crop graphs only:

```text
local crop PKL → global whole-brain PKL
```

This notebook:

1. loads the scaled crop `.pkl`;
2. loads the original `.mat`;
3. reads the crop bounding-box metadata from `mat["info"]`;
4. checks the graph bounds before correction;
5. adds the crop offset to the graph coordinates;
6. checks the graph bounds after correction;
7. saves a new global-coordinate `.pkl`.

This notebook is only needed for crop/subvolume graphs. Do not apply this offset to a whole-brain graph.

---

## 3. MAT to PKL conversion

The `.mat` file stores the vascular graph at the voxel-component level.

### Nodes

Ji nodes are stored as connected components of node voxels. A node can contain one or more voxel elements. The conversion code:

- reads the voxel indices for each node component;
- converts MATLAB linear indices into 3D voxel coordinates;
- converts `(Y, X, Z)` into `(X, Y, Z)`;
- computes the node coordinate as the centroid of the node voxels;
- assigns node diameter from the median node radius.

Relevant final vertex attributes:

```text
coords
index
diameter
annotation
```

`annotation` is included for compatibility, although this crop conversion does not add full atlas annotation.

---

### Edges / branches

Ji links are stored as connected components of link voxels. Each link corresponds to a vessel branch between two nodes. The conversion code:

- reads link voxel indices;
- converts them into 3D points;
- converts coordinate order from `(Y, X, Z)` to `(X, Y, Z)`;
- computes segment lengths between consecutive points;
- stores per-point radii and diameters;
- maps Ji vessel type labels to MVN labels.

The vessel type mapping is:

```python
JI_TO_MVN_NKIND = {
    1: 4,  # capillary
    2: 2,  # artery
    3: 3,  # vein
}
```

Relevant final edge attributes:

| Attribute | Meaning |
|---|---|
| `points` | full tortuous branch geometry |
| `lengths2` | center-to-center segment lengths between consecutive points |
| `length` | total edge length using Ji convention |
| `diameters` | per-point diameters |
| `diameter` | scalar edge diameter |
| `nkind` | vessel type: `2 = artery`, `3 = vein`, `4 = capillary` |
| `connectivity` | source and target node indices |

---

## 4. Coordinate and physical-unit scaling

### Raw coordinates

The Ji graph coordinates are reconstructed from voxel indices. 

```text
1 voxel = 1 µm isotropic
```

Therefore, this is not a voxel-to-micron conversion in the usual anisotropic sense. The coordinates are already in a 1 µm isotropic grid.

---

### Shrinkage correction

The fixed brain is slightly shrunken relative to the in vivo brain. The brain shrunks:

```text
(1/1.051)
```

Therefore, in order to get physical units we need to multiply by this factor geometry and longitude attributes. 

```python
rescale_graph_lengths_inplace(G_scaled_full, 1.051)
```

This affects:

```text
coords
node_points
points
lengths2
length
```

Conceptually:

```text
local_in_vivo_coords = local_fixed_coords × 1.051
```

This gives local physical coordinates and lengths in micrometers. 

Outputs are called ..._scaled.pkl

---

### Length convention

For each link connected component, the code follows the Ji/Guo convention:

```text
length = sum(center-to-center segment lengths) + voxel_size
```

For a one-voxel link:

```text
length = voxel_size
```

After shrinkage correction:

```text
voxel_size = 1.051 µm
```

The sanity check in `mat_to_pkl_scaling.py` verifies:

```text
scaled length / raw length ≈ 1.051
```

and confirms that `points` and `lengths2` remain geometrically consistent after scaling.

---

## 5. Radii and diameters

Radii/diameters are **not rescaled** by the `1.051` factor in this workflow.

Reason:

- Ji et al. calibrated vessel radii against in vivo measurements.
- In this dataset, the great majority of radii are above approximately `1.8 µm`.
- According to the Ji/Kleinfeld radius calibration logic, radii in this range are already treated as corrected physical radii.
- Therefore, radii and diameters are kept in physical units and are not multiplied by the shrinkage factor.

The output should report that diameters are unchanged after geometric scaling.

---

## 6. Why the MVN crop did not initially overlap in ParaView

When visualizing the scaled MVN crop together with the whole-brain volume in ParaView, the crop did not overlap the expected region.

The reason was not the scale. The issue was the coordinate origin.

The MVN crop `.pkl` after `mat_to_pkl_scaling.py` is in local crop coordinates:

```text
X ≈ [0, 1125]
Y ≈ [0, 1125]
Z ≈ [0, 1125]
```

The whole-brain reference volume is in global whole-brain coordinates:

```text
X: [0, 11264]
Y: [0, 10368]
Z: [0, 13312]
```

So ParaView was drawing the MVN crop near the origin of the whole-brain space instead of in its real anatomical location.

---

## 7. Crop bbox and offset correction

The `.mat` file stores the crop position inside the whole-brain dataset in:

```python
mat["info"].bbox_mmll
mat["info"].bbox_mmxx
```

For the current crop:

```text
bbox_mmll = [833, 2497, 6657, 1072, 1072, 1072]
bbox_mmxx = [833, 2497, 6657, 1904, 3568, 7728]
```

These fields encode the crop bounding box. In practice:

```text
bbox origin ≈ [833, 2497, 6657]
```

and the global crop location is approximately:

```text
X: 833  → 1904
Y: 2497 → 3568
Z: 6657 → 7728
```

The offset correction adds the bbox origin to every node coordinate and every edge point:

```text
global_coords = local_scaled_coords + bbox_origin
```

This is implemented in `mvn_offset_correction.ipynb`.

---

## 8. Important: the bbox offset is not scaled

The bbox offset is **not multiplied by `1.051`** in the final working workflow.

Reason:

- the bbox is already stored in the global coordinate system of the whole-brain reference volume;
- the whole-brain reference used in ParaView is in that global coordinate system;
- scaling the bbox would move the crop into a different coordinate system and would break the alignment.

Therefore, for visualization against the whole-brain volume, the correction is:

```text
global_coords = local_coords × 1.051 + raw_bbox_offset
```

not:

```text
global_coords = (local_coords + bbox_offset) × 1.051
```

The shrinkage correction is applied to the local graph geometry. The bbox is used only as a translation into whole-brain space.

---

## 9. Workflow summary

### For the crop / MVN graph

```text
ML20180815_240_c5o1_578.mat
        │
        ▼
mat_to_pkl_scaling.py
        │
        ▼
ML20180815_240_c5o1_578_mvn1_scaled.pkl
        │
        │  local crop coordinates
        │  geometry scaled by 1.051
        │  radii/diameters unchanged
        ▼
mvn_offset_correction.ipynb
        │
        ▼
ML20180815_240_c5o1_578_mvn1_scaled_offset.pkl
        │
        │  global whole-brain coordinates thanks to offset
        │  suitable for ParaView overlay with whole-brain volume
```

### For a whole-brain graph

```text
wholebrain.mat
        │
        ▼
mat_to_pkl_scaling.py
        │
        ▼
wholebrain_scaled.pkl
```

No bbox offset correction is needed if the whole-brain graph is already in global coordinates.

---

## 11. References

- Ji et al. 2021, *Neuron*: whole mouse brain microvascular connectome; whole-brain graph statistics; sample deformation and radius calibration.
- Ji et al. 2026, *PNAS*: cortical MVN / crop network used for network-flow analysis and RBC tracking context.
- `mat_to_pkl_scaling.py`: conversion from Ji `.mat` to MVN-style `igraph` pickle and shrinkage scaling.
- `mvn_offset_correction.ipynb`: post-processing notebook for moving a crop graph from local crop space into global whole-brain coordinates.
