## Paris Calculations

This document explains the key calculations used in the Paris code to help you understand how various measurements are derived.

### 1. Distance to Surface

**Measurement unit:** Voxels from the atlas at 25 μm resolution (referred to as "atlas voxels")

**How it is computed:** The code uses the `distance_transform_edt` function from `scipy.ndimage`. This function:
- Creates a binary mask from the atlas where brain voxels are set to 1 and voxels outside the brain are set to 0
- For each voxel inside the brain, calculates the shortest Euclidean distance to the nearest surface voxel
- Returns an array containing the distance from each voxel to the brain surface

**Important note:** The distance data is precomputed and loaded from the atlas (not calculated dynamically). The code uses the 25 μm Allen Brain Atlas as the reference, from which it extracts the distance-to-surface file (`ABA_25um_2017_distance_to_surface.tif`), along with annotations, hemisphere data, and reference volumes. These resources are loaded directly and then reoriented and cropped to match your specific dataset.

### 2. Radii of the Vertex

**Measurement unit:** Voxels from the original image (resolution 1.625 × 1.625 × 2.5 μm)

**How it is calculated:** For the non-tortuous graph, the radius assigned to a vertex is the maximum radius from all points in the polyline (the series of line segments forming the edge):

$$r_{vertex} = \max(r_1, r_2, r_3, ..., r_n)$$

where $r_1, r_2, ..., r_n$ are the radii of each individual point in the polyline. This maximum radius value is then assigned to both endpoints (vertices) of the edge.

### 3. Length of the Edge

**Measurement unit:** Voxels from the original image (resolution 1.625 × 1.625 × 2.5 μm)

**How it is calculated:** For the tortuous graph, the edge length is computed by summing the distances between all consecutive points along the line segment (polyline):

$$L_{edge} = \sum_{i=1}^{n-1} d(p_i, p_{i+1})$$

where $p_1, p_2, ..., p_n$ are the points in the polyline and $d(p_i, p_{i+1})$ is the Euclidean distance between consecutive points. This gives the total path length along the tortuous path rather than just the straight-line distance between endpoints.

### 4. Radii in Atlas Space

**How it is calculated:** The vertex radii are rescaled from the original image space to the atlas space (25 μm voxel grid). This rescaling uses a transformation factor computed from the dimensions of the source and target images:

$$r_{atlas} = r_{original} \times scaling\_factor$$

where the scaling factor is 0.0768 (precisely 0.076804589994077).

References:
- [TubeMap.py lines 134-146](https://github.com/ClearAnatomics/ClearMap/blob/6d6e37f98457b821fb9f8a56cd7b4cb048bf272e/ClearMap/Scripts/TubeMap.py#L134C2-L146C9)
- [TubeMap.py lines 482-492](https://github.com/ClearAnatomics/ClearMap/blob/6d6e37f98457b821fb9f8a56cd7b4cb048bf272e/ClearMap/Scripts/TubeMap.py#L482C1-L492C83)

**Scaling ratio:** You can calculate the ratio between the original and atlas space radii to verify the transformation:

$$ratio = \frac{r_{original}}{r_{atlas}} \approx 13.013$$

**To convert to micrometers:** Multiply the atlas-space radii by 25 μm (the atlas voxel size):

$$r_{micrometers} = r_{atlas} \times 25 \text{ μm}$$

**Additional notes:** The transformation factor 0.076804589994077 is computed from the source and sink image shapes. This same factor can be used in other contexts where radius rescaling between the original image space and the 25 μm atlas grid is needed.



