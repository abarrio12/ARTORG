## Paris calculation

These file aims to explain some of the calculations made in the Paris code for better understanding

1. Distance to surface

Measurement: voxels scaled to atlas at 25 um ("atlas voxels")

2. How it was computed: I found that they use the function distance_transform_edt from scipy.ndimage
This function creates a binary mask from the atlas (voxels inside the brain have a value > 0 and are set to 1, voxels outside are set to 0, defining the brain surface). Then, it computes for each voxel inside the brain the shortest euclidean distance to the nearest voxel outside the brain. The result is an array that contains all distances from the voxels to the surface.
Lines in Paris code
API of the function

I also found that the distance_file used is not computed dynamically (at least I have not found the parte of the code where they do it yet). They use the atlas 25 um (Link to Paris code) and from there they extract the components (distance to suface: ABA_25um_2017_distance_to_surface.tif ,annotation, hemispheres and reference volumes) and can be found in the folder Resources.
Decompress_atlas function
Prepare annotation files

 Then is loaded from this atlas resources folder directly (Link to Paris line)  and after is only reoriented and cropped to match the dataset (Link to Paris lines for reorientation).

Hope it makes some sense, is a bit tricky. I will write it in the readme

2. Radii of the vertex
Measurement: voxels of image (resolution 1.625 x 1.625 x 2.5)
For the radii of the vertex (not atlas careful!) what they did is they took the maximum of the radii of the points in the polyline and assign it to the verteces of the edge (non tortuous graph)

3. Length of the edge
Measurement: voxels of image (resolution 1.625 x 1.625 x 2.5)
For the edge of the length they sum the lengths2, meaning, the lengths of the segments of the tortuous = distance between points.

4. Radii atlas
Hi Sofia, since we were not fully sure about the radii units/scaling, I checked how radii_atlas is computed in the Paris code.
I found that for the vertex, radii_atlas is indeed rescaled to the atlas (25 µm voxel grid) using a resampling factor computed from the source and sink image shapes:
https://github.com/ClearAnatomics/ClearMap/blob/6d6e37f98457b821fb9f8a56cd7b4cb048bf272e/ClearMap/Scripts/TubeMap.py#L134C2-L146C9
https://github.com/ClearAnatomics/ClearMap/blob/6d6e37f98457b821fb9f8a56cd7b4cb048bf272e/ClearMap/Scripts/TubeMap.py#L482C1-L492C83

Basically, it converts radii from the original image voxel space into voxels of the 25 µm atlas grid (0.076804589994077 is the transformation factor in case we needed any other time)

In the end, to obtain radii_Atlas in micrometers, we simply multiply by 25, as we discussed the other day (but just to be sure I did the checking)



