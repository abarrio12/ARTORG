# Cutting: Extract Vessels from a 3D Box

This code cuts a vessel graph using a rectangular 3D box.

You can use it to select a region of interest from the full graph. The output is a new smaller graph containing only the vessels inside the box.

---

## Main script

### `Cut_The_Graph_formatted.py`

This code uses the MVN-format attributes, so be aware that your input .pkl is correct.

This script has two ways to work with the box:

1. classify edges only
2. classify and cut edges

---

## Option 1: classify edges only

```python
get_edges_in_boundingBox_vertex_based_2()
```


This is the simpler and faster option.

It separates edges into:

edges_in_box         → vessels fully inside the box
edges_across_border  → vessels crossing the box boundary
edges_outside_box    → vessels outside the box

Use this when you only want to know which edges are inside, outside, or crossing the box.

## Option 2: classify and cut edges

```python
get_edges_in_boundingBox_vertex_based()
```

This is the full clipping option.

It does the same classification as above, but also cuts vessels that cross the box boundary.

For crossing vessels, it:

1. finds where the vessel intersects the box surface
2. creates a new boundary vertex
3. keeps only the part of the vessel inside the box
4. recomputes the edge geometry and attributes

Use this when you want a new graph that contains only the vessel parts inside the box.

## Typical workflow
```
1. Draw or define a box in ParaView
2. Note the box center and size
3. Run the cutting script
4. Save the cropped graph
5. Visualize the result in ParaView
```


## Important note about units

ParaView does not know whether your numbers are voxels or micrometers. 
It only displays the coordinates stored in the file.

So before cutting, check whether your graph is in: `voxels` or `µm`

The box coordinates must use the same unit system as the graph.