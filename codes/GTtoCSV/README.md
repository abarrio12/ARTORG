# GTtoCSV: Graph-Tool Format to CSV Conversion

## Overview
This module converts **graph-tool graph files** (.gt format) into **tabular CSV format**

## Goal
Transform binary graph-tool objects into compatible CSV tables:
1. Extract graph topology (edges, vertices)
2. Export node attributes (coordinates, radii, annotations)
3. Export edge properties (connectivity, vessel type, diameter, length)
4. Enable analysis in non-graph tools (Excel)


## Folder Structure

```
GTtoCSV/
├── gt2CSV_SOFIA.py                    # Main GT → CSV converter
├── Franca_Extract_graph_info.py       # Extract and display graph info
└── README.md
```

## Main Scripts

### 1. **gt2CSV_SOFIA.py** - Graph-Tool to CSV Converter

Primary conversion utility implementing the `ReadWriteGraph` class.

**Class Methods:**

```python
class ReadWriteGraph:
    
    # File I/O
    loadGraphFromFile(file)           # Load .gt file
    saveFile(array, filepath)         # Save array to CSV
    
    # Edge properties → CSV
    writeEdges(file)                  # Edge connectivity (source, target)
    writeEdgePropertyRadii(file)      # Edge radii
    writeEdgePropertyVein(file)       # Vein annotation (binary: 0/1)
    writeEdgePropertyArtery(file)     # Artery annotation (binary: 0/1)
    
    # Coordinate/radii → CSV
    writeCoordinates(file)            # Vertex coordinates
    writeRadii(file)                  # Vertex radii
```

**Typical Usage:**

```python
from gt2CSV_SOFIA import ReadWriteGraph

# Initialize and load graph
rw = ReadWriteGraph(file="graph.gt")

# Set output folder
rw.folderPrefix("/home/admin/Ana/MicroBrain/CSV")

# Export all properties
rw.writeEdges("edges.csv")
rw.writeVertexPropertyAnnotation("annotation.csv")                            # vertex properties
rw.writeEdgePropertyRadii("radii_edge.csv")                                   # edge properties
rw.writeGraphPropertyEdgeGeometryCoordinates("edge_geometry_coordinates.csv") # graph properties
```

### 2. **Franca_Extract_graph_info.py** - Graph Structure Inspector

Before converting a graph-tool file to CSV, use this utility to understand what data is available:

**Purpose:**
- Inspect the graph-tool file without full conversion
- List all vertex properties (coordinates, radii, annotations, etc.)
- List all edge properties (vessel type, diameter, geometric info)
- Display graph-level metadata
- Get counts: number of vertices, edges, attributes

**Usage:**
```python
from Franca_Extract_graph_info import extract_graph_info

# Inspects graph.gt and prints available properties
info = extract_graph_info("graph.gt")
# Output shows what attributes are available to export
```

**Why use this?**
- Know exactly which properties exist before calling `writeEdges()`, `writeCoordinates()`, etc.
- Verify graph integrity and structure before processing
- Decide which properties to export to CSV
- Debug issues if certain attributes are missing or unexpected

## CSV Output Files

Generated files in `CSV/` folder:

### Vertex-Related CSVs
- `vertices.csv` - Vertex indices (implicit node list)
- `coordinates.csv` - Vertex coordinates (N × 3): x, y, z
- `radii.csv` - Vertex radii (N × 1)
- `coordinates_atlas.csv` - Atlas-space coordinates
- `radii_atlas.csv` - Atlas-space radii
- `annotation.csv` - Vertex annotations/labels

### Edge-Related CSVs
- `edges.csv` - Edge connectivity (E × 2): source, target
- `radii_edge.csv` - Per-edge radii
- `artery.csv` - Artery classification (binary: 0/1)
- `vein.csv` - Vein classification (binary: 0/1)
- `artery_raw.csv` - Raw artery values
- `artery_binary.csv` - Thresholded artery
- `length.csv` - Edge lengths
- `radii_atlas_edge.csv` - Per-edge radii atlas


### Geometric CSVs
- `edge_geometry_coordinates.csv` - Polyline points per edge
- `edge_geometry_radii.csv` - Radii along the polyline 
- `edge_geometry_radii_atlas.csv` - Radii atlas along the polyline
- `edge_geometry_annotation.csv` - Annotations along polyline
- `edge_geometry_indices.csv` - Point indices (geom_start, geom_end)
- `edge_geometry_artery_binary.csv` - Artery type along polyline

**Note:** The terms "geometry," "points," and "polyline" refer to the same thing: individual nodes along the tortuous (curved) vessel path. They are used interchangeably.

## Data Workflow

```
Original Source (Paris/ClearMap)
        ↓
[build_graph_gt] → graph.gt (graph-tool format) (Paris)
        ↓
[GTtoCSV] → CSV files (tabular format)
        ↓
[CSVtoPKL] → graph.pkl (Python pickle format)
        ↓
Further analysis: cutting, visualization, metrics
```

## Related Modules

- **CSVtoPKL:** Converts CSV → PKL (builds graph structure + geometry)
- **PKLtoVTP:** Converts PKL → VTP (visualization format)

## Author
Sofia (`gt2CSV.py`) and Franca (`Extract_graph_info.py`) dataset adaptation

Updated: Ana Barrio - Feb 2026
