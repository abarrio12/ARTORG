# GTtoCSV: Graph-Tool Format to CSV Conversion

## Overview
This module converts **graph-tool graph files** (.gt format) into **tabular CSV format**, enabling data interchange with other tools, databases, spreadsheet analysis, and cross-platform compatibility.

## Goal
Transform binary graph-tool objects into compatible CSV tables:
1. Extract graph topology (edges, vertices)
2. Export node attributes (coordinates, radii, annotations)
3. Export edge properties (connectivity, vessel type, diameter)
4. Enable downstream analysis in non-graph-specific tools (Excel)


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
rw.writeCoordinates("coordinates.csv")
rw.writeRadii("radii.csv")
rw.writeEdgePropertyRadii("radii_edge.csv")
rw.writeEdgePropertyArtery("artery.csv")
rw.writeEdgePropertyVein("vein.csv")
```

### 2. **Franca_Extract_graph_info.py** - Graph Metadata Extraction

Utility to inspect and display graph structure:
- Vertex properties (available attributes)
- Edge properties (available attributes)
- Graph metadata
- Counts and summaries

Used for **data exploration** before conversion.

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

### Geometric CSVs
- `edge_geometry_coordinates.csv` - Polyline points per edge
- `edge_geometry_radii.csv` - Radii along each polyline
- `edge_geometry_annotation.csv` - Annotations along polyline
- `edge_geometry_indices.csv` - Point indices (geom_start, geom_end)
- `edge_geometry_artery_binary.csv` - Artery type along polyline

## Data Workflow

```
Original Source (Paris/ClearMap)
        ↓
[build_graph] → graph.gt (graph-tool format)
        ↓
[GTtoCSV] → CSV files (tabular format)
        ↓
[CSVtoPKL] → graph.pkl (Python pickle format)
        ↓
Further analysis: cutting, visualization, metrics
```

## Format Specifications

### edges.csv
```
source,target
0,1
1,2
1,3
...
```

### coordinates.csv
```
x,y,z
100.5,200.3,50.1
101.2,201.5,51.3
...
```

### radii_edge.csv
```
radius
2.5
3.1
2.8
...
```

### artery.csv / vein.csv
```
classification
1      # True (is artery/vein)
0      # False (not artery/vein)
...
```

## Related Modules

- **CSVtoPKL:** Converts CSV → PKL (builds graph structure + geometry)
- **PKLtoVTP:** Converts PKL → VTP (visualization format)

## Author
Sofia and Franca dataset adaptation
Updated: Ana Barrio - Feb 2026
