# MicroBrain: Blood Vessel Analysis & Visualization

## 🎯 Project Aim

MicroBrain provides a clear, reproducible toolkit to turn vessel tables (CSV) into analysis-ready Python objects (PKL), preserve vessel geometry, and enable unit-aware measurements and region-based studies. It focuses on:

- Converting CSV vessel data into a single working PKL (`18_igraph.pkl`) used for analysis
- Preserving curved vessel geometry and per-point properties
- Supporting measurements in image pixels, atlas units, and micrometers (µm)
- Extracting regional subgraphs (rectangular ROIs) for targeted study
- Performing node- and topology-centered analyses (sizes, connectivity, density, redundancy)
- Exporting VTP files for 3D viewing in ParaView

The real goal: make vascular graph analysis transparent, reproducible, and easy to compare across brain regions and datasets.

---

## ✅ What's Been Built

A complete system from raw data to visualization:

- ✓ **Create analysis files** - Convert CSV vessel data into Python format (pkl) ready for analysis
- ✓ **Convert measurements** - Switch between image pixels, atlas references, and physical measurements (micrometers)
- ✓ **Keep vessel curves** - Preserve the realistic curved paths of blood vessels through all processing
- ✓ **Extract regional networks** - Focus on specific brain regions by cutting out rectangular areas
- ✓ **Analyze vessel properties** - Study vessel sizes, connectivity, density, branching patterns, and node properties
- ✓ **Create 3D visualizations** - Export data as files to view vessels in 3D inside ParaView
- ✓ **Compare brain regions** - Study how vessel patterns differ between areas like Hippocampus and Somatomotor cortex

---

## 📁 Folder Structure & Contents

### **`codes/`** — Main Analysis Toolkit
Complete pipeline for processing vessel data from raw data to visualization.

#### **`codes/CSVtoPKL/`** — Convert Raw Data to Usable Format
Takes the raw vessel data (in spreadsheet/table format) and converts it into a Python format that's easy to work with.

**What each script does:**
- **`CSV2Pickle_SOFIA.py`** - Quick conversion with basic information
- **`build_graph_outgeom_voxels.py`** - Full conversion that keeps the curved shape of vessels
- **`CSVtoPKL_FULLGEOM.py`** - Alternative way to store the curved vessel paths
- **`convert_outgeom_voxels_to_um.py`** - Converts from pixel measurements to physical micrometers

**Result:** A Python data file containing:
- The vessel network (all the junctions and connections)
- The 3D curved path each vessel takes
- Properties of each vessel (size, type, location)
- Both pixel-based and micrometer-based measurements

---

#### **`codes/GTtoCSV/`** — Convert From Raw Format to Spreadsheets
Takes data from the original brain imaging software (graph-tool format) and converts it to spreadsheet-friendly tables.

**What's included:**
- **`gt2CSV_SOFIA.py`** - Converts vessel connections, coordinates, sizes, and labels
- **`Franca_Extract_graph_info.py`** - Shows you information about the vessel network

**Why use this:** Lets you view and work with raw data in Excel or CSV readers before further processing.

---

#### **`codes/cutting/`** — Zoom Into Specific Brain Regions
Lets you select a rectangular box in the brain and extract just the blood vessels inside that region.

**What it does:**
- Keeps only vessels in your chosen box
- Cuts vessels that cross the box edge
- Recalculates vessel properties for the smaller region
- Works with both pixel and physical (micrometer) measurements

**Available tools:**
- **`cut_outgeom_roi_UM.py`** - Cut using real-world measurements (micrometers) — recommended
- **`cut_outgeom_roi_VOX.py`** - Cut using pixel measurements


---

#### **`codes/PKLtoVTP/`** — Create 3D Models for Viewing
Converts your vessel data into 3D files you can open and explore in ParaView.

**Available scripts:**
- **`PKLtoVTP_Tortuous_OUTGEOM.py`** - Best option: shows realistic curved vessel paths
- **`pkl2vtp_SOFIA.py`** - General-purpose converter with flexible options
- **`PKLtoVTP_nonT_basic.py`** - Quick preview using straight vessel connections (non tortuous graph)

**What gets included in the 3D file:**
- Vessel sizes (radius and diameter) at each point
- Vessel type (artery, vein, capillary)
- How curved each vessel is (tortuosity)
- Overall vessel length

If any more information is needed, you only need to add it to the code as Point/Cell Data to export.

---

#### **`codes/Graph Analysis & by region/`** — Measure & Analyze Vessels
Tools for understanding properties of the vessel network and comparing different brain regions.

**What you can measure:**

- **Basic properties:** Vessel lengths, diameters, how many connections each vessel has
- **Node analysis:** Junction points, their properties, where they're located (central vs. boundary)
- **Spatial patterns:** Where arteries vs. veins are located, vessel size variations
- **Vessel density:** How much of the region is filled with vessels, density at different depths
- **Redundancy:** How many alternate paths exist for blood to flow from arteries to veins

**Organization:**

```
Graph Analysis & by region/
├── graph_analysis_functions.py    # Main analysis library
├── General analysis templates                   
├── Brain region-specific notebooks:
│   ├── Hippocampal area analysis
│   ├── Somatomotor area analysis
│   └── Compare Hippocampus vs. Somatomotor
└── Tools to select specific brain regions (SelectBrainRegion)
```

**How to use:**
- Choose to work with voxels or physical micrometers
- Select a brain region (hippocampal or somatomotor in this case --> defined by annotation ID) or define a custom area you are interested in 
- Run analysis to get measurements and statistics

---

### **`CSV/`** — Raw Data in Spreadsheet Format
The original data extracted from brain scans, in table form.

**Key files:**
- `vertices.csv`, `edges.csv` - Which vessels connect to which, the basic network structure
- `coordinates.csv`, `coordinates_atlas.csv` - Where each vessel node is located
- `radii.csv`, `radii_atlas.csv` - Vessel sizes
- `edge_geometry_*.csv` - Detailed information about vessel geometry
- `annotation.csv`, `artery.csv`, `vein.csv` - Labels for what type of vessel (artery, vein, capillary)
- `length.csv` - Calculated measurements like length and distance to brain surface
- `distance_to_surface.csv`

---

### **`whole brain/`** — Complete Brain Network
The full vessel network for the entire brain, ready to view and analyze.

- **`18_vessels_graph.gt`** - Original vessel network data file (graph-tool format)
- **`18_igraph.pkl`** - Complete brain vessel network in Python analysis format (pkl) - the working file for all analysis
- **`18_igraph.vtp`** - 3D model of all brain vessels you can open in ParaView
- **`Extract Area Annotated from graph Brain/`** - Information about brain regions
  - `ABA_annotation_last.json` - Maps vessels to standard brain areas (Allen Brain Atlas)
  - Files for specific regions like Hippocampus and Somatomotor cortex


---

## 📏 Understanding Measurements & Units

Your vessel data uses **three different measurement systems**. This can be complex, but here's what matters:

### The Quick Reference

**Know which units are in your files and use this to convert:**

| From | To | Multiply by |
|---|---|---|
| Scan image pixels → Micrometers | 1.625 (X,Y) or 2.5 (Z) |
| Brain atlas units → Micrometers | 25 |
| Scan pixels → Atlas units | ÷ ~16 |

### Why Three Systems Exist

1. **Scan Image Pixels** - Grid of the brain scan (1.625 µm in X/Y, 2.5 µm in Z)
2. **Brain Atlas Units** - Standard reference used by neuroscientists (25 µm per unit)
3. **Micrometers** - Actual physical measurements

The key: **track which system your file uses**.

---

## ⚠️ Important: ParaView Has No Units

**ParaView doesn't know what units your measurements are in.** It just displays numbers.

- If your 3D file has coordinates like (100, 200, 300), ParaView shows those numbers
- It doesn't know if they're scan pixels, atlas points, or micrometers
- **The units come from what YOU put in your file** — you must track them yourself

### What This Means

1. **When exporting to 3D:** Decide if you want scan pixels or micrometers in your 3D file
2. **When measuring in ParaView:** Remember which units are in your file
3. **When recording regions:** Always write down the units (e.g., "box at pixels (500, 400, 300)" or "box at µm (810, 650, 750)")
4. **When comparing measurements:** Make sure everything uses the same units

### Example

```python
# In your code, you know the units
box_center = [500, 400, 300]  # in voxels (image)
box_size = [1000, 800, 600]   # in micrometers

# Convert carefully to one system
box_size_in_voxels = box_size / [1.625, 1.625, 2.5]  # now in voxels

# Export to 3D file 
# → ParaView will show the numbers, but won't label them

# YOU must remember: "The file has coordinates in voxels of image"
```

**Pro tip:** Write a comment in your code with the units, and check your exported 3D files to confirm before opening in ParaView.

---

## 🚀 Getting Started

### 1. **Create Analysis File from CSV Data**
```bash
cd codes/CSVtoPKL/
python build_graph_outgeom_voxels.py
# → Creates 18_igraph.pkl with the complete vessel network
```

### 2. **Convert to Physical Measurements (Optional)**
```bash
python convert_outgeom_voxels_to_um.py
# → Adds micrometer-space versions of coordinates and properties
```

### 3. **View in 3D**
```bash
cd ../PKLtoVTP/
python PKLtoVTP_Tortuous_OUTGEOM.py
# → Creates a .vtp file you can open in ParaView
```

### 4. **Select a Brain Region**
```
Step 1: Open the .vtp file in ParaView
Step 2: Explore and find the brain region you want (e.g., Hippocampus)
Step 3: Define a 3D box around your region
Step 4: In ParaView's built-in terminal or Python console, use the coordinates
       (Instructions in codes/cutting/README.md)
```

### 5. **Extract and Analyze Your Region**
```python
# Run in Python (or ParaView Python console)
from codes.cutting.cut_outgeom_roi_UM import *
cut_graph = cut_by_box_um(your_data, center, size, resolution)

# Then analyze
from codes.Graph Analysis & by region import *
results = analyze_vessels(cut_graph)
```

---

## 📊 How Data Flows Through the System

```
Brain scan data
    ↓
[Convert to usable format]
    ↓
Raw vessel network (in pixels)
    ↓
[Optional: Convert to real-world micrometers]
    ↓
Complete vessel network
    ↓
[Cut out specific regions]
    ↓
Regional vessel networks
    ↓
[Create 3D visualization files]
    ↓
View in ParaView + Analyze
    ↓
Vessel measurements, statistics, comparisons
```

---

## 📚 More Information

Each folder has its own detailed guide:

- **`codes/CSVtoPKL/README.md`** - Converting raw data to working format
- **`codes/GTtoCSV/README.md`** - Converting between file formats
- **`codes/cutting/README.md`** - How to cut out specific brain regions
- **`codes/PKLtoVTP/README.md`** - Creating 3D visualization files
- **`codes/Graph Analysis & by region/README.md`** - How to measure and analyze vessels

---

## 👤 Author

- **Ana Barrio** - Project creation and development

---

## 📝 Quick Reference

- **Vessel types:** 
  - Arterioles (small arteries)
  - Venules (small veins)
  - Capillaries (tiny vessels connecting arteries and veins)

- **Key measurements:**
  - **Diameter** = how wide the vessel is (twice the radius)
  - **Tortuosity** = how curved the vessel is (actual length / straight-line distance)
  - **Density** = how much of an area is filled with vessels

- **Remember:** ParaView shows numbers without units — you must track which units are in your files!

---

**Last Updated:** February 20, 2026
