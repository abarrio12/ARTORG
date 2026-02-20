import pickle
import vtk
import numpy as np

'''
Data structure in PKL:
- length = arc length along the polyline (from CSV)
- lengths2 = distance between consecutive geometry points (computed from diffs)
- radii_atlas = radius in atlas voxel space (25 µm grid)

VTP export:
- Points: geometry coordinates (x, y, z)
- CellData: per-edge attributes (nkind, radius_atlas, diameter_atlas, length, tortuosity)
- PointData: per-point attributes (annotation, radii_atlas, etc.)
'''

in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
out_vtp = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.vtp"

data = pickle.load(open(in_path, "rb"))

G = data["graph"]
geom = data["geom"]

x = np.asarray(geom["x"], dtype=np.float32)
y = np.asarray(geom["y"], dtype=np.float32)
z = np.asarray(geom["z"], dtype=np.float32)
nP = len(x)

# -------- pointwise arrays (optional) ----------
ann = np.asarray(geom["annotation"], dtype=np.int32) if "annotation" in geom else None
radii_atlas_p = np.asarray(geom["radii_atlas_geom"], dtype=np.float32) if "radii_atlas_geom" in geom else None

diam_atlas_p = None
if radii_atlas_p is not None:
    diam_atlas_p = (2.0 * radii_atlas_p).astype(np.float32)

lengths2_p = np.asarray(geom["lengths2"], dtype=np.float32) if "lengths2" in geom else None

def _check(name, arr):
    if arr is not None and len(arr) != nP:
        raise ValueError(f"geom['{name}'] must match x/y/z length.")
_check("annotation", ann)
_check("radii_atlas_geom", radii_atlas_p)
_check("diameters_atlas", diam_atlas_p)
_check("lengths2", lengths2_p)


# ============================
# VTK: points ONCE
# ============================
points = vtk.vtkPoints()
points.SetNumberOfPoints(nP)
for i in range(nP):
    points.SetPoint(i, float(x[i]), float(y[i]), float(z[i]))

lines = vtk.vtkCellArray()

# ============================
# CellData arrays
# ============================
def make_int_array(name):
    a = vtk.vtkIntArray(); a.SetName(name); return a
def make_float_array(name):
    a = vtk.vtkFloatArray(); a.SetName(name); return a

nkind_array        = make_int_array("nkind")
radius_atlas_array = make_float_array("radius_atlas")
diameter_atlas_array = make_float_array("diameter_atlas")
length_array       = make_float_array("length")
tortuosity_array   = make_float_array("tortuosity")

# ============================
# PointData arrays (attach directly, no loop)
# ============================
pointdata_arrays = []

if ann is not None:
    a = vtk.vtkIntArray(); a.SetName("annotation"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, int(ann[i]))
    pointdata_arrays.append(a)

if radii_atlas_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("radii_atlas"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(radii_atlas_p[i]))
    pointdata_arrays.append(a)

if diam_atlas_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("diameters_atlas"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(diam_atlas_p[i]))
    pointdata_arrays.append(a)

if lengths2_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("lengths2"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(lengths2_p[i]))
    pointdata_arrays.append(a)


# ============================
# Build polylines referencing GLOBAL point ids
# ============================
for e in range(G.ecount()):
    s = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])
    npts = en - s
    if npts < 2:
        continue

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)
    for i in range(npts):
        polyline.GetPointIds().SetId(i, s + i)  # <--- global point id

    lines.InsertNextCell(polyline)

    # ---- CellData ----
    nkind_array.InsertNextValue(int(G.es[e]["nkind"]) if "nkind" in G.es.attributes() else -1)

    if "radius_atlas" in G.es.attributes():
        r_atlas_edge = float(G.es[e]["radius_atlas"])
    elif radii_atlas_p is not None:
        r_atlas_edge = float(np.nanmax(radii_atlas_p[s:en]))
    else:
        r_atlas_edge = np.nan
    radius_atlas_array.InsertNextValue(r_atlas_edge)

    if "diameter_atlas" in G.es.attributes():
        d_atlas_edge = float(G.es[e]["diameter_atlas"])
    elif not np.isnan(r_atlas_edge):
        d_atlas_edge = 2.0 * r_atlas_edge
    elif diam_atlas_p is not None:
        d_atlas_edge = float(np.nanmax(diam_atlas_p[s:en])) # radii of the edge is max(radii of the points), so diameters of the edge is max(diameters of the points)
    else:
        d_atlas_edge = np.nan
    diameter_atlas_array.InsertNextValue(d_atlas_edge)

    if "length" in G.es.attributes():
        L = float(G.es[e]["length"])
    elif lengths2_p is not None:
        L = float(np.sum(lengths2_p[s:en]))
    else:
        P = np.column_stack([x[s:en], y[s:en], z[s:en]]).astype(np.float64)
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    length_array.InsertNextValue(L)

    tau = float(G.es[e]["tortuosity"]) if "tortuosity" in G.es.attributes() and G.es[e]["tortuosity"] is not None else np.nan
    tortuosity_array.InsertNextValue(tau)

# ============================
# PolyData + write
# ============================
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

# CellData
polydata.GetCellData().AddArray(nkind_array)
polydata.GetCellData().AddArray(radius_atlas_array)
polydata.GetCellData().AddArray(diameter_atlas_array)
polydata.GetCellData().AddArray(length_array)
polydata.GetCellData().AddArray(tortuosity_array)
polydata.GetCellData().SetActiveScalars("nkind")

# PointData
for a in pointdata_arrays:
    polydata.GetPointData().AddArray(a)

# choose default active scalar
if polydata.GetPointData().HasArray("annotation"):
    polydata.GetPointData().SetActiveScalars("annotation")
elif polydata.GetPointData().HasArray("radii_atlas"):
    polydata.GetPointData().SetActiveScalars("radii_atlas")
elif polydata.GetPointData().HasArray("diameters_atlas"):
    polydata.GetPointData().SetActiveScalars("diameters_atlas")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(out_vtp)
writer.SetInputData(polydata)
writer.SetDataModeToAppended()
writer.EncodeAppendedDataOff()
writer.Write()

print("Saved", out_vtp)
print("Points:", polydata.GetNumberOfPoints(), "Cells:", polydata.GetNumberOfCells())
