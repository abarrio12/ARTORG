import pickle
import vtk
import numpy as np

'''
length_tortuous = longitud real siguiendo la geometría del vaso (suma de distancias entre puntos consecutivos de la polyline).
length = la longitud “del edge” que te venía del CSV / atributo del grafo. En muchos pipelines suele ser la distancia recta entre nodos 
(chord) o una longitud “resumida” precomputada (depende de cómo exportaste length.csv).
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
radii_p = np.asarray(geom["radii"], dtype=np.float32) if "radii" in geom else None
diam_p = np.asarray(geom["diameters"], dtype=np.float32) if "diameters" in geom else None
if diam_p is None and radii_p is not None:
    diam_p = (2.0 * radii_p).astype(np.float32)

lengths2_p = np.asarray(geom["lengths2"], dtype=np.float32) if "lengths2" in geom else None
lengths_p  = np.asarray(geom["lengths"], dtype=np.float32)  if "lengths" in geom else None

def _check(name, arr):
    if arr is not None and len(arr) != nP:
        raise ValueError(f"geom['{name}'] must match x/y/z length.")
_check("annotation", ann)
_check("radii", radii_p)
_check("diameters", diam_p)
_check("lengths2", lengths2_p)
_check("lengths", lengths_p)

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
radius_array       = make_float_array("radius")
diameter_array     = make_float_array("diameter")
length_array       = make_float_array("length")
length_tort_array  = make_float_array("length_tortuous")
tortuosity_array   = make_float_array("tortuosity")

# ============================
# PointData arrays (attach directly, no loop)
# ============================
pointdata_arrays = []

if ann is not None:
    a = vtk.vtkIntArray(); a.SetName("annotation"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, int(ann[i]))
    pointdata_arrays.append(a)

if radii_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("radii"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(radii_p[i]))
    pointdata_arrays.append(a)

if diam_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("diameters"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(diam_p[i]))
    pointdata_arrays.append(a)

if lengths2_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("lengths2"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(lengths2_p[i]))
    pointdata_arrays.append(a)

if lengths_p is not None:
    a = vtk.vtkFloatArray(); a.SetName("lengths"); a.SetNumberOfTuples(nP)
    for i in range(nP): a.SetValue(i, float(lengths_p[i]))
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

    if "radius" in G.es.attributes():
        r_edge = float(G.es[e]["radius"])
    elif radii_p is not None:
        r_edge = float(np.nanmean(radii_p[s:en]))
    else:
        r_edge = np.nan
    radius_array.InsertNextValue(r_edge)

    if "diameter" in G.es.attributes():
        d_edge = float(G.es[e]["diameter"])
    elif not np.isnan(r_edge):
        d_edge = 2.0 * r_edge
    elif diam_p is not None:
        d_edge = float(np.nanmean(diam_p[s:en]))
    else:
        d_edge = np.nan
    diameter_array.InsertNextValue(d_edge)

    if "length" in G.es.attributes():
        L = float(G.es[e]["length"])
    elif lengths2_p is not None:
        L = float(np.sum(lengths2_p[s:en]))
    else:
        P = np.column_stack([x[s:en], y[s:en], z[s:en]]).astype(np.float64)
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    length_array.InsertNextValue(L)

    Lt = float(G.es[e]["length_tortuous"]) if "length_tortuous" in G.es.attributes() and G.es[e]["length_tortuous"] is not None else np.nan
    length_tort_array.InsertNextValue(Lt)

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
polydata.GetCellData().AddArray(radius_array)
polydata.GetCellData().AddArray(diameter_array)
polydata.GetCellData().AddArray(length_array)
polydata.GetCellData().AddArray(length_tort_array)
polydata.GetCellData().AddArray(tortuosity_array)
polydata.GetCellData().SetActiveScalars("nkind")

# PointData
for a in pointdata_arrays:
    polydata.GetPointData().AddArray(a)

# choose default active scalar
if polydata.GetPointData().HasArray("annotation"):
    polydata.GetPointData().SetActiveScalars("annotation")
elif polydata.GetPointData().HasArray("radii"):
    polydata.GetPointData().SetActiveScalars("radii")
elif polydata.GetPointData().HasArray("diameters"):
    polydata.GetPointData().SetActiveScalars("diameters")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(out_vtp)
writer.SetInputData(polydata)
writer.SetDataModeToAppended()
writer.EncodeAppendedDataOff()
writer.Write()

print("Saved", out_vtp)
print("Points:", polydata.GetNumberOfPoints(), "Cells:", polydata.GetNumberOfCells())
