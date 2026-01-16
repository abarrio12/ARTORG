import pickle
import vtk
import numpy as np

# ============================
# Input / Output
# ============================
in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"

out_tort = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT_TORTUOUS.vtp"
out_non  = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT_NON_TORTUOUS_AB.vtp"

# ============================
# Load OUTGEOM dict
# ============================
data = pickle.load(open(in_path, "rb"))
G = data["graph"]
x = np.asarray(data["coords"]["x"])
y = np.asarray(data["coords"]["y"])
z = np.asarray(data["coords"]["z"])

print("Loaded:", in_path)
print("Graph:", G.summary())
print("x/y/z dtype:", x.dtype, y.dtype, z.dtype)

# ============================
# Helper to add cell arrays
# ============================
def make_cell_arrays():
    nkind_array = vtk.vtkIntArray(); nkind_array.SetName("nkind")
    radius_array = vtk.vtkFloatArray(); radius_array.SetName("radius")
    length_array = vtk.vtkFloatArray(); length_array.SetName("length")
    tortuosity_array = vtk.vtkFloatArray(); tortuosity_array.SetName("tortuosity")
    return nkind_array, radius_array, length_array, tortuosity_array

def add_cell_values(eidx, nkind_array, radius_array, length_array, tortuosity_array):
    nkind_array.InsertNextValue(int(G.es[eidx]["nkind"]))
    radius_array.InsertNextValue(float(G.es[eidx]["radius"]))
    length_array.InsertNextValue(float(G.es[eidx]["length"]))
    t = G.es[eidx]["tortuosity"]
    tortuosity_array.InsertNextValue(float(t) if t is not None else np.nan)

# ============================
# 1) TORTUOUS export
# ============================
points_t = vtk.vtkPoints()
lines_t  = vtk.vtkCellArray()
nkind_t, radius_t, length_t, tort_t = make_cell_arrays()

pid = 0
kept_edges_t = 0

for e in range(G.ecount()):
    s  = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])
    npts = en - s
    if npts < 2:
        continue

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i in range(npts):
        points_t.InsertNextPoint(float(x[s+i]), float(y[s+i]), float(z[s+i]))
        polyline.GetPointIds().SetId(i, pid)
        pid += 1

    lines_t.InsertNextCell(polyline)
    add_cell_values(e, nkind_t, radius_t, length_t, tort_t)
    kept_edges_t += 1

poly_t = vtk.vtkPolyData()
poly_t.SetPoints(points_t)
poly_t.SetLines(lines_t)
poly_t.GetCellData().AddArray(nkind_t)
poly_t.GetCellData().AddArray(radius_t)
poly_t.GetCellData().AddArray(length_t)
poly_t.GetCellData().AddArray(tort_t)
poly_t.GetCellData().SetActiveScalars("nkind")

w = vtk.vtkXMLPolyDataWriter()
w.SetFileName(out_tort)
w.SetInputData(poly_t)
w.SetDataModeToAppended()
w.EncodeAppendedDataOff()
w.Write()

print("Saved tortuous:", out_tort, "| edges:", kept_edges_t)

# ============================
# 2) NON-TORTUOUS export (A,B endpoints from geometry)           !!!!!!!!!!!!!!!!!!!!
# ============================
points_n = vtk.vtkPoints()
lines_n  = vtk.vtkCellArray()
nkind_n, radius_n, length_n, tort_n = make_cell_arrays()

pid = 0
kept_edges_n = 0

for e in range(G.ecount()):
    s  = int(G.es[e]["geom_start"])
    en = int(G.es[e]["geom_end"])
    if en - s < 2:
        continue

    A = (float(x[s]),    float(y[s]),    float(z[s]))             # !!!!!!!!!!!!!!!!!
    B = (float(x[en-1]), float(y[en-1]), float(z[en-1]))

    points_n.InsertNextPoint(*A)
    points_n.InsertNextPoint(*B)

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, pid)
    line.GetPointIds().SetId(1, pid+1)
    pid += 2

    lines_n.InsertNextCell(line)
    add_cell_values(e, nkind_n, radius_n, length_n, tort_n)
    kept_edges_n += 1

poly_n = vtk.vtkPolyData()
poly_n.SetPoints(points_n)
poly_n.SetLines(lines_n)
poly_n.GetCellData().AddArray(nkind_n)
poly_n.GetCellData().AddArray(radius_n)
poly_n.GetCellData().AddArray(length_n)
poly_n.GetCellData().AddArray(tort_n)
poly_n.GetCellData().SetActiveScalars("nkind")

w = vtk.vtkXMLPolyDataWriter()
w.SetFileName(out_non)
w.SetInputData(poly_n)
w.SetDataModeToAppended()
w.EncodeAppendedDataOff()
w.Write()

print("Saved non-tortuous (A,B):", out_non, "| edges:", kept_edges_n)
