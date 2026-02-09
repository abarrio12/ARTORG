import pickle
import numpy as np
import vtk

in_pkl  = "/home/admin/Ana/MicroBrain/output/nonT_Hcut3.pkl"
out_vtp = "/home/admin/Ana/MicroBrain/output/nonT_Hcut3.vtp"

# load (puede ser Graph o dict con graph)
with open(in_pkl, "rb") as f:
    obj = pickle.load(f, encoding="latin1")
G = obj["graph"] if isinstance(obj, dict) and "graph" in obj else obj

# coords por vértice
coords = np.asarray(G.vs["coords"], dtype=float)  # (nV,3)

# vtk points
points = vtk.vtkPoints()
points.SetNumberOfPoints(coords.shape[0])
for i,(x,y,z) in enumerate(coords):
    points.SetPoint(i, float(x), float(y), float(z))

lines = vtk.vtkCellArray()

# cell data arrays
def farr(name):
    a = vtk.vtkFloatArray(); a.SetName(name); return a
def iarr(name):
    a = vtk.vtkIntArray(); a.SetName(name); return a

diam_arr  = farr("diameter")
len_arr   = farr("length")
nkind_arr = iarr("nkind")

eattrs = set(G.es.attributes())

for e in G.es:
    s, t = e.tuple

    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, int(s))
    line.GetPointIds().SetId(1, int(t))
    lines.InsertNextCell(line)

    diam_arr.InsertNextValue(float(e["diameter"]) if "diameter" in eattrs else np.nan)
    len_arr.InsertNextValue(float(e["length"]) if "length" in eattrs else np.nan)
    nkind_arr.InsertNextValue(int(e["nkind"]) if "nkind" in eattrs else -1)

poly = vtk.vtkPolyData()
poly.SetPoints(points)
poly.SetLines(lines)

poly.GetCellData().AddArray(diam_arr)
poly.GetCellData().AddArray(len_arr)
poly.GetCellData().AddArray(nkind_arr)
poly.GetCellData().SetActiveScalars("nkind")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(out_vtp)
writer.SetInputData(poly)
writer.Write()

print("Saved:", out_vtp)
print("Points:", poly.GetNumberOfPoints(), "Lines:", poly.GetNumberOfCells())
