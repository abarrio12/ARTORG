import pickle
import vtk
import numpy as np

# ============================
# Load FULL pkl
# ============================

data = pickle.load(open(
    "/home/admin/Ana/MicroBrain/output/graph_18_full.pkl", "rb"
))

G = data["graph"]
coords = data["coords"]

x, y, z = coords["x"], coords["y"], coords["z"]

# ============================
# VTK structures
# ============================

points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

nkind_array = vtk.vtkIntArray()
nkind_array.SetName("nkind")

radius_array = vtk.vtkFloatArray()
radius_array.SetName("radius")

length_array = vtk.vtkFloatArray()
length_array.SetName("length")

tortuosity_array = vtk.vtkFloatArray()
tortuosity_array.SetName("tortuosity")

point_id = 0

# ============================
# Loop over edges
# ============================

for e in range(G.ecount()):
    s = int(G.es[e]["geom_start"])
    e_ = int(G.es[e]["geom_end"])
    npts = e_ - s

    if npts < 2:
        continue

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i in range(npts):
        points.InsertNextPoint(
            float(x[s + i]),
            float(y[s + i]),
            float(z[s + i])
        )
        polyline.GetPointIds().SetId(i, point_id)
        point_id += 1

    lines.InsertNextCell(polyline)

    nkind_array.InsertNextValue(int(G.es[e]["nkind"]))
    radius_array.InsertNextValue(float(G.es[e]["radius"]))
    length_array.InsertNextValue(float(G.es[e]["length"]))
    tortuosity_array.InsertNextValue(float(G.es[e]["tortuosity"]))

# ============================
# Create PolyData
# ============================

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

polydata.GetCellData().AddArray(nkind_array)
polydata.GetCellData().AddArray(radius_array)
polydata.GetCellData().AddArray(length_array)
polydata.GetCellData().AddArray(tortuosity_array)

# Default coloring
polydata.GetCellData().SetActiveScalars("nkind")

# ============================
# Write VTP
# ============================

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/home/admin/Ana/MicroBrain/output/graph_18_full.vtp")
writer.SetInputData(polydata)
writer.SetDataModeToAppended()
writer.EncodeAppendedDataOff()
writer.Write()

print("Saved vascular_network.vtp")

