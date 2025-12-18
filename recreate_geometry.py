import numpy as np
import pickle
import vtk

# ============================
# Load data
# ============================

G = pickle.load(open(
    "/home/admin/Ana/MicroBrain/output/graph_18_igraph.pkl", "rb"
))
coords = np.load(
    "/home/admin/Ana/MicroBrain/output/edge_geometry_coords.npz"
)

x = coords["x"]
y = coords["y"]
z = coords["z"]

# ============================
# VTK structures
# ============================

points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

point_id = 0

# ============================
# Loop over edges
# ============================

for e in range(G.ecount()):
    s = G.es[e]["geom_start"]
    e_ = G.es[e]["geom_end"]

    npts = e_ - s
    if npts < 2:
        continue

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i in range(npts):
        points.InsertNextPoint(
            float(x[s+i]),
            float(y[s+i]),
            float(z[s+i])
        )
        polyline.GetPointIds().SetId(i, point_id)
        point_id += 1

    lines.InsertNextCell(polyline)

# ============================
# Create PolyData
# ============================

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

# ============================
# Write to file
# ============================

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("vascular_network.vtp")
writer.SetInputData(polydata)
writer.Write()

print("Saved vascular_network.vtp")
