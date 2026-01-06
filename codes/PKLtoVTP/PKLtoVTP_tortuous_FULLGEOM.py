'''
import pickle
import vtk
import numpy as np

# ============================
# Load FULLGEOM graph
# ============================

G = pickle.load(open(
    "/home/admin/Ana/MicroBrain/graph_18_FULLGEOM.pkl", "rb"
))

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
    geom = G.es[e]["points"]  
    geom = np.asarray(geom, dtype=np.float32) # secure (N,3) 

    if geom.shape[0] < 2:
        continue

    npts = geom.shape[0]

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i in range(npts):
        points.InsertNextPoint(
            float(geom[i, 0]),
            float(geom[i, 1]),
            float(geom[i, 2])
        )
        polyline.GetPointIds().SetId(i, point_id)
        point_id += 1

    lines.InsertNextCell(polyline)

    nkind_array.InsertNextValue(int(G.es[e]["nkind"]))
    radius_array.InsertNextValue(float(G.es[e]["radius"]))
    length_array.InsertNextValue(float(G.es[e]["length"]))
    t = G.es[e]["tortuosity"]
    tortuosity_array.InsertNextValue(float(t) if t == t else 0.0)  # nan -> 0

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

polydata.GetCellData().SetActiveScalars("nkind")

# ============================
# Write VTP
# ============================

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(
    "/home/admin/Ana/MicroBrain/output18/graph_18_FULLGEOM.vtp"
)
writer.SetInputData(polydata)
writer.Write()

print("Saved vascular_network FULLGEOM")

'''
import pickle
import vtk
import numpy as np

G = pickle.load(open(
    "/home/admin/Ana/MicroBrain/18_igraph.pkl", "rb"
))

points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

nkind_array = vtk.vtkIntArray(); nkind_array.SetName("nkind")
radius_array = vtk.vtkFloatArray(); radius_array.SetName("radius")
length_array = vtk.vtkFloatArray(); length_array.SetName("length")
tortuosity_array = vtk.vtkFloatArray(); tortuosity_array.SetName("tortuosity")

point_id = 0

MAX_EXPORT_EDGES = None  # e.g. 200000 for test
n_edges = G.ecount() if MAX_EXPORT_EDGES is None else min(G.ecount(), MAX_EXPORT_EDGES)

for e in range(n_edges):
    geom = np.asarray(G.es[e]["points"], dtype=np.float32)

    if geom.ndim != 2 or geom.shape[1] != 3 or geom.shape[0] < 2:
        continue

    npts = geom.shape[0]
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i in range(npts):
        points.InsertNextPoint(float(geom[i, 0]), float(geom[i, 1]), float(geom[i, 2]))
        polyline.GetPointIds().SetId(i, point_id)
        point_id += 1

    lines.InsertNextCell(polyline)

    nkind_array.InsertNextValue(int(G.es[e]["nkind"]))
    radius_array.InsertNextValue(float(G.es[e]["radius"]))
    length_array.InsertNextValue(float(G.es[e]["length"]))

    t = G.es[e]["tortuosity"]
    tortuosity_array.InsertNextValue(float(t) if t == t else 0.0)  # nan -> 0

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

polydata.GetCellData().AddArray(nkind_array)
polydata.GetCellData().AddArray(radius_array)
polydata.GetCellData().AddArray(length_array)
polydata.GetCellData().AddArray(tortuosity_array)
polydata.GetCellData().SetActiveScalars("nkind")

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("/home/admin/Ana/MicroBrain/output18/18_igraph.vtp")
writer.SetInputData(polydata)
writer.SetDataModeToBinary()
writer.Write()

print("Saved vascular_network FULLGEOM")
print("Edges written:", n_edges)
print("VTK points:", points.GetNumberOfPoints())
