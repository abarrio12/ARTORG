import pickle
import vtk
import numpy as np

# ============================
# Load pkl
# ============================

<<<<<<< HEAD
in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.pkl"
out_vtp = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom.vtp"
# ejemplo: si quieres el cut:
# in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"
# out_vtp = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.vtp"
=======
in_path = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT.pkl"
out_vtp = "/home/admin/Ana/MicroBrain/output/graph_18_OutGeom_CUT_TORTUOUS.vtp"
>>>>>>> 3eb0a63 (PKLtoVTP combine Tort/nonTort)

data = pickle.load(open(in_path, "rb"))
G = data["graph"]

x = np.asarray(data["coords"]["x"], dtype=np.float32)
y = np.asarray(data["coords"]["y"], dtype=np.float32)
z = np.asarray(data["coords"]["z"], dtype=np.float32)

ann = None
if "annotation" in data:
    ann = np.asarray(data["annotation"], dtype=np.int32)
    if len(ann) != len(x):
        raise ValueError("data['annotation'] must match coords length.")

# ============================
# VTK structures
# ============================

points = vtk.vtkPoints()
lines = vtk.vtkCellArray()

# Cell data (per edge)
nkind_array = vtk.vtkIntArray();   nkind_array.SetName("nkind")
radius_array = vtk.vtkFloatArray(); radius_array.SetName("radius")
length_array = vtk.vtkFloatArray(); length_array.SetName("length")
tortuosity_array = vtk.vtkFloatArray(); tortuosity_array.SetName("tortuosity")

# Point data (per geometry point)
ann_array = None
if ann is not None:
    ann_array = vtk.vtkIntArray()
    ann_array.SetName("geom_annotation")

point_id = 0

# ============================
# Loop over edges
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
        points.InsertNextPoint(float(x[s + i]), float(y[s + i]), float(z[s + i]))
        polyline.GetPointIds().SetId(i, point_id)

        if ann_array is not None:
            ann_array.InsertNextValue(int(ann[s + i]))

        point_id += 1

    lines.InsertNextCell(polyline)

    nkind_array.InsertNextValue(int(G.es[e]["nkind"]))
    radius_array.InsertNextValue(float(G.es[e]["radius"]))
    length_array.InsertNextValue(float(G.es[e]["length"]))

    # por si alguna tortuosity es nan
    tort = G.es[e]["tortuosity"]
    tortuosity_array.InsertNextValue(float(tort) if tort is not None else np.nan)

# ============================
# Create PolyData
# ============================

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

# CellData
polydata.GetCellData().AddArray(nkind_array)
polydata.GetCellData().AddArray(radius_array)
polydata.GetCellData().AddArray(length_array)
polydata.GetCellData().AddArray(tortuosity_array)
polydata.GetCellData().SetActiveScalars("nkind")

# PointData
if ann_array is not None:
    polydata.GetPointData().AddArray(ann_array)
    polydata.GetPointData().SetActiveScalars("geom_annotation")

# ============================
# Write VTP
# ============================

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(out_vtp)
writer.SetInputData(polydata)
writer.SetDataModeToAppended()
writer.EncodeAppendedDataOff()
writer.Write()

print("Saved", out_vtp)
print("Points:", polydata.GetNumberOfPoints(), "Cells:", polydata.GetNumberOfCells())
if ann_array is not None:
    print("PointData geom_annotation:", ann_array.GetNumberOfTuples())
