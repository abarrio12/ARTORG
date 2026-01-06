import pickle
import numpy as np

# abre el archivo en modo lectura binaria
with open(r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\MVN1_corrected_SI.pkl", "rb") as f:
    data = pickle.load(f)

# Ver tipo de objeto
print(type(data))

print(data.summary())

# ------- SINGLE CONNECTED COMPONENT -------

print(data.is_connected())

if data.is_connected():
    print("El grafo es un single connected component")
else:
    print("El grafo tiene más de un componente")
    
components = data.components()  # devuelve un objeto VertexClustering
print("Número de componentes conectados:", len(components))

# Útil para detectar nodos aislados o subconjuntos desconectados.
for i, comp in enumerate(components):
    print(f"Componente {i} tiene {len(comp)} nodos")
    
    
    

# ----- EDGES FOR EACH NKIND -----

edge_types = data.es["nkind"]

# opcion A: usar conjunto para obtener tipos únicos
unique_edge_types = set(edge_types)
print("Tipos únicos de aristas (opción A):", unique_edge_types)

# opcion B: contar ocurrencias de cada tipo
unique, counts = np.unique(edge_types, return_counts=True)
for i, n in zip(unique, counts):
    print(f"Tipo de arista: {i}, Cuenta: {n}")
    


# ----- DIAMETRO MEDIO PARA CADA NKIND -----

print(data.es.attributes())  
print(data.es["diameters"][:1])  # primer diámetro


# cada arista está formada por un diametro que a su vez es un array de valores(3)

# first i need the mean of diameters for each edge
diam_arista = np.array([np.mean(d) for d in data.es["diameters"]])

#classify by nkind
nkind = np.array(data.es["nkind"])

for k in np.unique(nkind):
    mean_d = diam_arista[nkind == k].mean()
    print(f"nkind = {k}: average diameter = {mean_d:.6e}")


# ---- Average length for each vessel type ----

print(data.es["lengths"][:2])  # first length --> again array of 3 numbers --> cada arista representa un segmento curvo dividido en 3 sub-segmentos

# first i need the mean of lengths for each edge
lengths_arista = np.array([np.mean(l) for l in data.es["lengths"]])
nkind = np.array(data.es["nkind"])
for k in np.unique(nkind):
    mean_l = lengths_arista[nkind == k].mean()
    print(f"nkind = {k}: average length = {mean_l:.6e}")


# ----  Degrees of the nodes? Are there nodes with degree > 3? -----
degrees = data.degree()
print("Degrees of nodes:", np.unique(degrees))

count = 0
for i, d in enumerate(degrees):
    if d > 3:
        print(f"Nodo {i} tiene grado {d}")
        count += 1
       
print(count)
 
 
 
 
# ---- Nodes with boundary conditions compared to the total number of nodes and location ----

print(data.vs.attributes())
print(data.vs["border_vertices"][:10])  # first 10 entries
print(data.vs["boundaryType"][:10])  # first 10 entries

# we have border vertices, boundary type and boundary value

# boolean for frontier nodes
frontier_nodes = np.array([fn is not None for fn in data.vs["border_vertices"]])

# booleano de nodos con boundaryType definido (no None)
has_boundary = np.array([bt is not None for bt in data.vs["boundaryType"]])

# nodos frontera que efectivamente tienen BC
node_with_bc = frontier_nodes & has_boundary

print(f"Nodes with BC: {np.sum(node_with_bc)} of {len(data.vs)}")
# location in the array
print(f"Location of nodes with BC: {np.where(node_with_bc)[0]}")
# get coordinates of these nodes
coords_with_bc = np.array(data.vs["coords"])[node_with_bc]
print("Coordinates of nodes with BC:", coords_with_bc[:5])

