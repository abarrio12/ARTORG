import igraph as ig
from igraph import plot

# ------------------  CREACION DE GRAFOS  ------------------

# Crear un grafo dirigido con 7 nodos y varias aristas
# Asignar atributos a los nodos y aristas
g = ig.Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]


# Mostrar atributos de nodos y aristas
print(g.vs.attributes())
print(g.es.attributes())

# Visualizar el grafo
print(g)
ig.plot(g, target="grafo.png") 




# ------------------  BETWENNESS DE ARISTAS  ------------------



# Creamos un grafo simple
g1 = ig.Graph([(0,1), (0,2), (1,3), (2,3), (0,3)])

# Calculamos la betweenness de las aristas
ebs = g.edge_betweenness()
print("Lista de betweenness:", ebs)

# Buscamos la máxima betweenness
max_eb = max(ebs)
print("Betweenness máxima:", max_eb)

# Enumeramos ebs para obtener índice y valor
for idx, eb in enumerate(ebs):
    print(f"Índice: {idx}, Betweenness: {eb}, Arista: {g.get_edgelist()[idx]}")
    
# Aristas con betweenness máxima
critical_edges = [g.get_edgelist()[idx] for idx, eb in enumerate(ebs) if eb == max_eb]
print("Aristas críticas (mayor betweenness):", critical_edges)




# ------------------  MATRIZ DE ADYACENCIA  ------------------



# Cada celda (i, j) vale: 1, si hay una arista entre el vértice i y j OR 0, si no la hay
print(g.get_adjacency())

print(g.get_adjacency_sparse())  # Matriz dispersa -- solo guarda las posiciones donde hay arista, para ahorrar memoria.

# si yo quiero acceder a un atributo en particular de las aristas:
print(g.get_adjacency(attribute="is_formal"))  # Matriz de adyacencia con el atributo DE LAS ARISTAS (g.es) (si los hubiera)

print(g.get_adjacency_sparse(attribute="is_formal"))  # Matriz de adyacencia con el atributo DE LAS ARISTAS (g.es) (si los hubiera), pero solo donde hay arista

print(g.Weighted_Adjacency) # Matriz de adyacencia con pesos (si los hubiera), si no no devuelve nada.




# ------------------  AUTOMORFISMOS E ISOMORFISMOS  ------------------



# Automorfismo: miden la similitud estructural de un grafo consigo mismo.Si un grafo tiene muchas maneras de reordenar sus nodos y 
# mantener la estructura igual, significa que es muy simétrico. Si tiene solo 1 automorfismo, significa que no tiene simetrías: 
# no puedes reordenar los nodos sin cambiar la estructura.

print(g.count_automorphisms_vf2())  # Número de automorfismos del grafo (isomorfismos de un grafo consigo mismo)

# Dos nodos (usuarios) son estructuralmente equivalentes si: Están conectados a los mismos otros nodos. Si los intercambias, la 
# estructura global de la red no cambia.

print(g.get_automorphisms_vf2())

print(g.isomorphic(g1))  # Comprueba si dos grafos son isomorfos (idénticos en estructura, aunque los nodos puedan tener etiquetas diferentes)

print(g1.get_isomorphisms_vf2(g))  # Encuentra todas las formas de mapear g1 en g si son isomorfos




# ------------------  GRAFOS DIRIGIDOS Y NO DIRIGIDOS  ------------------


print(g.as_directed())  # Convierte el grafo en dirigido (si no lo es ya)
print(g.as_undirected())  # Convierte el grafo en no dirigido (si no lo es ya)

print(g.is_directed())  # Comprueba si el grafo es dirigido
#print(g.is_undirected())  # Comprueba si el grafo es no dirigido

print(g.degree_distribution())  # Distribución de grados considerando la dirección de las aristas


print("Promedio de grado:", sum(g.degree())/g.vcount())



# ------------------  ANÁLISIS DE CENTRALIDAD  ------------------
print("Closeness:", g.closeness(), max(g.closeness()))  # cercanía de vértices
print("Betweenness:", g.betweenness())  # cuántos caminos pasan por cada vértice
print("Pagerank:", g.pagerank(), max(g.pagerank()))  # importancia de vértices
print(g.pagerank().index(max(g.pagerank())))  # índice del vértice más importante según pagerank
print("Eigenvector centrality:", g.eigenvector_centrality(), max(g.eigenvector_centrality()))  # importancia de vértices según sus vecinos

# Camino más corto
print("Camino de A a D:", g.get_shortest_paths(0, 3)) #vertice origen y destino

# Árbol de expansión mínima (spanning tree)
mst = g.spanning_tree(weights=None)


print("Componentes conectados:", g.components())
print("Bloques biconectados:", g.biconnected_components())


# Crear etiquetas combinando índice y nombre
g.vs["label"] = [f"{v.index}: {v['name']}" for v in g.vs]
# [f"{expresion}" for variable in iterable if condicion] unico order permitido

ig.plot(g, target="grafo_titulo.png", vertex_label=g.vs["label"], vertex_size=30)



# ------------------  MODIFICAR ARISTAS/VÉRTICES  ------------------
# Agregar un nuevo vértice
g.add_vertex(name="Hannah", age=29, gender="f")
print(g.vs["name"])

# Eliminar un vértice por su índice
g.delete_vertices(1)  # Elimina el vértice con índice 1 (Bob)
print(g.vs["name"])

# Agregar una nueva arista
g.add_edge("Alice", "Hannah", is_formal=False)
print(g.get_edgelist())

# Eliminar una arista por sus extremos      
g.delete_edges([(0,1)])  # Elimina la arista entre Alice y Claire
print(g.get_edgelist())

# Modificar atributos de un vértice
print(g.vs["age"])
g.vs.find(name="Claire")["age"] = 19
print(g.vs["age"])

# Modificar atributos de una arista -- igraph no acepta directamente nombres de nodos en _between, sino índices de vértices.
# Por eso usamos .index para obtener el índice del vértice correspondiente al nombre dado. Por otro lado, name="Alice" 
# no es un objeto. Tenemos que usar g.vs.find(name="Alice") para obtener el vértice con ese nombre. y luego volver a buscar bewteen 
# con los índices de esos vértices. 
# 
# g.es.find(_between=((name="Alice").index, (name="Esther").index))["is_formal"] = False NO FUNCIONA

source = g.vs.find(name="Alice").index
target = g.vs.find(name="Claire").index

# Busca todas las aristas donde source y target coinciden (DIRIGIDO)
eids = [e.index for e in g.es if e.source == source and e.target == target]

print(g.es["is_formal"])
# DIRIGIDO Y NO DIRIGIDO
for e in g.es: 
    if (e.source == source and e.target == target) or (e.source == target and e.target == source):
        e["is_formal"] = False

print(g.es["is_formal"])

# Modificar atributo
#for eid in eids:
    #g.es[eid]["is_formal"] = False


# Calcular métricas
print("Densidad:", g.density())
print("Diámetro:", g.diameter())



