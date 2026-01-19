import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ====================================================================================================================
#                                                    LOAD GRAPH FUNCTION
# ====================================================================================================================
def load_graph(path):
    """
    Load an igraph Graph object from a pickle file.

    Parameters
    ----------
    path : str
        Path to the .pkl file.

    Returns
    -------
    graph : igraph.Graph
        Loaded vascular graph.
    """
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph


# Vessel type mapping
vessel_type = {
    2: "arteriole",
    3: "venule",
    4: "capillary"
}


# ====================================================================================================================
#                                               FUNCTIONS FOR GRAPH ANALYSIS
# ====================================================================================================================


# =============================
# single connected component
# =============================
def single_connected_component(graph):
    """
    Analyze whether the graph consists of a single connected component
    and return useful information about all components.

    Returns
    -------
    is_single : bool
        True if the graph is fully connected (only one component).
    n_components : int
        Total number of connected components.
    components : list of lists
        Each element is a list of node indices belonging to that component.
    """

    # Check connectivity
    is_single = graph.is_connected()

    # igraph returns a VertexClustering object
    components = graph.components()
    n_components = len(components)

    # Print summary
    if is_single:
        print("The graph is a single connected component.")
    else:
        print("The graph has more than one connected component.")

    print("Number of connected components:", n_components)
    return is_single, n_components
   



# =============================
# Edge types and count
# =============================

def get_edges_types(graph):
    """
    Count how many edges belong to each nkind type.

    Parameters
    ----------
    graph : igraph.Graph
        The vascular graph.
    vessel_type : dict, optional
        Mapping from nkind integer to string name (e.g. {2: "artery"}).

    Returns
    -------
    unique : np.ndarray
        Unique nkind values.
    counts : np.ndarray
        Number of edges of each nkind type.
    """
    edge_types = graph.es["nkind"]
    unique, counts = np.unique(edge_types, return_counts=True)
    print("\nEdge types:\n")
    for i, n in zip(unique, counts):
        print(f" - {vessel_type[i]}, {i}, Count: {n}")
    return unique, counts
     

# =============================
#  Graph diameter by nkind
# =============================
        
def get_diameter_nkind(graph):
        """
        Compute the diameter of the graph.

        Parameters
        ----------
        graph : igraph.Graph
            The vascular graph.

        Returns
        -------
        diameter : int
            The diameter of the graph.
        """
        
        diam = np.array(graph.es["diameter"])
        
        # classify by nkind
        nkind = np.array(graph.es["nkind"])
        diameters = []
        print("\nAverage diameter by nkind:\n")
        for k in np.unique(nkind):
            mean_diam =  float(diam[nkind == k].mean())
            diameters.append(mean_diam)
            print(f"nkind = {k}: average diameter = {mean_diam:.6f} μm")
        return diameters




# =============================
# Length by nkind
# =============================

def get_length_nkind(graph):
        """
        Compute the length of the graph.

        Parameters
        ----------
        graph : igraph.Graph
            The vascular graph.

        Returns
        -------
        length : int
            The length of the graph.
        """
        
        length_att = np.array(graph.es["length"])
        # classify by nkind
        nkind = np.array(graph.es["nkind"])
        l = []

        print("\n Average lenght by nkind:\n")
        for k in np.unique(nkind):
            mean_l = length_att[nkind == k].mean()
            l.append(mean_l)
            print(f"nkind = {k}: average length (att) = {mean_l:.6f} μm")
        return l
    




# =============================
# Node degrees
# =============================

def get_degrees(graph, threshold = 4):
    """
    Compute node degrees and identify high-degree nodes based on a threshold.
    Mark high-degree nodes with their actual degree.
    Non-high-degree nodes receive a 0 in 'degree_hd'.

    Parameters
    ----------
    graph : igraph.Graph
        The vascular graph.

    Returns
    -------
    degrees : np.ndarray
        Array containing the degree of each node.
    high_degree_nodes : list
        List of node indices whose degree is above the threshold.
    """
   
    degrees = np.array(graph.degree())
    graph.vs["degree"] = degrees
    
    # For Paraview visualization => degree_hd = real degree only for high-degree nodes, else 0
    degree_hd = np.where(degrees >= threshold, degrees, 0)
    graph.vs["high_degree_node"] = degree_hd

    # List of high degree nodes 
    high_degree = np.where(degrees >= threshold)[0]
    
    print("\n Degrees of nodes:", np.unique(degrees))
    print(f"High-degree nodes (> {threshold}): {len(high_degree)}")
    return np.unique(degrees), high_degree



def get_location_degrees(graph, node_list):
    """
    Given a list of node indices, compute their distance-to-surface and
    print basic statistics.

    Parameters
    ----------
    graph : igraph.Graph
    node_list : list of ints
        High-degree node indices.
    max_print : int
        How many nodes to print individually.

    Returns
    -------
    distances : list of float
        Distance to surface of each high-degree node.
    """

    dist = graph.vs["distance_to_surface"]

    # Extract relevant distances
    distances = np.array([dist[i] for i in node_list])

    print(f"\nHigh-degree nodes analyzed: {len(node_list)}")

    for i in node_list[:10]:
        print(f"Node {i} -> distance to surface = {dist[i]:.2f} µm")
    
    
    # ---- Basic statistics ----
    print("\nDistance-to-surface statistics (µm):")
    print(f"  Min:   {distances.min():.2f}")
    print(f"  Mean:  {distances.mean():.2f}")
    print(f"  Median:{np.median(distances):.2f}")
    print(f"  Max:   {distances.max():.2f}")

    # ---- Classification ----
    superficial = np.sum(distances < 20)
    superficial_2 = np.sum((distances >= 20) & (distances < 50))
    middle      = np.sum((distances >= 50) & (distances < 200))
    deep        = np.sum(distances >= 200)

    print("\nClassification by depth:")
    print(f"  Superficial (<20 µm):     {superficial}  ({100*superficial/len(distances):.1f}%)")
    print(f"  Superficial_2 (20-50 µm):     {superficial_2}  ({100*superficial_2/len(distances):.1f}%)")
    print(f"  Middle (50–200 µm):       {middle}       ({100*middle/len(distances):.1f}%)")
    print(f"  Deep (>200 µm):           {deep}         ({100*deep/len(distances):.1f}%)")

    return distances


# =============================
# duplicated edges
# =============================

def find_duplicated_edges(graph):
    """
    Detect duplicated edges in the graph, regardless of direction.
    (u, v) and (v, u) are considered identical.

    Parameters
    ----------
    graph : igraph.Graph
        The vascular graph.

    Returns
    -------
    duplicated : list of tuples
        List of duplicated edges in (min, max) normalized form.
        (--> right now ommited to reduce output size.)
    len(duplicated) : int
        Number of duplicated edges found.
    """
    edges = graph.get_edgelist()
    sorted_edges = [tuple(sorted(edge)) for edge in edges]
    count = Counter(sorted_edges)
    duplicated = []
    for edge, cnt in count.items():
        if cnt > 1:
            duplicated.append(edge)
    print("\nNumber of duplicated edges:", len(duplicated))
    
    return len(duplicated)
    
    

# ============================= FALTARÍA AÑADIR FUNCION DE LAS BC, PRIMERO MIRAR CODIGO DE GAIA PARA VER COMO PODRÍA 
# HACERSE (IF/ELSE TORTUOUS) ======================================================






# ====================================================================================================================
#                                                      PLOTTING FUNCTIONS
# ====================================================================================================================




# Tip: Group before plotting (by nkind, mean values, etc.)

def plot_category_stats(categ, attribute_toplot, label_dict=None,
                        xlabel="Category", ylabel="Value",
                        title="Category statistics"):
    labels = [label_dict.get(c, c) if label_dict else c for c in categ] 
    # Map categories to labels if label_dict is provided cat = [2,3,4] + label dict = {2:"artery", 3:"vein", 4:"capillary"} 
    # --> labels = ["artery", "vein", "capillary"]

    plt.figure(figsize=(8,5))
    plt.bar(labels, attribute_toplot, color="steelblue", edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_histograms_by_category(attribute_toplot, category, label_dict=None,
                                xlabel="Value", plot_title="Category"):

    unique_cats = np.unique(category)

    # Compute shared global min/max for all categories
    global_min = attribute_toplot.min()
    global_max = attribute_toplot.max()

    # Shared bins for ALL subplots → ensures comparability
    bins = np.linspace(global_min, global_max, 80)

    plt.figure(figsize=(12, 4 * len(unique_cats)))

    for i, c in enumerate(unique_cats, 1):
        plt.subplot(len(unique_cats), 1, i)

        subset = attribute_toplot[category == c]
        mean_value = subset.mean()

        # Plot histogram using shared bins
        plt.hist(subset, bins=bins, alpha=0.75, density=True)

        # Mean line
        plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5)

        # Shared X-range for ALL subplots
        plt.xlim(global_min, global_max)

        # Label
        name = label_dict.get(c, "Unknown") if label_dict else c

        plt.title(f"{plot_title} {c} ({name})")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend([f"Mean = {mean_value:.2f}"])

    plt.tight_layout()
    plt.show()



# ============================================================
# BC detection (minimal, correct for box cut)
# ============================================================

"""
BC node analysis for a box cut
- Detect BC nodes on each of the 6 box faces using coordinates and a tolerance eps
- For each face: count BC nodes + classify vessel type (via edge 'nkind')
- Print percentages per face
"""

def bc_nodes_on_plane(graph, axis, value, coords_attr, eps=1e-3):
    """
    axis: 0=x, 1=y, 2=z
    value: plane coordinate
    returns: np.array of vertex indices lying on that plane within eps
    """
    if coords_attr not in graph.vs.attributes():
        raise ValueError(
            f"Missing vertex attribute '{coords_attr}'. "
            f"Available vertex attrs: {graph.vs.attributes()}"
        )

    coords = np.asarray(graph.vs[coords_attr], dtype=float)
    # Sanity check (coords exists and are Nx3)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"'{coords_attr}' must be Nx3. Got shape {coords.shape}.")

    # coords of node - coords of plane ~ True/False if node in plane
    mask = np.abs(coords[:, axis] - float(value)) <= float(eps)
    return np.where(mask)[0] # returns the index of the node (BC)


def bc_nodes_on_box_faces(graph, box, coords_attr, eps=1e-3):
    """
    box: dict with keys xmin,xmax,ymin,ymax,zmin,zmax
    returns dict face_name -> np.array(node_ids)
    """
    required = ["xmin","xmax","ymin","ymax","zmin","zmax"]
    missing = [k for k in required if k not in box]
    if missing:
        raise ValueError(f"box is missing keys: {missing}. Required: {required}")
    
    faces = {
        "x_min": bc_nodes_on_plane(graph, 0, box["xmin"], coords_attr, eps),
        "x_max": bc_nodes_on_plane(graph, 0, box["xmax"], coords_attr, eps),
        "y_min": bc_nodes_on_plane(graph, 1, box["ymin"], coords_attr, eps),
        "y_max": bc_nodes_on_plane(graph, 1, box["ymax"], coords_attr, eps),
        "z_min": bc_nodes_on_plane(graph, 2, box["zmin"], coords_attr, eps),
        "z_max": bc_nodes_on_plane(graph, 2, box["zmax"], coords_attr, eps),
    }
    return faces


def bc_node_nkind(graph, node_id):
    """
    Classify a node (node_id of BC nodes) by the most common nkind among its incident edges.
    This helps understand how many of each nkind are BC nodes 
    Returns nkind int or None if not available.
    """
    if "nkind" not in graph.es.attributes():
        raise ValueError(
            "Missing edge attribute 'nkind'. "
            f"Available edge attrs: {graph.es.attributes()}"
        )
    # v is incident of edge e if v is source/target of e 
    inc_edges = graph.incident(int(node_id))
    nk = [graph.es[e]["nkind"] for e in inc_edges]
    nk = [x for x in nk if x is not None]

    if len(nk) == 0:
        return None
    return Counter(nk).most_common(1)[0][0]


def bc_node_type_label(graph, node_id, vessel_type_map=vessel_type):
    nk = bc_node_nkind(graph, node_id)
    if nk is None:
        return "unknown"
    return vessel_type_map.get(nk, f"nkind_{nk}")


# ============================================================
# Sanity helpers
# ============================================================

# degree distribution of BC nodes per face
def _degree_distribution(graph, nodes):
    deg = np.asarray(graph.degree(), dtype=int)
    return Counter(deg[nodes]) if len(nodes) else Counter()

def _distance_to_surface_stats(graph, nodes):
    if "distance_to_surface" not in graph.vs.attributes():
        return None
    d = np.asarray(graph.vs["distance_to_surface"], dtype=float)
    if len(nodes) == 0:
        return None
    vals = d[nodes]
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_bc_for_box(graph, box, coords_attr, eps=1e-3):
    """
    Prints:
      - counts per face
      - vessel type counts + %
      - degree distribution per face
      - TOTAL unique BC nodes across faces (note: corners counted multiple times per-face)
      - optional distance_to_surface stats

    Returns dict with all computed results.
    """
    
    
    # -----------------------------------------------------
    # BC NODES PER FACES (planes)
    # -----------------------------------------------------
    
    faces = bc_nodes_on_box_faces(graph, box, coords_attr, eps=eps)

    results = {}
    all_bc_set = set()

    nV = graph.vcount()
    nE = graph.ecount()
    
    print(f"\n=== BC ANALYSIS ===")
    print(f"Graph: {nV} vertices, {nE} edges")
    print(f"Coords attr: '{coords_attr}' | eps: {eps}")


    # Per face
    for face, nodes in faces.items():
        nodes = np.array(nodes, dtype=int)
        all_bc_set.update(nodes.tolist())

        # type counts
        labels = [bc_node_type_label(graph, v) for v in nodes]
        type_counts = Counter(labels)
        total = len(nodes)

        # degree dist
        deg_counts = _degree_distribution(graph, nodes)

        # distance_to_surface
        dstat = _distance_to_surface_stats(graph, nodes)

        results[face] = {
            "nodes": nodes,
            "count": total,
            "type_counts": dict(type_counts),
            "type_percent": {k: (100.0*v/total) for k, v in type_counts.items()} if total else {},
            "degree_counts": dict(deg_counts),
            "distance_to_surface_stats": dstat,
        }

        # ---- printing ----
        print(f"\n--- Face {face} ---")
        print(f"BC nodes: {total}")

        if total == 0:
            print("WARNING: 0 BC nodes on this face. (eps too small? wrong coords_attr? wrong box units?)")
        elif total > 0.2 * nV:
            print("WARNING: Many nodes on this face (>20% of vertices). (eps too big? box value off?)")

        # type counts
        if total:
            for k, v in type_counts.most_common():
                print(f"  {k}: {v} ({100.0*v/total:.1f}%)")
        else:
            print("  (no types)")

        # degree dist
        if total:
            # print sorted by degree
            deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(deg_counts.items())])
            print(f"Degree distribution (degree:count): {deg_str}")

        # distance_to_surface stats
        if dstat is not None:
            print("distance_to_surface (µm): "
                  f"min={dstat['min']:.2f}, mean={dstat['mean']:.2f}, "
                  f"median={dstat['median']:.2f}, max={dstat['max']:.2f}")


    # -----------------------------------------------------
    # BC NODES PER BOX
    # -----------------------------------------------------

    all_bc = np.array(sorted(all_bc_set), dtype=int)
    total_unique = len(all_bc)

    tot_labels = [bc_node_type_label(graph, v) for v in all_bc]
    tot_counts = Counter(tot_labels)
    tot_deg = _degree_distribution(graph, all_bc)
    tot_dstat = _distance_to_surface_stats(graph, all_bc)

    results["TOTAL_unique"] = {
        "nodes": all_bc,
        "count": total_unique,
        "type_counts": dict(tot_counts),
        "type_percent": {k: (100.0*v/total_unique) for k, v in tot_counts.items()} if total_unique else {},
        "degree_counts": dict(tot_deg),
        "distance_to_surface_stats": tot_dstat,
    }

    print(f"\n=== TOTAL UNIQUE BC NODES ===")
    print(f"Unique BC nodes across all faces: {total_unique}")

    if total_unique == 0:
        print("FATAL WARNING: 0 BC nodes total. Almost certainly wrong box/coords_attr/eps.")
    else:
        for k, v in tot_counts.most_common():
            print(f"  {k}: {v} ({100.0*v/total_unique:.1f}%)")

        deg_str = ", ".join([f"{d}:{c}" for d, c in sorted(tot_deg.items())])
        print(f"Total degree distribution (degree:count): {deg_str}")

        if tot_dstat is not None:
            print("TOTAL distance_to_surface (µm): "
                  f"min={tot_dstat['min']:.2f}, mean={tot_dstat['mean']:.2f}, "
                  f"median={tot_dstat['median']:.2f}, max={tot_dstat['max']:.2f}")

    return results




# ====================================================================================================================
#                                                     SAVE GRAPH FUNCTION
# ====================================================================================================================

def dump_graph(graph, out_path):
    """
    Save the modified igraph Graph object (with new attributes) into a .pkl file.

    Parameters
    ----------
    graph : igraph.Graph
        The graph you want to save.
    out_path : str
        Path where you want the new .pkl file saved.
    """
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"\nGraph successfully saved to: {out_path}")



# ============================================================
# Minimal "how to run"
# ============================================================
if __name__ == "__main__":
    # ---- EDIT THESE (cut file if BC node recognition) ---- 
    pkl_path = r"C:\Users\Ana\OneDrive\Escritorio\ARTORG\half brain files (pkl-vtp)\18_igraph.pkl"

    # Your cut box limits (same units as coords_attr)
    box = dict(
        xmin=0, xmax=1000,
        ymin=0, ymax=1000,
        zmin=0, zmax=1000
    )

    # Choose coordinates:
    # - "coordinates_atlas" (usually µm)
    # - "coordinates" (if your vertices store image/voxel coords)
    coords_attr = "coords"

    # Tolerance. If µm and floats from interpolation, try 1e-2 or 1e-1.
    eps = 1e-2

    # ---- RUN ----
    G = load_graph(pkl_path)
    _ = analyze_bc_for_box(G, box, coords_attr=coords_attr, eps=eps)
