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
