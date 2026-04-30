import os
import pickle

def igraph_to_mvn_dicts(G, verbose=True):
    """
    Convert final MVN igraph into separate dictionaries:
      - vertices_dict
      - edges_dict
      - graph_dict
    """

    vertices_dict = {attr: G.vs[attr] for attr in G.vs.attributes()}
    edges_dict = {attr: G.es[attr] for attr in G.es.attributes()}

    if "connectivity" not in edges_dict:
        edges_dict["connectivity"] = [tuple(map(int, e.tuple)) for e in G.es]

    graph_dict = {attr: G[attr] for attr in G.attributes()}

    graph_dict["n_vertices"] = G.vcount()
    graph_dict["n_edges"] = G.ecount()
    graph_dict["vertex_attributes"] = sorted(list(G.vs.attributes()))
    graph_dict["edge_attributes"] = sorted(list(G.es.attributes()))

    if verbose:
        print("\n[MVN dict format]")
        print(f"  Vertices: {G.vcount():,}")
        print(f"  Edges:    {G.ecount():,}")
        print("  Vertex attrs:", graph_dict["vertex_attributes"])
        print("  Edge attrs:", graph_dict["edge_attributes"])

        if "unit" in G.attributes():
            print("  Unit:", G["unit"])
        if "diameter_unit" in G.attributes():
            print("  Diameter unit:", G["diameter_unit"])
        if "coord_space" in G.attributes():
            print("  Coord space:", G["coord_space"])

    return vertices_dict, edges_dict, graph_dict


def save_mvn_dicts_from_igraph(G, out_dir, base_name, verbose=True):
    """
    Save final MVN igraph attributes as three separate pickle dictionaries.
    """

    vertices_dict, edges_dict, graph_dict = igraph_to_mvn_dicts(
        G,
        verbose=verbose,
    )

    os.makedirs(out_dir, exist_ok=True)

    v_path = os.path.join(out_dir, f"{base_name}_verticesDict.pkl")
    e_path = os.path.join(out_dir, f"{base_name}_edgesDict.pkl")
    g_path = os.path.join(out_dir, f"{base_name}_graphDict.pkl")

    with open(v_path, "wb") as f:
        pickle.dump(vertices_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(e_path, "wb") as f:
        pickle.dump(edges_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(g_path, "wb") as f:
        pickle.dump(graph_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nSaved MVN dicts:")
    print(" ", v_path)
    print(" ", e_path)
    print(" ", g_path)

    return vertices_dict, edges_dict, graph_dict, v_path, e_path, g_path