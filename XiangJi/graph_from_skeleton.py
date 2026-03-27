# Labels identifican componentes
# connected_* identifica cómo se conectan esas componentes entre sí
# numInfo guarda los metadatos del volumen, info que no debe variar y la cual se consulta de manera continua
# Component struct me da la información básica/general de a qué componente pertenece un cierto voxel
# dentro de los voxeles yo puedo tener: 
#  - Node voxel
#  - Link voxel
#  - Endpoint
#  - Isopoint
# Esto son componentes como tal, pero luego hay un nivel superior en el que se determinan los CC para determinar
# no si un grupo es nodo, edge... sino determinar el nodo , el edge...
# mask_size lo guardo implicitamente , en python está en skel.shape, no hace falta guardarlo en SkeletonData

import numpy as np
import networkx as nx

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from typing import List, Set



# ============================================================
# Type aliases
# ============================================================
coord = Tuple[int, int, int]


# ============================================================
# Basic classes
# ============================================================
@dataclass
class NumInfo:
    """
    Basic metadata of the skeleton volume.
    No padding nor linear indices or linear offsets are used in this implementation, 
    as the idea is to convert everything to Python, and Python usually works with 3D coordinates.
  
    """
    mask_size: Tuple[int, int, int]
    skeleton_voxel: int
    block_voxel: int  # num total voxels in the volume => prod(mask_size_values) · 3D -> 1D
    neighborhood: int = 26



@dataclass
class ComponentStruct:
    """
    MATLAB-like component container. "given a voxel, to what component does it belong to?"

    A voxel can be: node voxel, link voxel, endopoint voxel or isopoint voxel.
    
    cc_ind:
        list of connected components, each stored as a 1D array of voxel IDs
        (voxel IDs refer to rows in data.coords)

    pos_ind:
        concatenation of all component voxel IDs

    label:
        for each entry in pos_ind, the component label it belongs to
    """
    num_cc: int = 0
    cc_ind: List[np.ndarray] = field(default_factory=list)
    pos_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # list of all voxels IDs from the CC
    num_voxel_all_components: int = 0
    num_voxel_per_cc: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # number of voxels per connected component
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # to what CC does the voxel of pos_ind belong to 
    map_voxel_to_label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) #array of the total size of the skeleton voxels, for each voxel id tells you in which CC it is (-1 if in none) 
    

@dataclass
class EndpointStruct:
    """
    Endpoint structure, similar in spirit to MATLAB endpoint block.
    Notice that endpoints are a subset of link points
    """
    pos_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    num_voxel: int = 0
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    map_voxel_to_label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    link_label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class IsoPointStruct:
    """
    Isolated single voxels (degree 0).
    """
    pos_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    num_voxel: int = 0
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    map_voxel_to_label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class LinkStruct(ComponentStruct):
    """
    Link structure.

    connected_node_label:
        shape (num_cc, 2), stores node labels at both ends, -1 if absent (no node)

    connected_endpoint_label:
        shape (num_cc, 2), stores endpoint labels at both ends, -1 if absent 

    end_kind:
        list of tuples like ("node", "endpoint"), ("node", "node"), etc.
    
    end_kind = ("node", "endpoint")
    connected_node_label = [3, -1]
    connected_endpoint_label = [-1, 5]
        --> Links starts in node 3 and finishes in endpoint 5
    """
    connected_node_label: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64)) # to which CC is the node connected 
    connected_endpoint_label: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64)) # to which link CC is the link connected
    num_node_per_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # how many nodes does the link touch (0, 1,2)
    num_endpoint_per_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # how many endpoints does the link touch (0,1,2)
    terminal_link_kinds: List[Tuple[str, str]] = field(default_factory=list) # stores topological type of link extrems (node, endpoint, isopoint). Nothing to do with nkind (A/V/C)
    num_voxels_in_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # # Number of voxels in each link CC
    link_length_euclidean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) # Euclidean length of each link chain, computed from consecutive voxel coordinates.
    is_isoloop: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool)) # True if the link is an isolated loop.


@dataclass
class NodeStruct(ComponentStruct):
    """
    Node structure.

    connected_link_label:for each node connected component, list of attached link labels
        connected_link_label = [
            np.array([0, 4]),      # node 0
            np.array([1, 2, 5]),   # node 1
            np.array([3])          # node 2
        ]
        num_links_per_node = [2,3,1]
    
    """
    connected_link_label: List[np.ndarray] = field(default_factory=list)
    num_links_per_node: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    centroid: List[Tuple[float, float, float]] = field(default_factory=list) # median pos of all voxels that form the node , #TODO: Do i need this? check in future


@dataclass
class SkeletonData:
    """
    Internal voxel-level representation.
    """
    skel: np.ndarray 
    coords: np.ndarray # in matlab voxel_list = find(skl), here coords = np.argwhere(skl) 
    coord_to_id: Dict[coord, int] # in matlab sp_skl(idx) -> returns id of the voxel using sparse matrix, here using a dict as {coord -> id}


# ============================================================
# 26-neighborhood offsets
# ============================================================
def offsets26() -> List[coord]:
    """
    Generate all 26 offsets of the 3D Moore neighborhood.
    """
    neighbors = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                neighbors.append((i, j, k))
    return neighbors

OFFS26 = offsets26()


# ============================================================
# Input normalization
# ============================================================
def prepare_skeleton(skel: np.ndarray) -> SkeletonData:
    """
    If the input is the skeleton (3D logical array) convert it to voxel list
    
    Normalize input skeleton to:
    - boolean 3D array
    - list of active coordinates (similar to voxel_list has list of lineal indices in matlab)
    - coordinate -> voxel_id lookup 
    """
    skel = np.asarray(skel, dtype=bool)
    if skel.ndim != 3: # x y z 
        raise ValueError("Input skeleton must be a 3D array.")

    coords = np.argwhere(skel)
    coord_to_id = {tuple(c): i for i, c in enumerate(coords)}

    return SkeletonData(
        skel=skel,
        coords=coords,
        coord_to_id=coord_to_id,
    )


# ============================================================
# Geometry helpers
# ============================================================
def in_bounds(p: coord, shape: Tuple[int, int, int]) -> bool:
    """
    Check whether a coordinate is inside the volume.
    Same idea of padding in matlab.
    """
    x,y,z = p 
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


def neighbors26_of(p: coord, skel: np.ndarray) -> List[coord]:
    """
    Return all active 26-neighbors of voxel p.
    """
    x,y,z = p 
    out = []
    for i, j, k in OFFS26:
        q = (x + i, y + j, z + k)
        if in_bounds(q, skel.shape) and skel[q]:
            out.append(q)
    return out


# ============================================================
# Voxel adjacency
# ============================================================
def build_voxel_adjacency(data: SkeletonData) -> List[List[int]]:
    """
    Build undirected voxel-to-voxel adjacency list (voxel_list of matlab)
   
    For each voxel it stores its neighbors. 
    
    Input: 
    -  skel: 3D volume 
    - coords: np.ndarray -> active voxels
    - coord_to_id: dict (x, y, z) -> id
    """
    n = len(data.coords) # total number of active voxels in the skeleton
    adj = [[] for v in range(n)] # create an empty list for each voxel

    for voxel_id, coord_voxel in enumerate(data.coords):
        coord_voxel = tuple(coord_voxel)
        for neigh in neighbors26_of(coord_voxel, data.skel):
            neigh_id = data.coord_to_id[tuple(neigh)] 
            if neigh_id > voxel_id: # in order to not add the same edge twice/check same pair again (0->1 i dont need 1->0)
                # undirected graph so adding conection in both ways
                adj[voxel_id].append(neigh_id)
                adj[neigh_id].append(voxel_id)

    return adj


def voxel_degrees(adj: List[List[int]]) -> np.ndarray:
    """
    Degree of each voxel in voxel graph
    """
    return np.array([len(v) for v in adj], dtype=np.int64)


# ============================================================
# Component builder helper
# ============================================================
def build_component_struct(cc_ind: List[np.ndarray], n_total_voxels: int) -> ComponentStruct:
    """
    Build a MATLAB-like component structure ("summary") from a list of connected components (each is a list of IDs of voxels)
    Fills all fields in ComponentStruct.
    This function does not classify, but only organizes info returned in group_node_voxels and trace_links
    
    Input:
    - cc_ind: list of each connected components. Each component is a list of voxel IDs
    - n_total_voxels: total number of voxels in that CC
    """
    out = ComponentStruct()
    out.num_cc = len(cc_ind)
    out.cc_ind = [np.asarray(c, dtype=np.int64) for c in cc_ind]

    if out.num_cc == 0:
        out.pos_ind = np.array([], dtype=np.int64)
        out.num_voxel_all_components = 0
        out.num_voxel_per_cc = np.array([], dtype=np.int64)
        out.label = np.array([], dtype=np.int64)
        out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)
        return out

    out.num_voxel_per_cc = np.array([len(c) for c in out.cc_ind], dtype=np.int64)
    out.pos_ind = np.concatenate(out.cc_ind) # all voxels of all components in one vector
    out.num_voxel_all_components = len(out.pos_ind) # num of voxels in all components together
    out.label = np.repeat(np.arange(out.num_cc, dtype=np.int64), out.num_voxel_per_cc) # created the label of each component for each voxel in pos_ind ("mapping")
    # if a voxel does not belong to a CC, -1
    out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)
    out.map_voxel_to_label[out.pos_ind] = out.label # adding the belonging to CC voxels label

    return out

# ============================================================
# Endpoint and isopoint structures
# ============================================================
def build_endpoint_struct(endpoint_voxels: np.ndarray, n_total_voxels: int) -> EndpointStruct:
    """
    Build endpoint structure.
    """
    endpoint_voxels = np.asarray(endpoint_voxels, dtype=np.int64)

    out = EndpointStruct()
    out.pos_ind = endpoint_voxels.copy()
    out.num_voxel = len(endpoint_voxels)
    out.label = np.arange(out.num_voxel, dtype=np.int64)
    out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)
    if out.num_voxel > 0:
        out.map_voxel_to_label[out.pos_ind] = out.label
    out.link_label = np.full(out.num_voxel, -1, dtype=np.int64)
    return out


def build_isopoint_struct(isopoint_voxels: np.ndarray, n_total_voxels: int) -> IsoPointStruct:
    """
    Build isolated-point structure.
    """
    isopoint_voxels = np.asarray(isopoint_voxels, dtype=np.int64)

    out = IsoPointStruct()
    out.pos_ind = isopoint_voxels.copy()
    out.num_voxel = len(isopoint_voxels)
    out.label = np.arange(out.num_voxel, dtype=np.int64)
    out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)
    if out.num_voxel > 0:
        out.map_voxel_to_label[out.pos_ind] = out.label
    return out


# ============================================================
# Classification
# ============================================================
def classify_voxels(adj: List[List[int]]) -> Dict[str, np.ndarray]:
    """
    Classify voxels by degree. Only separates by type
    Returns a dict with key=topological label, 
    value = array of all voxel ids with classified with that topology 
    """
    deg = voxel_degrees(adj)
    return {
        "degree": deg,
        "isopoint": np.where(deg == 0)[0],
        "endpoint": np.where(deg == 1)[0],
        "link": np.where(deg == 2)[0],
        "node": np.where(deg > 2)[0],
    }


# ============================================================
# Connected components
# ============================================================

# In matlab code they use fun_cc_in_sparse_matrix that builds the CC of any subset
# Following the same idea, connected_components_from_subset retrieves the different 
# CC in a given subset. Then with build_node_cc we retrieve each node_cc

# Note that CC are needed for: node voxels, link voxels and isoloop voxels,
# endpoint voxels and isopoint voxels are individual voxels (not grouping needed)

def connected_components_from_subset(subset_voxels: np.ndarray, adj: List[List[int]]) -> List[np.ndarray]:
    """
    Find connected components inside a subset of voxel IDs.

    Parameters
    ----------
    subset_voxels : np.ndarray
        1D array of voxel IDs belonging to one voxel class (e.g. all node voxels, all loop voxels, etc.)
        This is returned in classify_voxel()
        
    adj : list of lists
        Full voxel-to-voxel adjacency list of the skeleton graph.

    Returns
    -------
    cc_list : list of np.ndarray
        List of connected components. Each component is returned as a 1D array of voxel IDs.

    Notes
    -----
    This is the Python equivalent in spirit to MATLAB's fun_cc_in_sparse_matrix(...), but it works directly on the
    voxel adjacency list instead of sparse linear indexing. Also inside the logic of visited/unvisited voxels
    is implemented.
    """
    subset_voxels = np.asarray(subset_voxels, dtype=np.int64)

    if subset_voxels.size == 0:
        return []

    subset_set: Set[int] = set(map(int, subset_voxels))
    visited: Set[int] = set()
    cc_list: List[np.ndarray] = []

    for start in subset_voxels:
        start = int(start)

        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        component = []

        while queue:
            u = queue.popleft()
            component.append(u)

            for v in adj[u]:
                if v in subset_set and v not in visited:
                    visited.add(v)
                    queue.append(v)

        cc_list.append(np.asarray(component, dtype=np.int64))

    return cc_list


# ============================================================
# Node connected components
# ============================================================

def build_node_cc(node_voxels: np.ndarray, adj: List[List[int]], coords: np.ndarray) -> NodeStruct:
    """
    Build the NodeStruct (class) from the subset of node voxels returned in connected_components_from_subset(...)
    Notice that a node can have more than 1 voxel, this is the reason why this function is needed, to 
    join all of the voxels that belong to the same node. 

    Parameters
    ----------
    node_voxels : np.ndarray
        1D array of voxel IDs classified as node voxels.
    adj : list of lists
        Full voxel adjacency list.
    coords : np.ndarray
        Array of voxel coordinates of shape (N, 3), where row i contains the coordinate of voxel ID i.

    Returns
    -------
    NodeStruct: Node connected components (each node of the future graph) plus per-node metadata.
    """
    # Step 1: find connected components among node voxels
    cc_list = connected_components_from_subset(node_voxels, adj)

    # Step 2: build the MATLAB-like component fields
    base = build_component_struct(cc_list, len(coords))

    # Step 3: compute one centroid per node connected component
    centroids = []
    for comp in cc_list:
        comp_coords = coords[comp]
        centroids.append(tuple(np.mean(comp_coords, axis=0).tolist()))

    # Step 4: create NodeStruct
    out = NodeStruct(
        num_cc=base.num_cc,
        cc_ind=base.cc_ind,
        pos_ind=base.pos_ind,
        num_voxel_all_components=base.num_voxel_all_components,
        num_voxel_per_cc=base.num_voxel_per_cc,
        label=base.label,
        map_voxel_to_label=base.map_voxel_to_label,
        connected_link_label=[np.array([], dtype=np.int64) for _ in range(base.num_cc)],
        num_links_per_node=np.zeros(base.num_cc, dtype=np.int64),
        centroid=centroids,
    )

    return out


# ============================================================
# Link Connected Components (link tracing)
# ============================================================

# In matlab code, fun_skeleton_get_link_cc(...) is used, a subfunction of fun_skeleton_to_graph, which
# finds the link connected components. Except for the beginning voxels, each voxels in the link has exactly two neigbhors, whose indices in the
# voxel list in the first two row of voxel_neighbor_idx_list. 
# The start tracking points are the union of the link points in the neighbor to the node points or endpoints (in case the links has
# two endpoints). 


def get_link_start_voxels(
    node_voxels: np.ndarray,
    endpoint_voxels: np.ndarray,
    adj: List[List[int]],
    link_voxel_set: Set[int],
) -> np.ndarray:
    """
    Build the start voxel list for link tracing.
    According to XiangJi's code: "The start tracking points are the union of the link points in the neighbor
    to the node points, or the end points ( in case that the links have two endpoints)

    Equivalent in spirit to MATLAB's l_link_start_voxel_idx.
    """
    start_voxels = set()

    # 1) Endpoints are valid starts
    for ep in endpoint_voxels:
        ep = int(ep)
        for nb in adj[ep]:
            if nb in link_voxel_set:
                start_voxels.add(nb)

    # 2) Link voxels neighboring node voxels are also valid starts
    for nd in node_voxels:
        nd = int(nd)
        for nb in adj[nd]:
            if nb in link_voxel_set:
                start_voxels.add(nb)

    return np.array(sorted(start_voxels), dtype=np.int64)

def build_link_cc(
    adj: list,
    link_set: set,
    node_set: set,
    endpoint_set: set,
    link_start_ids: list,     # possible seeds (endpoints + node neighbors) where link can start, same as l_link_start_voxel_idx in matlab
    voxel_unvisited: np.ndarray, # important: includes endpoints and link voxels (not nodes/isopoints -> can't be part of the link)
) -> tuple:
    """
    Replicates fun_skeleton_get_link_cc.
    Only traces voxel link chains. Here is not important the starting point topology.
    
    Returns:
        chains          : list of lists of voxel IDs (each list = one chain = one link)
        terminals       : for each chain, returns the ending point (can be node or endpoint or None) 
        voxel_unvisited : boolean array updated where visited voxels turn False
    """
    chains    = []
    terminals = []  # final voxel of the chain can be a node, ep or none

    # same as matlab: iterates starting points in order 
    for start_id in link_start_ids:

        # if the starting node has already been visited, jump to next one (another chain has already gone through here) 
        if not voxel_unvisited[start_id]:
            continue

        # if it is still unvisited, i take it as new starting point of the new chain (link)
        chain    = [start_id]
        voxel_unvisited[start_id] = False # need to update to visited, we don't want to go through it again. Important, in XiangJi's code except for the beginning voxels, each voxel in the link has exactly 2 neighbors.

        current  = start_id
        previous = -1   # we come from nowhere at the beginning 
        terminal = None

        while True:
            keep_tracking = False # stop by default (chain completed), except if there is a next possible voxel (unvisited)

            # check neighbors of current voxel 
            # (adj[current] already has the voxels link neighbors if you use first build_link_neighbor_list)
            for nb in adj[current]:
                if nb == previous:
                    continue  # do not go backwards

                if nb in link_set and voxel_unvisited[nb]:
                    # move forward to the next unvisited voxel 
                    voxel_unvisited[nb] = False
                    chain.append(nb) # add link voxel to chain
                    previous = current # update positions for next link search
                    current  = nb
                    keep_tracking = True
                    break 

                if nb in node_set or nb in endpoint_set:
                    # final voxel of the chain reached/no more unvisited nodes
                    terminal = nb

            if not keep_tracking:
                break

        chains.append(chain)
        terminals.append(terminal)

    return chains, terminals, voxel_unvisited


def build_link_struct(
    chains: list,            # output of get_link_cc
    terminals: list,         # output of get_link_cc
    isoloop_chains: list,    # isolated loops (second call to get_link_cc in MATLAB) -> if i still have unvisited, redo this (line 177 fun_skeleton_to_graph.m)
    node_map: np.ndarray,    # map_voxel_to_label of nodes
    endpoint_map: np.ndarray,# map_voxel_to_label of endpoints
    coords: np.ndarray,
    node_set: set,
    endpoint_set: set,
) -> LinkStruct:
    """
    Equivalent to %% Construct graph of fun_skeleton_to_graph.m (line 139)
    Takes the raw chains and builds the LinkStruct with metadata.
    """

    all_chains = chains + isoloop_chains
    n_total    = len(coords)

    cc_ind     = [np.array(c, dtype=np.int64) for c in all_chains]
    num_cc     = len(cc_ind)

    # pos_ind = cat(1, link_cc.PixelIdxList{:}) en MATLAB
    pos_ind    = np.concatenate(cc_ind) if cc_ind else np.array([], dtype=np.int64)

    # num_voxel_per_cc = cellfun(@length, link_cc.PixelIdxList) en MATLAB
    num_voxel_per_cc = np.array([len(c) for c in cc_ind], dtype=np.int64)

    # label = repelem(1:num_cc, num_voxel_per_cc) en MATLAB (aquí 0-indexed)
    label = np.repeat(np.arange(num_cc, dtype=np.int64), num_voxel_per_cc)

    # map_voxel_to_label: para cualquier voxel id -> en qué link está (-1 si no está)
    # en MATLAB: sparse(pos_ind, ones, label, block_voxel, 1)
    map_voxel_to_label = np.full(n_total, -1, dtype=np.int64)
    if len(pos_ind) > 0:
        map_voxel_to_label[pos_ind] = label

    # --- para cada cadena, qué nodo/endpoint hay en cada extremo ---
    # equivale a graph_str.link.connected_node_label = zeros(num_cc, 2)
    connected_node_label     = np.full((num_cc, 2), -1, dtype=np.int64)
    connected_endpoint_label = np.full((num_cc, 2), -1, dtype=np.int64)
    end_kind                 = []
    is_isoloop               = np.zeros(num_cc, dtype=bool)

    for lid, (chain, terminal) in enumerate(zip(chains, terminals)):

        # el extremo de inicio: el voxel que inició la búsqueda está fuera del chain,
        # lo guardamos como "de dónde vino" al llamar get_link_cc
        # aquí terminal es el extremo final
        kind_end, id_end = _classify_terminal(terminal, node_set, endpoint_set,
                                              node_map, endpoint_map)

        # el extremo de inicio lo inferimos del primer voxel del chain:
        # miramos quién lo inició (nodo o endpoint) buscando en sus vecinos
        # (esto lo resuelves guardando el 'prev' en get_link_cc si quieres ser más explícita)
        end_kind.append(("unknown", kind_end))  # puedes refinar guardando el prev

        if kind_end == "node":
            connected_node_label[lid, 1] = id_end
        elif kind_end == "endpoint":
            connected_endpoint_label[lid, 1] = id_end

    for lid in range(len(chains), num_cc):
        # loops aislados
        end_kind.append(("loop", "loop"))
        is_isoloop[lid] = True

    num_node_per_link     = np.sum(connected_node_label >= 0,     axis=1)
    num_endpoint_per_link = np.sum(connected_endpoint_label >= 0, axis=1)

    lengths_euclidean = np.array([
        _euclidean_length(coords, c) for c in all_chains
    ], dtype=float)

    return LinkStruct(
        num_cc                   = num_cc,
        cc_ind                   = cc_ind,
        pos_ind                  = pos_ind,
        num_voxel                = len(pos_ind),
        num_voxel_per_cc         = num_voxel_per_cc,
        label                    = label,
        map_voxel_to_label       = map_voxel_to_label,
        connected_node_label     = connected_node_label,
        connected_endpoint_label = connected_endpoint_label,
        num_node_per_link        = num_node_per_link,
        num_endpoint_per_link    = num_endpoint_per_link,
        terminal_link_kinds      = end_kind,
        num_voxels_in_link       = num_voxel_per_cc,
        link_length_euclidean    = lengths_euclidean,
        is_isoloop               = is_isoloop,
    )


def _classify_terminal(v, node_set, endpoint_set, node_map, endpoint_map):
    if v is None:
        return "none", -1
    if v in node_set:
        return "node", int(node_map[v])
    if v in endpoint_set:
        return "endpoint", int(endpoint_map[v])
    return "none", -1


def _euclidean_length(coords, chain):
    if len(chain) < 2:
        return 0.0
    arr = coords[np.array(chain, dtype=np.int64)]
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))



def euclidean_chain_length(coords: np.ndarray, chain: List[int]) -> float:
    """
    Euclidean length of an ordered voxel chain.
    """
    if len(chain) < 2:
        return 0.0

    arr = coords[np.asarray(chain, dtype=np.int64)]
    dif = arr[1:] - arr[:-1]
    return float(np.sum(np.linalg.norm(dif, axis=1)))


def trace_links(
    adj: List[List[int]],
    node_voxels: np.ndarray,
    endpoint_voxels: np.ndarray,
    link_voxels: np.ndarray,
    node_map_voxel_to_label: np.ndarray,
    endpoint_map_voxel_to_label: np.ndarray,
    coords: np.ndarray,
) -> Tuple[LinkStruct, ComponentStruct]:
    """
    Trace all link connected components following MATLAB-like logic.

    Start from:
    - endpoint voxels
    - node-neighboring link voxels

    Unvisited remaining link voxels are treated as isolated loops.
    """
    n_total_voxels = len(coords)

    node_set = set(map(int, node_voxels))
    endpoint_set = set(map(int, endpoint_voxels))
    link_set = set(map(int, link_voxels))

    visited: Set[int] = set()
    unvisited: Set[int] = set(link_set)

    link_cc: List[np.ndarray] = []
    link_end_kind: List[Tuple[str, str]] = []
    link_node_labels: List[Tuple[int, int]] = []
    link_endpoint_labels: List[Tuple[int, int]] = []
    link_lengths_voxel: List[int] = []
    link_lengths_euclidean: List[float] = []
    link_is_loop: List[bool] = []

    isoloop_cc: List[np.ndarray] = []

    def classify_terminal(v: Optional[int]) -> Tuple[str, int]:
        """
        Convert a terminal voxel ID to a semantic endpoint descriptor.
        """
        if v is None:
            return ("none", -1)

        if v in node_set:
            return ("node", int(node_map_voxel_to_label[v]))

        if v in endpoint_set:
            return ("endpoint", int(endpoint_map_voxel_to_label[v]))

        return ("none", -1)

    def walk_from_seed(seed: int, prev: int) -> Tuple[List[int], Optional[int]]:
        """
        Walk along a chain of link voxels starting from `seed`.

        Parameters
        ----------
        seed : int
            First link voxel.
        prev : int
            Previous voxel already attached to the chain start
            (endpoint voxel or node voxel).

        Returns
        -------
        chain : list of int
            Ordered link voxel IDs
        terminal : Optional[int]
            Final non-link voxel reached at the far end, or None
        """
        chain = [seed]
        visited.add(seed)
        unvisited.discard(seed)

        current = seed
        previous = prev

        # replicates while idea from matlab code line 47 - 56 of fun_skeleton_get_link_cc(...)
        while True:
            candidates = [v for v in adj[current] if v != previous] # select candidates from the current voxel neighbors 

            next_link = [v for v in candidates if v in link_set and v not in visited] # if the next voxel is not yet been visited, i use it as start point of the new link
            terminal_candidates = [v for v in candidates if v in node_set or v in endpoint_set] 

            if next_link:
                nxt = next_link[0]
                chain.append(nxt)
                visited.add(nxt)
                unvisited.discard(nxt)
                previous = current
                current = nxt # update starting searching point for next link
                continue

            if terminal_candidates:
                return chain, terminal_candidates[0]

            return chain, None

    # --------------------------------------------------------
    # Start from endpoints
    # --------------------------------------------------------
    for ep in endpoint_voxels:
        ep = int(ep)
        for nb in adj[ep]:
            if nb in link_set and nb not in visited:
                chain, terminal = walk_from_seed(nb, prev=ep)

                start_kind, start_id = ("endpoint", int(endpoint_map_voxel_to_label[ep]))
                end_kind, end_id = classify_terminal(terminal)

                link_cc.append(np.asarray(chain, dtype=np.int64))
                link_end_kind.append((start_kind, end_kind))

                node_pair = [-1, -1]
                endpoint_pair = [-1, -1]

                if start_kind == "node":
                    node_pair[0] = start_id
                elif start_kind == "endpoint":
                    endpoint_pair[0] = start_id

                if end_kind == "node":
                    node_pair[1] = end_id
                elif end_kind == "endpoint":
                    endpoint_pair[1] = end_id

                link_node_labels.append(tuple(node_pair))
                link_endpoint_labels.append(tuple(endpoint_pair))
                link_lengths_voxel.append(len(chain))
                link_lengths_euclidean.append(euclidean_chain_length(coords, chain))
                link_is_loop.append(False)

    # --------------------------------------------------------
    # Start from node boundaries
    # --------------------------------------------------------
    for nd in node_voxels:
        nd = int(nd)
        for nb in adj[nd]:
            if nb in link_set and nb not in visited:
                chain, terminal = walk_from_seed(nb, prev=nd)

                start_kind, start_id = ("node", int(node_map_voxel_to_label[nd]))
                end_kind, end_id = classify_terminal(terminal)

                link_cc.append(np.asarray(chain, dtype=np.int64))
                link_end_kind.append((start_kind, end_kind))

                node_pair = [-1, -1]
                endpoint_pair = [-1, -1]

                if start_kind == "node":
                    node_pair[0] = start_id
                elif start_kind == "endpoint":
                    endpoint_pair[0] = start_id

                if end_kind == "node":
                    node_pair[1] = end_id
                elif end_kind == "endpoint":
                    endpoint_pair[1] = end_id

                link_node_labels.append(tuple(node_pair))
                link_endpoint_labels.append(tuple(endpoint_pair))
                link_lengths_voxel.append(len(chain))
                link_lengths_euclidean.append(euclidean_chain_length(coords, chain))
                link_is_loop.append(False)

    # --------------------------------------------------------
    # Remaining unvisited link voxels = isolated loops
    # --------------------------------------------------------
    def trace_isolated_loop(start: int) -> List[int]:
        """
        Trace a closed loop made only of link voxels.
        """
        chain = [start]
        visited.add(start)
        unvisited.discard(start)

        neighbors = [v for v in adj[start] if v in link_set]
        if not neighbors:
            return chain

        previous = start
        current = neighbors[0]

        while True:
            if current in visited:
                break

            chain.append(current)
            visited.add(current)
            unvisited.discard(current)

            candidates = [v for v in adj[current] if v in link_set and v != previous]
            if not candidates:
                break

            nxt = candidates[0]
            previous, current = current, nxt

            if current == start:
                break

        return chain

    for lv in list(unvisited):
        if lv not in visited:
            chain = trace_isolated_loop(lv)
            comp = np.asarray(chain, dtype=np.int64)
            isoloop_cc.append(comp)

            link_cc.append(comp)
            link_end_kind.append(("loop", "loop"))
            link_node_labels.append((-1, -1))
            link_endpoint_labels.append((-1, -1))
            link_lengths_voxel.append(len(comp))
            link_lengths_euclidean.append(euclidean_chain_length(coords, chain))
            link_is_loop.append(True)

    # Build link structure
    base = build_component_struct(link_cc, n_total_voxels)
    link_out = LinkStruct(
        num_cc=base.num_cc,
        cc_ind=base.cc_ind,
        pos_ind=base.pos_ind,
        num_voxel_all_components=base.num_voxel_all_components,
        num_voxel_per_cc=base.num_voxel_per_cc,
        label=base.label,
        map_voxel_to_label=base.map_voxel_to_label,
        connected_node_label=np.asarray(link_node_labels, dtype=np.int64) if link_node_labels else np.empty((0, 2), dtype=np.int64),
        connected_endpoint_label=np.asarray(link_endpoint_labels, dtype=np.int64) if link_endpoint_labels else np.empty((0, 2), dtype=np.int64),
        num_node=np.sum(np.asarray(link_node_labels, dtype=np.int64) >= 0, axis=1) if link_node_labels else np.array([], dtype=np.int64),
        num_endpoint=np.sum(np.asarray(link_endpoint_labels, dtype=np.int64) >= 0, axis=1) if link_endpoint_labels else np.array([], dtype=np.int64),
        end_kind=link_end_kind,
        num_voxels_in_link=np.asarray(link_lengths_voxel, dtype=np.int64),
        link_length_euclidean=np.asarray(link_lengths_euclidean, dtype=float),
        is_isoloop=np.asarray(link_is_loop, dtype=bool),
    )

    isoloop_base = build_component_struct(isoloop_cc, n_total_voxels)

    return link_out, isoloop_base

def build_isoloop_cc(
    isoloop_voxels: np.ndarray,
    adj: List[List[int]],
    n_total_voxels: int,
) -> ComponentStruct:
    """
    Build a ComponentStruct for isolated loop voxels.

    Parameters
    ----------
    isoloop_voxels : np.ndarray
        1D array of voxel IDs belonging to isolated loops.
    adj : list of lists
        Full voxel adjacency list.
    n_total_voxels : int
        Total number of skeleton voxels.

    Returns
    -------
    ComponentStruct
        Connected components of isolated loop voxels.
    """
    cc_list = connected_components_from_subset(isoloop_voxels, adj)
    return build_component_struct(cc_list, n_total_voxels)

# ============================================================
# Node-link connectivity
# ============================================================
def fill_node_link_connectivity(node: NodeStruct, link: LinkStruct) -> NodeStruct:
    """
    Fill:
    - node.connected_link_label
    - node.num_link
    """
    if node.num_cc == 0 or link.num_cc == 0:
        return node

    connected = [[] for _ in range(node.num_cc)]

    for lid in range(link.num_cc):
        a, b = link.connected_node_label[lid]
        if a >= 0:
            connected[a].append(lid)
        if b >= 0:
            connected[b].append(lid)

    node.connected_link_label = [
        np.unique(np.asarray(lst, dtype=np.int64)) if len(lst) > 0 else np.array([], dtype=np.int64)
        for lst in connected
    ]
    node.num_link = np.array([len(x) for x in node.connected_link_label], dtype=np.int64)
    return node


# ============================================================
# Graph builder
# ============================================================
def build_graph(node: NodeStruct, endpoint: EndpointStruct, link: LinkStruct) -> nx.Graph:
    """
    Build final graph.

    Node naming:
    - nodes are named "node_{id}"
    - endpoints are named "endpoint_{id}"

    This avoids fake placeholder nodes like -1.
    """
    G = nx.Graph()

    # Add node connected components
    for nid in range(node.num_cc):
        G.add_node(
            f"node_{nid}",
            kind="node",
            label=nid,
            voxel_ids=node.cc_ind[nid],
            centroid=node.centroid[nid] if nid < len(node.centroid) else None,
        )

    # Add endpoints
    for eid, vox in enumerate(endpoint.pos_ind):
        G.add_node(
            f"endpoint_{eid}",
            kind="endpoint",
            label=eid,
            voxel_id=int(vox),
        )

    # Add edges (links)
    for lid in range(link.num_cc):
        endk0, endk1 = link.end_kind[lid]

        # Build endpoint identifiers for graph edge ends
        def endpoint_name(kind: str, local_idx: int, side: int) -> Optional[str]:
            if kind == "node":
                nid = int(link.connected_node_label[local_idx, side])
                return f"node_{nid}" if nid >= 0 else None
            if kind == "endpoint":
                eid = int(link.connected_endpoint_label[local_idx, side])
                return f"endpoint_{eid}" if eid >= 0 else None
            return None

        u = endpoint_name(endk0, lid, 0)
        v = endpoint_name(endk1, lid, 1)

        # Loop links can be stored as self-loops only if attached to a node.
        # Isolated loops remain out of the networkx graph connectivity and are
        # still recorded in graph_str["isoloop"] and graph_str["link"].
        if u is None and v is None:
            continue
        if u is None or v is None:
            continue

        G.add_edge(
            u,
            v,
            kind="link",
            label=lid,
            voxel_ids=link.cc_ind[lid],
            length_voxel=int(link.num_voxels_in_link[lid]),
            length_euclidean=float(link.link_length_euclidean[lid]),
            is_loop=bool(link.is_isoloop[lid]),
        )

    return G


# ============================================================
# Main function
# ============================================================
def fun_skeleton_to_graph(skel: np.ndarray) -> dict:
    """
    Python implementation that follows the logic of the MATLAB
    fun_skeleton_to_graph pipeline.

    Input
    -----
    skel : np.ndarray
        3D binary skeleton array

    Output
    ------
    graph_str : dict
        Dictionary with fields:
        - num
        - node
        - link
        - endpoint
        - isopoint
        - isoloop
        - graph
        - coords
        - degree
    """
    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    data = prepare_skeleton(skel)
    adj = build_voxel_adjacency(data)
    topo = classify_voxels(adj)

    num = NumInfo(
        mask_size=tuple(data.skel.shape),
        skeleton_voxel=len(data.coords),
        block_voxel=int(np.prod(data.skel.shape)),
        neighborhood=26,
    )

    # --------------------------------------------------------
    # Endpoint / isopoint structures
    # --------------------------------------------------------
    endpoint = build_endpoint_struct(topo["endpoint"], len(data.coords))
    isopoint = build_isopoint_struct(topo["isopoint"], len(data.coords))

    # --------------------------------------------------------
    # Node connected components
    # --------------------------------------------------------
    node = build_node_cc(topo["node"], adj, data.coords)
    # --------------------------------------------------------
    # Link tracing + isolated loops
    # --------------------------------------------------------
    link, isoloop_base = trace_links(
        adj=adj,
        node_voxels=topo["node"],
        endpoint_voxels=topo["endpoint"],
        link_voxels=topo["link"],
        node_map_voxel_to_label=node.map_voxel_to_label,
        endpoint_map_voxel_to_label=endpoint.map_voxel_to_label,
        coords=data.coords,
    )

    # --------------------------------------------------------
    # Endpoint -> link label
    # --------------------------------------------------------
    endpoint.link_label = np.full(endpoint.num_voxel, -1, dtype=np.int64)
    for lid in range(link.num_cc):
        for side in (0, 1):
            eid = link.connected_endpoint_label[lid, side] if link.connected_endpoint_label.size > 0 else -1
            if eid >= 0:
                endpoint.link_label[eid] = lid

    # --------------------------------------------------------
    # Node -> link connectivity
    # --------------------------------------------------------
    node = fill_node_link_connectivity(node, link)

    # --------------------------------------------------------
    # Isolated loop structure
    # --------------------------------------------------------
    isoloop = ComponentStruct(
        num_cc=isoloop_base.num_cc,
        cc_ind=isoloop_base.cc_ind,
        pos_ind=isoloop_base.pos_ind,
        num_voxel_all_components=isoloop_base.num_voxel_all_components,
        num_voxel_per_cc=isoloop_base.num_voxel_per_cc,
        label=isoloop_base.label,
        map_voxel_to_label=isoloop_base.map_voxel_to_label,
    )

    # --------------------------------------------------------
    # Final graph
    # --------------------------------------------------------
    G = build_graph(node, endpoint, link)

    # --------------------------------------------------------
    # Assemble MATLAB-like output
    # --------------------------------------------------------
    graph_str = {
        "num": num,
        "node": node,
        "link": link,
        "endpoint": endpoint,
        "isopoint": isopoint,
        "isoloop": isoloop,
        "graph": G,
        "coords": data.coords,
        "coord_to_id": data.coord_to_id,
        "degree": topo["degree"],
        "adj": adj,
    }

    return graph_str


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    skel = np.zeros((20, 20, 20), dtype=np.uint8)

    # Main trunk
    for z in range(5, 15):
        skel[10, 10, z] = 1

    # Two branches
    for i in range(5):
        skel[10 - i, 10, 15 + i] = 1
        skel[10 + i, 10, 15 + i] = 1

    # Another branch
    for i in range(4):
        skel[10, 10 + i, 10 - i] = 1

    graph_str = fun_skeleton_to_graph(skel)

    print("num.skeleton_voxel:", graph_str["num"].skeleton_voxel)
    print("node.num_cc:", graph_str["node"].num_cc)
    print("link.num_cc:", graph_str["link"].num_cc)
    print("endpoint.num_voxel:", graph_str["endpoint"].num_voxel)
    print("isopoint.num_voxel:", graph_str["isopoint"].num_voxel)
    print("isoloop.num_cc:", graph_str["isoloop"].num_cc)
    print("graph nodes:", graph_str["graph"].number_of_nodes())
    print("graph edges:", graph_str["graph"].number_of_edges())