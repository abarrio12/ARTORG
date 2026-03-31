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

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple, Set







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
    
    Fields
    ------
    num_cc :
        Number of connected components.

    cc_ind :
        List of connected components. Each entry is a 1D array of voxel IDs.

    pos_ind :
        Concatenation of all voxel IDs from all connected components.

    num_voxel_all_components :
        Total number of voxels stored in this structure.

    num_voxel_per_cc :
        Number of voxels in each connected component.

    label :
        For each voxel in pos_ind, which CC label it belongs to.

    map_voxel_to_label :
        Array of length = total number of skeleton voxels.
        map_voxel_to_label[v] = CC label of voxel v, or -1 if voxel v is not
        present in this structure.
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
    Endpoint structure, similar to the MATLAB endpoint block.

    Endpoints are degree-1 voxels and are stored individually rather than
    grouped into connected components.

    In the MATLAB-like construction used here, endpoint voxels are also part of
    the traced link components, so endpoint.link_label can be recovered directly
    from link.map_voxel_to_label at the endpoint voxel positions.
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
    Matlab only stores pos_ind and num_voxel, but the label and map_voxel_to_label 
    was added for consistency with the other structures.
    """
    pos_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    num_voxel: int = 0
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    map_voxel_to_label: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class IsoLoopStruct:
    """
    MATLAB-like isolated-loop structure.

    MATLAB only stores:
    - cc_ind
    - num_cc
    - pos_ind
    """
    cc_ind: List[np.ndarray] = field(default_factory=list)
    num_cc: int = 0
    pos_ind: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


@dataclass
class LinkStruct(ComponentStruct):
    """
    Link structure.

    connected_node_label:
        shape (num_cc, 2), stores node labels at both ends, -1 if absent (no node)

    connected_endpoint_label:
        shape (num_cc, 2), stores endpoint labels at both ends, -1 if absent 
    
    num_node_per_link :
        Number of node terminals for each link (0, 1, or 2).
        
    num_endpoint_per_link :
        Number of endpoint terminals for each link (0, 1, or 2).

    #terminal_link_kinds:
    #    list of tuples describing the types of the link ends: ("node", "endpoint"), ("node", "node"), ("loop", "loop") etc.
    
    link_length_euclidean :
        Euclidean chain length computed from ordered voxel coordinates.
        This is not used in the original MATLAB code, but seems useful to also store geometric length of link.



    Note
    -----------
    A self-loop attached to the same node is represented as:
        connected_node_label[lid] = [nid, nid]
        terminal_link_kinds[lid] = ("node", "node")
        
    Example:
    terminal_link_kinds = ("node", "endpoint")
    connected_node_label = [3, -1]
    connected_endpoint_label = [-1, 5]
        --> Links starts in node 3 and finishes in endpoint 5
    """
    connected_node_label: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64)) # to which CC is the node connected 
    connected_endpoint_label: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64)) # to which link CC is the link connected
    num_node_per_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # how many nodes does the link touch (0, 1,2)
    num_endpoint_per_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))  # how many endpoints does the link touch (0,1,2)
    #terminal_link_kinds: List[Tuple[str, str]] = field(default_factory=list) # stores topological type of link extrems (node, endpoint, isopoint). Nothing to do with nkind (A/V/C)
    #num_voxels_in_link: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64)) # # Number of voxels in each link CC
    link_length_euclidean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) # Euclidean length of each link chain, computed from consecutive voxel coordinates.
  

@dataclass
class NodeStruct(ComponentStruct):
    """
    Node structure.

    connected_link_label :
        For each node connected component, list of attached link labels.

    num_links_per_node :
        Number of attached links per node CC.

    centroid :
        Mean coordinate of all voxels that form the node CC.
    
    
    Example:
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
    Internal voxel-level representation
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
    If the input is the skeleton (3D logical array) needed to be normalized to the internal representation used in the rest of the code.
    
    Normalize input skeleton to:
    - boolean 3D array
    - list of active coordinates (similar to voxel_list has list of linear indices in matlab)
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
    Check whether a 3D coordinate lies inside the volume.
    Matlab uses padding.
    """
    x, y, z = p
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]



def neighbors26_of(p: coord, skel: np.ndarray) -> List[coord]:
    """
    Return all active (skeleton = 1) 26-neighbors of voxel p.
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
   
    For each voxel it stores its adjacent neighbors. 
    
    Input: 
    - skel: 3D volume 
    - coords: np.ndarray -> active voxels
    - coord_to_id: dict (x, y, z) -> id
    """
    n = len(data.coords) # total number of active voxels in the skeleton
    adj = [[] for _ in range(n)] # create an empty list for each voxel

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
# Classification
# ============================================================
def classify_voxels(adj: List[List[int]]) -> Dict[str, np.ndarray]:
    """
    Classify voxels by topological degree.
    Returns a dict with key=topological label, value = array of all voxel 
    ids with classified with that topology 
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
    voxel adjacency list instead of sparse linear indexing. Also, inside the logic of visited/unvisited voxels
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
# Builders for node / endpoint / isopoint / isoloop
# ============================================================

def build_node_struct(node_voxels: np.ndarray, adj: List[List[int]], coords: np.ndarray) -> NodeStruct:
    """
    Build the NodeStruct (class) from the subset of node voxels returned in connected_components_from_subset(...)
    Notice that a node can have more than 1 voxel, this is the reason why this function is needed, to 
    group all of the voxels that belong to the same node. 

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



def build_endpoint_struct(endpoint_voxels: np.ndarray, n_total_voxels: int, link: LinkStruct) -> EndpointStruct:
    endpoint_voxels = np.asarray(endpoint_voxels, dtype=np.int64)

    out = EndpointStruct()
    out.pos_ind = endpoint_voxels.copy()
    out.num_voxel = len(endpoint_voxels)
    out.label = np.arange(out.num_voxel, dtype=np.int64)

    out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)
    if out.num_voxel > 0:
        out.map_voxel_to_label[out.pos_ind] = out.label

    # MATLAB:
    # graph_str.endpoint.link_label = full(graph_str.link.map_ind_2_label(graph_str.endpoint.pos_ind));
    out.link_label = np.full(out.num_voxel, -1, dtype=np.int64)
    if out.num_voxel > 0 and link.map_voxel_to_label.size > 0:
        out.link_label = link.map_voxel_to_label[out.pos_ind].copy()

    return out



def build_isopoint_struct(isopoint_voxels: np.ndarray, n_total_voxels: int) -> IsoPointStruct:
    isopoint_voxels = np.asarray(isopoint_voxels, dtype=np.int64)

    out = IsoPointStruct()
    out.pos_ind = isopoint_voxels.copy()
    out.num_voxel = len(isopoint_voxels)
    out.label = np.arange(out.num_voxel, dtype=np.int64)
    out.map_voxel_to_label = np.full(n_total_voxels, -1, dtype=np.int64)

    if out.num_voxel > 0:
        out.map_voxel_to_label[out.pos_ind] = out.label

    return out


def build_isoloop_struct(isoloop_cc: List[np.ndarray]) -> IsoLoopStruct:
    out = IsoLoopStruct()
    out.cc_ind = [np.asarray(c, dtype=np.int64) for c in isoloop_cc]
    out.num_cc = len(out.cc_ind)

    if out.num_cc > 0:
        out.pos_ind = np.concatenate(out.cc_ind)
    else:
        out.pos_ind = np.array([], dtype=np.int64)

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
    node_set: Set[int],
) -> np.ndarray:
    """
    Build the start voxel list for link tracing.

    MATLAB logic:
        start points = union(neighbors of node voxels that are not nodes,
                             endpoint voxels)
    """
    start_voxels = set(map(int, endpoint_voxels))  # endpoints themselves

    for nd in node_voxels:
        nd = int(nd)
        for nb in adj[nd]:
            if nb not in node_set:
                start_voxels.add(nb)

    return np.array(sorted(start_voxels), dtype=np.int64)


def initialize_voxel_unvisited(
    n_total_voxels: int,
    node_voxels: np.ndarray,
    isopoint_voxels: np.ndarray,
) -> np.ndarray:
    """
    Initialize the unvisited voxel mask for link tracing.
    
    MATLAB logic:
    voxel_unvisited = true(num.skeleton_voxel,1);
    voxel_unvisited(l_nd_idx) = false;
    voxel_unvisited(l_isop_Q) = false;
    """
    voxel_unvisited = np.ones(n_total_voxels, dtype=bool)
    voxel_unvisited[np.asarray(node_voxels, dtype=np.int64)] = False
    voxel_unvisited[np.asarray(isopoint_voxels, dtype=np.int64)] = False
    return voxel_unvisited


def build_link_cc(
    adj: List[List[int]],
    link_set: Set[int],
    node_set: Set[int],
    endpoint_set: Set[int],
    link_start_ids: np.ndarray,     # possible seeds (endpoints + node neighbors) where link can start, same as l_link_start_voxel_idx in matlab
    voxel_unvisited: np.ndarray, # important: includes endpoints and link voxels (not nodes/isopoints -> can't be part of the link)
) -> tuple:
    """
    Replicates fun_skeleton_get_link_cc.
    Only traces normal voxel link chains.
    
        Parameters
    ----------
    adj : list of lists
        Full voxel adjacency list.
    link_set : set
        Set of link voxel IDs.
    node_set : set
        Set of node voxel IDs.
    endpoint_set : set
        Set of endpoint voxel IDs.
    link_start_ids : np.ndarray
        Start voxels for tracing.
    voxel_unvisited : np.ndarray
        Boolean mask. True means this voxel can still be traced
        (typically endpoints + link voxels).
        
    Returns:
        chains          : list of lists of voxel IDs (each list = one chain = one link)
        terminals       : for each chain, returns the ending point (can be node or endpoint or None) 
        voxel_unvisited : boolean array updated where visited voxels turn False
    
    Notes
    -----
    - Node voxels are not added to the traced chain.
    - Endpoint voxels can appear in the chain if tracing starts from them.
    - Internal voxels of the chain are degree 2 link voxels.
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
                
                # If I reach a node or an endpoint, I stop and record it as terminal (but I don't add it to the chain, because chain is only link voxels)
                if nb in node_set or nb in endpoint_set:
                    # final voxel of the chain reached/no more unvisited nodes
                    terminal = nb

            if not keep_tracking:
                break

        chains.append(np.asarray(chain, dtype=np.int64))
        terminals.append(terminal)

    return chains, terminals, voxel_unvisited


def trace_isoloop_cc(
    adj: List[List[int]],
    voxel_unvisited: np.ndarray,
    link_set: Set[int],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Trace isolated loops from the remaining unvisited link voxels.

    This corresponds to the second MATLAB call to fun_skeleton_get_link_cc(...)
    on the leftover unvisited voxels.

    Parameters
    ----------
    adj : list of lists
        Full voxel adjacency list.
    voxel_unvisited : np.ndarray
        Boolean mask after normal link tracing.
    link_set : set
        Set of voxel IDs classified as link voxels.

    Returns
    -------
    isoloop_chains : list of np.ndarray
        Each entry is one isolated loop represented as an ordered voxel chain.
    voxel_unvisited : np.ndarray
        Updated visitation mask.
    """
    isoloop_chains = []

    # Only remaining link voxels can form isolated loops
    remaining = [int(v) for v in np.where(voxel_unvisited)[0] if int(v) in link_set]

    for start in remaining:
        if not voxel_unvisited[start]:
            continue

        chain = [start]
        voxel_unvisited[start] = False

        current = start
        previous = -1

        while True:
            next_candidates = []

            for nb in adj[current]:
                if nb == previous:
                    continue
                if nb in link_set and voxel_unvisited[nb]:
                    next_candidates.append(nb)

            if not next_candidates:
                break

            nxt = next_candidates[0]
            chain.append(nxt)
            voxel_unvisited[nxt] = False
            previous = current
            current = nxt

        isoloop_chains.append(np.asarray(chain, dtype=np.int64))

    return isoloop_chains, voxel_unvisited


def chain_euclidean_length(chain: np.ndarray, coords: np.ndarray) -> float:
    if len(chain) <= 1:
        return 0.0
    pts = coords[chain].astype(float)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ============================================================
# MATLAB-like graph construction
# ============================================================
def build_link_struct(chains: List[np.ndarray], coords: np.ndarray, n_total_voxels: int) -> LinkStruct:
    """
    MATLAB-like construction of graph_str.link from traced link CCs.
    Node-link connectivity is NOT assigned here yet.
    """
    base = build_component_struct(chains, n_total_voxels)

    num_cc = base.num_cc
    num_endpoint_per_link = np.zeros(num_cc, dtype=np.int64)
    connected_endpoint_label = np.full((num_cc, 2), -1, dtype=np.int64)

    return LinkStruct(
        num_cc=base.num_cc,
        cc_ind=base.cc_ind,
        pos_ind=base.pos_ind,
        num_voxel_all_components=base.num_voxel_all_components,
        num_voxel_per_cc=base.num_voxel_per_cc,
        label=base.label,
        map_voxel_to_label=base.map_voxel_to_label,
        connected_node_label=np.full((num_cc, 2), -1, dtype=np.int64),
        connected_endpoint_label=connected_endpoint_label,
        num_node_per_link=np.zeros(num_cc, dtype=np.int64),
        num_endpoint_per_link=num_endpoint_per_link,
        link_length_euclidean=np.array([chain_euclidean_length(c, coords) for c in chains], dtype=float) if num_cc > 0 else np.array([], dtype=float),
    )



def connect_nodes_with_links_voxelwise(
    node: NodeStruct,
    link: LinkStruct,
    adj: List[List[int]],
) -> Tuple[NodeStruct, LinkStruct]:
    """
    MATLAB-like voxelwise node-link connection.

    Equivalent in spirit to:
        for each node voxel:
            inspect neighbors
            detect link labels
            add info to node and link
    """
    if node.num_cc == 0 or link.num_cc == 0:
        return node, link

    node.connected_link_label = [[] for _ in range(node.num_cc)]

    connected_node_label = np.full((link.num_cc, 2), -1, dtype=np.int64)
    num_node_per_link = np.zeros(link.num_cc, dtype=np.int64)

    # Traverse each node voxel
    for local_idx, node_voxel_id in enumerate(node.pos_ind):
        node_voxel_id = int(node_voxel_id)
        node_label = int(node.label[local_idx])

        neighbor_link_labels = []

        for nb in adj[node_voxel_id]:
            link_label = int(link.map_voxel_to_label[nb])
            if link_label >= 0:
                neighbor_link_labels.append(link_label)

        if len(neighbor_link_labels) == 0:
            continue

        # MATLAB stores all hits on the node side, then unique() afterwards
        node.connected_link_label[node_label].extend(neighbor_link_labels)

        # MATLAB does NOT unique on the link side before filling the two slots.
        for link_label in neighbor_link_labels:
            num_node_per_link[link_label] += 1
            if num_node_per_link[link_label] > 2:
                raise AssertionError("The link connects to more than 2 node voxels")
            connected_node_label[link_label, num_node_per_link[link_label] - 1] = node_label

    # Unique only on node side, exactly like MATLAB
    node.connected_link_label = [
        np.unique(np.asarray(lst, dtype=np.int64)) if len(lst) > 0 else np.array([], dtype=np.int64)
        for lst in node.connected_link_label
    ]
    node.num_links_per_node = np.array([len(x) for x in node.connected_link_label], dtype=np.int64)

    link.connected_node_label = connected_node_label
    link.num_node_per_link = num_node_per_link

    return node, link


def fill_link_endpoint_connectivity(link: LinkStruct, endpoint: EndpointStruct) -> LinkStruct:
    """
    Fill link.connected_endpoint_label and link.num_endpoint_per_link
    from endpoint.link_label.

    This is not required by MATLAB to build endpoint.link_label, but it is a
    consistent derived representation.
    """
    if link.num_cc == 0 or endpoint.num_voxel == 0:
        return link

    connected_endpoint_label = np.full((link.num_cc, 2), -1, dtype=np.int64)
    num_endpoint_per_link = np.zeros(link.num_cc, dtype=np.int64)

    for ep_label, link_label in enumerate(endpoint.link_label):
        link_label = int(link_label)
        if link_label < 0:
            continue

        num_endpoint_per_link[link_label] += 1
        if num_endpoint_per_link[link_label] > 2:
            raise AssertionError("A link is connected to more than 2 endpoints")

        slot = num_endpoint_per_link[link_label] - 1
        connected_endpoint_label[link_label, slot] = ep_label

    link.connected_endpoint_label = connected_endpoint_label
    link.num_endpoint_per_link = num_endpoint_per_link
    return link

# ============================================================
# Main MATLAB-like function
# ============================================================
def fun_skeleton_to_graph(skel: np.ndarray) -> dict:
    """
    MATLAB-like graph construction.

    This version returns a structured graph_str-like dictionary

    Steps:
    - classify voxels (node, link, endpoint, isopoint)
    - trace links from endpoints and node neighbors
    - trace isolated loops from remaining link voxels
    - build link, node, endpoint, isopoint, isoloop structures
    - connect nodes with links voxelwise (26-neighborhood)
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
    # Trace normal links
    # --------------------------------------------------------
    node_set = set(map(int, topo["node"]))
    endpoint_set = set(map(int, topo["endpoint"]))
    link_set = set(map(int, topo["link"]))

    link_start_ids = get_link_start_voxels(
        node_voxels=topo["node"],
        endpoint_voxels=topo["endpoint"],
        adj=adj,
        node_set=node_set,
    )

    voxel_unvisited = initialize_voxel_unvisited(
        n_total_voxels=len(data.coords),
        node_voxels=topo["node"],
        isopoint_voxels=topo["isopoint"],
    )

    link_chains, _, voxel_unvisited = build_link_cc(
        adj=adj,
        link_set=link_set,
        node_set=node_set,
        endpoint_set=endpoint_set,
        link_start_ids=link_start_ids,
        voxel_unvisited=voxel_unvisited,
    )
    
    # --------------------------------------------------------
    # Construct graph_str.link
    # --------------------------------------------------------
    link = build_link_struct(
        chains=link_chains,
        coords=data.coords,
        n_total_voxels=len(data.coords),
    )

    # --------------------------------------------------------
    # Construct graph_str.endpoint
    # MATLAB endpoint.link_label = link.map_voxel_to_label(endpoint.pos_ind)
    # --------------------------------------------------------
    endpoint = build_endpoint_struct(topo["endpoint"], len(data.coords), link)

    link = fill_link_endpoint_connectivity(link, endpoint)
    # --------------------------------------------------------
    # Construct graph_str.isopoint
    # --------------------------------------------------------
    isopoint = build_isopoint_struct(topo["isopoint"], len(data.coords))

    # --------------------------------------------------------
    # Construct graph_str.isoloop
    # MATLAB stores only cc_ind, num_cc, pos_ind
    # --------------------------------------------------------
    isoloop_chains, voxel_unvisited = trace_isoloop_cc(
        adj=adj,
        voxel_unvisited=voxel_unvisited,
        link_set=link_set,
    )
    isoloop = build_isoloop_struct(isoloop_chains)

    # --------------------------------------------------------
    # Construct graph_str.node
    # --------------------------------------------------------
    node = build_node_struct(topo["node"], adj, data.coords)

    # --------------------------------------------------------
    # Connect nodes with links voxel by voxel
    # MATLAB-like block: "Connect nodes with links"
    # --------------------------------------------------------
    node, link = connect_nodes_with_links_voxelwise(node, link, adj)

    # --------------------------------------------------------
    # Assemble MATLAB-like output
    # --------------------------------------------------------
    graph_str = {
        "num": num,
        "link": link,
        "endpoint": endpoint,
        "isopoint": isopoint,
        "isoloop": isoloop,
        "node": node,
        "coords": data.coords,
        "coord_to_id": data.coord_to_id,
        "degree": topo["degree"],
        "adj": adj,
    }

    return graph_str




# ============================================================
# Skeleton sintético de test
# ============================================================
def make_test_skeleton() -> np.ndarray:
    skel = np.zeros((12, 12, 12), dtype=bool)

    # --------------------------------------------------------
    # 1) Link simple entre 2 endpoints
    # (1,1,1) -- (2,1,1) -- (3,1,1)
    # --------------------------------------------------------
    skel[1, 1, 1] = True
    skel[2, 1, 1] = True
    skel[3, 1, 1] = True

    # --------------------------------------------------------
    # 2) Nodo con 3 ramas, ahora sí contiguas
    #
    #             (6,3,6) endpoint
    #                  |
    #             (6,4,6) link
    #                  |
    #             (6,5,6) link
    #                  |
    # (3,6,6)e-(4,6,6)l-(5,6,6)l-(6,6,6)node-(7,6,6)l-(8,6,6)e
    # --------------------------------------------------------
    skel[6, 6, 6] = True   # node central

    skel[5, 6, 6] = True   # link izquierda
    skel[4, 6, 6] = True   # link izquierda
    skel[3, 6, 6] = True   # endpoint izquierda

    skel[7, 6, 6] = True   # link derecha
    skel[8, 6, 6] = True   # endpoint derecha

    skel[6, 5, 6] = True   # link arriba
    skel[6, 4, 6] = True   # link arriba
    skel[6, 3, 6] = True   # endpoint arriba

    # --------------------------------------------------------
    # 3) Isopoint
    # --------------------------------------------------------
    skel[1, 10, 1] = True

    # --------------------------------------------------------
    # 4) Isoloop aislado
    # --------------------------------------------------------
    loop_pts = [
        (8, 3, 9),
        (9, 2, 10),
        (10, 1, 9),
        (9, 2, 8),
    ]
    for p in loop_pts:
        skel[p] = True

    return skel
# ============================================================
# Helpers de inspección
# ============================================================

def voxel_ids_to_coords(ids: np.ndarray, coords: np.ndarray):
    return [tuple(coords[int(i)]) for i in ids]

def print_graph_summary(graph_str: dict):
    coords = graph_str["coords"]

    print("\n" + "="*60)
    print("RESUMEN GENERAL")
    print("="*60)
    print("mask_size          :", graph_str["num"].mask_size)
    print("skeleton_voxel     :", graph_str["num"].skeleton_voxel)
    print("block_voxel        :", graph_str["num"].block_voxel)
    print("neighborhood       :", graph_str["num"].neighborhood)

    print("\n" + "-"*60)
    print("CLASIFICACIÓN GLOBAL")
    print("-"*60)
    deg = graph_str["degree"]
    print("n degree==0 (isopoint):", np.sum(deg == 0))
    print("n degree==1 (endpoint):", np.sum(deg == 1))
    print("n degree==2 (link)    :", np.sum(deg == 2))
    print("n degree>2 (node)     :", np.sum(deg > 2))

    node = graph_str["node"]
    link = graph_str["link"]
    endpoint = graph_str["endpoint"]
    isopoint = graph_str["isopoint"]
    isoloop = graph_str["isoloop"]

    print("\n" + "-"*60)
    print("NODE")
    print("-"*60)
    print("num_cc:", node.num_cc)
    for i, comp in enumerate(node.cc_ind):
        print(f"  node_cc[{i}] voxels ids   :", comp.tolist())
        print(f"  node_cc[{i}] voxels coords:", voxel_ids_to_coords(comp, coords))
        print(f"  node_cc[{i}] centroid     :", node.centroid[i])
        print(f"  node_cc[{i}] connected links:", node.connected_link_label[i].tolist())

    print("\n" + "-"*60)
    print("LINK")
    print("-"*60)
    print("num_cc:", link.num_cc)
    for i, comp in enumerate(link.cc_ind):
        print(f"  link_cc[{i}] voxels ids   :", comp.tolist())
        print(f"  link_cc[{i}] voxels coords:", voxel_ids_to_coords(comp, coords))
        print(f"  link_cc[{i}] connected_node_label     :", link.connected_node_label[i].tolist())
        print(f"  link_cc[{i}] connected_endpoint_label :", link.connected_endpoint_label[i].tolist())
        print(f"  link_cc[{i}] num_node_per_link        :", int(link.num_node_per_link[i]))
        print(f"  link_cc[{i}] num_endpoint_per_link    :", int(link.num_endpoint_per_link[i]))
        print(f"  link_cc[{i}] euclidean_length         :", float(link.link_length_euclidean[i]))

    print("\n" + "-"*60)
    print("ENDPOINT")
    print("-"*60)
    print("num_voxel:", endpoint.num_voxel)
    for i, vid in enumerate(endpoint.pos_ind):
        print(f"  endpoint[{i}] id={int(vid)} coord={tuple(coords[int(vid)])} link_label={int(endpoint.link_label[i])}")

    print("\n" + "-"*60)
    print("ISOPOINT")
    print("-"*60)
    print("num_voxel:", isopoint.num_voxel)
    for i, vid in enumerate(isopoint.pos_ind):
        print(f"  isopoint[{i}] id={int(vid)} coord={tuple(coords[int(vid)])}")

    print("\n" + "-"*60)
    print("ISOLOOP")
    print("-"*60)
    print("num_cc:", isoloop.num_cc)
    for i, comp in enumerate(isoloop.cc_ind):
        print(f"  isoloop_cc[{i}] ids   :", comp.tolist())
        print(f"  isoloop_cc[{i}] coords:", voxel_ids_to_coords(comp, coords))


# ============================================================
# Checks automáticos
# ============================================================

def run_synthetic_check():
    skel = make_test_skeleton()
    graph_str = fun_skeleton_to_graph(skel)

    print_graph_summary(graph_str)

    node = graph_str["node"]
    link = graph_str["link"]
    endpoint = graph_str["endpoint"]
    isopoint = graph_str["isopoint"]
    isoloop = graph_str["isoloop"]

    print("\n" + "="*60)
    print("ASSERTS")
    print("="*60)

    # Esperados globales
    assert graph_str["num"].skeleton_voxel == int(np.count_nonzero(skel))

    # 1 isopoint
    assert isopoint.num_voxel == 1, f"Esperaba 1 isopoint, obtuve {isopoint.num_voxel}"

    # 1 node_cc (el voxel central)
    assert node.num_cc == 1, f"Esperaba 1 node_cc, obtuve {node.num_cc}"
    assert int(node.num_links_per_node[0]) == 3, (
        f"Esperaba 3 links conectados al nodo, obtuve {node.num_links_per_node[0]}"
    )
    # 5 endpoints:
    # 2 del link lineal + 3 del nodo ramificado
    assert endpoint.num_voxel == 5, f"Esperaba 5 endpoints, obtuve {endpoint.num_voxel}"

    # 4 links normales:
    # 1 lineal + 3 ramas del nodo
    assert link.num_cc == 4, f"Esperaba 4 link_cc, obtuve {link.num_cc}"

    # 1 isoloop
    assert isoloop.num_cc == 1, f"Esperaba 1 isoloop_cc, obtuve {isoloop.num_cc}"
    assert len(isoloop.cc_ind[0]) == 4, f"Esperaba loop de 4 voxels, obtuve {len(isoloop.cc_ind[0])}"

    # El nodo central debe tener 3 links conectados
    assert int(node.num_links_per_node[0]) == 3, (
        f"Esperaba 3 links conectados al nodo, obtuve {node.num_links_per_node[0]}"
    )

    # Distribuciones topológicas de los links:
    # - 1 link con 0 nodos y 2 endpoints   (el lineal)
    # - 3 links con 1 nodo y 1 endpoint    (las ramas)
    assert sorted(link.num_node_per_link.tolist()) == [0, 1, 1, 1], (
        f"num_node_per_link inesperado: {link.num_node_per_link.tolist()}"
    )
    assert sorted(link.num_endpoint_per_link.tolist()) == [1, 1, 1, 2], (
        f"num_endpoint_per_link inesperado: {link.num_endpoint_per_link.tolist()}"
    )

    print("Todos los checks han pasado correctamente.")
    return graph_str


# ============================================================
# Ejecutar test
# ============================================================

graph_test = run_synthetic_check()