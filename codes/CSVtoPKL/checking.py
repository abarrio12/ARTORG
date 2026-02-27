import pickle
import numpy as np



# ======================================================================
#  CHECKING FOR ONLY ONE EDGE
# ======================================================================
def print_edge_debug(pkl_path, ei=0, max_show=50):
    """
    Prints for one edge:
      - geom index range (s,en) and #points/#segments
      - per-segment lengths2 for that edge (from geom['lengths2'])
      - edge length (G.es['length']) if present
      - edge length_steps (G.es['length_steps']) if present
      - edge diameter_atlas (G.es['diameter_atlas']) if present
      - per-point atlas diameters along the polyline (from geom['diam_atlas_geom'] or 2*radii_atlas_geom)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]
    g = data.get("geom", {})

    # --- required geometry ---
    lengths2 = np.asarray(g.get("lengths2", None))
    if lengths2 is None:
        raise KeyError("geom['lengths2'] not found in PKL")

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    if ei < 0 or ei >= len(gs):
        raise IndexError(f"ei={ei} out of range (0..{len(gs)-1})")

    s = int(gs[ei]); en = int(ge[ei])
    npts = en - s
    nseg = max(0, npts - 1)

    # --- edge attrs ---
    def get_edge_attr(name):
        return None if name not in G.es.attributes() else G.es[name][ei]

    length = get_edge_attr("length")
    length_steps = get_edge_attr("length_steps")
    diam_atlas_edge = get_edge_attr("diameter_atlas")

    # --- per-point atlas diameter along geometry ---
    if "diam_atlas_geom" in g:
        diam_atlas_geom = np.asarray(g["diam_atlas_geom"], dtype=np.float32)
    elif "radii_atlas_geom" in g:
        diam_atlas_geom = 2.0 * np.asarray(g["radii_atlas_geom"], dtype=np.float32)
    else:
        diam_atlas_geom = None

    # --- slice ---
    l2_edge = lengths2[s:en-1] if nseg > 0 else np.array([], dtype=np.float32)  # segments
    if diam_atlas_geom is not None:
        dpts_edge = diam_atlas_geom[s:en]  # points
    else:
        dpts_edge = None

    print("\n==================== EDGE DEBUG ====================")
    print(f"PKL: {pkl_path}")
    print(f"edge ei={ei}")
    print(f"geom_start={s}, geom_end={en}  -> npts={npts}, nseg={nseg}")

    print("\n--- lengths2 (segments) ---")
    if l2_edge.size == 0:
        print("No segments (edge has <2 points).")
    else:
        show = l2_edge[:max_show]
        print(f"lengths2 count={l2_edge.size}  min={float(l2_edge.min()):.6f}  "
              f"med={float(np.median(l2_edge)):.6f}  max={float(l2_edge.max()):.6f}  sum={float(l2_edge.sum()):.6f}")
        print("first values:", np.array2string(show, precision=6, separator=", "))
        if l2_edge.size > max_show:
            print(f"... (showing first {max_show} of {l2_edge.size})")

    print("\n--- edge attributes ---")
    print("length (edge):", length)
    print("length_steps (edge):", length_steps)
    print("diameter_atlas (edge):", diam_atlas_edge)

    print("\n--- diameter atlas along points ---")
    if dpts_edge is None:
        print("No geom['diam_atlas_geom'] or geom['radii_atlas_geom'] found.")
    else:
        showp = dpts_edge[:max_show]
        print(f"diam_atlas_points count={dpts_edge.size}  min={float(dpts_edge.min()):.6f}  "
              f"med={float(np.median(dpts_edge)):.6f}  max={float(dpts_edge.max()):.6f}")
        print("first values:", np.array2string(showp, precision=6, separator=", "))
        if dpts_edge.size > max_show:
            print(f"... (showing first {max_show} of {dpts_edge.size})")

    print("====================================================\n")



#print_edge_debug("/home/ana/MicroBrain/output/um/graph_18_OutGeom_um.pkl", ei=12, max_show=30)


def print_edge_debug_um(pkl_path, ei=0, max_show=50):
    """
    UM-aware edge debug printer.

    It prints (when available):
      VOX:
        - geom[x,y,z], geom[lengths2]
        - edge length_steps, length
        - diameter_atlas (raw atlas units)
        - diam_atlas_geom (raw atlas units, per point)
      UM:
        - geom_R[x_R,y_R,z_R], geom_R[lengths2_R]
        - edge length_R / length_um (if present)
        - diameter_atlas_R (µm)
        - diam_atlas_geom_R (µm, per point)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]
    g  = data.get("geom", {})
    gR = data.get("geom_R", {})
    v  = data.get("vertex", {})
    vR = data.get("vertex_R", {})

    gs = np.asarray(G.es["geom_start"], dtype=np.int64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)

    if ei < 0 or ei >= len(gs):
        raise IndexError(f"ei={ei} out of range (0..{len(gs)-1})")

    s = int(gs[ei]); en = int(ge[ei])
    npts = en - s
    nseg = max(0, npts - 1)

    def get_edge_attr(name):
        return None if name not in G.es.attributes() else G.es[name][ei]

    # -------------------------
    # Edge attrs (VOX + UM)
    # -------------------------
    length_steps = get_edge_attr("length_steps")
    length_vox   = get_edge_attr("length")            # in your VOX builder: voxel arc length
    length_R     = get_edge_attr("length_R") or get_edge_attr("length_um")
    diam_atlas_edge_raw = get_edge_attr("diameter_atlas")
    diam_atlas_edge_um  = get_edge_attr("diameter_atlas_R")

    # -------------------------
    # Geometry per-segment (VOX + UM)
    # -------------------------
    lengths2_vox = g.get("lengths2", None)
    lengths2_um  = gR.get("lengths2_R", None)

    l2_edge_vox = None
    if lengths2_vox is not None and nseg > 0:
        lengths2_vox = np.asarray(lengths2_vox, dtype=np.float32)
        l2_edge_vox = lengths2_vox[s:en-1]

    l2_edge_um = None
    if lengths2_um is not None and nseg > 0:
        lengths2_um = np.asarray(lengths2_um, dtype=np.float32)
        l2_edge_um = lengths2_um[s:en-1]

    # -------------------------
    # Atlas diameter per-point (raw + um)
    # -------------------------
    # raw (atlas units)
    diam_pts_raw = None
    if "diam_atlas_geom" in g:
        diam_pts_raw = np.asarray(g["diam_atlas_geom"], dtype=np.float32)[s:en]
    elif "radii_atlas_geom" in g:
        diam_pts_raw = 2.0 * np.asarray(g["radii_atlas_geom"], dtype=np.float32)[s:en]

    # um
    diam_pts_um = None
    if "diam_atlas_geom_R" in gR:
        diam_pts_um = np.asarray(gR["diam_atlas_geom_R"], dtype=np.float32)[s:en]
    elif "radii_atlas_geom_R" in gR:
        diam_pts_um = 2.0 * np.asarray(gR["radii_atlas_geom_R"], dtype=np.float32)[s:en]

    # -------------------------
    # Print
    # -------------------------
    print("\n==================== EDGE DEBUG (UM-AWARE) ====================")
    print(f"PKL: {pkl_path}")
    print(f"edge ei={ei}")
    print(f"geom_start={s}, geom_end={en}  -> npts={npts}, nseg={nseg}")

    print("\n--- EDGE ATTRIBUTES ---")
    print("length_steps:", length_steps)
    print("length (vox arc length):", length_vox)
    print("length_R / length_um (µm):", length_R)
    print("diameter_atlas (raw atlas units):", diam_atlas_edge_raw)
    print("diameter_atlas_R (µm):", diam_atlas_edge_um)

    def _print_arr_stats(title, arr, unit=""):
        if arr is None:
            print(f"\n--- {title} ---")
            print("Not available.")
            return
        if arr.size == 0:
            print(f"\n--- {title} ---")
            print("Empty.")
            return
        show = arr[:max_show]
        print(f"\n--- {title} ---")
        print(f"count={arr.size}  min={float(arr.min()):.6f}{unit}  med={float(np.median(arr)):.6f}{unit}  "
              f"max={float(arr.max()):.6f}{unit}  sum={float(arr.sum()):.6f}{unit}")
        print("first values:", np.array2string(show, precision=6, separator=", "))
        if arr.size > max_show:
            print(f"... (showing first {max_show} of {arr.size})")

    _print_arr_stats("lengths2 (VOX, per segment)", l2_edge_vox, unit="")
    _print_arr_stats("lengths2_R (µm, per segment)", l2_edge_um, unit=" µm")

    _print_arr_stats("diam_atlas_geom (raw, per point)", diam_pts_raw, unit="")
    _print_arr_stats("diam_atlas_geom_R (µm, per point)", diam_pts_um, unit=" µm")

    print("===============================================================\n")


# Example:
#print_edge_debug_um("/home/ana/MicroBrain/output/um/graph_18_OutGeom_um.pkl", ei=12, max_show=30)




# ===========================================================================
#  CHECKING FOR WHOLE BRAIN
# ===========================================================================
import pickle
import numpy as np

def _safe_stats(arr):
    """Return (n, mean, max) over finite values; None if empty."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0, None, None
    return int(arr.size), float(arr.mean()), float(arr.max())


def summarize_outgeom_vox(pkl_path):
    """
    VOX summary:
      - V, E
      - mean/max of:
          * diameter_atlas (raw atlas units)
          * length (vox arc length, as stored in VOX build)
          * length_steps (if present)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]

    V = G.vcount()
    E = G.ecount()

    print("\n==================== VOX SUMMARY ====================")
    print(f"PKL: {pkl_path}")
    print(f"V={V:,}  E={E:,}")

    # diameter_atlas (raw atlas units)
    if "diameter_atlas" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["diameter_atlas"])
        print(f"diameter_atlas (raw atlas units): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    else:
        print("diameter_atlas: NOT FOUND")

    # length (vox arc length)
    if "length" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length"])
        print(f"length (vox): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    else:
        print("length: NOT FOUND")

    # length_steps
    if "length_steps" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length_steps"])
        print(f"length_steps (count): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    else:
        print("length_steps: NOT FOUND")

    print("=====================================================\n")
    return {"V": V, "E": E}


def summarize_outgeom_um(pkl_path):
    """
    UM summary:
      - V, E
      - mean/max of:
          * diameter_atlas_R (µm) if present, else compute from diameter_atlas*25 if present
          * length_R (µm) if present, else length_um if present
          * (also prints VOX length/steps if they are still present)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]

    V = G.vcount()
    E = G.ecount()

    print("\n==================== UM SUMMARY ======================")
    print(f"PKL: {pkl_path}")
    print(f"V={V:,}  E={E:,}")

    # diameter atlas in µm
    if "diameter_atlas_R" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["diameter_atlas_R"])
        print(f"diameter_atlas_R (µm): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    elif "diameter_atlas" in G.es.attributes():
        # fallback: assume atlas_um_per_voxel stored in data (else default 25)
        atlas_um = float(data.get("atlas_um_per_voxel", 25.0))
        darr = np.asarray(G.es["diameter_atlas"], dtype=np.float64) * atlas_um
        n, meanv, maxv = _safe_stats(darr)
        print(f"diameter_atlas*{atlas_um:g} (µm, fallback): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    else:
        print("diameter_atlas_R: NOT FOUND (and no diameter_atlas fallback)")

    # length in µm
    if "length_R" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length_R"])
        print(f"length_R (µm): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    elif "length_um" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length_um"])
        print(f"length_um (µm): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    else:
        print("length_R/length_um: NOT FOUND")

    # Also show what's still there from VOX (useful sanity)
    if "length" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length"])
        print(f"length (vox, still present): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")
    if "length_steps" in G.es.attributes():
        n, meanv, maxv = _safe_stats(G.es["length_steps"])
        print(f"length_steps (count, still present): n={n:,}  mean={meanv:.6f}  max={maxv:.6f}")

    print("=====================================================\n")
    return {"V": V, "E": E}


def compare_vox_vs_um(vox_pkl_path, um_pkl_path):
    """
    Prints VOX summary and UM summary, and checks V/E match.
    """
    vox = summarize_outgeom_vox(vox_pkl_path)
    um  = summarize_outgeom_um(um_pkl_path)

    print("============== V/E CONSISTENCY CHECK ==============")
    sameV = vox["V"] == um["V"]
    sameE = vox["E"] == um["E"]
    print(f"Same V? {sameV}  (vox={vox['V']:,}, um={um['V']:,})")
    print(f"Same E? {sameE}  (vox={vox['E']:,}, um={um['E']:,})")
    print("===================================================\n")



#compare_vox_vs_um(
#     "/home/ana/MicroBrain/output/graph_18_OutGeom.pkl",
#     "/home/ana/MicroBrain/output/um/graph_18_OutGeom_um.pkl",
#)



# =======================================================================00
# CHECKING WITH NKINDS AND ALSO TOTALS OF WHOLE BRAIN
#==========================================================================
import pickle
import numpy as np

def _finite(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]

def _get_attr(G, name):
    return None if name not in G.es.attributes() else np.asarray(G.es[name])

def _detect_space(data):
    return "um" if ("geom_R" in data and "vertex_R" in data) else "vox"

def summarize_by_nkind(pkl_path, space="auto"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]
    V = G.vcount()
    E = G.ecount()

    if space == "auto":
        space = _detect_space(data)

    # choose keys
    if space == "um":
        len_key  = "length_R" if "length_R" in G.es.attributes() else ("length_um" if "length_um" in G.es.attributes() else None)
        diam_key = "diameter_atlas_R" if "diameter_atlas_R" in G.es.attributes() else None
        unit_len = "µm"
        unit_d   = "µm"
    else:
        len_key  = "length" if "length" in G.es.attributes() else None  # voxel arc length
        diam_key = "diameter_atlas" if "diameter_atlas" in G.es.attributes() else None  # raw atlas units
        unit_len = "vox"
        unit_d   = "atlas-vox"

    nkind = _get_attr(G, "nkind")
    if nkind is None:
        raise KeyError("Edge attribute 'nkind' not found.")

    L = _get_attr(G, len_key) if len_key else None
    D = _get_attr(G, diam_key) if diam_key else None

    print(f"\n==================== {space.upper()} BY NKIND ====================")
    print(f"PKL: {pkl_path}")
    print(f"V={V:,}  E={E:,}")
    print(f"Using length key: {len_key} ({unit_len})")
    print(f"Using diam   key: {diam_key} ({unit_d})")

    for k in sorted(set(map(int, np.unique(nkind)))):
        m = (nkind == k)
        cnt = int(np.sum(m))

        if L is not None:
            Lk = _finite(L[m])
            sumL = float(np.sum(Lk)) if Lk.size else float("nan")
            meanL = float(np.mean(Lk)) if Lk.size else float("nan")
            maxL = float(np.max(Lk)) if Lk.size else float("nan")
        else:
            sumL = meanL = maxL = float("nan")

        if D is not None:
            Dk = _finite(D[m])
            meanD = float(np.mean(Dk)) if Dk.size else float("nan")
            maxD  = float(np.max(Dk)) if Dk.size else float("nan")
        else:
            meanD = maxD = float("nan")

        # meters if UM
        if space == "um" and np.isfinite(sumL):
            sumL_m = sumL / 1e6
            print(f"nkind={k}: n={cnt:,}  sumL={sumL:,.3f} {unit_len} ({sumL_m:,.3f} m)  meanL={meanL:.3f}  maxL={maxL:.3f}  meanD={meanD:.3f}  maxD={maxD:.3f}")
        else:
            print(f"nkind={k}: n={cnt:,}  sumL={sumL:,.3f} {unit_len}  meanL={meanL:.3f}  maxL={maxL:.3f}  meanD={meanD:.3f}  maxD={maxD:.3f}")

    print("=====================================================\n")


def total_length_summary(vox_pkl, um_pkl):
    # VOX
    with open(vox_pkl, "rb") as f:
        dvox = pickle.load(f)
    Gv = dvox["graph"]
    Vv, Ev = Gv.vcount(), Gv.ecount()

    Lvox = _get_attr(Gv, "length")
    Lsteps = _get_attr(Gv, "length_steps")
    Datlas = _get_attr(Gv, "diameter_atlas")

    sum_vox = float(np.sum(_finite(Lvox))) if Lvox is not None else float("nan")
    sum_steps = float(np.sum(_finite(Lsteps))) if Lsteps is not None else float("nan")
    mean_datlas = float(np.mean(_finite(Datlas))) if Datlas is not None else float("nan")

    # UM
    with open(um_pkl, "rb") as f:
        dum = pickle.load(f)
    Gu = dum["graph"]
    Vu, Eu = Gu.vcount(), Gu.ecount()

    Lr = _get_attr(Gu, "length_R")
    if Lr is None:
        Lr = _get_attr(Gu, "length_um")
    Dr = _get_attr(Gu, "diameter_atlas_R")

    sum_um = float(np.sum(_finite(Lr))) if Lr is not None else float("nan")
    sum_um_m = sum_um / 1e6 if np.isfinite(sum_um) else float("nan")
    mean_d_um = float(np.mean(_finite(Dr))) if Dr is not None else float("nan")

    print("\n==================== TOTAL LENGTH SUMMARY ====================")
    print(f"VOX: V={Vv:,} E={Ev:,}  sum(length_vox)={sum_vox:,.3f} vox  sum(length_steps)={sum_steps:,.0f} steps  mean(diam_atlas)={mean_datlas:.6f} atlas-vox")
    print(f" UM: V={Vu:,} E={Eu:,}  sum(length_R)={sum_um:,.3f} µm  = {sum_um_m:,.3f} m  mean(diam_atlas_R)={mean_d_um:.3f} µm")
    print(f"V match? {Vv==Vu}   E match? {Ev==Eu}")
    print("=============================================================\n")


def check_global_jump_is_zero(um_pkl, tol=1e-6):
    """
    Checks if lengths2_R[ge-1] is ~0 for all edges (kills the global jump).
    """
    with open(um_pkl, "rb") as f:
        data = pickle.load(f)

    G = data["graph"]
    gR = data.get("geom_R", {})
    if "lengths2_R" not in gR:
        raise KeyError("geom_R['lengths2_R'] not found.")

    lengths2_R = np.asarray(gR["lengths2_R"], dtype=np.float64)
    ge = np.asarray(G.es["geom_end"], dtype=np.int64)
    idx = ge - 1
    idx = idx[(idx >= 0) & (idx < lengths2_R.size)]
    vals = lengths2_R[idx]

    max_abs = float(np.max(np.abs(vals)))
    n_bad = int(np.sum(np.abs(vals) > tol))

    print("\n==================== GLOBAL JUMP CHECK (UM) ====================")
    print(f"max |lengths2_R[ge-1]| = {max_abs:.6e} µm")
    print(f"count > {tol} : {n_bad:,} / {len(idx):,}")
    print("===============================================================\n")


vox = "/home/ana/MicroBrain/output/graph_18_OutGeom.pkl"
um  = "/home/ana/MicroBrain/output/um/graph_18_OutGeom_um.pkl"

total_length_summary(vox, um)
summarize_by_nkind(vox, space="vox")
summarize_by_nkind(um,  space="um")

#check_global_jump_is_zero(um, tol=1e-6)

import pickle
import numpy as np
import igraph as ig


# ============================================================
# LOAD FORMATTED GRAPH (written with G.write_pickle)
# ============================================================
def load_formatted_graph(pkl_path):
    try:
        G = ig.Graph.Read_Pickle(pkl_path)
    except Exception:
        # fallback in case someone saved with plain pickle.dump(G)
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ig.Graph):
            raise TypeError("This file does not contain an igraph.Graph.")
        G = obj
    return G


def _finite(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_stats(arr):
    """Return (n, mean, max, min, sum) over finite values; None stats if empty."""
    a = _finite(arr)
    if a.size == 0:
        return 0, None, None, None, None
    return int(a.size), float(a.mean()), float(a.max()), float(a.min()), float(a.sum())


# ============================================================
# 1) EDGE DEBUG (FORMATTED)
# ============================================================
def print_edge_debug_formatted(pkl_path, ei=0, max_show=50, tol_len=1e-6, tol_diam=1e-6):
    """
    Prints for one edge in the formatted graph:
      - connectivity, nkind
      - points count, segments count
      - per-segment lengths2 stats + compare with scalar 'length'
      - per-point diameters stats + compare with scalar 'diameter'
    """
    G = load_formatted_graph(pkl_path)

    if ei < 0 or ei >= G.ecount():
        raise IndexError(f"ei={ei} out of range (0..{G.ecount()-1})")

    unit = G["unit"] if "unit" in G.attributes() else "unknown"

    e = G.es[ei]

    conn = e["connectivity"] if "connectivity" in G.es.attributes() else tuple(e.tuple)
    nkind = e["nkind"] if "nkind" in G.es.attributes() else None

    pts = e["points"] if "points" in G.es.attributes() else []
    l2  = e["lengths2"] if "lengths2" in G.es.attributes() else []
    dps = e["diameters"] if "diameters" in G.es.attributes() else []

    L_scalar = e["length"] if "length" in G.es.attributes() else None
    D_scalar = e["diameter"] if "diameter" in G.es.attributes() else None

    npts = len(pts)
    nseg = max(0, npts - 1)

    l2_arr = _finite(l2) if len(l2) else np.array([])
    dps_arr = _finite(dps) if len(dps) else np.array([])

    sum_l2 = float(l2_arr.sum()) if l2_arr.size else 0.0
    mean_d = float(dps_arr.mean()) if dps_arr.size else np.nan

    print("\n==================== EDGE DEBUG (FORMATTED) ====================")
    print(f"PKL: {pkl_path}")
    print(f"unit: {unit}")
    print(f"edge ei={ei}")
    print(f"connectivity={conn}  nkind={nkind}")
    print(f"npts={npts}  nseg={nseg}")
    print(f"len(lengths2)={len(l2)}  len(diameters)={len(dps)}")

    print("\n--- lengths2 (per segment) ---")
    if l2_arr.size == 0:
        print("Empty / non-finite.")
    else:
        show = l2_arr[:max_show]
        print(f"count={l2_arr.size}  min={float(l2_arr.min()):.6f}  med={float(np.median(l2_arr)):.6f}  "
              f"max={float(l2_arr.max()):.6f}  sum={float(l2_arr.sum()):.6f} ({unit})")
        print("first values:", np.array2string(show, precision=6, separator=", "))
        if l2_arr.size > max_show:
            print(f"... (showing first {max_show} of {l2_arr.size})")

    print("\n--- length scalar ---")
    print("length:", L_scalar, f"({unit})")
    if L_scalar is not None and np.isfinite(float(L_scalar)) and l2_arr.size > 0:
        diff = abs(float(L_scalar) - sum_l2)
        print(f"sum(lengths2)={sum_l2:.6f}  |length - sum|={diff:.6e}")
        if diff > tol_len:
            print(f"WARNING: length != sum(lengths2) by > {tol_len}")

    print("\n--- diameters (per point) ---")
    if dps_arr.size == 0:
        print("Empty / non-finite.")
    else:
        show = dps_arr[:max_show]
        print(f"count={dps_arr.size}  min={float(dps_arr.min()):.6f}  med={float(np.median(dps_arr)):.6f}  "
              f"max={float(dps_arr.max()):.6f}  mean={float(dps_arr.mean()):.6f} ({unit})")
        print("first values:", np.array2string(show, precision=6, separator=", "))
        if dps_arr.size > max_show:
            print(f"... (showing first {max_show} of {dps_arr.size})")

    print("\n--- diameter scalar ---")
    print("diameter:", D_scalar, f"({unit})")
    if D_scalar is not None and np.isfinite(float(D_scalar)) and np.isfinite(mean_d):
        diff = abs(float(D_scalar) - mean_d)
        print(f"mean(diameters)={mean_d:.6f}  |diameter - mean|={diff:.6e}")
        if diff > tol_diam:
            print(f"WARNING: diameter != mean(diameters) by > {tol_diam}")

    print("===============================================================\n")


# ============================================================
# 2) WHOLE GRAPH SUMMARY (FORMATTED)
# ============================================================
def summarize_formatted(pkl_path):
    """
    Summary for formatted graph:
      - V, E
      - mean/max/min/sum of length
      - mean/max/min of diameter
      - length_steps if present
    """
    G = load_formatted_graph(pkl_path)
    unit = G["unit"] if "unit" in G.attributes() else "unknown"

    V = G.vcount()
    E = G.ecount()

    print("\n==================== FORMATTED SUMMARY ====================")
    print(f"PKL: {pkl_path}")
    print(f"unit: {unit}")
    print(f"V={V:,}  E={E:,}")

    if "length" in G.es.attributes():
        n, meanv, maxv, minv, sumv = _safe_stats(G.es["length"])
        print(f"length: n={n:,}  mean={meanv:.6f}  min={minv:.6f}  max={maxv:.6f}  sum={sumv:,.3f} ({unit})")
        if unit == "um" and sumv is not None:
            print(f"       sum={sumv/1e6:,.3f} m")
    else:
        print("length: NOT FOUND")

    if "diameter" in G.es.attributes():
        n, meanv, maxv, minv, _ = _safe_stats(G.es["diameter"])
        print(f"diameter: n={n:,}  mean={meanv:.6f}  min={minv:.6f}  max={maxv:.6f} ({unit})")
    else:
        print("diameter: NOT FOUND")

    if "length_steps" in G.es.attributes():
        n, meanv, maxv, minv, sumv = _safe_stats(G.es["length_steps"])
        print(f"length_steps: n={n:,}  mean={meanv:.6f}  min={minv:.6f}  max={maxv:.6f}  sum={sumv:,.0f}")
    else:
        print("length_steps: NOT FOUND")

    print("===========================================================\n")
    return {"V": V, "E": E, "unit": unit}


# ============================================================
# 3) BY NKIND (FORMATTED)
# ============================================================
def summarize_formatted_by_nkind(pkl_path):
    """
    Counts + length/diam stats by nkind for formatted graph.
    """
    G = load_formatted_graph(pkl_path)
    unit = G["unit"] if "unit" in G.attributes() else "unknown"

    if "nkind" not in G.es.attributes():
        raise KeyError("Edge attribute 'nkind' not found.")

    nkind = np.asarray(G.es["nkind"])
    L = np.asarray(G.es["length"]) if "length" in G.es.attributes() else None
    D = np.asarray(G.es["diameter"]) if "diameter" in G.es.attributes() else None

    print(f"\n==================== FORMATTED BY NKIND ====================")
    print(f"PKL: {pkl_path}")
    print(f"unit: {unit}")
    print(f"V={G.vcount():,}  E={G.ecount():,}")

    for k in sorted(set(map(int, np.unique(nkind)))):
        m = (nkind == k)
        cnt = int(np.sum(m))

        if L is not None:
            Lk = _finite(L[m])
            sumL = float(np.sum(Lk)) if Lk.size else float("nan")
            meanL = float(np.mean(Lk)) if Lk.size else float("nan")
            maxL = float(np.max(Lk)) if Lk.size else float("nan")
        else:
            sumL = meanL = maxL = float("nan")

        if D is not None:
            Dk = _finite(D[m])
            meanD = float(np.mean(Dk)) if Dk.size else float("nan")
            maxD  = float(np.max(Dk)) if Dk.size else float("nan")
        else:
            meanD = maxD = float("nan")

        if unit == "um" and np.isfinite(sumL):
            print(f"nkind={k}: n={cnt:,}  sumL={sumL:,.3f} µm ({sumL/1e6:,.3f} m)  "
                  f"meanL={meanL:.3f}  maxL={maxL:.3f}  meanD={meanD:.3f}  maxD={maxD:.3f}")
        else:
            print(f"nkind={k}: n={cnt:,}  sumL={sumL:,.3f} ({unit})  "
                  f"meanL={meanL:.3f}  maxL={maxL:.3f}  meanD={meanD:.3f}  maxD={maxD:.3f}")

    print("=============================================================\n")


# ============================================================
# 4) INTEGRITY CHECKS (FORMATTED)
# ============================================================
def check_formatted_integrity(pkl_path, sample=20000, seed=0, tol_len=1e-6):
    """
    Checks (sampled edges):
      - len(lengths2) == len(points)-1
      - length scalar ≈ sum(lengths2)
    """
    rng = np.random.default_rng(seed)
    G = load_formatted_graph(pkl_path)
    unit = G["unit"] if "unit" in G.attributes() else "unknown"
    E = G.ecount()

    required = {"points", "lengths2", "length"}
    missing = [a for a in required if a not in G.es.attributes()]
    if missing:
        raise KeyError(f"Missing edge attributes: {missing}")

    n = min(sample, E)
    idx = rng.integers(0, E, size=n, dtype=np.int64)

    bad_counts = 0
    bad_len = 0
    worst_diff = 0.0
    worst_ei = None

    for ei in idx:
        e = G.es[int(ei)]
        pts = e["points"]
        l2  = e["lengths2"]
        L   = e["length"]

        if len(l2) != max(0, len(pts) - 1):
            bad_counts += 1

        if L is None or not np.isfinite(float(L)):
            continue

        s = float(np.sum(_finite(l2))) if len(l2) else 0.0
        diff = abs(float(L) - s)
        if diff > tol_len:
            bad_len += 1
            if diff > worst_diff:
                worst_diff = diff
                worst_ei = int(ei)

    print("\n==================== FORMATTED INTEGRITY CHECK ====================")
    print(f"PKL: {pkl_path}")
    print(f"unit: {unit}")
    print(f"sampled edges: {n:,} / {E:,}")
    print(f"len(lengths2) != len(points)-1 : {bad_counts:,} / {n:,}")
    print(f"|length - sum(lengths2)| > {tol_len} : {bad_len:,} / {n:,}")
    if worst_ei is not None:
        print(f"worst diff={worst_diff:.6e} at edge ei={worst_ei}")
        print("Tip: run print_edge_debug_formatted(..., ei=worst_ei) to inspect.")
    print("===================================================================\n")


# ============================================================
# 5) TOP-K OUTLIERS (FORMATTED)
# ============================================================
def topk_edges_formatted(pkl_path, attr="length", k=20):
    G = load_formatted_graph(pkl_path)
    unit = G["unit"] if "unit" in G.attributes() else "unknown"

    if attr not in G.es.attributes():
        raise KeyError(f"Edge attribute '{attr}' not found.")

    vals = np.asarray(G.es[attr], dtype=np.float64)
    m = np.isfinite(vals)
    idx = np.where(m)[0]
    if idx.size == 0:
        print(f"No finite values for {attr}.")
        return

    top_idx = idx[np.argsort(vals[idx])[-k:]][::-1]

    print(f"\n==================== TOP-{min(k, top_idx.size)} EDGES BY {attr} ====================")
    print(f"PKL: {pkl_path}")
    print(f"unit: {unit}")
    for r, ei in enumerate(top_idx, 1):
        print(f"{r:02d}) ei={int(ei):<8d}  {attr}={float(vals[ei]):.6f}")
    print("================================================================\n")


# ============================================================
# EXAMPLE RUN (EDIT PATHS)
# ============================================================
pkl_formatted = "/home/ana/MicroBrain/output/formatted/graph_18_OutGeom_um_formatted.pkl"

summarize_formatted(pkl_formatted)
summarize_formatted_by_nkind(pkl_formatted)

# Check for inconsistencies (sampled)
check_formatted_integrity(pkl_formatted, sample=50000, seed=0, tol_len=1e-6)

# If things look insane: find largest edges
topk_edges_formatted(pkl_formatted, attr="length", k=20)
topk_edges_formatted(pkl_formatted, attr="diameter", k=20)

# Inspect one edge in detail
print_edge_debug_formatted(pkl_formatted, ei=12, max_show=30)
