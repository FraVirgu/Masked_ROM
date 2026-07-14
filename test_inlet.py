"""
Check the inlet assumption.

CCO_Domain picks the inlet as the degree-1 vertex (leaf) with the largest
radius:

    leaves = [i for i in vertices if degree[i] == 1]
    inlet  = max(leaves, key=lambda i: radii[i])

That is a CHOICE, not a verified fact. In a CCO tree the root vessel is the
widest in the whole tree, so this should hold:

  1. The inlet is the widest vertex in the ENTIRE graph, not just among leaves.
     If some interior vertex is wider, the tree structure is not what we assume.

  2. The inlet is decisively wider than the runner-up leaf. If two leaves are
     nearly tied, the choice is fragile and could flip on a different mesh.

  3. Radii DECREASE away from the inlet (Murray's law / CCO construction):
     a daughter vessel is never wider than its parent.

Run:
    python test_inlet.py -name Prova_14_07
"""
import argparse
import numpy as np
from dolfin import Mesh, XDMFFile, MeshFunction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-name", type=str, default="Prova_14_07")
    ap.add_argument("-tol",  type=float, default=1e-9,
                    help="tolerance when comparing radii along an edge")
    args = ap.parse_args()

    base = f"./nets/{args.name}/{args.name}_"

    mesh = Mesh()
    with XDMFFile(f"{base}marked_mesh.xdmf") as f:
        f.read(mesh)

    markers = MeshFunction("size_t", mesh, 0)
    with XDMFFile(f"{base}markers.xdmf") as f:
        f.read(markers)

    radii = MeshFunction("double", mesh, 0)
    with XDMFFile(f"{base}radii.xdmf") as f:
        f.read(radii)

    r  = np.array(radii.array())
    mk = np.array(markers.array())
    nv = mesh.num_vertices()

    # adjacency + degree
    adj = [[] for _ in range(nv)]
    for a, b in mesh.cells():
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    deg = np.array([len(a) for a in adj])

    leaves = np.flatnonzero(deg == 1)
    inlet  = np.flatnonzero(mk == 111)

    print("\n" + "=" * 70)
    print(f"INLET CHECK — {args.name}")
    print("=" * 70)
    print(f"  vertices {nv}   edges {mesh.num_cells()}   leaves {len(leaves)}")
    print(f"  radii    min={r.min():.6f}  max={r.max():.6f}")

    if len(inlet) != 1:
        print(f"\n  FAIL  expected exactly 1 inlet (marker 111), found {len(inlet)}")
        return 1
    v_in = int(inlet[0])

    print(f"\n  inlet vertex   : {v_in}")
    print(f"  inlet radius   : {r[v_in]:.6f}")
    print(f"  inlet degree   : {deg[v_in]}   (must be 1: the root is a leaf)")

    ok = True

    # --- 1. is the inlet the widest vertex in the WHOLE graph? ---
    print("\n" + "-" * 70)
    print("1. Is the inlet the widest vertex in the entire tree?")
    print("-" * 70)
    v_widest = int(np.argmax(r))
    print(f"   widest vertex overall : {v_widest}  (r = {r[v_widest]:.6f}, "
          f"degree {deg[v_widest]})")
    if v_widest == v_in:
        print("   PASS  the inlet IS the widest vertex in the tree.")
    elif abs(r[v_widest] - r[v_in]) < args.tol:
        print(f"   PASS  tied with vertex {v_widest} at the same radius.")
    else:
        print(f"   FAIL  vertex {v_widest} is WIDER than the inlet "
              f"({r[v_widest]:.6f} > {r[v_in]:.6f}).")
        print(f"         It has degree {deg[v_widest]}, so it is "
              f"{'a leaf' if deg[v_widest] == 1 else 'an INTERIOR vertex'}.")
        if deg[v_widest] > 1:
            print("         An interior vertex wider than the root contradicts the")
            print("         CCO tree structure: the root should be the widest vessel.")
        ok = False

    # --- 2. how decisive is the choice among leaves? ---
    print("\n" + "-" * 70)
    print("2. Is the inlet decisively wider than the other leaves?")
    print("-" * 70)
    leaf_r = sorted(((float(r[v]), int(v)) for v in leaves), reverse=True)
    print("   widest leaves:")
    for rad, v in leaf_r[:5]:
        tag = "  <-- INLET" if v == v_in else ""
        print(f"     vertex {v:>3}   r = {rad:.6f}{tag}")

    if len(leaf_r) > 1:
        top, second = leaf_r[0][0], leaf_r[1][0]
        margin = (top - second) / max(top, 1e-30)
        print(f"\n   margin over runner-up : {margin:.1%}")
        if leaf_r[0][1] != v_in:
            print("   FAIL  the marked inlet is NOT the widest leaf.")
            ok = False
        elif margin < 0.05:
            print("   WARN  the inlet is only marginally wider than the next leaf.")
            print("         The choice is fragile — check it is the intended root.")
        else:
            print("   PASS  the inlet is clearly the widest leaf.")

    # --- 3. do radii decrease away from the inlet? ---
    print("\n" + "-" * 70)
    print("3. Do radii decrease from the inlet outwards? (CCO / Murray)")
    print("-" * 70)
    depth, order, queue = {v_in: 0}, [v_in], [v_in]
    while queue:
        v = queue.pop(0)
        for w in adj[v]:
            if w not in depth:
                depth[w] = depth[v] + 1
                order.append(w)
                queue.append(w)

    bad, worst, n_out = 0, 0.0, 0
    for v in order:
        for w in adj[v]:
            if depth.get(w, -1) == depth[v] + 1:     # w is further from inlet
                n_out += 1
                rise = r[w] - r[v]
                if rise > args.tol:
                    bad += 1
                    worst = max(worst, rise)

    print(f"   outward edges  : {n_out}")
    if bad:
        print(f"   FAIL  the radius INCREASES on {bad}/{n_out} outward edges "
              f"(worst +{worst:.6f}).")
        print("         A daughter vessel should never be wider than its parent.")
        ok = False
    else:
        print("   PASS  the radius never increases moving away from the inlet.")

    print("\n" + "=" * 70)
    if ok:
        print("  The inlet is the widest vertex, is a leaf, and radii taper away")
        print("  from it — consistent with it being the CCO root vessel.")
    else:
        print("  The inlet assumption does NOT hold. See the failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
