"""
Bipartite Process-stage index: subdomain latents <-> vessel latents.

Before any network is written, this answers three questions about the
decomposition that decide whether a vessel-mediated exchange can carry the
cross-box coupling at all:

  1. Does every box contain at least one 1D vertex? A box with none has no
     route to the rest of the world except face adjacency -- exactly the
     regime where Robin_residual's "Known limitation" says the flux term
     alone is insufficient.

  2. Do vessels crossing a cut plane really appear in BOTH boxes? That
     shared vertex IS the exchange; without it the bipartite graph is
     disconnected across the cut and the whole idea collapses.

  3. Does the vessel bus connect box pairs that are NOT face-adjacent? This
     is the value proposition over plain Schwarz: a 2-hop path through the
     centerline between boxes that share no face.

The membership map is read off subdomain["active_q_vertices"], which
decomposeDomain already builds (a bounding-box test padded by the largest
vessel radius). Vertex ids are GLOBAL: every subdomain stores the same meshQ
object, so ids collide across boxes on purpose and no remapping is needed.

Usage
-----
    python3 test_bipartite_index.py -name git_sphere_small -n 40
"""

import os
import argparse
import numpy as np

from dolfin import Function

from Solver_full_domain import Solver3D1D
from Boundary import SphereBoundary, random_sphere_points
from Decompose_Domain_Analytic_sphere import (
    decomposeDomain,
    check_sphere_domain_consistency,
)


# ---------------------------------------------------------------------------
# the index itself
# ---------------------------------------------------------------------------

def build_bipartite_index(subdomains, meshQ):
    """Edge lists for the bipartite Process graph.

    Returns
    -------
    box2q : (src_box, dst_q)
        Parallel int arrays. src_box indexes into the subdomain list (and so
        into the z_box latent tensor); dst_q is a GLOBAL meshQ vertex id (and
        so indexes into z_q). One entry per (box, vessel vertex) membership.
    q_edges : (n_cells, 2) int array
        The centerline's own connectivity, straight from meshQ.cells(). This
        is what lets information travel ALONG a vessel during Process.
    box_of : dict
        ijk tuple -> index in the latent tensor.
    """
    box_of = {}
    src_box, dst_q = [], []

    for b, sd in enumerate(subdomains):
        box_of[sd["ijk"]] = b
        active = np.asarray(sd["active_q_vertices"], dtype=int)
        src_box.append(np.full(active.shape, b, dtype=int))
        dst_q.append(active)

    if src_box:
        src_box = np.concatenate(src_box)
        dst_q = np.concatenate(dst_q)
    else:
        src_box = np.zeros(0, dtype=int)
        dst_q = np.zeros(0, dtype=int)

    q_edges = np.asarray(meshQ.cells(), dtype=int)

    return (src_box, dst_q), q_edges, box_of


def faces_adjacent(ijk_a, ijk_b):
    """True when two boxes share a face (differ by 1 on exactly one axis)."""
    d = np.abs(np.asarray(ijk_a) - np.asarray(ijk_b))
    return d.sum() == 1


# ---------------------------------------------------------------------------
# solver rebuild (same recipe as test_robin_ladder.build_solver)
# ---------------------------------------------------------------------------

def build_solver(args):
    if not os.path.isdir(os.path.join("nets", args.name)):
        raise SystemExit(f"nets/{args.name} does not exist.")
    sol_dir = args.solution or os.path.join(
        "solution",
        f"Simple{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")
    sol_npy = os.path.join(sol_dir, "solution.npy")
    if not os.path.isfile(sol_npy):
        raise SystemExit(f"no solution at {sol_npy}; run the solve first.")

    boundary = SphereBoundary(
        radius=args.radius,
        inlet_points=random_sphere_points(
            40, x_sign=-1, min_x=0.2, min_dist_to_boundary=0.06,
            radius=args.radius),
        outlet_points=random_sphere_points(
            40, x_sign=+1, min_x=0.2, min_dist_to_boundary=0.06,
            radius=args.radius),
        border_eps=10e-1,
    )
    check_sphere_domain_consistency(
        boundary=boundary, n_min=-args.radius, n_max=args.radius)

    solver = Solver3D1D(
        path_to_1D_mesh=os.path.join("nets", args.name, args.name) + "_",
        boundary=boundary, n=args.n, sigma3d=args.sigma3d,
        sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet").build()

    n_3d, n_1d = solver.W[0].dim(), solver.W[1].dim()
    x_np = np.load(sol_npy)
    if x_np.size != n_3d + n_1d:
        raise SystemExit(
            f"solution has {x_np.size} entries, operators want {n_3d + n_1d}. "
            f"Check -n / -radius against the solve.")

    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector().set_local(x_np[:n_3d])
    solver.u3d.vector().apply("insert")
    solver.u1d.vector().set_local(x_np[n_3d:])
    solver.u1d.vector().apply("insert")
    return solver, boundary


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def report(subdomains, meshQ, box2q, q_edges, box_of):
    src_box, dst_q = box2q
    n_boxes = len(subdomains)
    n_q = meshQ.num_vertices()
    bar = "=" * 78

    print(f"\n{bar}\nBIPARTITE INDEX\n{bar}")
    print(f"boxes                {n_boxes}")
    print(f"1D vertices (global) {n_q}")
    print(f"1D edges (centerline){q_edges.shape[0]:>5d}")
    print(f"membership edges     {src_box.size}")

    # ---- 1. every box has vessels? ----------------------------------------
    print(f"\n{bar}\n1. VESSEL COVERAGE PER BOX\n{bar}")
    per_box = {b: np.asarray(sd["active_q_vertices"], dtype=int)
               for b, sd in enumerate(subdomains)}
    empty = []
    print(f"{'box':>12}  {'1D verts':>9}  {'3D dofs':>9}  {'frac of 1D':>11}")
    for b, sd in enumerate(subdomains):
        k = per_box[b].size
        n_dofs = len(sd["local_to_global_dof"])
        print(f"{str(sd['ijk']):>12}  {k:>9d}  {n_dofs:>9d}  "
              f"{k / max(n_q, 1):>10.1%}")
        if k == 0:
            empty.append(sd["ijk"])

    if empty:
        print(f"\n  !! {len(empty)} box(es) contain NO 1D vertex: {empty}")
        print("     These reach the rest of the world only through face")
        print("     adjacency. The vessel bus cannot help them.")
    else:
        print("\n  OK: every box contains at least one 1D vertex.")

    # ---- 2. do cut-crossing vessels appear in two boxes? -------------------
    print(f"\n{bar}\n2. SHARED 1D VERTICES ACROSS CUTS\n{bar}")
    n_boxes_of_q = np.zeros(n_q, dtype=int)
    for b in range(n_boxes):
        n_boxes_of_q[per_box[b]] += 1

    shared = int(np.sum(n_boxes_of_q >= 2))
    unseen = int(np.sum(n_boxes_of_q == 0))
    print(f"1D vertices in >= 2 boxes : {shared:>5d}  ({shared / max(n_q,1):.1%})")
    print(f"1D vertices in exactly 1  : {int(np.sum(n_boxes_of_q == 1)):>5d}")
    print(f"1D vertices in NO box     : {unseen:>5d}")
    if n_q:
        print(f"max boxes sharing a vertex: {int(n_boxes_of_q.max()):>5d}")

    # An edge of the centerline whose endpoints sit in different box sets is
    # a vessel that genuinely crosses a cut.
    crossing = 0
    for a, b_ in q_edges:
        sa = {b for b in range(n_boxes) if a in set(per_box[b].tolist())}
        sb = {b for b in range(n_boxes) if b_ in set(per_box[b].tolist())}
        if sa != sb:
            crossing += 1
    print(f"centerline edges spanning different box sets: {crossing}")

    if shared == 0:
        print("\n  !! No 1D vertex is shared between boxes. The bipartite")
        print("     graph is DISCONNECTED across cuts -- the vessel bus")
        print("     carries nothing. Do not build the Process stage on it.")
    else:
        print("\n  OK: vessels crossing cuts appear in multiple boxes.")

    # ---- 3. non-adjacent box pairs joined by the vessel bus ----------------
    print(f"\n{bar}\n3. BOX PAIRS JOINED BY THE VESSEL BUS\n{bar}")

    # (a) direct: two boxes sharing a 1D vertex  (1 hop through a vessel node)
    direct = {}
    for q in range(n_q):
        if n_boxes_of_q[q] < 2:
            continue
        owners = sorted(b for b in range(n_boxes) if q in set(per_box[b].tolist()))
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                direct.setdefault((owners[i], owners[j]), 0)
                direct[(owners[i], owners[j])] += 1

    # (b) along-vessel: boxes connected through a centerline edge, i.e. the
    #     coupling that only appears once Process passes messages along the
    #     vessel rather than merely pooling at a shared node.
    along = set()
    for a, b_ in q_edges:
        oa = [b for b in range(n_boxes) if a in set(per_box[b].tolist())]
        ob = [b for b in range(n_boxes) if b_ in set(per_box[b].tolist())]
        for x in oa:
            for y in ob:
                if x != y:
                    along.add((min(x, y), max(x, y)))

    ijk = [sd["ijk"] for sd in subdomains]
    non_adj_direct = [p for p in direct if not faces_adjacent(ijk[p[0]], ijk[p[1]])]
    non_adj_along = [p for p in along if not faces_adjacent(ijk[p[0]], ijk[p[1]])]

    print(f"box pairs sharing a 1D vertex      : {len(direct)}")
    print(f"   of which NOT face-adjacent      : {len(non_adj_direct)}")
    print(f"box pairs joined along a vessel edge: {len(along)}")
    print(f"   of which NOT face-adjacent      : {len(non_adj_along)}")

    if non_adj_direct:
        print("\n  non-face-adjacent pairs reachable through the vessel bus:")
        for p in sorted(non_adj_direct):
            d = np.abs(np.asarray(ijk[p[0]]) - np.asarray(ijk[p[1]])).sum()
            kind = "edge-diag" if d == 2 else "corner-diag"
            print(f"    {ijk[p[0]]} <-> {ijk[p[1]]}   {kind:>12}  "
                  f"{direct[p]:>4d} shared vertices")

    # ---- verdict ----------------------------------------------------------
    print(f"\n{bar}\nVERDICT\n{bar}")
    ok = True
    if empty:
        print(f"[FAIL] {len(empty)} box(es) have no vessel.")
        ok = False
    else:
        print("[ OK ] every box is on the vessel bus.")
    if shared == 0:
        print("[FAIL] no shared 1D vertices -- bus is disconnected at cuts.")
        ok = False
    else:
        print(f"[ OK ] {shared} shared 1D vertices stitch the cuts.")
    if not non_adj_direct and not non_adj_along:
        print("[WARN] the bus only ever joins face-adjacent boxes, so it adds")
        print("       no reach beyond what plain Schwarz already has.")
    else:
        print(f"[ OK ] the bus reaches {len(set(non_adj_direct) | set(non_adj_along))} "
              f"non-face-adjacent box pair(s).")
    print(bar)
    return ok


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[1])
    ap.add_argument("-name", type=str, required=True)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("-sigma1d", type=float, default=1.0)
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-kappa", type=float, default=1.0)
    ap.add_argument("-radius", type=float, default=5.0)
    ap.add_argument("-solution", type=str, default=None)
    args = ap.parse_args()

    solver, boundary = build_solver(args)
    subdomains = decomposeDomain(solver, boundary)
    meshQ = subdomains[0]["meshQ"]

    # Every subdomain must share the SAME meshQ object, otherwise global
    # vertex ids do not collide and the whole index is meaningless.
    for sd in subdomains:
        assert sd["meshQ"] is meshQ, (
            "subdomains do not share one meshQ; global 1D ids would not align")

    box2q, q_edges, box_of = build_bipartite_index(subdomains, meshQ)
    report(subdomains, meshQ, box2q, q_edges, box_of)


if __name__ == "__main__":
    main()
