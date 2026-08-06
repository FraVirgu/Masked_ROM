"""
Minimal Robin-residual test: unit box, two subdomains, one 2-node vessel.

The full pipeline has many moving parts -- a sphere boundary, exterior-dof
elimination, a branching vascular tree, 8 subdomains with 3 face-neighbours
each. This strips all of it away to the smallest configuration where eqs. (4)
and (5) still mean something:

    domain   : [0,1]^3, EVERY cell interior (no boundary, no elimination)
    vessel   : a single straight segment through the middle, 2 nodes
    split    : one cut plane at x = 0.5 -> two boxes, one shared face

With no exterior dofs, `A_i` has no unit-diagonal placeholder rows and the
locally-unsupported-dof problem cannot arise. With one cut plane there is
exactly one interface and each box has exactly one neighbour, so the E_ij
transfer is as simple as it can be.

What this tests
---------------
Eq. (4) requires b_i - A_i u* to be supported on the artificial interface.
That holds only if A_i is the one-sided restriction of the global operator --
and a vessel whose averaging circle spans the cut breaks it, because neither
box can compute that circular average alone.

Vessel placements (-vessel):
    offset  x=0.25 -- circle entirely inside box 0, clear of the cut
    near    x=0.46 -- circle straddles the cut cleanly (the real test)
    middle  x=0.50 -- segment exactly ON the cut (degenerate, see below)

Measured results
----------------
    offset, no cross term          : residual 100% interface-supported,
                                     eq. (4)-(5) recovers u* to 2.9e-15
    near,   no cross term          : residual 38% / 86% supported,
                                     eq. (4)-(5) is WORSE than raw
    near,   -restrict_C -cross     : residual 100% / 100% supported,
                                     eq. (4)-(5) beats raw by 1.46x
    offset, -restrict_C -cross     : cross term is identically 0, and the
                                     2.9e-15 is unchanged (regression guard)

So the note's scheme is exact when no vessel crosses an interface, fails when
one does, and the cross term below repairs the extraction. The residual is
then perfectly interface-supported, but the solve still does not reach machine
precision: with -restrict_C the local operator keeps only 49-89% of a
straddling vessel's arc weight, so A_i is one-sided AND under-coupled. Closing
that last gap needs the coupling in the operator, not only on the rhs.

Run:
    python test_robin_minimal.py -vessel offset
    python test_robin_minimal.py -vessel near
    python test_robin_minimal.py -vessel near -restrict_C
    python test_robin_minimal.py -vessel near -restrict_C -cross
    python test_robin_minimal.py -vessel offset -restrict_C -cross
"""

import argparse

import numpy as np
from scipy.sparse.linalg import spsolve, splu

from dolfin import (
    BoxMesh,
    Point,
    Mesh,
    MeshEditor,
    MeshFunction,
)

from Solver_partition_domain import SolverPartitionDomain
from Decompose_Domain_Analytic import matrix_to_csr
from Robin_residual import _robin_interface

INTERIOR_TAG = 222


# =============================================================================
# Geometry
# =============================================================================

def build_line_mesh(points):
    """A 1D mesh in 3D space through the given points, one cell per segment."""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, 3)
    editor.init_vertices(len(points))
    editor.init_cells(len(points) - 1)
    for i, p in enumerate(points):
        editor.add_vertex(i, np.asarray(p, dtype=float))
    for i in range(len(points) - 1):
        editor.add_cell(i, np.array([i, i + 1], dtype="uintp"))
    editor.close()
    return mesh


def vessel_points(where):
    """Two nodes spanning the box in y, at the x the caller asks for.

    The segment runs along y, so its averaging circle lies in the x-z plane and
    its extent in x is +/- the radius about the segment's x.

    'offset' x=0.25  -- circle entirely inside box 0, clear of the cut.
    'near'   x=0.46  -- circle straddles the cut (0.46 + 0.08 > 0.5) but no
                        quadrature point lands exactly on it. This is the clean
                        test of a straddling vessel.
    'middle' x=0.5   -- segment exactly ON the cut. Degenerate: quadrature
                        points sitting exactly on the plane are accepted or
                        rejected by a cell-collision tie-break that differs
                        between the two boxes, so the two halves disagree even
                        though the configuration is mirror-symmetric.
    """
    x = {"middle": 0.5, "near": 0.46, "offset": 0.25}[where]
    return [(x, 0.25, 0.5), (x, 0.75, 0.5)]


def make_subdomain(mesh_lo, mesh_hi, n_sub, meshQ, q_radii, p_known,
                   sigma3d, kappa, ijk, V_global, coords_global,
                   C_global=None):
    """One box: its own mesh, its local solve, and its dof map into V_global.

    If C_global is given, the box's coupling operator is the GLOBAL C with the
    columns it does not own removed -- the arc weights are then exactly the
    global ones, which is what makes the cross term below well-posed. The
    default local quadrature instead renormalizes each partial arc by its own
    surviving measure, so C_i^(v) u_i + C_j^(v) u_j would not reconstruct the
    true full-circle average.
    """
    meshV = BoxMesh(Point(*mesh_lo), Point(*mesh_hi), *n_sub)

    # No boundary: every cell is interior, so nothing is ever eliminated.
    markers = MeshFunction("size_t", meshV, 3, INTERIOR_TAG)

    # l2g needs the space, which needs the solver built; build once without
    # C_global to get the dof map, then rebuild with it if asked.
    def _build(cg=None, l2g=None):
        ps = SolverPartitionDomain(
            meshV=meshV,
            meshV_markers=markers,
            meshQ=meshQ,
            q_radii=q_radii,
            p_known=p_known,
            sigma3d=sigma3d,
            kappa=kappa,
            gamma=1.0,
            interior_tag=INTERIOR_TAG,
            f3d=0.0,
            C_global=cg,
            l2g=l2g,
        ).build()
        ps.ext_dofs = np.zeros(0, dtype=int)   # nothing outside the domain
        return ps

    ps = _build()
    V_sub = ps.V
    sub_coords = V_sub.tabulate_dof_coordinates().reshape((V_sub.dim(), -1))
    l2g = match_dofs(sub_coords, coords_global)

    if C_global is not None:
        ps = _build(cg=C_global, l2g=l2g)
    ps.solve()

    return {
        "ijk": ijk,
        "meshV": meshV,
        "V_sub": V_sub,
        "partition_solver": ps,
        "local_to_global_dof": l2g,
    }


def rel_err(sd, u):
    """Relative error of u against u* on this box."""
    ref = float(np.linalg.norm(sd["u_star_local"]))
    if ref <= 1e-14:
        return 0.0
    return float(np.linalg.norm(sd["u_star_local"] - u)) / ref


def match_dofs(sub_coords, coords_global, tol=1e-10):
    """Index of each local dof in the global dof list (exact grid match)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_global)
    dist, idx = tree.query(sub_coords)
    if dist.max() > tol:
        raise RuntimeError(
            f"local dof {int(np.argmax(dist))} has no global match "
            f"(nearest {dist.max():.2e}); the sub-mesh is not grid-aligned."
        )
    return idx.astype(int)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[1],
    )
    ap.add_argument("-n", type=int, default=12,
                    help="cells per axis on the GLOBAL box (must be even so "
                         "the cut at x=0.5 lands on a grid plane)")
    ap.add_argument("-vessel", choices=("middle", "near", "offset"),
                    default="near",
                    help="where the vessel sits relative to the cut at x=0.5: "
                         "'offset' clear of it, 'near' straddling it cleanly, "
                         "'middle' exactly on it (degenerate)")
    ap.add_argument("-radius", type=float, default=0.08,
                    help="vessel radius (the averaging circle's radius)")
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-kappa", type=float, default=1.0)
    ap.add_argument("-rho", type=float, default=None,
                    help="Robin penalty; default sigma3d/h")
    ap.add_argument("-restrict_C", action="store_true",
                    help="build each box's coupling operator by restricting "
                         "the global C instead of re-running clipped local "
                         "quadrature. Independent of -cross so the two effects "
                         "can be told apart.")
    ap.add_argument("-cross", action="store_true",
                    help="add the 1D-network cross term for vessels whose "
                         "averaging circle straddles the cut (the proposed "
                         "extension to eq. 5). Implies -restrict_C: the term "
                         "is only well-posed with global arc weights.")
    ap.add_argument("-max_iter", type=int, default=200,
                    help="maximum Schwarz sweeps")
    ap.add_argument("-tol", type=float, default=1e-12,
                    help="stop when the largest iterate change falls below this")
    ap.add_argument("-verbose", action="store_true",
                    help="print the Schwarz convergence history")
    args = ap.parse_args()

    if args.n % 2:
        raise SystemExit(f"-n must be even so x=0.5 is a grid plane, got {args.n}")

    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    half = args.n // 2

    print("=" * 68)
    print(f"minimal Robin-residual test: [0,1]^3, 2 boxes, vessel '{args.vessel}'")
    print("=" * 68)

    # ---- 1D vessel ---------------------------------------------------------
    pts = vessel_points(args.vessel)
    meshQ = build_line_mesh(pts)
    q_radii = np.full(meshQ.num_vertices(), args.radius, dtype=float)
    # A known linear pressure drop along the vessel; any nonzero datum works,
    # the scheme never sees where it came from.
    p_known = np.array([1.0, 0.0], dtype=float)
    print(f"  vessel: {pts[0]} -> {pts[1]}  radius={args.radius}")
    reach = pts[0][0] + args.radius
    if args.vessel == "offset":
        where = f"is clear of it (reaches x={reach:.3f})"
    elif args.vessel == "near":
        where = f"STRADDLES it (reaches x={reach:.3f} > 0.5)"
    else:
        where = "sits exactly ON it (degenerate)"
    print(f"  cut plane at x=0.5; vessel circle {where}")

    # ---- global problem ----------------------------------------------------
    meshV = BoxMesh(Point(*lo), Point(*hi), args.n, args.n, args.n)
    markers = MeshFunction("size_t", meshV, 3, INTERIOR_TAG)
    glob = SolverPartitionDomain(
        meshV=meshV, meshV_markers=markers, meshQ=meshQ,
        q_radii=q_radii, p_known=p_known,
        sigma3d=args.sigma3d, kappa=args.kappa, gamma=1.0,
        interior_tag=INTERIOR_TAG, f3d=0.0,
    ).build()
    glob.ext_dofs = np.zeros(0, dtype=int)
    glob.solve()

    V_global = glob.V
    n_global = V_global.dim()
    coords_global = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    u_star = glob.u3d.vector().get_local()
    print(f"  global: {n_global} dofs, ||u*||={np.linalg.norm(u_star):.4e}")

    # ---- two subdomains, split at x = 0.5 ----------------------------------
    # -cross needs the global arc weights preserved, so it implies -restrict_C.
    # Kept as separate flags because -restrict_C alone CHANGES THE RAW BASELINE
    # (it drops the far-side arc instead of renormalizing), and conflating the
    # two makes the cross term impossible to evaluate.
    restrict_C = args.restrict_C or args.cross
    C_for_boxes = glob.C if restrict_C else None

    subdomains = [
        make_subdomain((0.0, 0.0, 0.0), (0.5, 1.0, 1.0),
                       (half, args.n, args.n), meshQ, q_radii, p_known,
                       args.sigma3d, args.kappa, (0, 0, 0),
                       V_global, coords_global, C_global=C_for_boxes),
        make_subdomain((0.5, 0.0, 0.0), (1.0, 1.0, 1.0),
                       (half, args.n, args.n), meshQ, q_radii, p_known,
                       args.sigma3d, args.kappa, (1, 0, 0),
                       V_global, coords_global, C_global=C_for_boxes),
    ]

    # ---- operator correctness ----------------------------------------------
    # Everything downstream assumes A_i is the one-sided restriction of the
    # global operator on box i's INTERIOR rows -- rows whose entire global
    # stencil lies inside the box. If that fails, no transmission condition can
    # work and the rest of this script is measuring the wrong thing.
    #
    # Checked in two pieces, because they can fail independently:
    #   diffusion  AD00_i  vs  AD00_global   -- pure FEM assembly
    #   coupling   C_i^T G C_i  vs  C^T G C  -- the clipped-quadrature suspect
    print("")
    print("  operator check: A_i vs the global operator on interior rows")
    Ag = (glob.A).tocsr()
    ADg = matrix_to_csr(glob.AD00).tocsr()
    Mg = (glob.C.T @ glob.G @ glob.C).tocsr()
    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        owned = np.zeros(n_global, dtype=bool)
        owned[l2g] = True
        g2l = -np.ones(n_global, dtype=np.int64)
        g2l[l2g] = np.arange(l2g.size)

        ADi = matrix_to_csr(ps.AD00).tocsr()
        Mi = (ps.C.T @ ps.G @ ps.C).tocsr()

        # Interior rows: full global stencil inside the box.
        d_ad = d_m = 0.0
        n_int = 0
        for dl in range(ps.V.dim()):
            gr = l2g[dl]
            cols = Ag.indices[Ag.indptr[gr]:Ag.indptr[gr + 1]]
            if not owned[cols].all():
                continue
            n_int += 1
            for Gm, Lm, acc in ((ADg, ADi, "ad"), (Mg, Mi, "m")):
                s, e = Gm.indptr[gr], Gm.indptr[gr + 1]
                rg = np.zeros(ps.V.dim())
                rg[g2l[Gm.indices[s:e]]] = Gm.data[s:e]
                s2, e2 = Lm.indptr[dl], Lm.indptr[dl + 1]
                rl = np.zeros(ps.V.dim())
                rl[Lm.indices[s2:e2]] = Lm.data[s2:e2]
                dd = float(np.abs(rg - rl).max())
                if acc == "ad":
                    d_ad = max(d_ad, dd)
                else:
                    d_m = max(d_m, dd)
        print(f"    {sd['ijk']}: {n_int} interior rows | "
              f"max|dAD00|={d_ad:.3e}  max|d C^T G C|={d_m:.3e}")
        if d_ad > 1e-12:
            print("        -> DIFFUSION assembly is not one-sided (unexpected)")
        if d_m > 1e-12:
            print("        -> COUPLING block differs from the global operator;"
                  " eq. (4)'s Key point cannot hold")

    # Reference restriction of u* to each box, and the raw local error.
    for sd in subdomains:
        l2g = sd["local_to_global_dof"]
        sd["u_star_local"] = u_star[l2g]
        u_raw = sd["partition_solver"].u3d.vector().get_local()
        sd["rel_raw"] = rel_err(sd, u_raw)

    print("")
    print("  raw local error (no transmission condition):")
    for sd in subdomains:
        print(f"    {sd['ijk']}: {sd['rel_raw']:.6e}")

    # ---- eq. (4): extract, and measure where the residual lives ------------
    rho = args.rho
    if rho is None:
        rho = float(args.sigma3d) / max(float(meshV.hmax()), 1e-30)

    g_min, g_max = coords_global.min(axis=0), coords_global.max(axis=0)

    print("")
    print(f"  rho = {rho:.4e}")
    if args.cross:
        print("  eq. (4) residual WITH the 1D-network cross term "
              "-- where does it live?")
    else:
        print("  eq. (4) residual  b_i - A_i u* + rho G u*"
              "  -- where does it live?")

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        u_i = sd["u_star_local"]

        G_gamma, n_tagged = _robin_interface(ps, g_min, g_max)
        sd["_G_gamma"] = G_gamma

        # Interface = dofs on the cut plane. With one cut this is just x=0.5.
        sub_coords = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
        on_if = np.abs(sub_coords[:, 0] - 0.5) < 1e-10
        sd["_sel"] = on_if

        # ---- the missing 1D-network term --------------------------------
        # The global coupling block is C^T G C. For a vessel node v whose
        # averaging circle straddles the cut, the global row splits by which
        # box owns each column:
        #
        #     C_v = C_v^(i) + C_v^(j)
        #
        # Box i assembles only (C_v^(i))^T G_vv C_v^(i). The global equation
        # restricted to box i's rows also carries
        #
        #     (C_v^(i))^T G_vv (C_v^(j) u_j)
        #
        # which eq. (5) has no slot for. C_v^(j) u_j is ONE SCALAR per
        # straddling node -- the partial arc average on the far side -- so
        # what crosses the interface is a short vector, not a field.
        #
        # This is why the arc weights must be the global ones: with the
        # default clipped quadrature each box renormalizes its own partial
        # arc, and C_v^(i) u_i + C_v^(j) u_j would not reconstruct the true
        # full-circle average.
        cross = np.zeros(ps.V.dim(), dtype=float)
        if args.cross:
            Ci = ps.C.tocsr()

            # The global equation on box i's rows carries C_i^T G (C u*), with
            # the FULL C. Box i's own operator supplies C_i^T G (C_i u_i). The
            # missing piece is therefore
            #
            #     C_i^T G (C u* - C_i u_i)
            #
            # computed against the global C directly. An earlier version used
            # C_j u_j for the second factor, which DOUBLE-COUNTS every column
            # on the cut plane itself: those dofs are owned by both boxes, so
            # they appear in C_i and in C_j. Subtracting C_i u_i from the true
            # global average avoids the partition question entirely.
            #
            # G is a CG1 mass matrix on the 1D mesh, NOT diagonal -- it couples
            # neighbouring vessel nodes -- so this is a full matrix product,
            # not a per-node scalar.
            far_avg = glob.C.dot(u_star) - Ci.dot(u_i)
            cross = Ci.T.dot(glob.G.dot(far_avg))

            # Straddling nodes: this box holds part of the circle but not all
            # of the global row's weight.
            w_i = np.asarray(np.abs(Ci).sum(axis=1)).ravel()
            w_g = np.asarray(np.abs(glob.C.tocsr()).sum(axis=1)).ravel()
            sd["_n_straddle"] = int(np.count_nonzero(
                (w_i > 1e-14) & (w_g - w_i > 1e-14 * np.maximum(w_g, 1.0))))
            sd["_far_norm"] = float(np.linalg.norm(far_avg))

        sd["_cross"] = cross
        flux = np.asarray(ps.rhs, dtype=float) - ps.A.dot(u_i) - cross
        r_i = flux + rho * G_gamma.dot(u_i)

        n_all = float(np.linalg.norm(r_i))
        n_if = float(np.linalg.norm(r_i[on_if]))
        n_in = float(np.linalg.norm(r_i[~on_if]))
        frac = n_if / max(n_all, 1e-30)
        # The flux term alone -- this is the quantity the note's Key point is
        # a statement about, before the trace term is added.
        f_all = float(np.linalg.norm(flux))
        f_if = float(np.linalg.norm(flux[on_if]))

        sd["_r"] = r_i
        sd["_frac"] = frac
        extra = ""
        if args.cross:
            n_str = sd.get("_n_straddle", 0)
            extra = (f", {n_str} straddling nodes "
                     f"(||cross||={np.linalg.norm(cross):.3e})")
            # Regression guard: with nothing straddling, the cross term must
            # vanish identically and the scheme must reduce to plain eq. (4).
            if n_str == 0 and np.linalg.norm(cross) > 1e-12:
                extra += "  <-- BUG: no straddling nodes but cross != 0"
        print(f"    {sd['ijk']}: {int(on_if.sum())} interface dofs, "
              f"{n_tagged} tagged facets{extra}")
        print(f"        ||r||={n_all:.4e}  interface={n_if:.4e} "
              f"interior={n_in:.4e}  -> {frac:6.1%} on interface")
        print(f"        flux term alone: {f_if / max(f_all, 1e-30):6.1%} "
              f"on interface")

    # Only 'middle' is mirror-symmetric about the cut, so only there must the
    # two boxes agree. 'near' puts the vessel axis inside box 0, which is a
    # genuinely asymmetric configuration -- an earlier version flagged it too
    # and that warning was meaningless.
    if args.vessel == "middle":
        f0, f1 = subdomains[0]["_frac"], subdomains[1]["_frac"]
        if abs(f0 - f1) > 0.05 * max(f0, f1, 1e-30):
            print(f"    NOTE: the two boxes disagree ({f0:.1%} vs {f1:.1%}) on "
                  f"a mirror-symmetric configuration.\n"
                  f"          Quadrature points sitting exactly on the cut are "
                  f"accepted by one box\n          and rejected by the other. "
                  f"Use -vessel near for a clean straddle.")

    # ---- E_ij: two boxes, one shared face ----------------------------------
    for i, sd in enumerate(subdomains):
        other = subdomains[1 - i]
        l2g_i = sd["local_to_global_dof"]

        f_star = np.zeros(n_global, dtype=float)
        f_star[other["local_to_global_dof"]] = np.where(
            other["_sel"], other["_r"], 0.0)

        iface_i = np.zeros(n_global, dtype=bool)
        iface_i[l2g_i[sd["_sel"]]] = True
        sd["_f_in"] = np.where(iface_i, f_star, 0.0)[l2g_i]

    # ---- eq. (5): local Robin solve ----------------------------------------
    print("")
    print("  eq. (5) corrected local solve:")
    for sd in subdomains:
        ps = sd["partition_solver"]
        A_robin = (ps.A + rho * sd["_G_gamma"]).tocsc()
        # The cross term belongs on the rhs of eq. (5) too: it is part of the
        # equation box i's rows actually satisfy, not only of the extraction.
        rhs_i = (np.asarray(ps.rhs, dtype=float) + sd["_f_in"] + sd["_cross"])
        u_corr = spsolve(A_robin, rhs_i)

        rel = rel_err(sd, u_corr)
        sd["rel_corr"] = rel
        print(f"    {sd['ijk']}: raw {sd['rel_raw']:.6e} -> "
              f"corrected {rel:.6e}"
              f"   {'better' if rel < sd['rel_raw'] else 'WORSE'}")

    raw_avg = float(np.mean([sd["rel_raw"] for sd in subdomains]))
    cor_avg = float(np.mean([sd["rel_corr"] for sd in subdomains]))
    print("")
    print(f"  avg raw {raw_avg:.6e}  ->  avg corrected {cor_avg:.6e}")
    if cor_avg < raw_avg:
        print(f"  eq. (4)-(5) IMPROVES on raw by {raw_avg / max(cor_avg, 1e-30):.2f}x")
    else:
        print("  eq. (4)-(5) does NOT improve on raw")

    # ---- the trace-exchange alternative ------------------------------------
    # Schwarz never forms the eq. (4) residual. Each box solves
    #
    #     (A_i + rho G_i) u_i^(k+1) = b_i + rho G_i u_nbr^(k)
    #
    # so the only thing crossing the interface is the neighbour's current
    # TRACE. Nothing here requires A_i to be the one-sided restriction of the
    # global operator, which is the assumption a straddling vessel violates.
    # The clipped coupling operator is still wrong in the same way, but that
    # error now sits inside the fixed point rather than corrupting the data
    # being transferred.
    #
    # Consequence: this converges to its OWN fixed point, not to u*. Expect it
    # to land near the exact-Dirichlet floor, not at machine precision.
    print("")
    print("  Schwarz trace exchange (same boxes, same rho):")

    lu = [splu((sd["partition_solver"].A + rho * sd["_G_gamma"]).tocsc())
          for sd in subdomains]
    u_it = [np.zeros(sd["partition_solver"].V.dim()) for sd in subdomains]

    hist = []
    for k in range(1, args.max_iter + 1):
        # Global trace field from the current iterates (averaged on the shared
        # face, where both boxes own the dof).
        acc = np.zeros(n_global)
        cnt = np.zeros(n_global)
        for sd, u in zip(subdomains, u_it):
            acc[sd["local_to_global_dof"]] += u
            cnt[sd["local_to_global_dof"]] += 1.0
        u_glob = np.zeros(n_global)
        nz = cnt > 0
        u_glob[nz] = acc[nz] / cnt[nz]

        delta = 0.0
        nxt = []
        for i, sd in enumerate(subdomains):
            ps = sd["partition_solver"]
            u_nbr = u_glob[sd["local_to_global_dof"]]
            rhs = np.asarray(ps.rhs, dtype=float) + rho * (sd["_G_gamma"] @ u_nbr)
            u_new = lu[i].solve(rhs)
            delta = max(delta, float(np.linalg.norm(u_new - u_it[i])))
            nxt.append(u_new)
        u_it = nxt

        rels = []
        for sd, u in zip(subdomains, u_it):
            rels.append(rel_err(sd, u))
        hist.append((k, delta, float(np.mean(rels))))
        if delta < args.tol:
            break

    if args.verbose:
        print("    iter |   delta    | avg rel local")
        for k, d, r in hist:
            print(f"    {k:4d} | {d:.4e} | {r:.6e}")
    for sd, u in zip(subdomains, u_it):
        rel = rel_err(sd, u)
        sd["rel_schwarz"] = rel
        print(f"    {sd['ijk']}: raw {sd['rel_raw']:.6e} -> "
              f"schwarz {rel:.6e}"
              f"   {'better' if rel < sd['rel_raw'] else 'WORSE'}")
    sch_avg = float(np.mean([sd["rel_schwarz"] for sd in subdomains]))
    print(f"    converged in {hist[-1][0]} sweeps (delta={hist[-1][1]:.2e})")

    # ---- side by side ------------------------------------------------------
    print("")
    print("=" * 68)
    print(f"  vessel '{args.vessel}':  "
          f"{'circle straddles the cut' if args.vessel != 'offset' else 'circle clear of the cut'}")
    print("")
    print(f"    {'':22s} {'avg rel local error':>20s}")
    print(f"    {'raw (no coupling)':22s} {raw_avg:20.6e}")
    label = "eq. (4)-(5) + cross" if args.cross else "eq. (4)-(5) residual"
    print(f"    {label:22s} {cor_avg:20.6e}"
          f"   {'ok' if cor_avg < raw_avg else '<-- fails'}")
    print(f"    {'Schwarz trace':22s} {sch_avg:20.6e}"
          f"   {'ok' if sch_avg < raw_avg else '<-- fails'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
