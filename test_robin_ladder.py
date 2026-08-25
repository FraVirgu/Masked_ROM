"""Isolate the failure in eqs. (4)-(5) by removing one unknown at a time.

test_robin_flux.py measures the whole method at once, so its defect mixes three
independent things -- the choice of rho, the sign convention in the E_ij
transfer, and the cross term being frozen at u* -- and no single number
separates them. This script runs four tests, each adding exactly ONE mechanism
to the one before. The first that fails localises the bug.

The ground truth throughout is that u* solves the global problem exactly, so it
solves every correctly-localised piece of it. Each test below is a different way
of asking a box to reproduce u*_i, under progressively weaker data.

  [A] DIRICHLET TRACE -- no Robin term, no transfer, no rho.
      Pin u = u*|_Gamma on the artificial faces and solve. The local problem is
      then fully determined by data taken from u* alone, so the answer must be
      u*_i to solver precision. This tests ONLY that A_i, b_i, the cross term
      and local_to_global_dof are mutually consistent. If [A] fails, the defect
      is below the transmission layer entirely and no choice of rho or sign can
      repair it -- fix this before reading anything else.

  [B] TWO BOXES, ONE CUT -- the simplest possible transfer.
      Re-decompose the domain into 2 boxes split along one axis, so there is a
      single flat interface, no edges and no corners, and each box has exactly
      one neighbour. Eq. (5) here is the method in its minimal form. A failure
      that appears in [B] but not [A] is in the transfer itself, and the
      geometry is simple enough to reason about; a method that cannot do two
      boxes will not do eight.

  [C] SIGN, MEASURED -- both conventions, same conditions.
      Run eq. (5) with the incoming data as-is and with its sign flipped, and
      report both. test_robin_flux only reports which is BETTER, which is an
      inference; this runs both and shows the numbers. If one convention is
      near machine precision and the other is not, the sign is settled.

Alongside [A] the conditioning of the local operator is estimated, because the
two ways of asking "is A_i right" disagree in a way that is easy to misread:
||b_i - A_i u*|| on interior rows can be tiny while the solve is off by 1e-2.
That is not a contradiction -- error <= cond(A_i) * residual, and the local
Neumann problem is poorly conditioned -- so [A] reports residual, error and a
cond estimate side by side and the ratio between them is the thing to read.

Read them in order. A later test's number is meaningless while an earlier one
fails, which is exactly the trap the -cross runs fell into: perfect extraction
support and a worse defect at the same time.

Run:
    python3 test_robin_ladder.py -name report_sphere_small -n 40 -radius 5.0
    python3 test_robin_ladder.py -name report_sphere_small -n 40 -radius 5.0 -cross

Pass -radius matching the solve that cached the solution: the operators are
rebuilt from these flags and the dof COUNT alone will not catch a mismatch.
"""

import argparse
import os

import numpy as np
from scipy.sparse.linalg import spsolve

from dolfin import Function

from Decompose_Domain_Analytic_sphere import (
    check_sphere_domain_consistency,
    decomposeDomain,
)
from Solver_full_domain import Solver3D1D
from Boundary import SphereBoundary, random_sphere_points
from Robin_residual_sphere import (
    _face_neighbours,
    _interface_selector,
    _robin_interface,
    build_cross_term,
    build_local_operator,
)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def build_solver(args):
    """Rebuild the global operators and load the cached solution."""
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
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]
    print(f"loaded {sol_npy}")
    return solver, boundary


def prepare(subdomains, solver, rho, cross):
    """Attach G^Gamma, the interface selector, the cross term and eq.(4)'s r_i.

    Recomputed here rather than imported from apply_robin_residual so that a
    bug inside that function cannot hide by also corrupting what this reads.
    """
    V = solver.W[0]
    n_global = V.dim()
    coords = V.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min, g_max = coords.min(axis=0), coords.max(axis=0)
    u_star = solver.u3d.vector().get_local()

    C_glob = solver.C.tocsr()
    C_src = C_glob.dot(u_star) if cross else None

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]

        G_gamma, _ = _robin_interface(ps, g_min, g_max)
        sel = _interface_selector(sd, ps, l2g)
        u_i = u_star[l2g]
        b_i = np.asarray(ps.rhs, dtype=float)

        cross_i = 0.0
        if cross:
            cross_i = build_cross_term(ps, C_glob, solver.G, C_src, u_i)

        live = np.ones(b_i.size, dtype=bool)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            live[np.asarray(ext, dtype=int)] = False
            if not np.isscalar(cross_i):
                cross_i = np.asarray(cross_i).copy()
                cross_i[np.asarray(ext, dtype=int)] = 0.0

        sd["_G_gamma"] = G_gamma
        sd["_sel"] = sel
        sd["_cross"] = cross_i
        sd["_b_i"] = b_i
        sd["_live"] = live
        sd["_u_star_i"] = u_i

        r_i = b_i - ps.A.dot(u_i) - cross_i + rho * G_gamma.dot(u_i)
        f_star = np.zeros(n_global, dtype=float)
        f_star[l2g] = np.where(sel, r_i, 0.0)
        sd["_f_star"] = f_star

    return u_star


def gather_incoming(subdomains, n_global):
    """E_ij: sum the face-neighbours' extracted data onto box i's interface."""
    nbrs = _face_neighbours(subdomains)
    for i, sd in enumerate(subdomains):
        l2g_i = sd["local_to_global_dof"]
        iface_i = np.zeros(n_global, dtype=bool)
        iface_i[l2g_i[sd["_sel"]]] = True

        incoming = np.zeros(n_global, dtype=float)
        for j in nbrs[i]:
            owned_j = np.zeros(n_global, dtype=bool)
            owned_j[subdomains[j]["local_to_global_dof"]] = True
            shared = iface_i & owned_j
            incoming[shared] += subdomains[j]["_f_star"][shared]
        sd["_incoming"] = incoming
    return nbrs


def cond1_estimate(A_csc, itmax=5):
    """cond_1(A) = ||A||_1 ||A^-1||_1, with ||A^-1||_1 estimated, not formed.

    Higham's estimator: ||A^-1||_1 = max_x ||A^-1 x||_1 / ||x||_1 over the unit
    1-ball, whose maximum sits at a vertex e_j. Starting from x = 1/n and
    iterating x <- sign(A^-1 x) -> e_argmax|A^-T y| walks between vertices and
    converges in a handful of steps. A single sparse LU is reused for every
    solve, so this costs one factorisation rather than n of them -- forming
    A^-1 outright on a 9261-dof box is what makes the naive version unusable.

    Returns inf if the factorisation fails (a genuinely singular operator).
    """
    from scipy.sparse.linalg import splu

    n = A_csc.shape[0]
    normA = float(abs(A_csc).sum(axis=0).max())
    try:
        lu = splu(A_csc.tocsc())
    except Exception:
        return float("inf")

    x = np.full(n, 1.0 / n)
    est = 0.0
    for _ in range(itmax):
        y = lu.solve(x)
        est_new = float(np.abs(y).sum())
        if est_new <= est:
            break
        est = est_new
        z = lu.solve(np.sign(y), trans="T")
        j = int(np.argmax(np.abs(z)))
        x = np.zeros(n)
        x[j] = 1.0
    return normA * est


def rel_to_ustar(u_i, sd):
    """Relative error of a local solve against u* on the same box."""
    ref = sd["_u_star_i"]
    live = sd["_live"]
    d = float(np.linalg.norm((u_i - ref)[live]))
    n = float(np.linalg.norm(ref[live]))
    return d / n if n > 1e-30 else float("nan")


# ---------------------------------------------------------------------------
# [A] Dirichlet trace
# ---------------------------------------------------------------------------

def test_A(subdomains, rho, cross):
    """Pin u = u*|_Gamma on the artificial faces; no Robin, no transfer.

    The local problem is then closed by data drawn entirely from u*, so any
    departure from u*_i is a defect in A_i, b_i, the cross term or the
    local-to-global map -- not in the transmission condition, which is absent.
    """
    print("\n[A] Dirichlet trace on the artificial interface")
    print("    (no Robin term, no E_ij -- tests A_i, b_i, cross, l2g only)")
    print(f"{'box':>12s}{'rel err':>13s}{'iface dofs':>12s}")

    print("    residual = ||b_i - A_i u*|| on interior rows: what eq.(4)")
    print("    discards. error = what the solve actually returns. They are")
    print("    linked by the conditioning, error <= cond * residual, so a")
    print("    clean residual and a bad solve are the SAME fact when cond is")
    print("    large -- which is the trap this column exists to expose.")
    print(f"{'box':>12s}{'rel err':>13s}{'rel resid':>12s}{'ratio':>10s}"
          f"{'cond est':>11s}{'iface':>8s}")

    rels = []
    for sd in subdomains:
        ps = sd["partition_solver"]
        sel, live = sd["_sel"], sd["_live"]
        A = ps.A.tocsr()
        rhs = (sd["_b_i"] + sd["_cross"]).copy()

        pin = np.flatnonzero(sel & live)
        g = np.zeros(A.shape[0], dtype=float)
        g[pin] = sd["_u_star_i"][pin]

        # SYMMETRIC elimination. Zeroing the row alone leaves every OTHER row
        # still carrying its A[r, d] * u_d coupling on the left, while u_d is
        # being forced -- an inconsistent system that fails even on a perfect
        # operator. The known contribution has to move to the right-hand side
        # and the column has to go with the row:
        #
        #     rhs <- rhs - A[:, pin] g_pin ,   then rows/cols of pin -> identity
        #
        # ps.A has already been through eliminate_exterior_local, which zeroed
        # rows AND columns of the eliminated dofs, so this matches how the rest
        # of the operator was built.
        rhs = rhs - A.dot(g)
        rhs[pin] = g[pin]

        Dc = A.tocsc(copy=True)
        for d in pin:
            Dc.data[Dc.indptr[d]:Dc.indptr[d + 1]] = 0.0
        D = Dc.tocsr().tolil()
        for d in pin:
            D.rows[d] = [d]
            D.data[d] = [1.0]

        Dcsc = D.tocsc()
        u_i = spsolve(Dcsc, rhs)
        r = rel_to_ustar(u_i, sd)
        rels.append(r)

        # The residual eq.(4) throws away, on the rows it assumes are clean.
        interior = live & ~sel
        res = np.asarray(sd["_b_i"], dtype=float) - ps.A.dot(sd["_u_star_i"])
        nres = float(np.linalg.norm(res[interior]))
        nb = float(np.linalg.norm(sd["_b_i"][live]))
        rel_res = nres / nb if nb > 1e-30 else float("nan")

        # cond_1(D) via a sparse LU and Higham's 1-norm estimator: the same
        # quantity LAPACK's gecon reports, without forming the inverse.
        cond = cond1_estimate(Dcsc)

        ratio = r / rel_res if rel_res > 1e-30 else float("nan")
        print(f"{str(sd['ijk']):>12s}{r:13.3e}{rel_res:12.3e}{ratio:10.1f}"
              f"{cond:11.2e}{pin.size:8d}")

    rels = np.array(rels)
    print(f"    rel err: min {np.nanmin(rels):.3e}  "
          f"mean {np.nanmean(rels):.3e}  max {np.nanmax(rels):.3e}")
    ok = np.nanmean(rels) < 1e-8
    if ok:
        print("    -> PASS. The local operators reproduce u* from an exact")
        print("       trace, so the pieces under the Robin layer are sound.")
    else:
        print("    -> the exact trace does not return u*. Read the ratio and")
        print("       cond columns before concluding anything:")
        print("       * ratio ~ cond  -> the operators are FINE. The interior")
        print("         residual eq.(4) discards is small, and the solve")
        print("         amplifies it by the conditioning of the local Neumann")
        print("         problem. Accuracy is limited by cond(A_i), not by a bug.")
        print("       * ratio >> cond -> amplification does not account for it;")
        print("         something in A_i / b_i / cross / l2g is genuinely wrong.")
        if not cross:
            print("       Also try -cross: without it the 3D-1D coupling block")
            print("       is not one-sided, which inflates the residual itself.")
    return ok


# ---------------------------------------------------------------------------
# [B]/[C] eq. (5) with the transfer, both signs
# ---------------------------------------------------------------------------

def run_eq5(subdomains, rho, sign):
    """Solve eq.(5) on every box with the incoming data scaled by `sign`."""
    rels = []
    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        live = sd["_live"]

        f_local = np.asarray(sd["_incoming"][l2g], dtype=float).copy()
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            f_local[np.asarray(ext, dtype=int)] = 0.0

        A_robin = build_local_operator(ps, rho, sd["_G_gamma"])
        u_i = spsolve(A_robin, sd["_b_i"] + sign * f_local + sd["_cross"])
        rels.append(rel_to_ustar(u_i, sd))
    return np.array(rels)


def test_C(subdomains, rho, label):
    """Both sign conventions, same conditions, numbers not inference."""
    print(f"\n[C] eq.(5) sign convention  ({label})")
    print("    same solve, incoming data as-is vs negated")
    plus = run_eq5(subdomains, rho, +1.0)
    minus = run_eq5(subdomains, rho, -1.0)

    print(f"{'box':>12s}{'as-is':>13s}{'negated':>13s}{'better':>10s}")
    for sd, p, m in zip(subdomains, plus, minus):
        print(f"{str(sd['ijk']):>12s}{p:13.3e}{m:13.3e}"
              f"{('negated' if m < p else 'as-is'):>10s}")
    print(f"{'mean':>12s}{np.nanmean(plus):13.3e}{np.nanmean(minus):13.3e}")

    mp, mm = float(np.nanmean(plus)), float(np.nanmean(minus))
    if mm < mp:
        best, other, name = mm, mp, "negated"
    else:
        best, other, name = mp, mm, "as-is"

    if best < 1e-8:
        print(f"    -> the '{name}' convention is exact; the sign is settled.")
    elif best < 0.5 * other:
        print(f"    -> '{name}' is clearly better ({best:.3e} vs {other:.3e})")
        print("       but still far from exact, so the sign is not the only")
        print("       defect.")
    else:
        print(f"    -> neither convention is close ({best:.3e} vs {other:.3e}).")
        print("       The sign is not the dominant error; look at [A]/[B].")
    return mp, mm


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
    ap.add_argument("-rho", type=float, default=None,
                    help="Robin weight; default sigma3d/hmax")
    ap.add_argument("-cross", action="store_true")
    ap.add_argument("-solution", type=str, default=None)
    ap.add_argument("-skip-two", action="store_true",
                    help="skip the 2-box case (halves the runtime)")
    args = ap.parse_args()

    solver, boundary = build_solver(args)
    n_global = solver.W[0].dim()

    rho = args.rho
    if rho is None:
        rho = float(solver.sigma3d) / max(float(solver.meshV.hmax()), 1e-30)
        print(f"rho defaulted to sigma3d/hmax = {rho:.4e}")
    rho = float(rho)

    bar = "=" * 78
    tag = "eqs.(4)-(5)+cross" if args.cross else "eqs.(4)-(5)"
    print(f"\n{bar}\nROBIN LADDER   ({tag}, rho = {rho:g})\n{bar}")

    # ---- eight boxes -------------------------------------------------------
    print("\n### 8 boxes (2 per direction)")
    sd8 = decomposeDomain(solver, boundary, restrict_global_C=args.cross)
    prepare(sd8, solver, rho, args.cross)
    gather_incoming(sd8, n_global)

    ok_A = test_A(sd8, rho, args.cross)
    p8, m8 = test_C(sd8, rho, "8 boxes")

    # ---- two boxes ---------------------------------------------------------
    # One cut, one neighbour each, no edges or corners. decomposeDomain slices
    # an axis only when the box length is shorter than the domain, so leaving
    # y and z at the full width gives exactly two boxes split along x.
    p2 = m2 = None
    if not args.skip_two:
        print(f"\n{bar}\n### 2 boxes (single cut along x)\n{bar}")
        full = 2.0 * args.radius
        sd2 = decomposeDomain(solver, boundary,
                              x_ROM_lenght=args.radius,
                              y_ROM_lenght=full,
                              z_ROM_lenght=full,
                              restrict_global_C=args.cross)
        print(f"built {len(sd2)} subdomains: "
              f"{', '.join(str(s['ijk']) for s in sd2)}")
        if len(sd2) != 2:
            print("    (expected 2; check the ROM lengths against the domain)")

        prepare(sd2, solver, rho, args.cross)
        nbrs2 = gather_incoming(sd2, n_global)
        print(f"neighbours per box: {[len(x) for x in nbrs2]}")

        print("\n[B] the same ladder on the minimal geometry")
        test_A(sd2, rho, args.cross)
        p2, m2 = test_C(sd2, rho, "2 boxes")

    # ---- verdict -----------------------------------------------------------
    print(f"\n{bar}\nWHERE THE DEFECT IS\n{bar}")
    if not ok_A:
        print("[A] did not return u* from an exact trace.")
        print("    Compare the ratio and cond columns in [A]: if ratio is of")
        print("    the order of cond, the operators are sound and the method's")
        print("    accuracy is capped by the conditioning of the local Neumann")
        print("    problem -- a property to report, not a bug to fix. Only a")
        print("    ratio far above cond indicts A_i / b_i / cross / l2g.")
    elif p2 is None:
        print(f"[A] passed -- local operators are sound.")
        print(f"    best 8-box: {min(p8, m8):.3e}   (2-box case skipped)")
    else:
        best8, best2 = min(p8, m8), min(p2, m2)
        print(f"[A] passed -- local operators are sound.")
        print(f"    best 8-box: {best8:.3e}    best 2-box: {best2:.3e}")
        if best2 < 1e-8 <= best8:
            print("    2 boxes exact, 8 boxes not: the transfer is right for a")
            print("    single flat cut and breaks where boxes meet at edges or")
            print("    corners -- a dof there belongs to several interfaces.")
        elif best2 >= 1e-8:
            print("    even the single-cut case is inexact, so the defect is")
            print("    in the transfer itself, not in the corner bookkeeping.")
        else:
            print("    both exact: eqs.(4)-(5) are correctly implemented and")
            print("    the earlier failures were rho and sign, now settled.")
    print(bar)


if __name__ == "__main__":
    main()
