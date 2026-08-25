"""Is the Robin flux transfer of eq. (5) consistent with the extraction of eq. (4)?

The method is exact by construction, and that is what makes it testable. Eq. (4)
extracts, from the KNOWN global solution,

    f_i^{*,-} = P_i^Gamma ( b_i - A_i u*_i + rho G_i^Gamma u*_i )        (4)

and eq. (5) solves each box against the data its neighbours extracted,

    ( A_i + rho G_i^Gamma ) u_i = b_i + sum_{j in N(i)} E_ij f_j^{*,-}   (5)

Substituting u_i = u*_i into the left of (5) must therefore reproduce the right
of (5) identically -- u* solves the global problem, so it solves every localised
piece of it. That is a closed identity: it fixes the sign, the scaling and the
dof matching of E_ij all at once, WITHOUT this test having to know what the
intended convention is. If the identity holds, the transfer is right and a bad
result comes from somewhere else. If it fails, the defect vector says how.

    residual_i = (A_i + rho G^Gamma) u*_i - b_i - sum_j E_ij f_j^{*,-}

Read the output in this order; each check presupposes the ones above it.

  [1] extraction support   -- is b_i - A_i u* + rho G u* really confined to the
                              artificial interface? Eq. (4) throws away whatever
                              is not, so if this is far from 100% the rest of the
                              method is built on sand and every later number is
                              meaningless. (The module docstring's "Known
                              limitation" is exactly this.)

  [2] the identity          -- the check above. Reported per box, relative to
                              ||b_i||. Passing means eq. (5) is a correct
                              localisation of the global problem.

  [3] sign of the transfer  -- only if [2] fails. For each face-neighbour pair,
                              compare what i extracts on the shared face against
                              what j extracts there. A Robin flux leaves i and
                              enters j through antiparallel normals, so a
                              correlation near -1 means the two sides disagree
                              in sign as they should, and near +1 means a sign
                              has been dropped. Also reported: whether flipping
                              the sign of the incoming data would REDUCE the
                              defect from [2], which is the direct test.

  [4] spatial pattern       -- the per-box defect against the box index, to see
                              whether failures track position (a systematic
                              orientation bug) or scatter (something else).

Run:
    python3 test_robin_flux.py -name report_sphere_small -n 40 -radius 5.0
    python3 test_robin_flux.py -name report_sphere_small -n 40 -radius 5.0 -cross

Pass -radius matching the solve that produced the cached solution: this script
rebuilds the operators from the flags, and a mismatch silently compares against
a different geometry (the dof COUNT is the same for equal -n).
"""

import argparse
import os

import numpy as np

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
    count_straddling_nodes,
)


def build(args):
    """Rebuild operators and load the cached global solution."""
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
    print(f"loaded {sol_npy}")

    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]
    return solver, boundary


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
    ap.add_argument("-rho", type=float, default=1.0)
    ap.add_argument("-cross", action="store_true")
    ap.add_argument("-solution", type=str, default=None)
    args = ap.parse_args()

    solver, boundary = build(args)
    subdomains = decomposeDomain(solver, boundary,
                                 restrict_global_C=args.cross)

    V = solver.W[0]
    n_global = V.dim()
    coords = V.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min, g_max = coords.min(axis=0), coords.max(axis=0)
    u_star = solver.u3d.vector().get_local()
    rho = float(args.rho)

    C_glob = solver.C.tocsr()
    C_src = C_glob.dot(u_star) if args.cross else None

    bar = "=" * 78
    print(f"\n{bar}\nROBIN FLUX CONSISTENCY"
          f"   ({'eqs.(4)-(5)+cross' if args.cross else 'eqs.(4)-(5)'},"
          f" rho = {rho:g})\n{bar}")

    # ---- eq. (4), reproduced here rather than imported ---------------------
    # apply_robin_residual stores these on the subdomain dicts, but recomputing
    # them keeps the test independent of that function: a bug inside it cannot
    # hide by also corrupting the values this test reads.
    print("\n[1] extraction support -- fraction of the eq.(4) residual that")
    print("    lives on the artificial interface (eq. 4 discards the rest)")
    print(f"{'box':>12s}{'on iface':>11s}{'off iface':>12s}{'support':>10s}"
          f"{'straddling':>12s}")

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]

        G_gamma, _ = _robin_interface(ps, g_min, g_max)
        sel = _interface_selector(sd, ps, l2g)
        u_i = u_star[l2g]
        b_i = np.asarray(ps.rhs, dtype=float)

        cross_i = 0.0
        n_str = 0
        if args.cross:
            cross_i = build_cross_term(ps, C_glob, solver.G, C_src, u_i)
            n_str = count_straddling_nodes(ps, C_glob)

        r_i = b_i - ps.A.dot(u_i) - cross_i + rho * G_gamma.dot(u_i)

        live = np.ones(r_i.size, dtype=bool)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            live[np.asarray(ext, dtype=int)] = False

        on = float(np.linalg.norm(r_i[sel & live]))
        off = float(np.linalg.norm(r_i[~sel & live]))
        tot = float(np.linalg.norm(r_i[live]))
        frac = on / tot if tot > 1e-30 else float("nan")

        sd["_G_gamma"] = G_gamma
        sd["_sel"] = sel
        sd["_cross"] = cross_i
        sd["_r_i"] = r_i
        sd["_b_i"] = b_i
        sd["_live"] = live

        f_star = np.zeros(n_global, dtype=float)
        f_star[l2g] = np.where(sel, r_i, 0.0)
        sd["_f_star"] = f_star

        print(f"{str(sd['ijk']):>12s}{on:11.3e}{off:12.3e}{frac:9.1%}"
              f"{n_str:12d}")

    supports = []
    for sd in subdomains:
        r, sel, live = sd["_r_i"], sd["_sel"], sd["_live"]
        t = float(np.linalg.norm(r[live]))
        supports.append(float(np.linalg.norm(r[sel & live])) / t
                        if t > 1e-30 else float("nan"))
    supports = np.array(supports)
    print(f"    support: min {np.nanmin(supports):.1%}  "
          f"mean {np.nanmean(supports):.1%}  max {np.nanmax(supports):.1%}")
    if np.nanmean(supports) < 0.99:
        print("    -> NOT confined to the interface. Eq.(4) discards the")
        print("       off-interface part, so eq.(5) is solving a different")
        print("       problem than the one u* satisfies. Fix this before")
        print("       reading [2]: the defect below is then expected, and")
        print("       says nothing about the sign of the transfer.")

    # ---- E_ij, exactly as apply_robin_residual assembles it -----------------
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

    # ---- [2] the identity ---------------------------------------------------
    print(f"\n[2] consistency of eq.(5) at u_i = u*_i")
    print("    (A_i + rho G^Gamma) u*_i  -  b_i  -  sum_j E_ij f_j^*")
    print("    must vanish; reported relative to ||b_i||")
    print(f"{'box':>12s}{'defect':>13s}{'rel':>12s}{'||b_i||':>12s}"
          f"{'nbrs':>6s}{'flip better':>13s}")

    rels, flips = [], []
    for i, sd in enumerate(subdomains):
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        u_i = u_star[l2g]

        f_local = np.asarray(sd["_incoming"][l2g], dtype=float)
        cross_i = sd["_cross"]
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            e = np.asarray(ext, dtype=int)
            f_local = f_local.copy()
            f_local[e] = 0.0
            if not np.isscalar(cross_i):
                cross_i = np.asarray(cross_i).copy()
                cross_i[e] = 0.0

        A_robin = build_local_operator(ps, rho, sd["_G_gamma"])
        lhs = A_robin.dot(u_i)
        rhs = sd["_b_i"] + f_local + cross_i

        live = sd["_live"]
        d = float(np.linalg.norm((lhs - rhs)[live]))
        # Would the opposite sign on the incoming data do better? This is the
        # direct form of the sign question -- no correlation heuristic needed.
        d_flip = float(np.linalg.norm((lhs - (sd["_b_i"] - f_local + cross_i))[live]))
        nb = float(np.linalg.norm(sd["_b_i"][live]))
        rel = d / nb if nb > 1e-30 else float("nan")
        rels.append(rel)
        flips.append(d_flip < d)

        print(f"{str(sd['ijk']):>12s}{d:13.3e}{rel:12.3e}{nb:12.3e}"
              f"{len(nbrs[i]):6d}{('YES' if d_flip < d else 'no'):>13s}")

    rels = np.array(rels)
    print(f"    relative defect: min {np.nanmin(rels):.3e}  "
          f"mean {np.nanmean(rels):.3e}  max {np.nanmax(rels):.3e}")
    n_flip = int(np.sum(flips))
    ok = np.nanmean(rels) < 1e-8
    if ok:
        print("    -> IDENTITY HOLDS. Eq.(5) is a correct localisation; the")
        print("       transfer sign and dof matching are right. A poor")
        print("       corrected solution is NOT coming from this step.")
    else:
        print("    -> IDENTITY FAILS. Eq.(5) at u* does not reproduce its own")
        print("       right-hand side, so the localisation is not exact.")
        if n_flip:
            print(f"       Flipping the incoming sign helps on {n_flip}/"
                  f"{len(subdomains)} boxes -- see [3].")
        if np.nanmean(supports) < 0.99:
            print("       NOTE [1] already failed, which is sufficient to")
            print("       explain this. Do not read a sign error into it")
            print("       until the extraction is confined to the interface.")

    # ---- [3] sign of the transfer ------------------------------------------
    print(f"\n[3] pairwise sign on shared faces")
    print("    corr(f_i, f_j) on the dofs both boxes call interface.")
    print("    Antiparallel normals => a correct Robin flux correlates < 0;")
    print("    corr ~ +1 means both sides carry the same sign (one dropped).")
    print(f"{'pair':>22s}{'dofs':>7s}{'corr':>9s}{'|f_i|':>12s}{'|f_j|':>12s}"
          f"{'ratio':>9s}")

    seen, corrs = set(), []
    for i, sd_i in enumerate(subdomains):
        for j in nbrs[i]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            sd_j = subdomains[j]

            li = sd_i["local_to_global_dof"][sd_i["_sel"]]
            lj = sd_j["local_to_global_dof"][sd_j["_sel"]]
            shared = np.intersect1d(li, lj)
            if shared.size < 3:
                continue

            fi = sd_i["_f_star"][shared]
            fj = sd_j["_f_star"][shared]
            ni, nj = np.linalg.norm(fi), np.linalg.norm(fj)
            if ni < 1e-30 or nj < 1e-30:
                continue
            c = float(np.dot(fi, fj) / (ni * nj))
            corrs.append(c)
            print(f"{str(sd_i['ijk']) + '-' + str(sd_j['ijk']):>22s}"
                  f"{shared.size:7d}{c:9.3f}{ni:12.3e}{nj:12.3e}"
                  f"{nj / ni:9.3f}")

    if corrs:
        corrs = np.array(corrs)
        print(f"    corr: min {corrs.min():+.3f}  mean {corrs.mean():+.3f}  "
              f"max {corrs.max():+.3f}")
        if corrs.mean() > 0.5:
            print("    -> both sides agree in sign. For a flux through")
            print("       antiparallel normals that is the signature of a")
            print("       dropped sign in the E_ij transfer.")
        elif corrs.mean() < -0.5:
            print("    -> sides are opposed, as a Robin flux should be.")
        else:
            print("    -> no clean sign relation; the shared-face data is not")
            print("       a simple reflection either way.")

    # ---- [4] spatial pattern ------------------------------------------------
    print(f"\n[4] defect against box position")
    print("    a defect that tracks the index is a systematic orientation")
    print("    bug; scattered values point elsewhere")
    for axis, nm in enumerate("ijk"):
        lo = [rels[i] for i, sd in enumerate(subdomains) if sd["ijk"][axis] == 0]
        hi = [rels[i] for i, sd in enumerate(subdomains) if sd["ijk"][axis] == 1]
        if lo and hi:
            print(f"    {nm}=0: mean {np.nanmean(lo):.3e}   "
                  f"{nm}=1: mean {np.nanmean(hi):.3e}   "
                  f"ratio {np.nanmean(hi) / np.nanmean(lo):.2f}"
                  if np.nanmean(lo) > 1e-30 else f"    {nm}: degenerate")
    print(bar)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
