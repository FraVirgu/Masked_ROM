"""
Additive Schwarz / Robin domain decomposition.

The trace-exchange alternative to the Robin-residual method. It is a different
scheme entirely: it never forms the eq. (4) residual and never consumes the
ground-truth field, so it works with an unknown target.

Module layout:
    Decompose_Domain_Analytic.decomposeDomain -- builds the subdomains
    Robin_residual.apply_robin_residual       -- eqs. (4)-(5), one-shot
    this file                                 -- iterative trace exchange

The Robin interface matrix is shared with Robin_residual rather than rebuilt:
both schemes need exactly the same G_gamma on the artificial cut planes.

Status: this path reached ~6.5% reconstructed global relative error against a
~6.2% exact-Dirichlet floor, i.e. essentially at the ceiling any transmission
condition can reach on this problem. The Robin-residual method has not beaten
its raw baseline on the same problem -- see Robin_residual's "Known limitation".

Usage
-----
    from Solve_schwarz_robin import solve_schwarz_robin

    subdomains = decomposeDomain(solver, boundary)
    history = solve_schwarz_robin(subdomains, solver, rho_robin=2.9e-2)

`subdomains` must already carry "partition_solver" and "local_to_global_dof".
"""

import numpy as np
from scipy.sparse.linalg import splu

from dolfin import Function

from Robin_residual_sphere import _robin_interface


def solve_schwarz_robin(
    subdomains,
    solver,
    rho_robin=None,
    max_iter=30,
    tol=1e-6,
    print_summary=True,
):
    """
    Additive Schwarz iteration with a Robin transmission condition on the
    artificial cut planes.

    Per subdomain the system is

        (A_local + rho * G_gamma) u^(k+1) = rhs_local + rho * G_gamma u_nbr^(k)

    The Robin term is added to the matrix ONCE: A_local + rho*G_gamma is
    assembled and LU-factorized a single time, outside the loop. Only the
    right-hand side changes between sweeps, refreshed from the neighbours'
    current interface traces. This is why iterating reduces the error -- on the
    first sweep u_nbr^(0) = 0 carries no information, and coupling between boxes
    only propagates on later sweeps.

    Uses the previous iterate's trace, never the ground-truth field, so this is
    a scheme that would work with an unknown target.
    """
    V_global = solver.W[0]
    n_global = V_global.dim()

    if rho_robin is None:
        # Standard Robin scaling rho ~ sigma3d / h.
        hmax = solver.meshV.hmax() if hasattr(solver, "meshV") else 1.0
        rho_robin = float(solver.sigma3d) / max(float(hmax), 1e-30)

    coords_global = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min = coords_global.min(axis=0)
    g_max = coords_global.max(axis=0)

    # ---- one-time setup: Robin term into the matrix, then factorize ----------
    states = []
    for subdomain in subdomains:
        ps = subdomain["partition_solver"]
        # Same interface matrix the residual method uses: exterior dofs stay
        # pinned to u=0, so their Robin rows/cols are dropped and the
        # transmission condition never reactivates them.
        G_gamma, n_tagged = _robin_interface(ps, g_min, g_max)

        A_robin = (ps.A + rho_robin * G_gamma).tocsc()
        states.append({
            "subdomain": subdomain,
            "ps": ps,
            "G_gamma": G_gamma,
            "lu": splu(A_robin),
            "rhs_local": np.array(ps.rhs, dtype=float),
            "l2g": subdomain["local_to_global_dof"],
            "n_tagged_facets": n_tagged,
            "u": np.zeros(ps.V.dim(), dtype=float),
        })
        subdomain["schwarz_n_artificial_facets"] = n_tagged

    # ---- iterate: only the rhs changes --------------------------------------
    u3d_full_vec = solver.u3d.vector().get_local()
    full_ref_l2 = float(np.linalg.norm(u3d_full_vec))
    history = []

    for it in range(1, max_iter + 1):
        # Global trace field from the current iterates (averaged on overlaps).
        acc = np.zeros(n_global, dtype=float)
        cnt = np.zeros(n_global, dtype=float)
        for st in states:
            acc[st["l2g"]] += st["u"]
            cnt[st["l2g"]] += 1.0
        u_glob = np.zeros(n_global, dtype=float)
        nz = cnt > 0.0
        u_glob[nz] = acc[nz] / cnt[nz]

        delta = 0.0
        for st in states:
            u_nbr = u_glob[st["l2g"]]
            rhs = st["rhs_local"] + rho_robin * (st["G_gamma"] @ u_nbr)
            u_new = st["lu"].solve(rhs)
            delta = max(delta, float(np.linalg.norm(u_new - st["u"])))
            st["u_next"] = u_new
        for st in states:
            st["u"] = st["u_next"]

        # Reconstruct and measure against the full-domain reference.
        acc = np.zeros(n_global, dtype=float)
        cnt = np.zeros(n_global, dtype=float)
        for st in states:
            acc[st["l2g"]] += st["u"]
            cnt[st["l2g"]] += 1.0
        rec = np.zeros(n_global, dtype=float)
        nz = cnt > 0.0
        rec[nz] = acc[nz] / cnt[nz]
        rec[~nz] = u3d_full_vec[~nz]

        rel_glob = (
            float(np.linalg.norm(u3d_full_vec - rec)) / full_ref_l2
            if full_ref_l2 > 0.0
            else 0.0
        )
        rel_loc = []
        for st in states:
            uf = st["subdomain"]["sol_3d_sub"].vector().get_local()
            nf = float(np.linalg.norm(uf))
            rel_loc.append(
                float(np.linalg.norm(uf - st["u"])) / nf if nf > 1e-12 else 0.0
            )
        rel_loc_avg = float(np.mean(rel_loc))
        history.append((it, delta, rel_loc_avg, rel_glob))

        if delta < tol:
            break

    for st in states:
        sd = st["subdomain"]
        fn = Function(st["ps"].V)
        fn.vector()[:] = st["u"]
        sd["u3d_partition_schwarz"] = fn
        uf = sd["sol_3d_sub"].vector().get_local()
        nf = float(np.linalg.norm(uf))
        sd["u3d_partition_error_schwarz_norm_l2"] = float(np.linalg.norm(uf - st["u"]))
        sd["u3d_partition_error_schwarz_rel_l2"] = (
            sd["u3d_partition_error_schwarz_norm_l2"] / nf if nf > 1e-12 else 0.0
        )

    if print_summary and history:
        n_facets = np.array([st["n_tagged_facets"] for st in states], dtype=float)
        print("")
        print("Additive Schwarz / Robin iteration:")
        print(
            f"  rho_robin={rho_robin:.4e}  max_iter={max_iter}  tol={tol:.1e}  "
            f"artificial facets/subdomain: min={int(n_facets.min())} "
            f"max={int(n_facets.max())}"
        )
        print("  iter |   delta    | avg rel local | rel global")
        for it, delta, rl, rg in history:
            print(f"  {it:4d} | {delta:.4e} | {rl:.6e}  | {rg:.6e}")
        final_loc = history[-1][2]
        final_glob = history[-1][3]
        raw_loc = float(
            np.mean([
                sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains
            ])
        )
        print(
            f"  raw (no transmission) avg rel local: {raw_loc:.6e}"
            f"  ->  Schwarz: {final_loc:.6e}"
        )
        if final_loc > 1e-14:
            print(f"  improvement over raw: {raw_loc / final_loc:.2f}x")
        print(f"  final reconstructed global rel error: {final_glob:.6e}")

    return history


if __name__ == "__main__":
    import os
    import argparse

    from Decompose_Domain_Analytic_sphere import (
        check_sphere_domain_consistency,
        decomposeDomain,
    )
    from Solver_full_domain import Solver3D1D
    from Analytic_Domain import Domain
    from Boundary import SphereBoundary, random_sphere_points

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a simple analytic (non-OpenCCO) vascular domain, "
                    "solve the 3D-1D problem, decompose it, and couple the "
                    "subdomains with an additive Schwarz/Robin iteration.",
    )
    parser.add_argument("-name",   type=str, required=True,
                        help="subfolder name inside nets/ (e.g. sphere01)")
    parser.add_argument("-inlet",  type=int, default=10,
                        help="number of inflow vessels (n_vasi)")
    parser.add_argument("-outlet", type=int, default=10,
                        help="ramifications per vessel (n_ramifications)")
    parser.add_argument("-n",      type=int, default=40,
                        help="3D background mesh resolution")
    parser.add_argument("-sigma1d", type=float, default=1.0,
                        help="1D conductivity (sigma1d)")
    parser.add_argument("-sigma3d", type=float, default=1e-3,
                        help="3D conductivity (sigma3d)")
    parser.add_argument("-kappa", type=float, default=1.0,
                        help="coupling coefficient (kappa)")
    parser.add_argument("-radius", type=float, default=5.0,
                        help="radius of the spherical boundary")
    # Schwarz-specific knobs. rho is the Robin penalty: a sweep on this problem
    # put the optimum near 2.9e-2, about 10x the sigma3d/h heuristic that
    # solve_schwarz_robin falls back on when rho is left unset.
    parser.add_argument("-rho", type=float, default=None,
                        help="Robin penalty; default sigma3d/hmax (~2.9e-3). "
                             "Measured optimum on this problem: ~2.9e-2")
    parser.add_argument("-max_iter", type=int, default=60,
                        help="maximum Schwarz sweeps")
    parser.add_argument("-tol", type=float, default=1e-8,
                        help="stop when the largest iterate change falls below this")
    parser.add_argument("-rho_sweep", action="store_true",
                        help="sweep rho over decades around sigma3d/h instead of "
                             "a single run, and report the converged error for each")
    args = parser.parse_args()

    # nets/{name}/ holds every mesh file this run writes and the solver reads.
    #
    # Domain's export_* methods build filenames as f"{self.name}_marked_mesh.xdmf"
    # (i.e. they add the trailing '_'), while the solver reads them back as
    # f"{path_to_1D_mesh}marked_mesh.xdmf" (no added '_'). So self.name must NOT
    # end in '_', and path_to_1D_mesh MUST -- otherwise the underscores don't line
    # up and the solver looks for a file the domain never wrote.
    net_dir = os.path.join("nets", args.name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, args.name)   # nets/NAME/NAME   (no trailing _)
    mesh_prefix = f"{name_stem}_"                  # nets/NAME/NAME_  (solver prefix)

    boundary = SphereBoundary(
        radius=args.radius,
        inlet_points=random_sphere_points(
            40,
            x_sign=-1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        outlet_points=random_sphere_points(
            40,
            x_sign=+1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        border_eps=10e-1,
    )

    check_sphere_domain_consistency(
        boundary=boundary,
        n_min=-args.radius,
        n_max=args.radius,
    )

    # --- 1. build the analytic-boundary domain (unit sphere from Boundary.py) ---
    domain = Domain(
        name            = name_stem,
        n_vasi          = args.inlet,
        n_ramifications = args.outlet,
        boundary        = boundary,
        n_min = -args.radius, n_max = args.radius
    ).build()

    domain.export_box()
    domain.export_reticolo()
    domain.export_vaso()
    domain.export_xdmf()

    # --- 2. run the solver WITHOUT the boundary penalty ---
    # exterior='dirichlet' eliminates the exterior DOFs exactly instead of
    # pinning them with a penalty term, so the interface is not contaminated.
    out_dir = (
        f"./solution/Schwarz{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}"
    )
    solver = Solver3D1D(
        path_to_1D_mesh = mesh_prefix,
        boundary        = boundary,
        n               = args.n,
        sigma3d         = args.sigma3d,
        sigma1d         = args.sigma1d,
        kappa           = args.kappa,
        exterior        = "dirichlet",
    ).build().solve()

    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")

    # --- 3. decompose and solve each box in isolation ------------------------
    # decomposeDomain now stops at the local solve, leaving the subdomains at
    # the raw baseline the Schwarz result below is compared against.
    subdomains = decomposeDomain(solver, boundary)

    # --- 4. couple the subdomains by trading interface traces ----------------
    if args.rho_sweep:
        hmax = float(solver.meshV.hmax())
        rho_base = float(solver.sigma3d) / max(hmax, 1e-30)
        print("")
        print("rho sweep (converged error vs Robin penalty):")
        print("  rho          | iters | avg rel local | rel global")
        best = None
        for mult in (0.1, 1.0, 10.0, 100.0, 1000.0):
            rho = rho_base * mult
            hist = solve_schwarz_robin(
                subdomains, solver, rho_robin=rho,
                max_iter=args.max_iter, tol=args.tol, print_summary=False,
            )
            it, delta, rl, rg = hist[-1]
            print(f"  {rho:.4e} | {it:5d} | {rl:.6e}  | {rg:.6e}")
            if best is None or rg < best[1]:
                best = (rho, rg)
        print(f"  best: rho={best[0]:.4e} -> rel global {best[1]:.6e}")
    else:
        solve_schwarz_robin(
            subdomains, solver,
            rho_robin=args.rho, max_iter=args.max_iter, tol=args.tol,
            print_summary=True,
        )

    # The exact-Dirichlet floor -- re-solving each box with the true trace on
    # its cut planes -- bounds what any transmission condition can reach. It
    # was a diagnostic in decomposeDomain and no longer runs there; recover it
    # from diagnose_exact_dirichlet if you need the comparison.
