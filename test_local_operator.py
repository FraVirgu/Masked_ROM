"""Is A_i the exact one-sided restriction of the global A? Matrix algebra only.

test_robin_ladder.py's [A] showed that a box given the EXACT trace of u* on its
cut faces still does not return u*_i, which places the defect below the
transmission layer: in A_i, b_i, the cross term or local_to_global_dof. This
script finds which, by comparing operators directly. There is no solve here and
no rho -- a solve turns a structured discrepancy into one number and loses the
structure, which is exactly what has been happening.

The global operator is (Solver_full_domain)

    A = AD + M,        M_00 = C^T G C

and each box builds (Solver_partition_domain)

    A_i = AD00_i + gamma * C_i^T G_i C_i.

Eq. (4) needs A_i to be the restriction of A to this box's dofs, i.e.

    A_i == A[l2g][:, l2g]                                          (*)

on every row the box owns strictly inside its own volume. It cannot hold on
interface rows -- those genuinely have neighbour couplings the box does not see,
and that is precisely the flux eq. (4) extracts. So the comparison is reported
SEPARATELY for interior rows and interface rows: (*) must hold on the interior,
and the interface difference should be confined to columns the box does not own.

Blocks are compared one at a time, because they fail for different reasons:

  [1] diffusion AD00   -- pure FEM assembly on the box vs on the global mesh.
      A mismatch here is a meshing or marker problem: the local BoxMesh
      tetrahedralizes boundary-straddling cells differently from the global
      mesh (see eliminate_exterior_local's docstring), so cells cut by the
      sphere can carry different quadrature.

  [2] coupling C       -- the 3D-1D averaging operator. With -cross this is a
      column selection from the global C and should match EXACTLY on owned
      columns. Without it, the local circle quadrature renormalizes by the
      surviving arc, which Solver_partition_domain's own comment records as a
      77% discrepancy. That comment predicts [2] fails without -cross and
      passes with it; this test checks whether it does.

  [3] G, the 1D mass matrix -- assembled per box from the shared meshQ. It
      should be identical across boxes and equal to the global G; if it is not,
      every C^T G C differs for that reason alone.

  [4] full A_i and b_i -- the composition, and the residual b_i - A_i u*_i
      split into interior and interface parts. The interior part is what eq. (4)
      assumes vanishes.

Run:
    python3 test_local_operator.py -name report_sphere_small -n 40 -radius 5.0
    python3 test_local_operator.py -name report_sphere_small -n 40 -radius 5.0 -cross
"""

import argparse
import os

import numpy as np
from scipy.sparse import csr_matrix

from dolfin import Function

from Decompose_Domain_Analytic_sphere import (
    check_sphere_domain_consistency,
    decomposeDomain,
    matrix_to_csr,
)
from Solver_full_domain import Solver3D1D
from Boundary import SphereBoundary, random_sphere_points
from Robin_residual_sphere import _interface_selector


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
            f"solution has {x_np.size} entries, operators want {n_3d + n_1d}.")
    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]
    print(f"loaded {sol_npy}")
    return solver, boundary


def restrict(M_glob, l2g):
    """M_glob[l2g][:, l2g] -- the block of the global operator this box owns."""
    return M_glob.tocsr()[l2g, :][:, l2g]


def block_report(name, local, glob_restricted, interior, interface):
    """Compare two operators on interior rows and on interface rows.

    Interior rows must agree: eq. (4) assumes A_i is the one-sided restriction
    there. Interface rows are expected to differ -- that difference IS the flux
    the method extracts -- so it is reported but never counted as a failure.
    """
    D = (local - glob_restricted).tocsr()

    def rownorm(mask):
        if not mask.any():
            return 0.0, 0.0
        d = D[mask, :]
        g = glob_restricted.tocsr()[mask, :]
        return (float(np.sqrt((d.data ** 2).sum())) if d.nnz else 0.0,
                float(np.sqrt((g.data ** 2).sum())) if g.nnz else 0.0)

    di, gi = rownorm(interior)
    df, gf = rownorm(interface)
    ri = di / gi if gi > 1e-30 else float("nan")
    rf = df / gf if gf > 1e-30 else float("nan")
    print(f"  {name:22s} interior {di:11.3e} ({ri:8.2e} rel)   "
          f"interface {df:11.3e} ({rf:8.2e} rel)")
    return ri


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
    ap.add_argument("-cross", action="store_true",
                    help="restrict_global_C: build C_i by column selection")
    ap.add_argument("-two", action="store_true",
                    help="2 boxes (single cut) instead of 8")
    ap.add_argument("-solution", type=str, default=None)
    args = ap.parse_args()

    solver, boundary = build_solver(args)
    V = solver.W[0]
    n_global = V.dim()
    u_star = solver.u3d.vector().get_local()

    if args.two:
        full = 2.0 * args.radius
        subdomains = decomposeDomain(
            solver, boundary, x_ROM_lenght=args.radius,
            y_ROM_lenght=full, z_ROM_lenght=full,
            restrict_global_C=args.cross)
    else:
        subdomains = decomposeDomain(solver, boundary,
                                     restrict_global_C=args.cross)

    # Global blocks. A = AD + M with M_00 = C^T G C; AD's 3D-3D block is the
    # pure diffusion part. Both are pulled out separately so a mismatch can be
    # attributed rather than merely observed.
    A_glob = matrix_to_csr(solver.A[0][0])
    AD_glob = matrix_to_csr(solver.AD[0][0])
    C_glob = solver.C.tocsr()
    G_glob = solver.G.tocsr()
    M_glob = (C_glob.T @ G_glob @ C_glob).tocsr()

    bar = "=" * 78
    tag = "restricted C (-cross)" if args.cross else "local quadrature C"
    print(f"\n{bar}\nLOCAL OPERATOR vs GLOBAL RESTRICTION   ({tag})\n{bar}")
    print(f"global: A {A_glob.shape} nnz {A_glob.nnz}   "
          f"C {C_glob.shape} nnz {C_glob.nnz}   G nnz {G_glob.nnz}")

    print("\nEach block, local vs A[l2g][:,l2g]. Interior rows MUST agree;")
    print("interface rows are expected to differ (that is the flux).")

    worst = {}
    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = np.asarray(sd["local_to_global_dof"], dtype=int)
        sel = _interface_selector(sd, ps, l2g)

        live = np.ones(l2g.size, dtype=bool)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            live[np.asarray(ext, dtype=int)] = False
        interior = live & ~sel
        interface = live & sel

        print(f"\n{sd['ijk']}  local dofs {l2g.size}  "
              f"interior {int(interior.sum())}  interface {int(interface.sum())}")

        # [1] diffusion
        AD_loc = matrix_to_csr(ps.AD00)
        r1 = block_report("[1] diffusion AD00", AD_loc,
                          restrict(AD_glob, l2g), interior, interface)

        # [3] G -- same meshQ for every box, so this should be exact
        G_loc = ps.G.tocsr()
        dG = (G_loc - G_glob)
        nG = float(np.sqrt((G_glob.data ** 2).sum()))
        rG = (float(np.sqrt((dG.data ** 2).sum())) / nG
              if dG.nnz and nG > 1e-30 else 0.0)
        print(f"  {'[3] 1D mass G':22s} rel diff {rG:11.3e}"
              f"   (shared meshQ: expect 0)")

        # [2] coupling C: compare on the columns this box owns
        C_loc = ps.C.tocsr()
        C_res = C_glob[:, l2g]
        dC = (C_loc - C_res)
        nC = float(np.sqrt((C_res.data ** 2).sum()))
        rC = (float(np.sqrt((dC.data ** 2).sum())) / nC
              if dC.nnz and nC > 1e-30 else 0.0)
        w_loc = float(np.abs(C_loc).sum())
        w_res = float(np.abs(C_res).sum())
        print(f"  {'[2] coupling C':22s} rel diff {rC:11.3e}"
              f"   weight local {w_loc:.4e} vs restricted {w_res:.4e}"
              f"  ({w_loc / w_res if w_res > 1e-30 else float('nan'):.3f}x)")

        # [2b] the assembled coupling block
        M_loc = (C_loc.T @ G_loc @ C_loc).tocsr()
        r2 = block_report("[2b] coupling CtGC", M_loc,
                          restrict(M_glob, l2g), interior, interface)

        # [4] the full operator
        r4 = block_report("[4] full A_i", ps.A.tocsr(),
                          restrict(A_glob, l2g), interior, interface)

        # [4b] the eq.(4) residual, split. Eq.(4) keeps only the interface part
        # and discards the interior part; the interior part must therefore be
        # negligible for the extraction to mean anything.
        u_i = u_star[l2g]
        b_i = np.asarray(ps.rhs, dtype=float)
        r_i = b_i - ps.A.dot(u_i)
        ni = float(np.linalg.norm(r_i[interior]))
        nf = float(np.linalg.norm(r_i[interface]))
        nb = float(np.linalg.norm(b_i[live]))
        print(f"  {'[4b] b_i - A_i u*':22s} interior {ni:11.3e}"
              f"   interface {nf:11.3e}   (||b_i|| {nb:.3e})")
        print(f"  {'':22s} interior/total {ni / max(np.hypot(ni, nf), 1e-30):8.1%}"
              f"   <- eq.(4) discards this")

        worst[str(sd["ijk"])] = (r1, rC, r2, r4,
                                 ni / max(np.hypot(ni, nf), 1e-30))

    # ---- verdict ------------------------------------------------------------
    print(f"\n{bar}\nSUMMARY (relative difference on INTERIOR rows)\n{bar}")
    print(f"{'box':>12s}{'AD00':>12s}{'C':>12s}{'CtGC':>12s}{'A_i':>12s}"
          f"{'r interior':>12s}")
    for k, (r1, rC, r2, r4, fi) in worst.items():
        print(f"{k:>12s}{r1:12.3e}{rC:12.3e}{r2:12.3e}{r4:12.3e}{fi:11.1%}")

    a1 = np.array([v[0] for v in worst.values()])
    aC = np.array([v[1] for v in worst.values()])
    a2 = np.array([v[2] for v in worst.values()])
    a4 = np.array([v[3] for v in worst.values()])

    print(f"\n{bar}\nWHERE THE DEFECT IS\n{bar}")
    tol = 1e-10
    if np.nanmax(a1) > tol:
        print(f"[1] DIFFUSION differs on interior rows "
              f"(max {np.nanmax(a1):.3e}).")
        print("    The local BoxMesh does not reproduce the global assembly on")
        print("    cells it shares. This is a meshing/marker problem and is")
        print("    independent of the 3D-1D coupling -- it would break eq.(4)")
        print("    even with no vessels at all.")
    else:
        print(f"[1] diffusion matches on interior rows "
              f"(max {np.nanmax(a1):.3e}). OK.")

    if np.nanmax(aC) > tol:
        print(f"[2] COUPLING C differs from the global restriction "
              f"(max {np.nanmax(aC):.3e}).")
        if not args.cross:
            print("    Expected without -cross: the local circle quadrature")
            print("    renormalizes by the surviving arc. Solver_partition_domain")
            print("    records this as ~77%. Re-run with -cross.")
        else:
            print("    NOT expected with -cross: that path is a pure column")
            print("    selection from the global C and should be exact.")
    else:
        print(f"[2] coupling C matches (max {np.nanmax(aC):.3e}). OK.")

    if np.nanmax(a4) > tol:
        print(f"[4] A_i is not the restriction of A on interior rows "
              f"(max {np.nanmax(a4):.3e}).")
        print("    This is the direct cause of test_robin_ladder [A] failing.")
        dom = "diffusion" if np.nanmax(a1) > np.nanmax(a2) else "coupling"
        print(f"    Dominated by the {dom} block.")
    else:
        print(f"[4] A_i IS the restriction of A on interior rows. OK --")
        print("    the defect is then in b_i or in the exterior elimination,")
        print("    not in the operator.")
    print(bar)


if __name__ == "__main__":
    main()
