"""Is the cross term an operator evaluated at u*, and can it move into A_i?

The cross term of test_robin_minimal.py is added to the RIGHT-HAND SIDE of
eq. (5), while the operator being inverted stays A_i + rho G_i^Gamma. Since
the term is built from the known global solution,

    far_avg = C u* - C_i u*|_i        cross = C_i^T G far_avg

both factors are the true solution, so `cross` is really a MATRIX applied to
u*:

    cross = K u*,      K = C_i^T G (C - C_i)

and a matrix acting on the unknown belongs on the left. This script checks:

  1. that the identity cross == K u* holds numerically;
  2. how much of K has columns box i owns -- the part that could actually be
     folded into A_i without coupling the two solves;
  3. whether moving that part to the left changes the answer.

Point 2 is the crux. If K is mostly non-local, "put it in the operator" is not
a local fix and the one-sidedness of A_i cannot be repaired box by box.

Run:
    python test_cross_operator.py
    python test_cross_operator.py -axis y
    python test_cross_operator.py -sigma3d 1.0
"""

import argparse

import numpy as np
from scipy.sparse.linalg import spsolve

from dolfin import BoxMesh, Point, MeshFunction

import test_robin_minimal as T
from Solver_partition_domain import SolverPartitionDomain
from Robin_residual_sphere import _robin_interface


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[1])
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("-axis", choices=("x", "y", "z"), default="x")
    ap.add_argument("-vessel", choices=("middle", "near", "offset"),
                    default="near")
    ap.add_argument("-radius", type=float, default=0.08)
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-sigma1d", type=float, default=1.0)
    ap.add_argument("-kappa", type=float, default=1.0)
    args = ap.parse_args()

    ax = {"x": 0, "y": 1, "z": 2}[args.axis]
    half = args.n // 2
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    pts = T.vessel_points(args.vessel, ax)
    meshQ = T.build_line_mesh(pts)
    q_radii = np.full(meshQ.num_vertices(), args.radius)

    meshV = BoxMesh(Point(*lo), Point(*hi), args.n, args.n, args.n)
    markers = MeshFunction("size_t", meshV, 3, T.INTERIOR_TAG)

    # Same two-stage reference as the main test: coupled solve for p_sol,
    # then the p_known-driven solve whose operator the boxes restrict.
    _, p_sol, _ = T.solve_coupled_global(
        meshV, pts, args.radius, args.sigma3d, args.sigma1d, args.kappa)

    glob = SolverPartitionDomain(
        meshV=meshV, meshV_markers=markers, meshQ=meshQ,
        q_radii=q_radii, p_known=p_sol, sigma3d=args.sigma3d,
        kappa=args.kappa, gamma=1.0, interior_tag=T.INTERIOR_TAG,
        f3d=0.0).build()
    glob.ext_dofs = np.zeros(0, dtype=int)
    glob.solve()

    V_global = glob.V
    n_global = V_global.dim()
    coords_global = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    u_star = glob.u3d.vector().get_local()

    boxes = []
    for side in (0, 1):
        b_lo, b_hi = list(lo), list(hi)
        n_sub = [args.n, args.n, args.n]
        if side == 0:
            b_hi[ax] = 0.5
        else:
            b_lo[ax] = 0.5
        n_sub[ax] = half
        ijk = [0, 0, 0]
        ijk[ax] = side
        boxes.append(T.make_subdomain(
            tuple(b_lo), tuple(b_hi), tuple(n_sub), meshQ, q_radii, p_sol,
            args.sigma3d, args.kappa, tuple(ijk), V_global, coords_global,
            C_global=glob.C))

    g_min, g_max = coords_global.min(axis=0), coords_global.max(axis=0)
    rho = float(args.sigma3d) / max(float(meshV.hmax()), 1e-30)
    Cg = glob.C.tocsr()

    print("=" * 70)
    print(f"cut {args.axis}=0.5, vessel '{args.vessel}', "
          f"sigma3d={args.sigma3d}, rho={rho:.4e}")
    print("=" * 70)

    for sd in boxes:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        u_i = u_star[l2g]
        Ci = ps.C.tocsr()

        # 1. the cross term exactly as test_robin_minimal builds it
        far_avg = Cg.dot(u_star) - Ci.dot(u_i)
        cross = Ci.T.dot(glob.G.dot(far_avg))

        # 2. the same quantity as a matrix acting on the global u*.
        #    C_i has local columns; widen it to global so C - C_i makes sense.
        Ci_glob = np.zeros((Ci.shape[0], n_global))
        Ci_glob[:, l2g] = Ci.toarray()
        K = Ci.T @ (glob.G @ (Cg.toarray() - Ci_glob))    # (n_loc, n_global)
        d = np.abs(cross - K @ u_star).max()

        owned = np.zeros(n_global, dtype=bool)
        owned[l2g] = True
        w_in = np.abs(K[:, owned]).sum()
        w_out = np.abs(K[:, ~owned]).sum()
        frac_out = 100.0 * w_out / max(w_in + w_out, 1e-30)

        print(f"\nbox {sd['ijk']}")
        print(f"  ||cross||           = {np.linalg.norm(cross):.6e}")
        print(f"  max|cross - K u*|   = {d:.3e}  "
              f"-> identity {'HOLDS' if d < 1e-10 else 'FAILS'}")
        print(f"  |K| on owned cols   = {w_in:.6e}")
        print(f"  |K| on foreign cols = {w_out:.6e}  ({frac_out:.1f}% non-local)")

        G_gamma, _ = _robin_interface(ps, g_min, g_max)
        sub_coords = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
        on_if = np.abs(sub_coords[:, ax] - 0.5) < 1e-10

        flux = np.asarray(ps.rhs, float) - ps.A.dot(u_i) - cross
        sd.update(_r=flux + rho * G_gamma.dot(u_i), _sel=on_if, _G=G_gamma,
                  _cross=cross, _K=K, _u=u_i, _l2g=l2g, _ps=ps)

    # interface transfer, as in eq. (4)-(5)
    for i, sd in enumerate(boxes):
        other = boxes[1 - i]
        f_star = np.zeros(n_global)
        f_star[other["_l2g"]] = np.where(other["_sel"], other["_r"], 0.0)
        iface = np.zeros(n_global, dtype=bool)
        iface[sd["_l2g"][sd["_sel"]]] = True
        sd["_f_in"] = np.where(iface, f_star, 0.0)[sd["_l2g"]]

    def rel(sd, u):
        return float(np.linalg.norm(sd["_u"] - u) / np.linalg.norm(sd["_u"]))

    print("")
    print("  eq. (5) solved two ways:")
    print(f"    {'box':10s} {'cross on rhs':>16s} {'K_loc in operator':>20s}")
    for sd in boxes:
        ps, K, l2g = sd["_ps"], sd["_K"], sd["_l2g"]
        A_rob = (ps.A + rho * sd["_G"]).tocsc()

        # (a) what the test does now: the whole term on the right
        u_a = spsolve(A_rob, np.asarray(ps.rhs, float)
                      + sd["_f_in"] + sd["_cross"])

        # (b) the local columns of K moved into the operator. The foreign
        #     columns cannot follow -- box i does not own those dofs -- so
        #     they stay on the right, still frozen at u*.
        K_loc = K[:, l2g]
        K_out = K.copy()
        K_out[:, l2g] = 0.0
        u_b = spsolve(A_rob + K_loc, np.asarray(ps.rhs, float)
                      + sd["_f_in"] + K_out @ u_star)

        print(f"    {str(sd['ijk']):10s} {rel(sd, u_a):16.6e} "
              f"{rel(sd, u_b):20.6e}")
    print("")

    # ---- the coupled two-box solve -----------------------------------------
    # K cannot enter a LOCAL operator: its columns belong to the neighbour. But
    # it can enter a JOINT one. Assemble both boxes' rows over the shared
    # global dof vector,
    #
    #     row block i:   (A_i + K_i) u = b_i
    #
    # where A_i is scattered from local to global indices and K_i already is
    # global. Interface dofs are owned by both boxes and would be assembled
    # twice, so each such row is taken from one box only -- the global equation
    # on that row is the same either way.
    #
    # If the deficit really is exactly K, this must recover u* to solver
    # precision, and the gap between it and the local solves above is the price
    # of insisting the solve stay local.
    print("  coupled two-box solve (K in the operator, no frozen u*):")

    from scipy.sparse import lil_matrix

    A_joint = lil_matrix((n_global, n_global))
    b_joint = np.zeros(n_global)
    claimed = np.zeros(n_global, dtype=bool)

    for sd in boxes:
        ps, l2g, K = sd["_ps"], sd["_l2g"], sd["_K"]
        A_loc = ps.A.tocoo()
        rows_mine = ~claimed[l2g]          # rows this box is first to claim

        for r, c, v in zip(A_loc.row, A_loc.col, A_loc.data):
            if rows_mine[r]:
                A_joint[l2g[r], l2g[c]] += v
        nz = np.nonzero(K)
        for r, c in zip(*nz):
            if rows_mine[r]:
                A_joint[l2g[r], c] += K[r, c]
        b_joint[l2g[rows_mine]] += np.asarray(ps.rhs, float)[rows_mine]
        claimed[l2g] = True

    u_joint = spsolve(A_joint.tocsc(), b_joint)
    e_joint = float(np.linalg.norm(u_joint - u_star)
                    / np.linalg.norm(u_star))
    print(f"    global error vs u*: {e_joint:.6e}")
    for sd in boxes:
        print(f"    {str(sd['ijk']):10s} {rel(sd, u_joint[sd['_l2g']]):.6e}")
    print("")


if __name__ == "__main__":
    main()
