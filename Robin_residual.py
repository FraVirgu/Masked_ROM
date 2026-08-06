"""
Robin-residual domain decomposition -- eqs. (4) and (5) of the DD-ROM note.

The whole method is two formulas. Extraction, per subdomain, evaluated at the
restriction of the known global solution u* to box i:

    f_i^{*,-} = P_i^Gamma ( b_i - A_i u*_i + rho G_i^Gamma u*_i )        (4)

Local solve, with the data arriving from the face-neighbours:

    ( A_i + rho G_i^Gamma ) u_i = b_i + sum_{j in N(i)} E_ij f_j^{*,-}   (5)

That is all this file does. Diagnostics, exactness checks and the alternative
transmission schemes live elsewhere; `Decompose_Domain_Analytic.decomposeDomain`
is used unchanged to build the subdomains.

Usage
-----
    from Decompose_Domain_Analytic import decomposeDomain
    from Robin_residual import apply_robin_residual

    subdomains = decomposeDomain(solver, boundary)
    result = apply_robin_residual(subdomains, solver)

`decomposeDomain` must run first: it populates
"partition_solver", "local_to_global_dof", "P" and "ijk", which are the only
things this module reads.

Known limitation
----------------
Eq. (4) assumes b_i - A_i u* is supported on the artificial interface. That
holds only if A_i is the one-sided restriction of the global operator. The 3D
diffusion block is; the 3D-1D coupling block is not, because the local circle
quadrature clips vessels at the box boundary. For a vessel crossing a cut plane
most of its circular average lives in the neighbour, so the two boxes are
coupled through the 1D network as well as through the 3D flux -- and eq. (5)
carries only the latter. See `Solver_partition_domain.SolverPartitionDomain`
(C_global/l2g arguments) for the restricted-C variant that makes the extraction
one-sided at the cost of dropping that weight entirely.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from dolfin import Function

from Decompose_Domain_Analytic import (
    mark_artificial_facets,
    assemble_robin_interface,
)


def _robin_interface(ps, g_min, g_max):
    """G_i^Gamma on the artificial cut planes, eliminated rows/cols dropped.

    Two steps: mark_artificial_facets says WHICH faces of the box are internal
    cuts (as opposed to the true outer boundary, which keeps its physical
    condition), then assemble_robin_interface integrates u*v over just those.

    Eliminated exterior dofs carry a unit diagonal in A_i, so any load placed
    there is returned verbatim as the solution. The Robin term must not
    reactivate them, hence the D @ . @ D projection -- the facet tagging is
    pure grid geometry and knows nothing about the sphere, so it marks every
    cut-plane dof including the ~400 per box outside the physical domain.
    """
    facet_markers, n_tagged = mark_artificial_facets(ps, g_min, g_max)
    G_gamma = assemble_robin_interface(ps, facet_markers)

    ext = getattr(ps, "ext_dofs", None)
    if ext is not None and np.size(ext):
        keep = np.ones(G_gamma.shape[0], dtype=bool)
        keep[np.asarray(ext, dtype=int)] = False
        D = csr_matrix(
            (keep.astype(float), (np.arange(keep.size), np.arange(keep.size))),
            shape=G_gamma.shape,
        )
        G_gamma = D @ G_gamma @ D
    return G_gamma, n_tagged


def _interface_selector(subdomain, ps, l2g):
    """P_i^Gamma as a boolean mask over this box's local dofs.

    Built from the global selector (which already excludes true-boundary faces)
    and further stripped of eliminated exterior dofs, so it matches G_gamma.
    """
    sel = np.asarray(subdomain["P"].diagonal(), dtype=float)[l2g] > 0.0
    ext = getattr(ps, "ext_dofs", None)
    if ext is not None and np.size(ext):
        sel = sel.copy()
        sel[np.asarray(ext, dtype=int)] = False
    return sel


def _face_neighbours(subdomains):
    """Index pairs of boxes sharing a FACE.

    Not a dof-set intersection: on a structured grid the internal cut planes
    span the whole cross-section, so every pair of boxes touches somewhere
    (diagonal boxes along an edge, opposite corners at a single dof). Face
    adjacency is ijk differing by exactly 1 along exactly one axis.
    """
    nbrs = [[] for _ in subdomains]
    for i, sd_i in enumerate(subdomains):
        for j, sd_j in enumerate(subdomains):
            if i == j:
                continue
            diff = sorted(abs(a - b) for a, b in zip(sd_i["ijk"], sd_j["ijk"]))
            if diff == [0, 0, 1]:
                nbrs[i].append(j)
    return nbrs


def apply_robin_residual(subdomains, solver, rho_robin=None, state=None,
                         cross=False):
    """
    Run eqs. (4) and (5) once over all subdomains.

    Parameters
    ----------
    subdomains : list of dict
        Output of decomposeDomain.
    solver : Solver3D1D
        The converged global solve. Its u3d is the u* eq. (4) is evaluated at.
    rho_robin : float, optional
        Robin penalty. Defaults to sigma3d / hmax, the scale that balances the
        two terms of the residual.
    state : ndarray, optional
        Global-dof vector to evaluate the extraction at instead of u*. Passing
        the previous output makes the scheme iterative; leaving it None is the
        one-shot offline method as the note specifies it.
    cross : bool
        Add the 1D-network term for vessels whose averaging circle straddles a
        cut. REQUIRES the subdomains to have been built with
        decomposeDomain(..., restrict_global_C=True): the term is only
        well-posed when C_i carries the global arc weights (see below).

    Returns
    -------
    dict with "rho", "rel_global", "rel_local", "state", "iface_fraction".
    Each subdomain also gains "u3d_robin" and "f_star_minus".
    """
    if rho_robin is None:
        rho_robin = float(solver.sigma3d) / max(float(solver.meshV.hmax()), 1e-30)
    rho_robin = float(rho_robin)

    V_global = solver.W[0]
    n_global = V_global.dim()
    coords = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min, g_max = coords.min(axis=0), coords.max(axis=0)

    u_star = solver.u3d.vector().get_local()
    src = u_star if state is None else np.asarray(state, dtype=float)

    if cross:
        C_glob = solver.C.tocsr()
        C_src = C_glob.dot(src)          # the TRUE full-circle average

    # ---- eq. (4): extract, per subdomain ------------------------------------
    iface_num = 0.0
    iface_den = 0.0
    n_straddle_tot = 0

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]

        G_gamma, n_tagged = _robin_interface(ps, g_min, g_max)
        sel = _interface_selector(sd, ps, l2g)
        sd["_G_gamma"] = G_gamma
        sd["_sel"] = sel
        sd["_n_artificial_facets"] = n_tagged

        u_i = src[l2g]
        b_i = np.asarray(ps.rhs, dtype=float)

        # The 1D-network term. The global coupling block acting on box i's rows
        # is C_i^T G (C u*), with the FULL C: for a vessel whose averaging
        # circle straddles a cut, part of that circle lies in a neighbour. The
        # local operator only supplies C_i^T G (C_i u_i), so the difference
        #
        #     X_i = C_i^T G (C u* - C_i u_i)
        #
        # is missing from eq. (5) entirely. It is a coupling THROUGH THE VESSEL
        # NETWORK rather than through the flux across Gamma, and without it
        # b_i - A_i u* does not vanish in the interior -- which is exactly the
        # hypothesis eq. (4)'s P_i^Gamma restriction relies on.
        #
        # Written against the global C so nothing has to be assumed about how
        # the columns on the cut plane are shared between boxes; that also makes
        # it correct when a circle straddles several cuts at once (box edges and
        # corners), where a pairwise C_j u_j formulation would double-count.
        cross_i = 0.0
        if cross:
            C_i = ps.C.tocsr()
            cross_i = C_i.T.dot(solver.G.dot(C_src - C_i.dot(u_i)))
            w_i = np.asarray(np.abs(C_i).sum(axis=1)).ravel()
            w_g = np.asarray(np.abs(C_glob).sum(axis=1)).ravel()
            n_str = int(np.count_nonzero(
                (w_i > 1e-14) & (w_g - w_i > 1e-14 * np.maximum(w_g, 1.0))))
            sd["n_straddling_nodes"] = n_str
            n_straddle_tot += n_str
        sd["_cross"] = cross_i

        r_i = b_i - ps.A.dot(u_i) - cross_i + rho_robin * G_gamma.dot(u_i)

        # How much of the residual actually lives on the interface? Eq. (4)
        # discards everything else, so this fraction has to be ~1 for the
        # extraction to be meaningful. Measured at ~1% without the cross term.
        live = np.ones(r_i.size, dtype=bool)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            live[np.asarray(ext, dtype=int)] = False
        iface_num += float(np.linalg.norm(r_i[sel & live])) ** 2
        iface_den += float(np.linalg.norm(r_i[live])) ** 2

        f_star = np.zeros(n_global, dtype=float)
        f_star[l2g] = np.where(sel, r_i, 0.0)
        sd["f_star_minus"] = f_star

    # ---- E_ij: gather the face-neighbours' data on the shared face ----------
    # The grid is structured and local_to_global_dof is exact, so E_ij is a
    # plain index transfer -- no reorientation. Eq. (5) is a plain sum over
    # j in N(i): a dof on an edge shared by two face-neighbours genuinely
    # belongs to two pieces of Gamma^art_i, so it is not averaged.
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
            incoming[shared] += subdomains[j]["f_star_minus"][shared]
        sd["_f_incoming"] = incoming

    # ---- eq. (5): local Robin solve ----------------------------------------
    acc = np.zeros(n_global, dtype=float)
    cnt = np.zeros(n_global, dtype=float)
    rel_local = []

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]

        f_local = np.asarray(sd["_f_incoming"][l2g], dtype=float)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            f_local = f_local.copy()
            f_local[np.asarray(ext, dtype=int)] = 0.0

        # The cross term belongs on the rhs of eq. (5) as well: it is part of
        # the equation box i's rows actually satisfy, not only of the
        # extraction. Note it is frozen at u*, while the solve produces a
        # different u_i -- that inconsistency is why this does not reach
        # machine precision even when the extraction is exact.
        cross_i = sd["_cross"]
        if not np.isscalar(cross_i) and ext is not None and np.size(ext):
            cross_i = np.asarray(cross_i).copy()
            cross_i[np.asarray(ext, dtype=int)] = 0.0

        A_robin = (ps.A + rho_robin * sd["_G_gamma"]).tocsc()
        u_i = spsolve(A_robin,
                      np.asarray(ps.rhs, dtype=float) + f_local + cross_i)

        fn = Function(ps.V)
        fn.vector()[:] = u_i
        sd["u3d_robin"] = fn

        u_ref = sd["sol_3d_sub"].vector().get_local()
        nref = float(np.linalg.norm(u_ref))
        rel = float(np.linalg.norm(u_ref - u_i)) / nref if nref > 1e-12 else 0.0
        sd["u3d_robin_rel_l2"] = rel
        rel_local.append(rel)

        acc[l2g] += u_i
        cnt[l2g] += 1.0

    rec = np.zeros(n_global, dtype=float)
    nz = cnt > 0.0
    rec[nz] = acc[nz] / cnt[nz]
    state_out = rec.copy()
    rec[~nz] = u_star[~nz]

    ref = float(np.linalg.norm(u_star))
    rel_global = float(np.linalg.norm(u_star - rec)) / ref if ref > 0.0 else 0.0

    return {
        "rho": rho_robin,
        "rel_global": rel_global,
        "rel_local": float(np.mean(rel_local)),
        "state": state_out,
        # Fraction of the eq. (4) residual living on the artificial interface.
        # Eq. (4) throws the rest away, so this must be ~1 for the extraction
        # to mean anything.
        "iface_fraction": (float(np.sqrt(iface_num) / np.sqrt(iface_den))
                           if iface_den > 0.0 else 0.0),
        "n_straddling_nodes": n_straddle_tot,
    }


if __name__ == "__main__":
    import argparse
    import os

    from Decompose_Domain_Analytic import (
        check_sphere_domain_consistency,
        decomposeDomain,
    )
    from Solver_full_domain import Solver3D1D
    from Boundary import SphereBoundary, random_sphere_points

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Load a previously computed global 3D-1D solution, "
                    "decompose it, and run the Robin-residual correction "
                    "(eqs. 4-5). Does NOT re-solve the global problem.",
    )
    parser.add_argument(
        "-solution", type=str,
        default="./solution/SimpleDecomposition_n50_s1d1.0_s3d0.001_k1.0",
        help="folder holding solution.npy (its paraview/ subfolder holds the "
             "xdmf views of the same field)",
    )
    parser.add_argument("-name", type=str, default="Decomposition",
                        help="subfolder in nets/ holding the 1D mesh this "
                             "solution was computed on")
    parser.add_argument("-n", type=int, default=50,
                        help="3D background mesh resolution of that solution")
    parser.add_argument("-sigma1d", type=float, default=1.0)
    parser.add_argument("-sigma3d", type=float, default=1e-3)
    parser.add_argument("-kappa", type=float, default=1.0)
    parser.add_argument("-radius", type=float, default=5.0)
    parser.add_argument("-rho", type=float, default=None,
                        help="Robin penalty; default sigma3d/hmax")
    parser.add_argument("-restrict_global_C", action="store_true",
                        help="build each box's coupling operator by restricting "
                             "the global C instead of re-running clipped local "
                             "circle quadrature")
    parser.add_argument("-cross", action="store_true",
                        help="add the 1D-network term for vessels whose "
                             "averaging circle straddles a cut. Implies "
                             "-restrict_global_C: the term needs the global "
                             "arc weights to be well-posed.")
    args = parser.parse_args()

    sol_npy = os.path.join(args.solution, "solution.npy")
    if not os.path.isfile(sol_npy):
        raise SystemExit(f"no solution.npy in {args.solution}")

    # --- 1. rebuild the operators, but NOT the solution ----------------------
    # decomposeDomain reads solver.A, solver.C, solver.ext_dofs and the meshes,
    # none of which are stored in the saved field -- so build() must run. Only
    # solve() is skipped, which is the expensive part (the Krylov solve).
    #
    # The mesh is regenerated deterministically from nets/{name}, so the
    # operators match the ones the saved solution was computed with. The
    # boundary's inlet/outlet points are random, but they only affect the 1D
    # network that was already written to disk and is reloaded here.
    mesh_prefix = os.path.join("nets", args.name, args.name) + "_"
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
        path_to_1D_mesh=mesh_prefix,
        boundary=boundary,
        n=args.n,
        sigma3d=args.sigma3d,
        sigma1d=args.sigma1d,
        kappa=args.kappa,
        exterior="dirichlet",
    ).build()

    # --- 2. install the saved solution instead of solving --------------------
    # Same split solve() performs: the vector is [3D block | 1D block].
    x_np = np.load(sol_npy)
    n_3d = solver.W[0].dim()
    n_1d = solver.W[1].dim()
    if x_np.size != n_3d + n_1d:
        raise SystemExit(
            f"solution.npy has {x_np.size} entries but this mesh needs "
            f"{n_3d}+{n_1d}={n_3d + n_1d}. Check -n/-name/-radius match the "
            f"run that produced {args.solution}."
        )
    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]
    print(f"Loaded global solution from {sol_npy}  "
          f"({n_3d} 3D dofs, {n_1d} 1D dofs)")

    # --- 3. decompose and solve each box in isolation ------------------------
    # -cross needs C_i to carry the global arc weights, not a locally
    # renormalized partial arc, so it forces the restricted assembly.
    subdomains = decomposeDomain(
        solver, boundary,
        restrict_global_C=args.restrict_global_C or args.cross)

    # --- 4. eqs. (4)-(5) -----------------------------------------------------
    result = apply_robin_residual(
        subdomains, solver, rho_robin=args.rho, cross=args.cross)

    raw_glob = subdomains[0]["u3d_partition_full_error_raw_rel_l2"]
    raw_loc = float(np.mean([
        sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains]))
    print("")
    print("Robin-residual correction (eqs. 4-5)"
          + ("  + 1D-network cross term" if args.cross else "") + ":")
    print(f"  rho = {result['rho']:.4e}")
    # The structural diagnostic: eq. (4) keeps only the interface part of the
    # residual, so this fraction must be ~100% for the extraction to be valid.
    print(f"  residual on artificial interface: "
          f"{result['iface_fraction']:.1%}   (eq. 4 requires ~100%)")
    if args.cross:
        print(f"  straddling vessel nodes corrected: "
              f"{result['n_straddling_nodes']}")
    print(f"  avg rel local error:  raw {raw_loc:.6e}"
          f"  ->  corrected {result['rel_local']:.6e}")
    print(f"  reconstructed global: raw {raw_glob:.6e}"
          f"  ->  corrected {result['rel_global']:.6e}")
    print("  per-subdomain (ijk: raw -> corrected):")
    for sd in sorted(subdomains,
                     key=lambda s: -s["u3d_partition_error_local_raw_rel_l2"]):
        print(f"    {sd['ijk']}: "
              f"{sd['u3d_partition_error_local_raw_rel_l2']:.3e} -> "
              f"{sd['u3d_robin_rel_l2']:.3e}")



