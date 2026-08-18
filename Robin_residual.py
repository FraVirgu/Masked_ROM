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


def build_cross_term(ps, C_glob, G_glob, C_src, u_i):
    """X_i = C_i^T G (C u_src - C_i u_i), the 1D-network coupling term.

    The global coupling block acting on box i's rows is C_i^T G (C u), with the
    FULL C. Box i's own operator only supplies C_i^T G (C_i u_i), so the
    difference is missing from eq. (5) entirely. It is a coupling THROUGH THE
    VESSEL NETWORK rather than through the flux across Gamma.

    Two distinct mechanisms make it nonzero, and both matter:

      transverse -- a node's averaging circle is split by the cut, so C_i keeps
                    only part of the global arc weight. Driven by the radius.
      axial      -- the circles are whole but G, a CG1 mass matrix on the 1D
                    mesh, has off-diagonal entries between vessel nodes that
                    the cut separates. Independent of the radius.

    Written against the global C so nothing has to be assumed about how the
    columns on a cut plane are shared between boxes; that also makes it correct
    when a circle straddles several cuts at once (box edges and corners), where
    a pairwise C_j u_j formulation would double-count.

    NOTE both factors are evaluated at the SOURCE state, so when C_src comes
    from u* this is the exact missing operator applied to u*:

        X_i = C_i^T G (C - C_i) u*

    A matrix acting on the unknown belongs on the left-hand side. It is kept on
    the right only because u* is known here; see build_local_operator for the
    operator-side alternative. Returns a vector over box i's local dofs.
    """
    C_i = ps.C.tocsr()
    return C_i.T.dot(G_glob.dot(C_src - C_i.dot(u_i)))


def count_straddling_nodes(ps, C_glob):
    """Vessel nodes where this box holds part of the circle but not all of it."""
    C_i = ps.C.tocsr()
    w_i = np.asarray(np.abs(C_i).sum(axis=1)).ravel()
    w_g = np.asarray(np.abs(C_glob).sum(axis=1)).ravel()
    return int(np.count_nonzero(
        (w_i > 1e-14) & (w_g - w_i > 1e-14 * np.maximum(w_g, 1.0))))


def build_local_operator(ps, rho_robin, G_gamma):
    """A_i + rho G_i^Gamma, the operator eq. (5) inverts.

    Note the two G's are different objects and must not be confused:
    G_gamma is the artificial-interface SURFACE mass matrix (n_V x n_V) from
    assemble_robin_interface, while the G inside A_i's coupling block is the
    1D vessel mass matrix (n_Q x n_Q).

    The coupling missing from A_i is NOT repaired here, and cannot be. It is

        C_i^T G (C - C_i)

    and C - C_i annihilates exactly the columns C_i owns, so every nonzero
    entry sits in columns belonging to a NEIGHBOUR. No local operator can carry
    it: A_i + C_i^T G (C - C_i) is not a map from this box's dofs to itself.
    Restoring it would mean solving the subdomains jointly, which is what the
    decomposition exists to avoid. build_cross_term instead supplies the same
    quantity on the right-hand side, frozen at u* -- exact for the extraction,
    but leaving eq. (5) inverting an operator that omits a term its datum
    contains.
    """
    return (ps.A.tocsr() + rho_robin * G_gamma).tocsc()


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

    C_glob = solver.C.tocsr()
    C_src = C_glob.dot(src) if cross else None   # the TRUE full-circle average

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
            cross_i = build_cross_term(ps, C_glob, solver.G, C_src, u_i)
            n_str = count_straddling_nodes(ps, C_glob)
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
        A_robin = build_local_operator(ps, rho_robin, sd["_G_gamma"])
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
    # A run is identified by ONE thing: the name it was saved under. Everything
    # else -- which 1D network, which mesh resolution, which physics -- is
    # recovered from that name, because the saved folder is called
    #
    #     solution/Simple{name}_n{n}_s1d{sigma1d}_s3d{sigma3d}_k{kappa}
    #
    # and the 1D mesh lives in nets/{name}. Passing -solution and -name
    # independently, as an earlier version did, let them disagree: the operators
    # were rebuilt from one network while the field came from another, and the
    # only thing catching it was the dof-count check below.
    parser.add_argument("-name", type=str, default="robing_dd",
                        help="run name: reads the 1D mesh from nets/{name} and "
                             "the field from the matching solution/Simple{name}"
                             "_n..._s1d..._s3d..._k... folder")
    parser.add_argument("-n", type=int, default=40,
                        help="3D background mesh resolution of that solution")
    parser.add_argument("-sigma1d", type=float, default=1.0)
    parser.add_argument("-sigma3d", type=float, default=1e-3)
    parser.add_argument("-kappa", type=float, default=1.0)
    parser.add_argument(
        "-solution", type=str, default=None,
        help="override the solution folder. Normally left unset: it is derived "
             "from -name and the physics flags so the field and the 1D mesh "
             "cannot come from different runs.",
    )
    parser.add_argument("-radius", type=float, default=5.0)
    parser.add_argument("-rho", type=float, default=1.0,
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

    # Derive the solution folder from the run name unless explicitly overridden.
    # This MUST stay byte-identical to Decompose_Domain_Analytic's out_dir:
    #
    #     ./solution/Simple{name}_n{n}_s1d{sigma1d}_s3d{sigma3d}_k{kappa}
    #
    # A saved field is only reusable if every one of those parameters matches,
    # since each changes the operators. Encoding them in the name means a
    # mismatch shows up as "folder not found" -> re-solve, rather than as a
    # field silently paired with the wrong mesh.
    if args.solution is None:
        args.solution = os.path.join(
            "solution",
            f"Simple{args.name}_n{args.n}"
            f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")

    sol_npy = os.path.join(args.solution, "solution.npy")
    have_saved = os.path.isfile(sol_npy)

    # The 1D mesh is needed either way: decomposeDomain rebuilds the operators
    # from nets/{name} whether or not the field is already on disk.
    if not os.path.isdir(os.path.join("nets", args.name)):
        raise SystemExit(
            f"nets/{args.name} does not exist. Generate the 1D network first "
            f"(Decompose_Domain_Analytic.py -name {args.name})."
        )

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

    # --- 2. install the saved solution, or compute it if absent --------------
    # Reuse the saved field when it exists AND matches this mesh; otherwise run
    # the global solve once and save it, so the next invocation is cheap. A
    # stale field (right name, wrong mesh) is discarded rather than trusted:
    # the dof count is the only thing distinguishing it from a valid one.
    n_3d = solver.W[0].dim()
    n_1d = solver.W[1].dim()

    x_np = None
    if have_saved:
        cand = np.load(sol_npy)
        if cand.size == n_3d + n_1d:
            x_np = cand
            print(f"Loaded global solution from {sol_npy}  "
                  f"({n_3d} 3D dofs, {n_1d} 1D dofs)")
        else:
            print(f"Ignoring {sol_npy}: {cand.size} entries but this mesh "
                  f"needs {n_3d}+{n_1d}={n_3d + n_1d} "
                  f"({cand.size - (n_3d + n_1d):+d}). Re-solving.")

    if x_np is None:
        print(f"No usable solution in {args.solution} -- running the global "
              f"3D-1D solve (this is the expensive step).")
        raise RuntimeError("file's name must agree")

    # Same split solve() performs: the vector is [3D block | 1D block]. Done
    # unconditionally so the loaded and freshly-solved paths end identically.
    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]

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




    



