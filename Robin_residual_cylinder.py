"""
Robin-residual domain decomposition on a CYLINDER -- eqs. (4) and (5).

Companion to Robin_residual_sphere.py. The method itself does not depend on the
shape of the domain: apply_robin_residual and its helpers work off
`partition_solver`, the local-to-global dof map and the box extents, and they
reach the geometry only through the boundary object that decomposeDomain
already applied. So every function is IMPORTED from the sphere module and only
the driver below is rewritten, to build a CylinderBoundary instead of a
SphereBoundary.

    from Decompose_Domain_Analytic_cylinder import decomposeDomain
    from Robin_residual_cylinder import apply_robin_residual

    subdomains = decomposeDomain(solver, boundary)
    result = apply_robin_residual(subdomains, solver)

See Robin_residual_sphere.py for the derivation of eqs. (4)-(5) and for the
known limitation of the extraction when vessels straddle a cut plane -- both
apply here unchanged.
"""

import numpy as np

from dolfin import Function

# The method, verbatim. Re-exported so this module is a drop-in replacement for
# the sphere one; a fix there is a fix here.
from Robin_residual_sphere import (  # noqa: F401
    _face_neighbours,
    _interface_selector,
    _robin_interface,
    apply_robin_residual,
    build_cross_term,
    build_local_operator,
    count_straddling_nodes,
)


if __name__ == "__main__":
    import argparse
    import os

    from Decompose_Domain_Analytic_cylinder import (
        check_cylinder_domain_consistency,
        cylinder_background_mesh,
        cylinder_bbox,
        cylinder_rom_lengths,
        decomposeDomain,
    )
    from Solver_full_domain import Solver3D1D
    from Boundary import CylinderBoundary, random_cylinder_points

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Load a previously computed global 3D-1D solution on a "
                    "cylindrical domain, decompose it, and run the "
                    "Robin-residual correction (eqs. 4-5). Does NOT re-solve "
                    "the global problem.",
    )
    # A run is identified by ONE thing: the name it was saved under. Everything
    # else -- which 1D network, which mesh resolution, which physics -- is
    # recovered from that name, because the saved folder is called
    #
    #     solution/Simple{name}_n{n}_s1d{sigma1d}_s3d{sigma3d}_k{kappa}
    #
    # and the 1D mesh lives in nets/{name}.
    parser.add_argument("-name", type=str, default="cylinder_dd",
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
    parser.add_argument("-radius", type=float, default=5.0,
                        help="radius of the cylindrical boundary")
    parser.add_argument("-height", type=float, default=10.0,
                        help="total extent of the cylinder along its axis")
    parser.add_argument("-axis", choices=("x", "y", "z"), default="z",
                        help="direction of the cylinder axis")
    parser.add_argument("-parts", type=int, default=2,
                        help="subdomains per direction; the ROM lengths are "
                             "derived from the extents so a non-cubic cylinder "
                             "still splits evenly")
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
    # This MUST stay byte-identical to the driver that wrote it
    # (Decompose_Domain_Analytic_cylinder.py):
    #
    #     ./solution/Simple{name}_n{n}_s1d{sigma1d}_s3d{sigma3d}_k{kappa}
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
            f"(Decompose_Domain_Analytic_cylinder.py -name {args.name})."
        )

    # --- 1. rebuild the operators, but NOT the solution ----------------------
    # decomposeDomain reads solver.A, solver.C, solver.ext_dofs and the meshes,
    # none of which are stored in the saved field -- so build() must run. Only
    # solve() is skipped, which is the expensive part (the Krylov solve).
    #
    # The boundary MUST be built with the same radius/height/axis the solution
    # was computed with: it decides which dofs are exterior, and a mismatch
    # silently changes the operator rather than raising.
    mesh_prefix = os.path.join("nets", args.name, args.name) + "_"
    boundary = CylinderBoundary(
        radius=args.radius,
        height=args.height,
        axis=args.axis,
        inlet_points=random_cylinder_points(
            40, sign=-1, radius=args.radius, height=args.height,
            axis=args.axis, min_offset=0.2, min_dist_to_boundary=0.06),
        outlet_points=random_cylinder_points(
            40, sign=+1, radius=args.radius, height=args.height,
            axis=args.axis, min_offset=0.2, min_dist_to_boundary=0.06),
        border_eps=10e-1,
    )
    bounds = cylinder_bbox(args.radius, args.height, args.axis)
    check_cylinder_domain_consistency(boundary=boundary, bounds=bounds)

    # Must be the SAME mesh the solution was computed on -- the enclosing cube,
    # not boundary._bbox. Anything else changes the dof count and the saved
    # field is rejected (or, worse, silently pairs with a different geometry).
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix,
        full_domain_mesh=cylinder_background_mesh(
            args.radius, args.height, args.n, args.axis),
        boundary=boundary,
        n=args.n,
        sigma3d=args.sigma3d,
        sigma1d=args.sigma1d,
        kappa=args.kappa,
        exterior="dirichlet",
    ).build()

    # --- 2. install the saved solution, or fail loudly if absent -------------
    # Reuse the saved field when it exists AND matches this mesh; a stale field
    # (right name, wrong mesh) is discarded rather than trusted: the dof count
    # is the only thing distinguishing it from a valid one.
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

    # Same split solve() performs: the vector is [3D block | 1D block].
    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]

    # --- 3. decompose and solve each box in isolation ------------------------
    # The ROM lengths come from the cylinder's own extents: with a single shared
    # length a non-cubic domain would get a different box count per direction,
    # since decomposeDomain derives it as ceil(extent / ROM_length).
    lx, ly, lz = cylinder_rom_lengths(
        args.radius, args.height, args.axis, args.parts)
    print(f"ROM box lengths: x={lx:.4f} y={ly:.4f} z={lz:.4f} "
          f"({args.parts} per direction)")

    # -cross needs C_i to carry the global arc weights, not a locally
    # renormalized partial arc, so it forces the restricted assembly.
    subdomains = decomposeDomain(
        solver, boundary,
        x_ROM_lenght=lx, y_ROM_lenght=ly, z_ROM_lenght=lz,
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
    print(f"  domain: cylinder r={args.radius} h={args.height} "
          f"axis={args.axis}")
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
