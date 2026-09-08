"""Plot an artificial cut plane of a CYLINDRICAL domain -- Schwarz variant.

Sits at the intersection of two existing scripts:

    plot_interface_wall_cylinder.py         cylinder geometry, Robin residual
    plot_interface_wall_sphere_schwarz.py   sphere geometry,   Schwarz iteration
    this file                               cylinder geometry, Schwarz iteration

so it takes the cylinder setup from the first and the transmission scheme from
the second. As in plot_interface_wall_cylinder.py the drawing helpers are
geometry-agnostic -- they work off scattered dof coordinates -- and are
IMPORTED from the sphere script rather than duplicated; only `main` is written
here, to build a CylinderBoundary, size the mesh from the cylinder's extents,
and drive solve_schwarz_robin.

The distinction from the residual method matters for what the picture means.
apply_robin_residual evaluates eq. (4) at the ground-truth u*, so its corrected
wall is informed by the answer. The Schwarz iteration never touches u* -- each
sweep feeds a box the neighbours' PREVIOUS traces -- so the wall shown here is
reachable without knowing the target. A residual panel that looks better is
therefore not automatically the better scheme.

Three panels, the two error ones on a shared colour scale:

    global      u* restricted to the plane -- the target
    raw         the boxes touching the plane, solved in isolation
    schwarz     the same boxes after the converged Schwarz iteration

Red dashed lines mark the seams between subdomains on the plane.

There is no -cross here: the cross term corrects the eq. (4) extraction, and
the Schwarz scheme forms no such residual. The knobs that replace it are -rho,
-max_iter and -tol.

Run (reuses the saved global solution):

    python plot_interface_wall_cylinder_schwarz.py -name cylinder_dd
    python plot_interface_wall_cylinder_schwarz.py -name cylinder_dd -rho 2.9e-2
    python plot_interface_wall_cylinder_schwarz.py -name cylinder_dd -axis z -out wall.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from dolfin import Function

from Decompose_Domain_Analytic_cylinder import (
    check_cylinder_domain_consistency,
    cylinder_background_mesh,
    cylinder_bbox,
    cylinder_rom_lengths,
    decomposeDomain,
    enclosing_cube,
)
from Solver_full_domain import Solver3D1D
from Boundary import CylinderBoundary, random_cylinder_points
from Solve_schwarz_robin import solve_schwarz_robin

# Geometry-agnostic drawing helpers, reused verbatim.
from plot_interface_wall_sphere import (
    draw_subdomain_seams,
    plane_dofs,
    subdomain_seams,
    surface_plane,
)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[0])
    ap.add_argument("-name", type=str, default="cylinder_dd")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("-sigma1d", type=float, default=1.0)
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-kappa", type=float, default=1.0)
    ap.add_argument("-radius", type=float, default=5.0,
                    help="radius of the cylindrical boundary")
    ap.add_argument("-height", type=float, default=10.0,
                    help="total extent of the cylinder along its axis")
    ap.add_argument("-cyl_axis", choices=("x", "y", "z"), default="z",
                    help="direction of the cylinder axis (the geometry)")
    ap.add_argument("-parts", type=int, default=2,
                    help="subdomains per direction")
    # Schwarz knobs. Unlike the residual script, whose -rho defaults to 1.0,
    # rho defaults to None so solve_schwarz_robin applies its own sigma3d/hmax
    # heuristic -- the two rho's scale the same G_gamma but the schemes put it
    # to different use, so the residual default does not carry over.
    ap.add_argument("-rho", type=float, default=None,
                    help="Robin penalty; default sigma3d/hmax. "
                         "Measured optimum on the sphere problem: ~2.9e-2")
    ap.add_argument("-max_iter", type=int, default=60,
                    help="maximum Schwarz sweeps")
    ap.add_argument("-tol", type=float, default=1e-8,
                    help="stop when the largest iterate change falls below this")
    ap.add_argument("-axis", choices=("x", "y", "z"), default="y",
                    help="normal of the cut plane to draw (the view)")
    ap.add_argument("-restrict_global_C", action="store_true")
    ap.add_argument("-out", type=str, default=None,
                    help="output png; defaults to "
                         "interface_wall_schwarz_{name}_{axis}.png")
    ap.add_argument("-solution", type=str, default=None)
    args = ap.parse_args()

    ax_i = {"x": 0, "y": 1, "z": 2}[args.axis]
    if args.solution is None:
        args.solution = os.path.join(
            "solution",
            f"Simple{args.name}_n{args.n}"
            f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")
    sol_npy = os.path.join(args.solution, "solution.npy")
    if not os.path.isdir(os.path.join("nets", args.name)):
        raise SystemExit(f"nets/{args.name} does not exist.")

    # ---- rebuild operators, reuse the global field -------------------------
    mesh_prefix = os.path.join("nets", args.name, args.name) + "_"
    boundary = CylinderBoundary(
        radius=args.radius,
        height=args.height,
        axis=args.cyl_axis,
        inlet_points=random_cylinder_points(
            40, sign=-1, radius=args.radius, height=args.height,
            axis=args.cyl_axis, min_offset=0.2, min_dist_to_boundary=0.06),
        outlet_points=random_cylinder_points(
            40, sign=+1, radius=args.radius, height=args.height,
            axis=args.cyl_axis, min_offset=0.2, min_dist_to_boundary=0.06),
        border_eps=10e-1,
    )
    bounds = cylinder_bbox(args.radius, args.height, args.cyl_axis)
    check_cylinder_domain_consistency(boundary=boundary, bounds=bounds)

    # Same enclosing-cube mesh the solution was computed on; see the driver.
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix,
        full_domain_mesh=cylinder_background_mesh(
            args.radius, args.height, args.n, args.cyl_axis),
        boundary=boundary, n=args.n,
        sigma3d=args.sigma3d, sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet").build()

    n_3d, n_1d = solver.W[0].dim(), solver.W[1].dim()
    x_np = None
    if os.path.isfile(sol_npy):
        cand = np.load(sol_npy)
        if cand.size == n_3d + n_1d:
            x_np = cand
            print(f"Loaded global solution from {sol_npy}")
    if x_np is None:
        print(f"No usable solution in {args.solution} -- running the global "
              f"3D-1D solve (this is the expensive step).")
        raise RuntimeError("file's name must agree")

    solver.x_np = x_np
    solver.u3d = Function(solver.W[0])
    solver.u1d = Function(solver.W[1])
    solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d.vector()[:] = x_np[n_3d:]

    # The ROM lengths follow the cylinder's extents so a non-cubic domain still
    # splits into `parts` boxes per direction.
    lx, ly, lz = cylinder_rom_lengths(
        args.radius, args.height, args.cyl_axis, args.parts)
    subdomains = decomposeDomain(
        solver, boundary,
        x_ROM_lenght=lx, y_ROM_lenght=ly, z_ROM_lenght=lz,
        restrict_global_C=args.restrict_global_C)

    # Raw local fields must be captured BEFORE solve_schwarz_robin, which
    # stores its iterate under its own key alongside them.
    for sd in subdomains:
        sd["_u_raw"] = sd["partition_solver"].u3d.vector().get_local().copy()

    history = solve_schwarz_robin(
        subdomains, solver, rho_robin=args.rho,
        max_iter=args.max_iter, tol=args.tol, print_summary=True)
    # solve_schwarz_robin reports but does not return the converged errors, so
    # read them off the last sweep of the history it hands back.
    n_sweeps, delta_final, rel_local_final, rel_global_final = history[-1]
    rho_used = (args.rho if args.rho is not None
                else float(solver.sigma3d) / max(float(solver.meshV.hmax()), 1e-30))

    # ---- locate the cut plane ----------------------------------------------
    V = solver.W[0]
    coords_g = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    u_star = solver.u3d.vector().get_local()

    idx_lo = [sd for sd in subdomains if sd["ijk"][ax_i] == 0]
    idx_hi = [sd for sd in subdomains if sd["ijk"][ax_i] == 1]
    if not idx_hi:
        raise SystemExit(
            f"the decomposition has a single slab along {args.axis}; there is "
            f"no artificial plane normal to it to plot.")

    cut = max(float(sd["partition_solver"].V.tabulate_dof_coordinates()
                    .reshape((-1, 3))[:, ax_i].max()) for sd in idx_lo)

    # Spacing comes from the MESHED cube, which is what the dofs actually live
    # on -- not from the cylinder's tight bbox.
    n_min, n_max = enclosing_cube(args.radius, args.height, args.cyl_axis)
    spacing = (n_max - n_min) / args.n
    tol = 0.25 * spacing
    print(f"cut plane: {args.axis} = {cut:.6f}  (tol {tol:.2e})")
    print(f"boxes touching it: {len(idx_lo)} below, {len(idx_hi)} above")

    gsel = plane_dofs(coords_g, ax_i, cut, tol)
    if gsel.size == 0:
        raise SystemExit("no global dofs on the cut plane -- wrong -axis?")
    print(f"global dofs on the plane: {gsel.size}")

    # Exterior dofs carry no physical value; drop them from the colour scale.
    ext_g = np.zeros(V.dim(), dtype=bool)
    if getattr(solver, "ext_dofs", None) is not None:
        ext_g[np.asarray(solver.ext_dofs, dtype=int)] = True
    keep = ~ext_g[gsel]
    gsel, gvals = gsel[keep], u_star[gsel][keep]

    norm = Normalize(vmin=float(gvals.min()), vmax=float(gvals.max()))
    cmap = "viridis"
    ecmap = "inferno"

    # ---- gather the same plane from every box that touches it --------------
    def collect(field_key):
        pts, vals, ref = [], [], []
        for sd in idx_lo + idx_hi:
            ps = sd["partition_solver"]
            l2g = sd["local_to_global_dof"]
            c = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
            u = (sd["_u_raw"] if field_key == "raw"
                 else sd["u3d_partition_schwarz"].vector().get_local())
            s = plane_dofs(c, ax_i, cut, tol)
            ext = getattr(ps, "ext_dofs", None)
            if ext is not None and np.size(ext):
                live = np.ones(ps.V.dim(), dtype=bool)
                live[np.asarray(ext, dtype=int)] = False
                s = s[live[s]]
            pts.append(c[s])
            vals.append(u[s])
            ref.append(u_star[l2g[s]])
        return np.vstack(pts), np.concatenate(vals), np.concatenate(ref)

    p_raw, v_raw, r_raw = collect("raw")
    p_sch, v_sch, r_sch = collect("schwarz")

    e_raw = np.abs(v_raw - r_raw)
    e_sch = np.abs(v_sch - r_sch)

    enorm = Normalize(vmin=0.0,
                      vmax=float(max(e_raw.max(), e_sch.max(), 1e-30)))

    # ---- draw ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    s0 = surface_plane(axes[0], coords_g[gsel], gvals, ax_i, norm, cmap,
                       f"global $u^*$  on {args.axis} = {cut:.2f}")
    fig.colorbar(s0, ax=axes[0], shrink=0.85, label="pressure")

    lbl = f"Schwarz/Robin ({n_sweeps} sweeps)"
    # Each error panel also reports the RECONSTRUCTED GLOBAL error of the field
    # it comes from. The panel itself only shows the cut plane, so without this
    # the reader cannot tell whether a visually better wall corresponds to a
    # better solution overall -- the two need not move together.
    rel_glob_raw = float(subdomains[0]["u3d_partition_full_error_raw_rel_l2"])
    surface_plane(axes[1], p_raw, e_raw, ax_i, enorm, ecmap,
                  f"raw error  $|u_i-u^*|$\n"
                  f"max {e_raw.max():.2e}   "
                  f"global rel $L^2$ {rel_glob_raw:.3e}")
    s2 = surface_plane(axes[2], p_sch, e_sch, ax_i, enorm, ecmap,
                       f"{lbl} error  $|u_i-u^*|$\n"
                       f"max {e_sch.max():.2e}   "
                       f"global rel $L^2$ {rel_global_final:.3e}")
    fig.colorbar(s2, ax=axes[1:], shrink=0.85, label="|error|")

    a_lines, b_lines = subdomain_seams(idx_lo + idx_hi, ax_i)
    print(f"subdomain seams on the plane: {len(a_lines)} + {len(b_lines)}")
    for ax in axes:
        # Freeze the data limits first: axvline/axhline span the full axis and
        # would otherwise let a seam at the rim rescale the panel.
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        draw_subdomain_seams(ax, a_lines, b_lines)

    sub = ", ".join(str(sd["ijk"]) for sd in idx_lo + idx_hi)
    raw_local = float(np.mean(
        [sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains]))
    print(f"domain: cylinder r={args.radius} h={args.height} "
          f"axis={args.cyl_axis}")
    print(f"boxes on the wall: {sub}")
    # Local and global are reported on separate lines and named for what they
    # are: they are different measures and need not move by the same factor.
    print(f"averaged local:  raw {raw_local:.3e}"
          f"  ->  corrected {rel_local_final:.3e}")
    print(f"reconstructed global:  raw {rel_glob_raw:.3e}"
          f"  ->  corrected {rel_global_final:.3e}")

    out = args.out or f"interface_wall_schwarz_{args.name}_{args.axis}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    print(f"  max |error| on the wall: raw {e_raw.max():.4e}  "
          f"-> corrected {e_sch.max():.4e}")
    print(f"  mean|error| on the wall: raw {e_raw.mean():.4e}  "
          f"-> corrected {e_sch.mean():.4e}")
    # The convergence state has no analogue in the one-shot residual method and
    # is the thing to check when the corrected wall disappoints: a run that
    # stopped on max_iter rather than tol has not converged.
    print(f"  Schwarz: rho = {rho_used:.4e}, {n_sweeps} sweeps, "
          f"final delta {delta_final:.4e} (tol {args.tol:.1e}"
          f"{', HIT max_iter' if n_sweeps >= args.max_iter else ''})")


if __name__ == "__main__":
    main()
