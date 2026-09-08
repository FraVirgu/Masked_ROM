"""Plot the field on an artificial cut plane of a CYLINDRICAL domain.

Companion to plot_interface_wall_sphere.py. The drawing itself does not care
what shape the domain is -- plane_dofs, surface_plane, subdomain_seams and
draw_subdomain_seams all work off scattered dof coordinates -- so those are
IMPORTED from the sphere script and only `main` is rewritten, to build a
CylinderBoundary and to size the mesh spacing from the cylinder's own extents.

Three panels, the two error ones on a shared colour scale:

    global      u* restricted to the plane -- the target
    raw         the boxes touching the plane, solved in isolation
    robin       the same boxes after eqs. (4)-(5)

Red dashed lines mark the seams between subdomains on the plane.

Run (reuses the saved global solution):

    python plot_interface_wall_cylinder.py -name cylinder_dd
    python plot_interface_wall_cylinder.py -name cylinder_dd -cross
    python plot_interface_wall_cylinder.py -name cylinder_dd -axis z -out wall.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from Analytic_Domain import Domain
from dolfin import Function

from Decompose_Domain_Analytic_cylinder import (
    DISCRETIZATION_POINT_SUB_DOMAIN,
    LENGTH_SUB,
    check_cylinder_domain_consistency,
    cylinder_background_mesh,
    cylinder_bbox,
    decomposeDomain,
    enclosing_cube,
)
from Solver_full_domain import Solver3D1D
from Boundary import CylinderBoundary, random_cylinder_points
from Robin_residual_sphere import apply_robin_residual

# Geometry-agnostic drawing helpers, reused verbatim.
from plot_interface_wall_sphere import (
    draw_subdomain_seams,
    plane_dofs,
    subdomain_seams,
    surface_plane,
)


def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Build a simple analytic (non-OpenCCO) vascular domain on a "
                        "CYLINDER and solve the 3D-1D problem with the no-penalty "
                        "solver.",
        )
    parser.add_argument("-name",   type=str, required=True,
                        help="subfolder name inside nets/ (e.g. cylinder01)")
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
                        help="radius of the cylindrical boundary")
    parser.add_argument("-height", type=float, default=3.0,
                        help="total extent of the cylinder along its axis")

    parser.add_argument("-cyl_axis", choices=("x", "y", "z"), default="z",
                        help="direction of the cylinder axis (the geometry)")
    parser.add_argument("-rho", type=float, default=1.0,
                        help="Robin penalty for eqs. (4)-(5)")
    parser.add_argument("-cross", action="store_true",
                        help="add the 1D cross term for vessels straddling a cut")
    parser.add_argument("-restrict_global_C", action="store_true")
    parser.add_argument("-out", type=str, default=None,
                        help="output png stem; one file per axis is written "
                             "as {stem}_{axis}.png "
                             "(default: interface_wall_{name}_{axis}.png)")
    args = parser.parse_args()

    # nets/{name}/ holds every mesh file this run writes and the solver reads.
    #
    # Domain's export_* methods build filenames as f"{self.name}_marked_mesh.xdmf"
    # (i.e. they add the trailing '_'), while the solver reads them back as
    # f"{path_to_1D_mesh}marked_mesh.xdmf" (no added '_'). So self.name must NOT
    # end in '_', and path_to_1D_mesh MUST — otherwise the underscores don't line
    # up and the solver looks for a file the domain never wrote.
    net_dir = os.path.join("nets", args.name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, args.name)   # nets/NAME/NAME      (no trailing _)
    mesh_prefix = f"{name_stem}_"                   # nets/NAME/NAME_     (solver prefix)
    

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

    # --- 1. build the analytic-boundary domain --------------------------------
    n_min, n_max = enclosing_cube(args.radius, args.height, args.cyl_axis)
    print(f"cylinder r={args.radius} h={args.height} axis={args.cyl_axis}  ->  "
            f"background cube [{n_min}, {n_max}]^3 at n={args.n}")

    domain = Domain(
        name            = name_stem,
        n_vasi          = args.inlet,
        n_ramifications = args.outlet,
        boundary        = boundary,
    ).build()

    domain.export_box()
    domain.export_reticolo()
    domain.export_vaso()
    domain.export_xdmf()

    # Same enclosing-cube mesh the solution was computed on; see the driver.
    # Built exactly as Decompose_Domain_Analytic_cylinder's driver builds it.
    # full_domain_mesh is what keeps the cylinder a cylinder (the solver's own
    # default meshes the tight slab and silently solves a BOX), and
    # lenght_sub_domain/n_sub let the solver size that cube from the
    # decomposition, so the mesh solved here is the one the decomposition
    # below will split.
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix,
        full_domain_mesh=cylinder_background_mesh(
            args.radius, args.height, args.n, args.cyl_axis),
        boundary=boundary, n=args.n,
        sigma3d=args.sigma3d, sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet",
        lenght_sub_domain=LENGTH_SUB,
        n_sub=DISCRETIZATION_POINT_SUB_DOMAIN).build().solve()

    # The global solve is ALWAYS recomputed, never loaded from disk. The
    # background mesh now depends on LENGTH_SUB and
    # DISCRETIZATION_POINT_SUB_DOMAIN (Solver3D1D sizes the cube from them), so
    # a solution.npy written by an earlier run may belong to a different mesh.
    # Its size alone does not prove otherwise -- two different decompositions
    # can produce the same dof count -- so reusing it risks drawing one mesh's
    # field on another mesh's geometry. Solving here is the expensive step and
    # is the price of that guarantee.
    # Same layout the Decompose_Domain_Analytic_cylinder driver writes to,
    # so a run from either entry point lands in one place. Derived here
    # rather than taken from a flag: the mesh depends on LENGTH_SUB and
    # n_sub, so the directory name has to follow the run, not the caller.
    out_dir = os.path.join(
        "solution",
        f"Simple{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")
    solver.save(out_dir)

    # LENGTH_SUB is the single source of truth: the solver has just sized
    # the background cube to a multiple of LENGTH_SUB, so decomposing on any
    # other length would split a mesh that was built for this one and put the
    # cut planes off the grid again.
    subdomains = decomposeDomain(
        solver, boundary,
        x_ROM_lenght=LENGTH_SUB, y_ROM_lenght=LENGTH_SUB,
        z_ROM_lenght=LENGTH_SUB,
        restrict_global_C=args.restrict_global_C or args.cross)

    # Raw local fields must be captured BEFORE apply_robin_residual.
    for sd in subdomains:
        sd["_u_raw"] = sd["partition_solver"].u3d.vector().get_local().copy()

    result = apply_robin_residual(
        subdomains, solver, rho_robin=args.rho, cross=args.cross)

    # ---- draw one cut plane -------------------------------------------------
    # Called once per axis. Everything expensive -- the global solve and the
    # decomposition -- has already happened above and is shared by the three
    # calls; only the plane selection and the drawing repeat.
    def draw_axis(axis_name):
        ax_i = {"x": 0, "y": 1, "z": 2}[axis_name]
        print(f"\n=== cut plane normal to {axis_name} ===")
        # ---- locate the cut plane ----------------------------------------------
        V = solver.W[0]
        coords_g = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
        u_star = solver.u3d.vector().get_local()

        idx_lo = [sd for sd in subdomains if sd["ijk"][ax_i] == 0]
        idx_hi = [sd for sd in subdomains if sd["ijk"][ax_i] == 1]
        if not idx_hi:
            print(f"  skipping {axis_name}: a single slab along it, so there is "
                  f"no artificial plane normal to it.")
            return

        cut = max(float(sd["partition_solver"].V.tabulate_dof_coordinates()
                        .reshape((-1, 3))[:, ax_i].max()) for sd in idx_lo)

        # Spacing comes from the MESHED cube, which is what the dofs actually live
        # on -- not from the cylinder's tight bbox.
        n_min, n_max = enclosing_cube(args.radius, args.height, args.cyl_axis)
        spacing = (n_max - n_min) / args.n
        tol = 0.25 * spacing
        print(f"cut plane: {axis_name} = {cut:.6f}  (tol {tol:.2e})")
        print(f"boxes touching it: {len(idx_lo)} below, {len(idx_hi)} above")

        gsel = plane_dofs(coords_g, ax_i, cut, tol)
        if gsel.size == 0:
            print(f"  skipping {axis_name}: no global dofs on the cut plane.")
            return
        print(f"global dofs on the plane: {gsel.size}")

        # Exterior dofs are KEPT and drawn. The Dirichlet elimination pins them to
        # u = 0, so that is their value -- it is background, not missing data, and
        # dropping it left the triangulation covering only the wetted band and the
        # rest of the panel blank. They are still excluded from the colour SCALE,
        # since a large block of zeros would otherwise stretch it and flatten the
        # contrast over the region that carries the solution.
        ext_g = np.zeros(V.dim(), dtype=bool)
        if getattr(solver, "ext_dofs", None) is not None:
            ext_g[np.asarray(solver.ext_dofs, dtype=int)] = True
        gvals = u_star[gsel]
        interior = gvals[~ext_g[gsel]]
        scale = interior if interior.size else gvals

        norm = Normalize(vmin=float(scale.min()), vmax=float(scale.max()))
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
                     else sd["u3d_robin"].vector().get_local())
                # Exterior dofs kept for the same reason as the global panel: they
                # are pinned to 0 by the elimination, so they are background the
                # error panels should show as zero error, not holes in the mesh.
                s = plane_dofs(c, ax_i, cut, tol)
                pts.append(c[s])
                vals.append(u[s])
                ref.append(u_star[l2g[s]])
            return np.vstack(pts), np.concatenate(vals), np.concatenate(ref)

        p_raw, v_raw, r_raw = collect("raw")
        p_rob, v_rob, r_rob = collect("robin")

        e_raw = np.abs(v_raw - r_raw)
        e_rob = np.abs(v_rob - r_rob)

        enorm = Normalize(vmin=0.0,
                          vmax=float(max(e_raw.max(), e_rob.max(), 1e-30)))

        # ---- draw ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

        s0 = surface_plane(axes[0], coords_g[gsel], gvals, ax_i, norm, cmap,
                           f"global $u^*$  on {axis_name} = {cut:.2f}")
        fig.colorbar(s0, ax=axes[0], shrink=0.85, label="pressure")

        lbl = "eqs. (4)-(5)" + (" + cross" if args.cross else "")
        # Each error panel also reports the RECONSTRUCTED GLOBAL error of the field
        # it comes from. The panel itself only shows the cut plane, so without this
        # the reader cannot tell whether a visually better wall corresponds to a
        # better solution overall -- the two need not move together.
        rel_glob_raw = float(subdomains[0]["u3d_partition_full_error_raw_rel_l2"])
        rel_glob_rob = float(result["rel_global"])
        surface_plane(axes[1], p_raw, e_raw, ax_i, enorm, ecmap,
                      f"raw error  $|u_i-u^*|$\n"
                      f"max {e_raw.max():.2e}   "
                      f"global rel $L^2$ {rel_glob_raw:.3e}")
        s2 = surface_plane(axes[2], p_rob, e_rob, ax_i, enorm, ecmap,
                           f"{lbl} error  $|u_i-u^*|$\n"
                           f"max {e_rob.max():.2e}   "
                           f"global rel $L^2$ {rel_glob_rob:.3e}")
        fig.colorbar(s2, ax=axes[1:], shrink=0.85, label="|error|")

        a_lines, b_lines = subdomain_seams(idx_lo + idx_hi, ax_i)
        print(f"subdomain seams on the plane: {len(a_lines)} + {len(b_lines)}")
        # Frame the whole DOMAIN, not just the coloured data. Exterior dofs are
        # dropped before plotting, so on a domain that does not fill its background
        # cube -- a cylinder of height < 2*radius, say -- the surviving dofs cover
        # only a band of the cut plane. Letting matplotlib fit the axes to that band
        # (and set_aspect('equal') then shrink them) draws a rectangle and hides
        # that the subdomains are equal cubes. Taking the limits from the global
        # mesh instead keeps every panel square when the decomposition is square,
        # and keeps all the subdomain seams visible whether or not they carry data.
        a_dim, b_dim = [d for d in (0, 1, 2) if d != ax_i]
        dom_lo, dom_hi = coords_g.min(axis=0), coords_g.max(axis=0)
        for ax in axes:
            ax.set_xlim(dom_lo[a_dim], dom_hi[a_dim])
            ax.set_ylim(dom_lo[b_dim], dom_hi[b_dim])
            draw_subdomain_seams(ax, a_lines, b_lines)

        sub = ", ".join(str(sd["ijk"]) for sd in idx_lo + idx_hi)
        print(f"domain: cylinder r={args.radius} h={args.height} "
              f"axis={args.cyl_axis}")
        print(f"boxes on the wall: {sub}")
        print(f"global: raw "
              f"{np.mean([sd['u3d_partition_error_local_raw_rel_l2'] for sd in subdomains]):.3e}"
              f"  ->  corrected {result['rel_local']:.3e}"
              f"   (interface support {result['iface_fraction']:.1%})")

        stem = args.out or f"interface_wall_{args.name}"
        out = f"{stem}_{axis_name}.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")
        print(f"  max |error| on the wall: raw {e_raw.max():.4e}  "
              f"-> corrected {e_rob.max():.4e}")
        print(f"  mean|error| on the wall: raw {e_raw.mean():.4e}  "
              f"-> corrected {e_rob.mean():.4e}")

    # All three cuts, always: the decomposition is the same for each, so the
    # only cost is the drawing, and the three planes together show the whole
    # decomposition rather than one arbitrary view of it.
    for axis_name in ("x", "y", "z"):
        draw_axis(axis_name)


if __name__ == "__main__":
    main()
