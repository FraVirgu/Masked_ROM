"""Plot the field on an artificial cut plane of a CYLINDER+SPHERE domain.

Companion to plot_interface_wall_cylinder.py. The drawing itself does not care
what shape the domain is -- plane_dofs, surface_plane, subdomain_seams and
draw_subdomain_seams all work off scattered dof coordinates -- so those are
IMPORTED from the sphere script and only `main` is rewritten, to build a
CylinderSphereBoundary and to size the mesh from the union's own extents.

The domain is the set union of a finite cylinder and a sphere sitting on one of
its caps, so it is the first domain here that is NOT symmetric about its centre.
That matters in two places, both handled in Decompose_Domain_Analytic_cylinder_sphere:
the enclosing cube is built about the hull's midpoint rather than the origin,
and the consistency check is per-axis.

Three panels, the two error ones on a shared colour scale:

    global      u* restricted to the plane -- the target
    raw         the boxes touching the plane, solved in isolation
    robin       the same boxes after eqs. (4)-(5)

Red dashed lines mark the seams between subdomains on the plane.

Run:

    python plot_interface_wall_cylinder_sphere.py -name cylsph_dd
    python plot_interface_wall_cylinder_sphere.py -name cylsph_dd -cross
    python plot_interface_wall_cylinder_sphere.py -name cylsph_dd -sphere_radius 1.5
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from Analytic_Domain import Domain
from dolfin import Function, XDMFFile

from Decompose_Domain_Analytic_cylinder_sphere import (
    DISCRETIZATION_POINT_SUB_DOMAIN,
    LENGTH_SUB,
    check_cylinder_sphere_domain_consistency,
    cylinder_sphere_background_mesh,
    cylinder_sphere_bbox,
    decomposeDomain,
    enclosing_cube,
)
from Solver_full_domain import Solver3D1D
from Boundary import CylinderSphereBoundary, random_cylinder_sphere_points
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
            description="Build a simple analytic (non-OpenCCO) vascular domain on "
                        "the UNION of a cylinder and a sphere and solve the 3D-1D "
                        "problem with the no-penalty solver.",
        )
    parser.add_argument("-name",   type=str, required=True,
                        help="subfolder name inside nets/ (e.g. cylsph01)")
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
    parser.add_argument("-radius", type=float, default=2.0,
                        help="radius of the cylindrical part")
    parser.add_argument("-height", type=float, default=2.0,
                        help="total extent of the cylinder along its axis")
    parser.add_argument("-sphere_radius", type=float, default=None,
                        help="radius of the sphere lobe (default: -radius)")
    parser.add_argument("-sphere_offset", type=float, default=None,
                        help="sphere centre along the axis, from the cylinder "
                             "centre (default: half the height, i.e. on the cap)")

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
    # end in '_', and path_to_1D_mesh MUST -- otherwise the underscores don't line
    # up and the solver looks for a file the domain never wrote.
    net_dir = os.path.join("nets", args.name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, args.name)   # nets/NAME/NAME      (no trailing _)
    mesh_prefix = f"{name_stem}_"                   # nets/NAME/NAME_     (solver prefix)

    # Inlets in the far half of the tube, outlets in the sphere lobe, so the
    # vasculature runs the length of the union and crosses the junction. A split
    # that kept both sets on one side would leave the artificial planes near the
    # junction carrying almost no field, and the correction nothing to correct.
    sampler_kw = dict(
        radius=args.radius, height=args.height,
        sphere_radius=args.sphere_radius, sphere_offset=args.sphere_offset,
        axis=args.cyl_axis, min_offset=0.2, min_dist_to_boundary=0.06,
    )
    boundary = CylinderSphereBoundary(
        radius=args.radius,
        height=args.height,
        sphere_radius=args.sphere_radius,
        sphere_offset=args.sphere_offset,
        axis=args.cyl_axis,
        inlet_points=random_cylinder_sphere_points(40, sign=-1, **sampler_kw),
        outlet_points=random_cylinder_sphere_points(40, sign=+1, **sampler_kw),
        border_eps=10e-1,
    )
    bounds = cylinder_sphere_bbox(
        args.radius, args.height, args.sphere_radius, args.sphere_offset,
        args.cyl_axis)
    check_cylinder_sphere_domain_consistency(boundary=boundary, bounds=bounds)

    # --- 1. build the analytic-boundary domain --------------------------------
    n_min, n_max = enclosing_cube(
        args.radius, args.height, args.sphere_radius, args.sphere_offset,
        args.cyl_axis)
    print(f"cylinder r={args.radius} h={args.height} axis={args.cyl_axis} "
          f"+ sphere r={boundary.sphere_radius} at offset "
          f"{boundary.sphere_offset}  ->  background cube "
          f"[{n_min}, {n_max}]^3 at n={args.n}")

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
    # full_domain_mesh is what keeps the union a union (the solver's own default
    # meshes the tight hull and silently solves a BOX), and
    # lenght_sub_domain/n_sub let the solver size that cube from the
    # decomposition, so the mesh solved here is the one the decomposition below
    # will split.
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix,
        full_domain_mesh=cylinder_sphere_background_mesh(
            args.radius, args.height, args.n, args.sphere_radius,
            args.sphere_offset, args.cyl_axis),
        boundary=boundary, n=args.n,
        sigma3d=args.sigma3d, sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet",
        lenght_sub_domain=LENGTH_SUB,
        n_sub=DISCRETIZATION_POINT_SUB_DOMAIN).build().solve()

    # The global solve is ALWAYS recomputed, never loaded from disk. The
    # background mesh depends on LENGTH_SUB and DISCRETIZATION_POINT_SUB_DOMAIN
    # (Solver3D1D sizes the cube from them), so a solution.npy written by an
    # earlier run may belong to a different mesh. Its size alone does not prove
    # otherwise -- two different decompositions can produce the same dof count --
    # so reusing it risks drawing one mesh's field on another mesh's geometry.
    out_dir = os.path.join(
        "solution",
        f"Simple{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")
    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")

    # LENGTH_SUB is the single source of truth: the solver has just sized the
    # background cube to a multiple of LENGTH_SUB, so decomposing on any other
    # length would split a mesh built for this one and put the cut planes off
    # the grid again.
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

    # Post-method subdomain fields, one file per box. This is u3d_robin -- the
    # local solve AFTER the Robin transmission term, not partition_solver.u3d,
    # which is the isolated solve with no communication.
    sub_dir = os.path.join(out_dir, "subdomain")
    os.makedirs(sub_dir, exist_ok=True)
    for sd in subdomains:
        i, j, k = sd["ijk"]
        fn = sd["u3d_robin"]
        fn.rename("u3d_robin", "3D pressure after Robin transmission")
        with XDMFFile(f"{sub_dir}/u3d_robin_{i}_{j}_{k}.xdmf") as f:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            f.write(fn)
    print(f"Subdomain solutions saved → {sub_dir} "
          f"({len(subdomains)} boxes)")

    # ---- draw one cut plane -------------------------------------------------
    # Called once per axis. Everything expensive -- the global solve and the
    # decomposition -- has already happened above and is shared by the three
    # calls; only the plane selection and the drawing repeat.
    def draw_axis(axis_name):
        ax_i = {"x": 0, "y": 1, "z": 2}[axis_name]
        print(f"\n=== cut plane normal to {axis_name} ===")
        # ---- locate the cut plane ------------------------------------------
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

        # Spacing comes from the MESHED cube, which is what the dofs actually
        # live on -- not from the union's tight hull.
        n_min, n_max = enclosing_cube(
            args.radius, args.height, args.sphere_radius, args.sphere_offset,
            args.cyl_axis)
        spacing = (n_max - n_min) / args.n
        tol = 0.25 * spacing
        print(f"cut plane: {axis_name} = {cut:.6f}  (tol {tol:.2e})")
        print(f"boxes touching it: {len(idx_lo)} below, {len(idx_hi)} above")

        gsel = plane_dofs(coords_g, ax_i, cut, tol)
        if gsel.size == 0:
            print(f"  skipping {axis_name}: no global dofs on the cut plane.")
            return
        print(f"global dofs on the plane: {gsel.size}")

        # Exterior dofs are KEPT and drawn. The Dirichlet elimination pins them
        # to u = 0, so that is their value -- it is background, not missing
        # data. They are still excluded from the colour SCALE, since a large
        # block of zeros would stretch it and flatten the contrast over the
        # region that carries the solution. On this domain that block is large:
        # the union fills much less of its enclosing cube than a sphere does.
        ext_g = np.zeros(V.dim(), dtype=bool)
        if getattr(solver, "ext_dofs", None) is not None:
            ext_g[np.asarray(solver.ext_dofs, dtype=int)] = True
        gvals = u_star[gsel]
        interior = gvals[~ext_g[gsel]]
        scale = interior if interior.size else gvals

        norm = Normalize(vmin=float(scale.min()), vmax=float(scale.max()))
        cmap = "viridis"
        ecmap = "inferno"

        # ---- gather the same plane from every box that touches it ----------
        def collect(field_key):
            pts, vals, ref = [], [], []
            for sd in idx_lo + idx_hi:
                ps = sd["partition_solver"]
                l2g = sd["local_to_global_dof"]
                c = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
                u = (sd["_u_raw"] if field_key == "raw"
                     else sd["u3d_robin"].vector().get_local())
                # Exterior dofs kept for the same reason as the global panel.
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

        # ---- draw -----------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

        s0 = surface_plane(axes[0], coords_g[gsel], gvals, ax_i, norm, cmap,
                           f"global $u^*$  on {axis_name} = {cut:.2f}")
        fig.colorbar(s0, ax=axes[0], shrink=0.85, label="pressure")

        lbl = "eqs. (4)-(5)" + (" + cross" if args.cross else "")
        # Each error panel also reports the RECONSTRUCTED GLOBAL error of the
        # field it comes from. The panel itself only shows the cut plane, so
        # without this the reader cannot tell whether a visually better wall
        # corresponds to a better solution overall -- the two need not move
        # together.
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
        # Frame the whole DOMAIN, not just the coloured data -- see the cylinder
        # script. It matters more here: the union covers even less of its
        # enclosing cube, so fitting the axes to the data would crop the panel
        # to the lobe or the tube depending on where the cut falls, and the
        # three panels would not share a frame.
        a_dim, b_dim = [d for d in (0, 1, 2) if d != ax_i]
        dom_lo, dom_hi = coords_g.min(axis=0), coords_g.max(axis=0)
        for ax in axes:
            ax.set_xlim(dom_lo[a_dim], dom_hi[a_dim])
            ax.set_ylim(dom_lo[b_dim], dom_hi[b_dim])
            draw_subdomain_seams(ax, a_lines, b_lines)

        sub = ", ".join(str(sd["ijk"]) for sd in idx_lo + idx_hi)
        print(f"domain: cylinder r={args.radius} h={args.height} "
              f"axis={args.cyl_axis} + sphere r={boundary.sphere_radius} "
              f"offset={boundary.sphere_offset}")
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
