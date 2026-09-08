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
    DISCRETIZATION_POINT_SUB_DOMAIN,
    LENGTH_SUB,
    check_cylinder_domain_consistency,
    cylinder_background_mesh,
    cylinder_bbox,
    decomposeDomain,
    enclosing_cube,
)
from Analytic_Domain import Domain
from Solver_full_domain import Solver3D1D
from Boundary import CylinderBoundary, random_cylinder_points
from Solve_schwarz_robin import solve_schwarz_robin



def plane_dofs(coords, axis, value, tol):
    """Indices of dofs lying on the plane {x_axis = value}."""
    return np.flatnonzero(np.abs(coords[:, axis] - value) < tol)


def subdomain_seams(subdomains, axis):
    """In-plane coordinates of the artificial cuts between boxes.

    The plane normal to `axis` is tiled by every box at fixed index along that
    axis, so the seams on it are the box edges in the two OTHER directions.
    The extents are read back from each box's own mesh rather than recomputed
    from *_ROM_lenght: decomposeDomain snaps them to mesh nodes, so the stored
    bounds are the only ones that match the drawn field.

    Returns (a_lines, b_lines): seam positions along the two in-plane axes.
    Outermost bounds are dropped -- those are the domain rim, not a cut.
    """
    a, b = [d for d in (0, 1, 2) if d != axis]
    edges = {a: set(), b: set()}
    for sd in subdomains:
        c = (sd["partition_solver"].V.tabulate_dof_coordinates()
             .reshape((-1, 3)))
        for d in (a, b):
            edges[d].add(round(float(c[:, d].min()), 9))
            edges[d].add(round(float(c[:, d].max()), 9))
    # An outer bound appears only as a global extreme; interior seams are the
    # rest, i.e. every box edge that is not the overall min or max.
    out = []
    for d in (a, b):
        e = sorted(edges[d])
        out.append(e[1:-1] if len(e) > 2 else [])
    return out[0], out[1]


def draw_subdomain_seams(ax, a_lines, b_lines):
    """Overlay the box-to-box interfaces as faint red dashed lines.

    Kept deliberately subdued: the seams are an annotation on the field, not
    the subject, so they sit at low alpha and thin. zorder still puts them
    above the tripcolor mesh -- gouraud shading would otherwise hide them
    entirely -- but the low alpha lets the colour read through.
    """
    style = dict(color="red", linestyle="--", linewidth=0.8, alpha=0.35,
                 zorder=5)
    for v in a_lines:
        ax.axvline(v, **style)
    for v in b_lines:
        ax.axhline(v, **style)


def surface_plane(ax, coords, vals, axis, norm, cmap, title):
    """Draw the cut plane as a filled 2D field, seen face-on.

    The two in-plane axes are the ones that are not `axis` -- for the default
    y-cut, the x-z face. The field is interpolated across the mesh triangles
    (Gouraud) rather than drawn as discrete markers, so the whole surface
    carries colour.

    Triangulated rather than gridded: the sphere makes the plane a disc, not a
    rectangle, and the boxes contribute overlapping dof sets along their shared
    edges. Delaunay over the scattered in-plane points handles both. Triangles
    spanning much more than the mesh spacing are masked out, which stops the
    triangulation from bridging the concave outer rim and painting colour
    outside the physical domain.
    """
    from matplotlib.tri import Triangulation

    a, b = [d for d in (0, 1, 2) if d != axis]
    x, y = coords[:, a], coords[:, b]

    tri = Triangulation(x, y)
    xt, yt = x[tri.triangles], y[tri.triangles]
    sides = np.sqrt(np.diff(np.column_stack([xt, xt[:, :1]]), axis=1) ** 2
                    + np.diff(np.column_stack([yt, yt[:, :1]]), axis=1) ** 2)
    step = np.median(sides[sides > 0]) if np.any(sides > 0) else 1.0
    tri.set_mask(sides.max(axis=1) > 2.5 * step)

    surf = ax.tripcolor(tri, vals, cmap=cmap, norm=norm, shading="gouraud")
    ax.set_aspect("equal")
    ax.set_xlabel("xyz"[a])
    ax.set_ylabel("xyz"[b])
    ax.set_title(title, fontsize=10)
    return surf



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
    parser.add_argument("-rho", type=float, default=None,
                    help="Robin penalty; default sigma3d/hmax")
    parser.add_argument("-cross", action="store_true",
                        help="add the 1D cross term for vessels straddling a cut")
    parser.add_argument("-restrict_global_C", action="store_true")
    parser.add_argument("-out", type=str, default=None,
                        help="output png stem; one file per axis is written "
                             "as {stem}_{axis}.png (default: "
                             "interface_wall_schwarz_{name}_{axis}.png)")
    # Schwarz knobs; rho defaults to None so solve_schwarz_robin applies its
    # own sigma3d/hmax heuristic.
    
    parser.add_argument("-max_iter", type=int, default=60,
                        help="maximum Schwarz sweeps")
    parser.add_argument("-tol", type=float, default=1e-8,
                        help="stop when the largest iterate change falls below this")
    
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
                     else sd["u3d_partition_schwarz"].vector().get_local())
                # Exterior dofs kept for the same reason as the global panel: they
                # are pinned to 0 by the elimination, so they are background the
                # error panels should show as zero error, not holes in the mesh.
                s = plane_dofs(c, ax_i, cut, tol)
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
                           f"global $u^*$  on {axis_name} = {cut:.2f}")
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

        stem = args.out or f"interface_wall_schwarz_{args.name}"
        out = f"{stem}_{axis_name}.png"
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

    # All three cuts, always: the decomposition is the same for each, so the
    # only cost is the drawing, and the three planes together show the whole
    # decomposition rather than one arbitrary view of it.
    for axis_name in ("x", "y", "z"):
        draw_axis(axis_name)


if __name__ == "__main__":
    main()
