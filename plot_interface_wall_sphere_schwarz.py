"""Plot the field on the artificial cut plane, global vs subdomains -- Schwarz.

Companion to plot_interface_wall_sphere.py. Same plane, same panels, same
colour rules; the only thing that changes is WHICH transmission scheme
produced the corrected field:

    plot_interface_wall_sphere.py          Robin residual, eqs. (4)-(5), one shot
    this file                              additive Schwarz / Robin, iterative

The distinction matters for what the picture means. The residual method
evaluates eq. (4) at the ground-truth u*, so its corrected wall is informed by
the answer. The Schwarz iteration in Solve_schwarz_robin never touches u* --
each sweep feeds a box the neighbours' PREVIOUS traces -- so the wall shown
here is reachable without knowing the target. A residual panel that looks
better is therefore not automatically the better scheme.

Three panels, the two error ones on the SAME colour scale:

    global      u* restricted to the plane -- the target
    raw         the two subdomains touching the plane, solved in isolation
    schwarz     the same two after the converged Schwarz iteration

The plane is shared by more than two boxes in general: at fixed y it is tiled
by every (i, k) pair. The wall is therefore split into num_x * num_z pieces per
side, and the script draws each piece from the box that owns it, leaving the
seams visible -- those seams are the other artificial interfaces.

There is no -cross here: the cross term is a correction to the eq. (4)
extraction, and the Schwarz scheme forms no such residual. The knobs that
replace it are -rho, -max_iter and -tol.

Run (reuses the saved global solution, same resolution rules as the
residual plotter):

    python plot_interface_wall_sphere_schwarz.py -name robing_dd
    python plot_interface_wall_sphere_schwarz.py -name robing_dd -rho 2.9e-2
    python plot_interface_wall_sphere_schwarz.py -name robing_dd -axis y -out wall.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from dolfin import Function

from Decompose_Domain_Analytic_sphere import (
    DISCRETIZATION_POINT_SUB_DOMAIN,
    LENGTH_SUB,
    check_sphere_domain_consistency,
    decomposeDomain,
)
from Solver_full_domain import Solver3D1D
from Boundary import SphereBoundary, random_sphere_points
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
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[1])
    ap.add_argument("-name", type=str, default="final_r_dd_2")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("-sigma1d", type=float, default=1.0)
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-kappa", type=float, default=1.0)
    ap.add_argument("-radius", type=float, default=5.0)
    # Schwarz knobs. Unlike the residual method, rho defaults to None so
    # solve_schwarz_robin applies its own sigma3d/hmax heuristic; the measured
    # optimum on this problem is ~10x that, see Solve_schwarz_robin's -rho help.
    ap.add_argument("-rho", type=float, default=None,
                    help="Robin penalty; default sigma3d/hmax (~2.9e-3). "
                         "Measured optimum on this problem: ~2.9e-2")
    ap.add_argument("-max_iter", type=int, default=60,
                    help="maximum Schwarz sweeps")
    ap.add_argument("-tol", type=float, default=1e-8,
                    help="stop when the largest iterate change falls below this")
    ap.add_argument("-axis", choices=("x", "y", "z"), default="y",
                    help="normal of the cut plane to draw")
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

    # ---- rebuild operators, reuse or recompute the global field ------------
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

    # Built exactly as Decompose_Domain_Analytic_sphere's driver builds it:
    # lenght_sub_domain/n_sub make the solver size the background cube from the
    # decomposition (see Solver3D1D._load_meshes), so the mesh solved here is
    # the one the decomposition below will split. Omitting them would solve on
    # a different -- unaugmented -- mesh whose cut planes miss the grid.
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix, boundary=boundary, n=args.n,
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
    solver.save(args.solution)

    subdomains = decomposeDomain(
        solver, boundary, restrict_global_C=args.restrict_global_C)

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

    # The plane is the boundary between consecutive slabs along `axis`. Take it
    # from the boxes themselves rather than assuming the midpoint: it is the
    # max of the low slab's extent, which is also the min of the high slab's.
    idx_lo = [sd for sd in subdomains if sd["ijk"][ax_i] == 0]
    idx_hi = [sd for sd in subdomains if sd["ijk"][ax_i] == 1]
    if not idx_hi:
        raise SystemExit(
            f"the decomposition has a single slab along {args.axis}; there is "
            f"no artificial plane normal to it to plot.")

    cut = max(float(sd["partition_solver"].V.tabulate_dof_coordinates()
                    .reshape((-1, 3))[:, ax_i].max()) for sd in idx_lo)
    spacing = 2.0 * args.radius / args.n
    tol = 0.25 * spacing
    print(f"cut plane: {args.axis} = {cut:.6f}  (tol {tol:.2e})")
    print(f"boxes touching it: {len(idx_lo)} below, {len(idx_hi)} above")

    gsel = plane_dofs(coords_g, ax_i, cut, tol)
    if gsel.size == 0:
        raise SystemExit("no global dofs on the cut plane -- wrong -axis?")
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
    # Errors get a sequential map that reads as "zero is good", distinct from
    # the pressure map so the two panel types are not confused at a glance.
    ecmap = "inferno"

    # ---- gather the same plane from every box that touches it --------------
    # Also returns u* at the SAME dofs, via local_to_global_dof, so the error
    # panels compare like with like rather than interpolating.
    # `owner` records which box each plane point came from, so the wall error
    # can be broken down per subdomain instead of pooled into one array.
    def collect(field_key):
        pts, vals, ref, owner = [], [], [], []
        for b, sd in enumerate(idx_lo + idx_hi):
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
            owner.append(np.full(s.size, b, dtype=int))
        return (np.vstack(pts), np.concatenate(vals), np.concatenate(ref),
                np.concatenate(owner))

    p_raw, v_raw, r_raw, o_raw = collect("raw")
    p_sch, v_sch, r_sch, o_sch = collect("schwarz")

    e_raw = np.abs(v_raw - r_raw)
    e_sch = np.abs(v_sch - r_sch)

    # The two error panels share a scale so the before/after is readable; the
    # global panel keeps its own, since it is a pressure and they are errors.
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
    # The two error panels share `enorm`, so their colours are comparable.
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

    out = args.out or f"interface_wall_schwarz_{args.name}_{args.axis}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # ---- report -------------------------------------------------------------
    # Everything below is report material: the error ON the cut plane and the
    # error IN each subdomain, raw vs corrected, in one place.
    boxes = idx_lo + idx_hi
    raw_all = np.array([sd["u3d_partition_error_local_raw_rel_l2"]
                        for sd in subdomains])
    sch_all = np.array([sd["u3d_partition_error_schwarz_rel_l2"]
                        for sd in subdomains])

    def _rel(err, refv):
        d = float(np.linalg.norm(refv))
        return float(np.linalg.norm(err)) / d if d > 1e-30 else float("nan")

    def _gain(a, b):
        return a / b if b > 1e-30 else float("inf")

    tag = f"Schwarz/Robin, {n_sweeps} sweeps"
    bar = "=" * 78

    print(f"\n{bar}\nINTERFACE ERROR  --  cut plane {args.axis} = {cut:.4f}"
          f"   ({tag}, rho = {rho_used:g})\n{bar}")
    print(f"points on the wall: {e_raw.size}  "
          f"(from {len(boxes)} boxes, {len(idx_lo)} below / {len(idx_hi)} above)")
    print(f"{'':14s}{'raw':>13s}{'corrected':>13s}{'gain':>9s}")
    for label, a, b in (
            ("max |e|", e_raw.max(), e_sch.max()),
            ("mean |e|", e_raw.mean(), e_sch.mean()),
            ("median |e|", float(np.median(e_raw)), float(np.median(e_sch))),
            ("p95 |e|", float(np.percentile(e_raw, 95)),
             float(np.percentile(e_sch, 95))),
            ("L2 |e|", float(np.linalg.norm(e_raw)),
             float(np.linalg.norm(e_sch))),
            ("rel L2", _rel(e_raw, r_raw), _rel(e_sch, r_sch)),
    ):
        print(f"{label:14s}{a:13.4e}{b:13.4e}{_gain(a, b):9.2f}x")

    print(f"\nper-box error ON the wall (relative L2 of the plane slice):")
    print(f"{'box':>12s}{'pts':>7s}{'raw':>13s}{'corrected':>13s}{'gain':>9s}"
          f"{'max raw':>12s}{'max corr':>12s}")
    for b, sd in enumerate(boxes):
        mr, mc = o_raw == b, o_sch == b
        if not mr.any():
            continue
        rr, rc = _rel(e_raw[mr], r_raw[mr]), _rel(e_sch[mc], r_sch[mc])
        print(f"{str(sd['ijk']):>12s}{int(mr.sum()):7d}{rr:13.4e}{rc:13.4e}"
              f"{_gain(rr, rc):9.2f}x{e_raw[mr].max():12.4e}"
              f"{e_sch[mc].max():12.4e}")

    print(f"\n{bar}\nSUBDOMAIN ERROR  --  full volume, all {len(subdomains)} boxes"
          f"\n{bar}")
    print(f"{'box':>12s}{'raw':>13s}{'corrected':>13s}{'gain':>9s}"
          f"{'on wall':>9s}")
    on_wall = {id(sd) for sd in boxes}
    for sd in sorted(subdomains,
                     key=lambda s: -s["u3d_partition_error_schwarz_rel_l2"]):
        r0 = sd["u3d_partition_error_local_raw_rel_l2"]
        r1 = sd["u3d_partition_error_schwarz_rel_l2"]
        print(f"{str(sd['ijk']):>12s}{r0:13.4e}{r1:13.4e}{_gain(r0, r1):9.2f}x"
              f"{('yes' if id(sd) in on_wall else '-'):>9s}")
    print(f"{'mean':>12s}{raw_all.mean():13.4e}{sch_all.mean():13.4e}"
          f"{_gain(raw_all.mean(), sch_all.mean()):9.2f}x")
    print(f"{'min':>12s}{raw_all.min():13.4e}{sch_all.min():13.4e}")
    print(f"{'max':>12s}{raw_all.max():13.4e}{sch_all.max():13.4e}")

    # The convergence history has no analogue in the one-shot residual method
    # and is the thing to look at when the corrected wall disappoints: a run
    # that stopped on max_iter rather than tol has not converged.
    print(f"\nSchwarz convergence:  rho = {rho_used:.4e}, "
          f"{n_sweeps} sweeps, final delta {delta_final:.4e} "
          f"(tol {args.tol:.1e}"
          f"{', HIT max_iter' if n_sweeps >= args.max_iter else ''})")
    print(f"reconstructed global:  corrected {rel_global_final:.4e}"
          f"   (raw {rel_glob_raw:.4e})")
    print(f"averaged local:  raw {raw_all.mean():.4e}"
          f"  ->  corrected {rel_local_final:.4e}")
    print(bar)


if __name__ == "__main__":
    main()
