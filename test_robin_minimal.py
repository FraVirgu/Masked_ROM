"""
Minimal Robin-residual test: unit box, two subdomains, one 2-node vessel.

The full pipeline has many moving parts -- a sphere boundary, exterior-dof
elimination, a branching vascular tree, 8 subdomains with 3 face-neighbours
each. This strips all of it away to the smallest configuration where eqs. (4)
and (5) still mean something:

    domain   : [0,1]^3, EVERY cell interior (no boundary, no elimination)
    vessel   : a single straight segment through the middle, 2 nodes
    split    : one cut plane at x = 0.5 -> two boxes, one shared face

With no exterior dofs, `A_i` has no unit-diagonal placeholder rows and the
locally-unsupported-dof problem cannot arise. With one cut plane there is
exactly one interface and each box has exactly one neighbour, so the E_ij
transfer is as simple as it can be.

What this tests
---------------
Eq. (4) requires b_i - A_i u* to be supported on the artificial interface.
That holds only if A_i is the one-sided restriction of the global operator --
and a vessel whose averaging circle spans the cut breaks it, because neither
box can compute that circular average alone.

Vessel placements (-vessel):
    offset  x=0.25 -- circle entirely inside box 0, clear of the cut
    near    x=0.46 -- circle straddles the cut cleanly (the real test)
    middle  x=0.50 -- segment exactly ON the cut (degenerate, see below)

Measured results
----------------
    offset, no cross term          : residual 100% interface-supported,
                                     eq. (4)-(5) recovers u* to 2.9e-15
    near,   no cross term          : residual 38% / 86% supported,
                                     eq. (4)-(5) is WORSE than raw
    near,   -restrict_C -cross     : residual 100% / 100% supported,
                                     eq. (4)-(5) beats raw by 1.46x
    offset, -restrict_C -cross     : cross term is identically 0, and the
                                     2.9e-15 is unchanged (regression guard)

So the note's scheme is exact when no vessel crosses an interface, fails when
one does, and the cross term below repairs the extraction. The residual is
then perfectly interface-supported, but the solve still does not reach machine
precision: with -restrict_C the local operator keeps only 49-89% of a
straddling vessel's arc weight, so A_i is one-sided AND under-coupled. Closing
that last gap needs the coupling in the operator, not only on the rhs.

Run:
    python test_robin_minimal.py -vessel offset
    python test_robin_minimal.py -vessel near
    python test_robin_minimal.py -vessel near -restrict_C
    python test_robin_minimal.py -vessel near -restrict_C -cross
    python test_robin_minimal.py -vessel offset -restrict_C -cross
"""

import argparse
import os
import tempfile

import numpy as np
from scipy.sparse.linalg import spsolve, splu

from dolfin import (
    BoxMesh,
    Point,
    Mesh,
    MeshEditor,
    MeshFunction,
    XDMFFile,
)

from Solver_partition_domain import SolverPartitionDomain
from Solver_full_domain import Solver3D1D
from Decompose_Domain_Analytic_sphere import matrix_to_csr
from Robin_residual_sphere import (
    _robin_interface,
    build_cross_term,
    build_local_operator,
    count_straddling_nodes,
)

INTERIOR_TAG = 222
INLET_TAG = 111


# =============================================================================
# Geometry
# =============================================================================

def build_line_mesh(points):
    """A 1D mesh in 3D space through the given points, one cell per segment."""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, 3)
    editor.init_vertices(len(points))
    editor.init_cells(len(points) - 1)
    for i, p in enumerate(points):
        editor.add_vertex(i, np.asarray(p, dtype=float))
    for i in range(len(points) - 1):
        editor.add_cell(i, np.array([i, i + 1], dtype="uintp"))
    editor.close()
    return mesh


def vessel_points(where, axis):
    """Two nodes spanning the box in y, displaced along the cut axis.

    The segment always runs along y, so its averaging circle lies in the x-z
    plane. The cut axis therefore falls into two qualitatively different cases:

    axis 0 (x) or 2 (z) -- the cut plane is PARALLEL to the vessel axis and
        slices through the averaging circle. The vessel's extent along the cut
        axis is +/- the radius about its offset, so the circle straddles
        whenever offset + radius > 0.5. This is the case the whole test is
        about, and x and z are related by a 90-degree rotation about y: they
        must give identical numbers.

    axis 1 (y) -- the cut plane is PERPENDICULAR to the vessel axis and slices
        the SEGMENT, not the circle. Every averaging circle lies wholly in one
        box (its plane is parallel to the cut), so nothing ever straddles no
        matter where 'where' puts it. The vessel is genuinely cut -- one node
        each side, the 1D mass matrix G couples across -- but eq. (4)'s Key
        point is not threatened. This is the control.

    'offset' 0.25  -- circle entirely inside box 0, clear of the cut.
    'near'   0.46  -- circle straddles the cut (0.46 + 0.08 > 0.5) but no
                      quadrature point lands exactly on it. The clean test.
    'middle' 0.5   -- segment exactly ON the cut. Degenerate: quadrature points
                      sitting exactly on the plane are accepted or rejected by
                      a cell-collision tie-break that differs between the two
                      boxes, so the two halves disagree even though the
                      configuration is mirror-symmetric.
    """
    off = {"middle": 0.5, "near": 0.46, "offset": 0.25}[where]

    if axis == 1:
        # Cutting across the vessel: 'where' would move the segment's ENDS, not
        # its distance from the plane, which is not the same knob at all. Keep
        # the segment centred and spanning the cut so the plane always crosses
        # it; the circles stay parallel to the plane whatever we do.
        return [(0.5, 0.25, 0.5), (0.5, 0.75, 0.5)]

    # Cut parallel to the vessel: displace along the cut axis.
    p0 = [0.5, 0.25, 0.5]
    p1 = [0.5, 0.75, 0.5]
    p0[axis] = p1[axis] = off
    return [tuple(p0), tuple(p1)]


def solve_coupled_global(meshV, pts, radius, sigma3d, sigma1d, kappa):
    """The coupled 3D-1D solve on the whole box: returns (u_star, p_sol).

    This is what the real pipeline does (see Robin_residual.py's __main__):
    Solver3D1D solves the MONOLITHIC system for [3D block | 1D block], so the
    1D pressure is an UNKNOWN, determined by the coupling. Only afterwards is
    that solved field handed to SolverPartitionDomain as known data.

    Prescribing p_known by hand instead -- as an earlier version of this test
    did -- makes u_star and p_known mutually inconsistent: u_star is then the
    3D response to an arbitrary 1D field rather than to the field the coupled
    problem actually produces. On a cut normal to the vessel that matters a
    great deal, because a hand-picked datum like [1,0] imposes an asymmetry on
    a geometrically symmetric configuration and the measured interface support
    inherits it.

    Solver3D1D reads its 1D mesh from XDMF, so the segment is written out and
    reloaded rather than passing the in-memory mesh. The 3D mesh and markers
    are injected directly, which is what lets us skip the sphere boundary.
    """
    tmp = tempfile.mkdtemp(prefix="robin_min_")
    prefix = os.path.join(tmp, "seg_")

    meshQ = build_line_mesh(pts)

    with XDMFFile(f"{prefix}marked_mesh.xdmf") as f:
        f.write(meshQ)

    # Vertex 0 is the inlet: that is where the Nitsche term drives p_in. The
    # far node is left untagged and picks up the natural condition.
    markers = MeshFunction("size_t", meshQ, 0, 0)
    markers[0] = INLET_TAG
    with XDMFFile(f"{prefix}markers.xdmf") as f:
        f.write(markers)

    radii = MeshFunction("double", meshQ, 0)
    radii.array()[:] = radius
    with XDMFFile(f"{prefix}radii.xdmf") as f:
        f.write(radii)

    cell_markers = MeshFunction("size_t", meshV, 3, INTERIOR_TAG)

    solver = Solver3D1D(
        path_to_1D_mesh=prefix,
        boundary=None,
        n=meshV.num_cells(),          # unused: full_domain_mesh wins
        sigma3d=sigma3d,
        sigma1d=sigma1d,
        kappa=kappa,
        exterior="dirichlet",
        full_domain_mesh=meshV,
        full_domain_markers=cell_markers,
        inlet_tag=INLET_TAG,
    ).build()
    solver.solve()

    return (solver.u3d.vector().get_local(),
            solver.u1d.vector().get_local(),
            solver)


def make_subdomain(mesh_lo, mesh_hi, n_sub, meshQ, q_radii, p_known,
                   sigma3d, kappa, ijk, V_global, coords_global,
                   C_global=None):
    """One box: its own mesh, its local solve, and its dof map into V_global.

    If C_global is given, the box's coupling operator is the GLOBAL C with the
    columns it does not own removed -- the arc weights are then exactly the
    global ones, which is what makes the cross term below well-posed. The
    default local quadrature instead renormalizes each partial arc by its own
    surviving measure, so C_i^(v) u_i + C_j^(v) u_j would not reconstruct the
    true full-circle average.
    """
    meshV = BoxMesh(Point(*mesh_lo), Point(*mesh_hi), *n_sub)

    # No boundary: every cell is interior, so nothing is ever eliminated.
    markers = MeshFunction("size_t", meshV, 3, INTERIOR_TAG)

    # l2g needs the space, which needs the solver built; build once without
    # C_global to get the dof map, then rebuild with it if asked.
    def _build(cg=None, l2g=None):
        ps = SolverPartitionDomain(
            meshV=meshV,
            meshV_markers=markers,
            meshQ=meshQ,
            q_radii=q_radii,
            p_known=p_known,
            sigma3d=sigma3d,
            kappa=kappa,
            gamma=1.0,
            interior_tag=INTERIOR_TAG,
            f3d=0.0,
            C_global=cg,
            l2g=l2g,
        ).build()
        ps.ext_dofs = np.zeros(0, dtype=int)   # nothing outside the domain
        return ps

    ps = _build()
    V_sub = ps.V
    sub_coords = V_sub.tabulate_dof_coordinates().reshape((V_sub.dim(), -1))
    l2g = match_dofs(sub_coords, coords_global)

    if C_global is not None:
        ps = _build(cg=C_global, l2g=l2g)
    ps.solve()

    return {
        "ijk": ijk,
        "meshV": meshV,
        "V_sub": V_sub,
        "partition_solver": ps,
        "local_to_global_dof": l2g,
    }


def rel_err(sd, u):
    """Relative error of u against u* on this box."""
    ref = float(np.linalg.norm(sd["u_star_local"]))
    if ref <= 1e-14:
        return 0.0
    return float(np.linalg.norm(sd["u_star_local"] - u)) / ref


def match_dofs(sub_coords, coords_global, tol=1e-10):
    """Index of each local dof in the global dof list (exact grid match)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_global)
    dist, idx = tree.query(sub_coords)
    if dist.max() > tol:
        raise RuntimeError(
            f"local dof {int(np.argmax(dist))} has no global match "
            f"(nearest {dist.max():.2e}); the sub-mesh is not grid-aligned."
        )
    return idx.astype(int)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[1],
    )
    ap.add_argument("-n", type=int, default=12,
                    help="cells per axis on the GLOBAL box (must be even so "
                         "the cut at 0.5 lands on a grid plane)")
    ap.add_argument("-axis", choices=("x", "y", "z"), default="x",
                    help="which axis the cut plane is normal to. The vessel "
                         "runs along y, so 'x' and 'z' cut THROUGH the "
                         "averaging circle (equivalent by symmetry) while 'y' "
                         "cuts across the segment and never straddles.")
    ap.add_argument("-vessel", choices=("middle", "near", "offset"),
                    default="near",
                    help="where the vessel sits relative to the cut at 0.5: "
                         "'offset' clear of it, 'near' straddling it cleanly, "
                         "'middle' exactly on it (degenerate). Ignored for "
                         "-axis y, where the segment always spans the cut.")
    ap.add_argument("-radius", type=float, default=0.001,
                    help="vessel radius (the averaging circle's radius)")
    ap.add_argument("-sigma1d", type=float, default=1.0,
                    help="1D conductivity for the coupled global solve that "
                         "produces p_sol")
    ap.add_argument("-sigma3d", type=float, default=1e-3)
    ap.add_argument("-kappa", type=float, default=1.0)
    ap.add_argument("-rho", type=float, default=None,
                    help="Robin penalty; default sigma3d/h")
    ap.add_argument("-restrict_C", action="store_true",
                    help="build each box's coupling operator by restricting "
                         "the global C instead of re-running clipped local "
                         "quadrature. Independent of -cross so the two effects "
                         "can be told apart.")
    ap.add_argument("-cross", action="store_true",
                    help="add the 1D-network cross term for vessels whose "
                         "averaging circle straddles the cut (the proposed "
                         "extension to eq. 5). Implies -restrict_C: the term "
                         "is only well-posed with global arc weights.")
    ap.add_argument("-max_iter", type=int, default=200,
                    help="maximum Schwarz sweeps")
    ap.add_argument("-tol", type=float, default=1e-12,
                    help="stop when the largest iterate change falls below this")
    ap.add_argument("-verbose", action="store_true",
                    help="print the Schwarz convergence history")
    args = ap.parse_args()

    if args.n % 2:
        raise SystemExit(f"-n must be even so 0.5 is a grid plane, got {args.n}")

    ax = {"x": 0, "y": 1, "z": 2}[args.axis]
    an = args.axis
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    half = args.n // 2

    print("=" * 68)
    print(f"minimal Robin-residual test: [0,1]^3, 2 boxes, cut {an}=0.5, "
          f"vessel '{args.vessel}'")
    print("=" * 68)

    # ---- 1D vessel ---------------------------------------------------------
    pts = vessel_points(args.vessel, ax)
    meshQ = build_line_mesh(pts)
    q_radii = np.full(meshQ.num_vertices(), args.radius, dtype=float)
    print(f"  vessel: {pts[0]} -> {pts[1]}  radius={args.radius}")
    if ax == 1:
        # The cut is normal to the vessel axis, so it slices the segment. The
        # circles are parallel to the plane and each lies wholly on one side.
        print(f"  cut plane at {an}=0.5; it cuts the SEGMENT "
              f"({pts[0][1]:.2f} -> {pts[1][1]:.2f} spans it), but every "
              f"averaging circle\n  is parallel to the plane, so none straddle")
    else:
        # Whether the circle actually straddles is a fact about the radius, not
        # about the -vessel label: at small -radius even 'near' is clear of the
        # cut. Compute it rather than assuming it from the label.
        reach = pts[0][ax] + args.radius
        if abs(pts[0][ax] - 0.5) < 1e-12:
            where = "sits exactly ON it (degenerate)"
        elif reach > 0.5:
            where = f"STRADDLES it (reaches {an}={reach:.3f} > 0.5)"
        else:
            note = ("; -vessel near but radius too small to straddle"
                    if args.vessel == "near" else "")
            where = f"is clear of it (reaches {an}={reach:.3f} < 0.5{note})"
        print(f"  cut plane at {an}=0.5; vessel circle {where}")

    # ---- global problem ----------------------------------------------------
    # Two solves, in the order the real pipeline uses them:
    #
    #   1. the COUPLED 3D-1D solve, in which the 1D pressure is an unknown.
    #      This is the only place p_sol can come from; prescribing it by hand
    #      would make it inconsistent with u_star.
    #   2. the SolverPartitionDomain solve on the whole box, with p_sol now
    #      treated as known data. This is the operator the subdomain solves
    #      are restrictions of, so the residual test must compare against it.
    #
    # Step 2 reproduces step 1's 3D field when the coupling is consistent, and
    # the agreement between them is printed as a sanity check: a large gap
    # means the two solvers disagree about the same physics and nothing
    # downstream is trustworthy.
    meshV = BoxMesh(Point(*lo), Point(*hi), args.n, args.n, args.n)
    markers = MeshFunction("size_t", meshV, 3, INTERIOR_TAG)

    u_coupled, p_sol, coupled = solve_coupled_global(
        meshV, pts, args.radius, args.sigma3d, args.sigma1d, args.kappa)
    p_known = p_sol
    print(f"  coupled solve: p_sol = {np.array2string(p_sol, precision=4)}")

    glob = SolverPartitionDomain(
        meshV=meshV, meshV_markers=markers, meshQ=meshQ,
        q_radii=q_radii, p_known=p_known,
        sigma3d=args.sigma3d, kappa=args.kappa, gamma=1.0,
        interior_tag=INTERIOR_TAG, f3d=0.0,
    ).build()
    glob.ext_dofs = np.zeros(0, dtype=int)
    glob.solve()

    V_global = glob.V
    n_global = V_global.dim()
    coords_global = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    u_star = glob.u3d.vector().get_local()
    d_cpl = (np.linalg.norm(u_star - u_coupled)
             / max(np.linalg.norm(u_coupled), 1e-30))
    print(f"  global: {n_global} dofs, ||u*||={np.linalg.norm(u_star):.4e}")
    print(f"  coupled vs p_known-driven 3D field: rel diff {d_cpl:.3e}")

    # ---- two subdomains, split at x = 0.5 ----------------------------------
    # -cross needs the global arc weights preserved, so it implies -restrict_C.
    # Kept as separate flags because -restrict_C alone CHANGES THE RAW BASELINE
    # (it drops the far-side arc instead of renormalizing), and conflating the
    # two makes the cross term impossible to evaluate.
    restrict_C = args.restrict_C or args.cross
    C_for_boxes = glob.C if restrict_C else None

    # Split the unit box in two along the chosen axis: the low box gets
    # [0,0.5] on that axis, the high box [0.5,1], both full-width on the other
    # two. ijk differs by 1 along the cut axis only, so the boxes are face
    # neighbours whichever axis is chosen.
    def _box(side):
        b_lo, b_hi = list(lo), list(hi)
        n_sub = [args.n, args.n, args.n]
        if side == 0:
            b_hi[ax] = 0.5
        else:
            b_lo[ax] = 0.5
        n_sub[ax] = half
        ijk = [0, 0, 0]
        ijk[ax] = side
        return make_subdomain(tuple(b_lo), tuple(b_hi), tuple(n_sub),
                              meshQ, q_radii, p_known, args.sigma3d,
                              args.kappa, tuple(ijk), V_global, coords_global,
                              C_global=C_for_boxes)

    subdomains = [_box(0), _box(1)]

    # ---- operator correctness ----------------------------------------------
    # Everything downstream assumes A_i is the one-sided restriction of the
    # global operator on box i's INTERIOR rows -- rows whose entire global
    # stencil lies inside the box. If that fails, no transmission condition can
    # work and the rest of this script is measuring the wrong thing.
    #
    # Checked in two pieces, because they can fail independently:
    #   diffusion  AD00_i  vs  AD00_global   -- pure FEM assembly
    #   coupling   C_i^T G C_i  vs  C^T G C  -- the clipped-quadrature suspect
    print("")
    print("  operator check: A_i vs the global operator on interior rows")
    Ag = (glob.A).tocsr()
    ADg = matrix_to_csr(glob.AD00).tocsr()
    Mg = (glob.C.T @ glob.G @ glob.C).tocsr()
    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        owned = np.zeros(n_global, dtype=bool)
        owned[l2g] = True
        g2l = -np.ones(n_global, dtype=np.int64)
        g2l[l2g] = np.arange(l2g.size)

        ADi = matrix_to_csr(ps.AD00).tocsr()
        Mi = (ps.C.T @ ps.G @ ps.C).tocsr()

        # Interior rows: full global stencil inside the box.
        d_ad = d_m = 0.0
        n_int = 0
        for dl in range(ps.V.dim()):
            gr = l2g[dl]
            cols = Ag.indices[Ag.indptr[gr]:Ag.indptr[gr + 1]]
            if not owned[cols].all():
                continue
            n_int += 1
            for Gm, Lm, acc in ((ADg, ADi, "ad"), (Mg, Mi, "m")):
                s, e = Gm.indptr[gr], Gm.indptr[gr + 1]
                rg = np.zeros(ps.V.dim())
                rg[g2l[Gm.indices[s:e]]] = Gm.data[s:e]
                s2, e2 = Lm.indptr[dl], Lm.indptr[dl + 1]
                rl = np.zeros(ps.V.dim())
                rl[Lm.indices[s2:e2]] = Lm.data[s2:e2]
                dd = float(np.abs(rg - rl).max())
                if acc == "ad":
                    d_ad = max(d_ad, dd)
                else:
                    d_m = max(d_m, dd)
        print(f"    {sd['ijk']}: {n_int} interior rows | "
              f"max|dAD00|={d_ad:.3e}  max|d C^T G C|={d_m:.3e}")
        if d_ad > 1e-12:
            print("        -> DIFFUSION assembly is not one-sided (unexpected)")
        if d_m > 1e-12:
            print("        -> COUPLING block differs from the global operator;"
                  " eq. (4)'s Key point cannot hold")

        # ---- the rows the interior test SKIPS ---------------------------
        # The loop above discards every row whose global stencil leaves the
        # box, and those are exactly the rows where A_i can differ from the
        # global operator. Reporting only the interior rows therefore proves
        # the operator correct on the part of the domain where nothing was
        # ever in doubt. Split the skipped rows by mechanism, because the
        # two are repaired by different things:
        #
        #   diffusion-only  -- stencil crosses the cut, no vessel involved.
        #                      This is what the Robin condition is FOR; the
        #                      rho*G_gamma term is the intended repair.
        #   coupling        -- C_i^T G C_i row differs from C^T G C. No
        #                      boundary term repairs this: the row is
        #                      quantitatively wrong, not just one-sided.
        d_ad_x = d_m_x = 0.0
        n_ad_x = n_m_x = 0
        for dl in range(ps.V.dim()):
            gr = l2g[dl]
            cols = Ag.indices[Ag.indptr[gr]:Ag.indptr[gr + 1]]
            if owned[cols].all():
                continue
            for Gm, Lm, acc in ((ADg, ADi, "ad"), (Mg, Mi, "m")):
                s, e = Gm.indptr[gr], Gm.indptr[gr + 1]
                gc, gv = Gm.indices[s:e], Gm.data[s:e]
                # Only columns this box owns can be compared at all; the
                # rest are the neighbour's half of the row by construction.
                keep = owned[gc]
                rg = np.zeros(ps.V.dim())
                rg[g2l[gc[keep]]] = gv[keep]
                s2, e2 = Lm.indptr[dl], Lm.indptr[dl + 1]
                rl = np.zeros(ps.V.dim())
                rl[Lm.indices[s2:e2]] = Lm.data[s2:e2]
                dd = float(np.abs(rg - rl).max())
                if acc == "ad":
                    d_ad_x = max(d_ad_x, dd)
                    n_ad_x += dd > 1e-12
                else:
                    d_m_x = max(d_m_x, dd)
                    n_m_x += dd > 1e-12
        n_skip = ps.V.dim() - n_int
        print(f"        skipped rows: {n_skip} | "
              f"diffusion differs on {n_ad_x} (max {d_ad_x:.3e}) | "
              f"COUPLING differs on {n_m_x} (max {d_m_x:.3e})")
        sd["_n_bad_coupling"] = n_m_x

    # Reference restriction of u* to each box, and the raw local error.
    for sd in subdomains:
        l2g = sd["local_to_global_dof"]
        sd["u_star_local"] = u_star[l2g]
        u_raw = sd["partition_solver"].u3d.vector().get_local()
        sd["rel_raw"] = rel_err(sd, u_raw)

    print("")
    print("  raw local error (no transmission condition):")
    for sd in subdomains:
        print(f"    {sd['ijk']}: {sd['rel_raw']:.6e}")

    # ---- eq. (4): extract, and measure where the residual lives ------------
    rho = args.rho
    if rho is None:
        rho = float(args.sigma3d) / max(float(meshV.hmax()), 1e-30)

    g_min, g_max = coords_global.min(axis=0), coords_global.max(axis=0)

    print("")
    print(f"  rho = {rho:.4e}")
    if args.cross:
        print("  eq. (4) residual WITH the 1D-network cross term "
              "-- where does it live?")
    else:
        print("  eq. (4) residual  b_i - A_i u* + rho G u*"
              "  -- where does it live?")

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        u_i = sd["u_star_local"]

        G_gamma, n_tagged = _robin_interface(ps, g_min, g_max)
        sd["_G_gamma"] = G_gamma

        # Interface = dofs on the cut plane. With one cut this is just the
        # chosen axis at 0.5.
        sub_coords = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
        on_if = np.abs(sub_coords[:, ax] - 0.5) < 1e-10
        sd["_sel"] = on_if

        # ---- the missing 1D-network term --------------------------------
        # The global coupling block is C^T G C. For a vessel node v whose
        # averaging circle straddles the cut, the global row splits by which
        # box owns each column:
        #
        #     C_v = C_v^(i) + C_v^(j)
        #
        # Box i assembles only (C_v^(i))^T G_vv C_v^(i). The global equation
        # restricted to box i's rows also carries
        #
        #     (C_v^(i))^T G_vv (C_v^(j) u_j)
        #
        # which eq. (5) has no slot for. C_v^(j) u_j is ONE SCALAR per
        # straddling node -- the partial arc average on the far side -- so
        # what crosses the interface is a short vector, not a field.
        #
        # This is why the arc weights must be the global ones: with the
        # default clipped quadrature each box renormalizes its own partial
        # arc, and C_v^(i) u_i + C_v^(j) u_j would not reconstruct the true
        # full-circle average.
        # Built by Robin_residual.build_cross_term, the same routine the
        # production path uses, so the two cannot drift apart.
        cross = np.zeros(ps.V.dim(), dtype=float)
        if args.cross:
            Ci = ps.C.tocsr()
            Cg = glob.C.tocsr()
            cross = build_cross_term(ps, Cg, glob.G, Cg.dot(u_star), u_i)

            # TWO independent ways the cross term can be nonzero. The x/z cut
            # exercises only the first, the y cut only the second, and an
            # earlier guard here assumed the first was the whole story.
            #
            # (a) circle straddling: this box holds part of a node's averaging
            #     circle but not all of the global row's weight. Geometry of
            #     the CIRCLE vs the plane.
            w_i = np.asarray(np.abs(Ci).sum(axis=1)).ravel()
            w_g = np.asarray(np.abs(Cg).sum(axis=1)).ravel()
            owns = w_i > 1e-14
            sd["_n_straddle"] = count_straddling_nodes(ps, Cg)

            # (b) network coupling: G is a CG1 mass matrix on the 1D mesh, so
            #     it has off-diagonal entries between neighbouring vessel
            #     nodes. If the cut separates two coupled nodes, this box's
            #     rows carry G_vw * (far average at w) even though every
            #     circle it touches is whole. Geometry of the SEGMENT vs the
            #     plane -- entirely independent of (a).
            Gg = glob.G.tocsr()
            n_link = 0
            for v in np.flatnonzero(owns):
                s, e = Gg.indptr[v], Gg.indptr[v + 1]
                for w, gvw in zip(Gg.indices[s:e], Gg.data[s:e]):
                    if w != v and abs(gvw) > 1e-14 and not owns[w]:
                        n_link += 1
            sd["_n_link"] = n_link
            sd["_far_norm"] = float(np.linalg.norm(far_avg))

        sd["_cross"] = cross
        flux = np.asarray(ps.rhs, dtype=float) - ps.A.dot(u_i) - cross
        r_i = flux + rho * G_gamma.dot(u_i)

        n_all = float(np.linalg.norm(r_i))
        n_if = float(np.linalg.norm(r_i[on_if]))
        n_in = float(np.linalg.norm(r_i[~on_if]))
        frac = n_if / max(n_all, 1e-30)
        # The flux term alone -- this is the quantity the note's Key point is
        # a statement about, before the trace term is added.
        f_all = float(np.linalg.norm(flux))
        f_if = float(np.linalg.norm(flux[on_if]))

        sd["_r"] = r_i
        sd["_frac"] = frac
        extra = ""
        if args.cross:
            n_str = sd.get("_n_straddle", 0)
            n_lnk = sd.get("_n_link", 0)
            extra = (f", {n_str} straddling nodes, {n_lnk} cut network links "
                     f"(||cross||={np.linalg.norm(cross):.3e})")
            # Regression guard: the cross term must vanish only when BOTH
            # mechanisms are absent -- no split circle AND no 1D link across
            # the cut. Checking straddling alone falsely flags the y cut,
            # where whole circles sit either side of a cut segment and the
            # off-diagonal of G legitimately carries the coupling.
            if n_str == 0 and n_lnk == 0 and np.linalg.norm(cross) > 1e-12:
                extra += "  <-- BUG: nothing crosses the cut but cross != 0"
        print(f"    {sd['ijk']}: {int(on_if.sum())} interface dofs, "
              f"{n_tagged} tagged facets{extra}")
        print(f"        ||r||={n_all:.4e}  interface={n_if:.4e} "
              f"interior={n_in:.4e}  -> {frac:6.1%} on interface")
        print(f"        flux term alone: {f_if / max(f_all, 1e-30):6.1%} "
              f"on interface")

    # For a cut PARALLEL to the vessel only 'middle' is mirror-symmetric about
    # the plane, so only there must the two boxes agree. 'near' puts the vessel
    # axis inside box 0, which is a genuinely asymmetric configuration -- an
    # earlier version flagged it too and that warning was meaningless.
    #
    # For a cut ACROSS the vessel the segment is centred on the plane whatever
    # -vessel says, so the configuration is always mirror-symmetric and the two
    # boxes must always agree. Only p_known breaks the symmetry, and it does so
    # antisymmetrically (1 -> 0), which leaves the residual NORMS equal.
    f0, f1 = subdomains[0]["_frac"], subdomains[1]["_frac"]
    symmetric = (ax == 1) or (args.vessel == "middle")
    if symmetric and abs(f0 - f1) > 0.05 * max(f0, f1, 1e-30):
        print(f"    NOTE: the two boxes disagree ({f0:.1%} vs {f1:.1%}) on "
              f"a mirror-symmetric configuration.")
        if ax == 1:
            print("          A cut across the vessel should split it evenly; "
                  "this asymmetry is not\n          explained by the geometry.")
        else:
            print(f"          Quadrature points sitting exactly on the cut are "
                  f"accepted by one box\n          and rejected by the other. "
                  f"Use -vessel near for a clean straddle.")

    # ---- E_ij: two boxes, one shared face ----------------------------------
    for i, sd in enumerate(subdomains):
        other = subdomains[1 - i]
        l2g_i = sd["local_to_global_dof"]

        f_star = np.zeros(n_global, dtype=float)
        f_star[other["local_to_global_dof"]] = np.where(
            other["_sel"], other["_r"], 0.0)

        iface_i = np.zeros(n_global, dtype=bool)
        iface_i[l2g_i[sd["_sel"]]] = True
        sd["_f_in"] = np.where(iface_i, f_star, 0.0)[l2g_i]

    # ---- eq. (5): local Robin solve ----------------------------------------
    print("")
    print("  eq. (5) corrected local solve:")
    for sd in subdomains:
        ps = sd["partition_solver"]

        # Built by the same Robin_residual routine the production path uses.
        # The coupling missing from A_i is not repaired here and cannot be: it
        # has all its columns in the neighbour's dofs.
        A_robin = build_local_operator(ps, rho, sd["_G_gamma"])

        # The cross term belongs on the rhs of eq. (5) too: it is part of the
        # equation box i's rows actually satisfy, not only of the extraction.
        rhs_i = (np.asarray(ps.rhs, dtype=float) + sd["_f_in"] + sd["_cross"])
        u_corr = spsolve(A_robin, rhs_i)

        rel = rel_err(sd, u_corr)
        sd["rel_corr"] = rel
        print(f"    {sd['ijk']}: raw {sd['rel_raw']:.6e} -> "
              f"corrected {rel:.6e}"
              f"   {'better' if rel < sd['rel_raw'] else 'WORSE'}")

    raw_avg = float(np.mean([sd["rel_raw"] for sd in subdomains]))
    cor_avg = float(np.mean([sd["rel_corr"] for sd in subdomains]))
    print("")
    print(f"  avg raw {raw_avg:.6e}  ->  avg corrected {cor_avg:.6e}")
    if cor_avg < raw_avg:
        print(f"  eq. (4)-(5) IMPROVES on raw by {raw_avg / max(cor_avg, 1e-30):.2f}x")
    else:
        print("  eq. (4)-(5) does NOT improve on raw")

    # ---- the trace-exchange alternative ------------------------------------
    # Schwarz never forms the eq. (4) residual. Each box solves
    #
    #     (A_i + rho G_i) u_i^(k+1) = b_i + rho G_i u_nbr^(k)
    #
    # so the only thing crossing the interface is the neighbour's current
    # TRACE. Nothing here requires A_i to be the one-sided restriction of the
    # global operator, which is the assumption a straddling vessel violates.
    # The clipped coupling operator is still wrong in the same way, but that
    # error now sits inside the fixed point rather than corrupting the data
    # being transferred.
    #
    # Consequence: this converges to its OWN fixed point, not to u*. Expect it
    # to land near the exact-Dirichlet floor, not at machine precision.
    print("")
    print("  Schwarz trace exchange (same boxes, same rho):")

    lu = [splu((sd["partition_solver"].A + rho * sd["_G_gamma"]).tocsc())
          for sd in subdomains]
    u_it = [np.zeros(sd["partition_solver"].V.dim()) for sd in subdomains]

    hist = []
    for k in range(1, args.max_iter + 1):
        # Global trace field from the current iterates (averaged on the shared
        # face, where both boxes own the dof).
        acc = np.zeros(n_global)
        cnt = np.zeros(n_global)
        for sd, u in zip(subdomains, u_it):
            acc[sd["local_to_global_dof"]] += u
            cnt[sd["local_to_global_dof"]] += 1.0
        u_glob = np.zeros(n_global)
        nz = cnt > 0
        u_glob[nz] = acc[nz] / cnt[nz]

        delta = 0.0
        nxt = []
        for i, sd in enumerate(subdomains):
            ps = sd["partition_solver"]
            u_nbr = u_glob[sd["local_to_global_dof"]]
            rhs = np.asarray(ps.rhs, dtype=float) + rho * (sd["_G_gamma"] @ u_nbr)
            u_new = lu[i].solve(rhs)
            delta = max(delta, float(np.linalg.norm(u_new - u_it[i])))
            nxt.append(u_new)
        u_it = nxt

        rels = []
        for sd, u in zip(subdomains, u_it):
            rels.append(rel_err(sd, u))
        hist.append((k, delta, float(np.mean(rels))))
        if delta < args.tol:
            break

    if args.verbose:
        print("    iter |   delta    | avg rel local")
        for k, d, r in hist:
            print(f"    {k:4d} | {d:.4e} | {r:.6e}")
    for sd, u in zip(subdomains, u_it):
        rel = rel_err(sd, u)
        sd["rel_schwarz"] = rel
        print(f"    {sd['ijk']}: raw {sd['rel_raw']:.6e} -> "
              f"schwarz {rel:.6e}"
              f"   {'better' if rel < sd['rel_raw'] else 'WORSE'}")
    sch_avg = float(np.mean([sd["rel_schwarz"] for sd in subdomains]))
    print(f"    converged in {hist[-1][0]} sweeps (delta={hist[-1][1]:.2e})")

    # ---- side by side ------------------------------------------------------
    print("")
    print("=" * 68)
    if ax == 1:
        verdict = "cut across the vessel, no circle straddles"
    elif args.vessel == "offset":
        verdict = "circle clear of the cut"
    else:
        verdict = "circle straddles the cut"
    print(f"  cut {an}=0.5, vessel '{args.vessel}':  {verdict}")
    print("")
    print(f"    {'':22s} {'avg rel local error':>20s}")
    print(f"    {'raw (no coupling)':22s} {raw_avg:20.6e}")
    label = "eq. (4)-(5) + cross" if args.cross else "eq. (4)-(5) residual"
    print(f"    {label:22s} {cor_avg:20.6e}"
          f"   {'ok' if cor_avg < raw_avg else '<-- fails'}")
    print(f"    {'Schwarz trace':22s} {sch_avg:20.6e}"
          f"   {'ok' if sch_avg < raw_avg else '<-- fails'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
