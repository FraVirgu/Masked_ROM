# =============================================================================
# Entry point -- cylindrical domain
#
# Companion to Decompose_Domain_Analytic_sphere.py. The decomposition machinery
# itself is geometry-agnostic: decomposeDomain reads the mesh extents and only
# ever touches the geometry through boundary(point), so the box loop, the
# artificial-facet tagging and the exterior elimination are IMPORTED from the
# sphere module rather than copied. What actually differs for a cylinder is
#
#   * the bbox consistency check -- a cylinder is not cubic unless height=2r,
#     so check_sphere_domain_consistency's "all three axes equal" rule is wrong
#     here and is replaced by a per-axis one, and
#   * the driver, which builds a CylinderBoundary and its inlet/outlet points.
#
# Keeping one copy of the numerics means a fix in the sphere module is a fix
# here too; only the geometry lives in this file.
# =============================================================================

import math
import numpy as np

from Solver_full_domain import Solver3D1D
from Analytic_Domain import Domain
from Boundary import CylinderBoundary, random_cylinder_points

# The geometry-independent core. Re-exported so callers can do everything
# through this module, exactly as they would with the sphere one.
from Decompose_Domain_Analytic_sphere import (  # noqa: F401
    ARTIFICIAL_FACET_TAG,
    assemble_robin_interface,
    decomposeDomain,
    eliminate_exterior_local,
    mark_artificial_facets,
    matrix_to_csr,
    solve_partition_domain,
    LENGTH_SUB,
    DISCRETIZATION_POINT_SUB_DOMAIN
)

AXES = {"x": 0, "y": 1, "z": 2}


def cylinder_bbox(radius, height, axis="z", center=(0.0, 0.0, 0.0)):
    """Per-axis (min, max) of the cylinder's bounding box.

    Returned in the form the mesh generator wants: the background BoxMesh has
    to cover the cylinder exactly, and unlike the sphere that box is NOT a cube
    unless height == 2*radius.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {tuple(AXES)}.")
    ai = AXES[axis]
    ext = [float(radius)] * 3
    ext[ai] = 0.5 * float(height)
    c = np.asarray(center, dtype=float)
    return tuple((c[d] - ext[d], c[d] + ext[d]) for d in range(3))


def check_cylinder_domain_consistency(
    boundary, bounds, atol=1e-12
):
    """Verify the boundary's bbox matches the meshed box, axis by axis.

    The sphere version compares every axis against a single (n_min, n_max)
    pair, which silently encodes "the domain is a cube". A cylinder of height
    != 2*radius is not, so `bounds` is a per-axis sequence of (min, max) and
    each axis is checked against its own entry.

    Also cross-checks radius/height against the bbox when the boundary exposes
    them, which catches a CylinderBoundary built with a different orientation
    than the one the mesh was generated for -- a mismatch that otherwise shows
    up only as a silently wrong domain.
    """
    if not hasattr(boundary, "_bbox"):
        raise RuntimeError("Boundary has no _bbox; cannot verify consistency.")

    bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
    if len(bounds) != 3:
        raise ValueError("bounds must hold one (min, max) pair per axis.")

    for name, (bmin, bmax), (dmin, dmax) in zip(
        ("x", "y", "z"), boundary._bbox, bounds
    ):
        if not np.isclose(0.5 * (bmin + bmax), 0.5 * (dmin + dmax),
                          atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Inconsistent {name}-center: boundary bbox center="
                f"{0.5 * (bmin + bmax)}, domain center={0.5 * (dmin + dmax)}."
            )
        if not np.isclose(bmax - bmin, dmax - dmin, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Inconsistent {name}-length: boundary bbox length="
                f"{bmax - bmin}, domain length={dmax - dmin}."
            )

    axis = getattr(boundary, "axis", None)
    if axis is not None and hasattr(boundary, "radius"):
        ai = AXES[axis]
        for d in range(3):
            expected = (2.0 * float(boundary.radius) if d != ai
                        else float(boundary.height))
            got = bounds[d][1] - bounds[d][0]
            if not np.isclose(got, expected, atol=atol, rtol=0.0):
                raise RuntimeError(
                    f"Inconsistent cylinder extent along {'xyz'[d]}: expected "
                    f"{expected} from radius/height (axis={axis!r}), but the "
                    f"domain length is {got}."
                )


def enclosing_cube(radius, height, axis="z", center=(0.0, 0.0, 0.0)):
    """(n_min, n_max) of the smallest cube containing the cylinder.

    The 3D background mesh must be this cube, NOT the cylinder's tight bbox.
    Solver3D1D meshes boundary._bbox by default, which for a cylinder is the
    slab [-r,r]^2 x [-h/2,h/2]: every cell midpoint then lies inside the
    domain, nothing is marked exterior, and the geometry silently degenerates
    into a box instead of a cylinder. Meshing the cube instead lets the
    solver's own midpoint test carve the cylinder out of it, exactly the way
    the sphere case works.
    """
    bounds = cylinder_bbox(radius, height, axis, center)
    return min(lo for lo, _ in bounds), max(hi for _, hi in bounds)


def cylinder_background_mesh(radius, height, n, axis="z", center=(0.0, 0.0, 0.0)):
    """BoxMesh over the enclosing cube, for Solver3D1D(full_domain_mesh=...).

    Passing this explicitly is what keeps the cylinder a cylinder: the solver
    only builds its own mesh when full_domain_mesh is None, and the mesh it
    would build is the tight (wrong) one. Handing it the cube leaves the
    boundary object and the shared solver untouched.
    """
    from dolfin import BoxMesh, Point

    n_min, n_max = enclosing_cube(radius, height, axis, center)
    return BoxMesh(
        Point(n_min, n_min, n_min), Point(n_max, n_max, n_max), n, n, n
    )


def cylinder_rom_lengths(radius, height, axis="z", n_per_axis=2):
    """ROM box lengths that split each axis into `n_per_axis` slabs.

    Keyed off the ENCLOSING CUBE, not the cylinder's tight bbox: the mesh that
    decomposeDomain partitions is the cube, and it derives its box count as
    ceil(extent / ROM_length). Using the tight extents here would ask for boxes
    shorter than the mesh along the cylinder axis and split it far more finely
    than requested -- with height=1 and radius=5 that is 10 slabs in z against
    2 in x and y.
    """
    if n_per_axis < 1:
        raise ValueError("n_per_axis must be at least 1.")
    n_min, n_max = enclosing_cube(radius, height, axis)
    side = (n_max - n_min) / n_per_axis
    return side, side, side


if __name__ == "__main__":
    import os
    import argparse

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
    parser.add_argument("-axis", choices=("x", "y", "z"), default="z",
                        help="direction of the cylinder axis")
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

    # Inlets and outlets are split along the cylinder axis, so one set sits near
    # each cap -- the cylinder analogue of the sphere's -x / +x split.
    boundary = CylinderBoundary(
        radius=args.radius,
        height=args.height,
        axis=args.axis,
        inlet_points=random_cylinder_points(
            40,
            sign=-1,
            radius=args.radius,
            height=args.height,
            axis=args.axis,
            min_offset=0.2,
            min_dist_to_boundary=0.06,
        ),
        outlet_points=random_cylinder_points(
            40,
            sign=+1,
            radius=args.radius,
            height=args.height,
            axis=args.axis,
            min_offset=0.2,
            min_dist_to_boundary=0.06,
        ),
        border_eps=10e-1,
    )

    bounds = cylinder_bbox(args.radius, args.height, args.axis)
    check_cylinder_domain_consistency(boundary=boundary, bounds=bounds)

    # --- 1. build the analytic-boundary domain --------------------------------
    n_min, n_max = enclosing_cube(args.radius, args.height, args.axis)
    print(f"cylinder r={args.radius} h={args.height} axis={args.axis}  ->  "
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

    # --- 2. run the solver WITHOUT the boundary penalty ---
    # exterior='dirichlet' eliminates the exterior DOFs exactly instead of
    # pinning them with a penalty term, so the interface is not contaminated.
    out_dir = (
        f"./solution/Simple{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}"
    )
    # full_domain_mesh is the whole point here: left to itself Solver3D1D meshes
    # boundary._bbox, i.e. the tight slab, and then every cell midpoint is
    # inside the cylinder so nothing is ever marked exterior -- the run silently
    # solves a BOX. Handing it the enclosing cube restores the intended
    # behaviour: the solver's midpoint test marks cells 222/111 and the
    # Dirichlet elimination carves the cylinder out.
    solver  = Solver3D1D(
        path_to_1D_mesh = mesh_prefix,
        full_domain_mesh = cylinder_background_mesh(
            args.radius, args.height, args.n, args.axis),
        boundary        = boundary,
        n               = args.n,
        sigma3d         = args.sigma3d,
        sigma1d         = args.sigma1d,
        kappa           = args.kappa,
        exterior        = "dirichlet",
        lenght_sub_domain = LENGTH_SUB,
        n_sub = DISCRETIZATION_POINT_SUB_DOMAIN
    ).build().solve()

    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")
