# =============================================================================
# Entry point -- cylinder + sphere UNION domain
#
# Companion to Decompose_Domain_Analytic_cylinder.py. As there, the numerics are
# imported, not copied: decomposeDomain only ever touches the geometry through
# boundary(point), so the box loop, the artificial-facet tagging and the
# exterior elimination come from the sphere module unchanged.
#
# What is genuinely different here is that the union is the first domain in the
# project whose bounding box is NOT centred on `center`. The sphere rides on one
# end of the cylinder axis, so the hull runs from -h/2 to offset+sphere_radius
# along that axis. Every helper below therefore derives the cube from the hull
# rather than from a half-extent about the origin -- the cylinder version can
# assume symmetry and this one cannot.
# =============================================================================

import numpy as np

from Boundary import CylinderSphereBoundary  # noqa: F401

# The geometry-independent core. Re-exported so callers can do everything
# through this module, exactly as they would with the cylinder one.
from Decompose_Domain_Analytic_sphere import (  # noqa: F401
    ARTIFICIAL_FACET_TAG,
    assemble_robin_interface,
    decomposeDomain,
    eliminate_exterior_local,
    mark_artificial_facets,
    matrix_to_csr,
    solve_partition_domain,
    LENGTH_SUB,
    DISCRETIZATION_POINT_SUB_DOMAIN,
)

AXES = {"x": 0, "y": 1, "z": 2}


def cylinder_sphere_bbox(
    radius,
    height,
    sphere_radius=None,
    sphere_offset=None,
    axis="z",
    center=(0.0, 0.0, 0.0),
):
    """Per-axis (min, max) of the union's bounding box.

    The component-wise hull of the cylinder's slab and the sphere's box. Along
    the cylinder axis this is asymmetric about `center`: the sphere sticks out
    on one side only, so the returned box is NOT centred on the domain centre
    the way the sphere and cylinder ones are.

    Kept identical to CylinderSphereBoundary._bbox -- the consistency check
    below compares the two, so they must be computed the same way.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {tuple(AXES)}.")
    ai = AXES[axis]

    radius, height = float(radius), float(height)
    sr = radius if sphere_radius is None else float(sphere_radius)
    half = 0.5 * height
    offset = half if sphere_offset is None else float(sphere_offset)

    c = np.asarray(center, dtype=float)
    sc = c.copy()
    sc[ai] += offset

    ext = [radius, radius, radius]
    ext[ai] = half
    return tuple(
        (min(c[d] - ext[d], sc[d] - sr), max(c[d] + ext[d], sc[d] + sr))
        for d in range(3)
    )


def check_cylinder_sphere_domain_consistency(boundary, bounds, atol=1e-12):
    """Verify the boundary's bbox matches the meshed box, axis by axis.

    Same per-axis contract as check_cylinder_domain_consistency -- centre and
    length compared independently for x, y, z -- because the union is neither
    cubic nor centred and the sphere module's "all three axes equal" rule would
    reject every valid geometry here.

    The extra cross-check against radius/height/sphere_radius catches a boundary
    built with a different axis or offset than the mesh was generated for, which
    otherwise shows up only as a silently wrong domain.
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
    if axis is not None and hasattr(boundary, "sphere_radius"):
        expected = cylinder_sphere_bbox(
            boundary.radius, boundary.height, boundary.sphere_radius,
            boundary.sphere_offset, axis, boundary.center,
        )
        for d in range(3):
            got = bounds[d][1] - bounds[d][0]
            want = expected[d][1] - expected[d][0]
            if not np.isclose(got, want, atol=atol, rtol=0.0):
                raise RuntimeError(
                    f"Inconsistent union extent along {'xyz'[d]}: expected "
                    f"{want} from radius/height/sphere_radius (axis={axis!r}), "
                    f"but the domain length is {got}."
                )


def enclosing_cube(
    radius,
    height,
    sphere_radius=None,
    sphere_offset=None,
    axis="z",
    center=(0.0, 0.0, 0.0),
):
    """(n_min, n_max) of the smallest cube containing the union.

    Same reason as the cylinder case: Solver3D1D meshes boundary._bbox by
    default, and for this domain that tight hull would leave most cell midpoints
    inside, so the geometry would silently degenerate into a box instead of a
    capped tube. Meshing a cube lets the solver's midpoint test carve the union
    out of it.

    The cube is centred on the HULL's centre, not on `center`. Because the
    sphere is off to one side, a cube built symmetrically about `center` would
    have to be twice as large to reach the far pole -- and would still be
    lopsided. Growing the hull to a cube about its own midpoint is the smallest
    cube that actually contains the domain.
    """
    bounds = cylinder_sphere_bbox(
        radius, height, sphere_radius, sphere_offset, axis, center)
    side = max(hi - lo for lo, hi in bounds)
    mids = [0.5 * (lo + hi) for lo, hi in bounds]
    # A single (n_min, n_max) pair has to serve all three axes, so the cube is
    # placed about the most extreme midpoints -- this is the tightest cube that
    # covers the hull on every axis at once.
    n_min = min(m - 0.5 * side for m in mids)
    n_max = max(m + 0.5 * side for m in mids)
    return n_min, n_max


def cylinder_sphere_background_mesh(
    radius,
    height,
    n,
    sphere_radius=None,
    sphere_offset=None,
    axis="z",
    center=(0.0, 0.0, 0.0),
):
    """BoxMesh over the enclosing cube, for Solver3D1D(full_domain_mesh=...).

    Passing this explicitly is what keeps the union a union: the solver only
    builds its own mesh when full_domain_mesh is None, and the mesh it would
    build is the tight (wrong) one.
    """
    from dolfin import BoxMesh, Point
    from Solver_full_domain import Solver3D1D

    # The cube touches the domain wall, putting the outer Dirichlet boundary on
    # the domain surface. Pad it by the same clearance the solver applies to a
    # bbox it meshes itself, so the two paths agree.
    eps = Solver3D1D.BBOX_EPS
    n_min, n_max = enclosing_cube(
        radius, height, sphere_radius, sphere_offset, axis, center)
    n_min, n_max = n_min - eps, n_max + eps
    return BoxMesh(
        Point(n_min, n_min, n_min), Point(n_max, n_max, n_max), n, n, n
    )


def cylinder_sphere_rom_lengths(
    radius,
    height,
    sphere_radius=None,
    sphere_offset=None,
    axis="z",
    n_per_axis=2,
    center=(0.0, 0.0, 0.0),
):
    """ROM box lengths that split each axis into `n_per_axis` slabs.

    Keyed off the ENCLOSING CUBE, not the tight hull, for the same reason as the
    cylinder version: decomposeDomain partitions the cube and derives its box
    count as ceil(extent / ROM_length), so tight extents would split the domain
    far more finely along one axis than requested.
    """
    if n_per_axis < 1:
        raise ValueError("n_per_axis must be at least 1.")
    n_min, n_max = enclosing_cube(
        radius, height, sphere_radius, sphere_offset, axis, center)
    side = (n_max - n_min) / n_per_axis
    return side, side, side
