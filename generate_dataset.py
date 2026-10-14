"""Generate the surrogate-model training set: one folder per random case.

Every case is independent -- its own geometry, vessel network, global solve,
decomposition and Robin correction -- so cases can be generated in any order,
in separate jobs, and resumed: a case whose folder already holds `done` is
skipped.

Per case, in order:

    1. shape        sphere | cylinder | cylinder_sphere
    2. dimensions   radii, height, axis, sphere offset
    3. inlets       number of inlet/outlet points and where they sit,
                    n_vasi x n_ramifications vessel paths between them
    4. network      vessel radius distribution and Voronoi point density
    5. simulate     global 3D-1D solve, decomposition, eqs. (4)-(5)
    6. save

Layout:

    dataset/case_00000/
        params.json          sampled parameters, seed, solver stats, errors
        network/             1D mesh files the solver read (xdmf)
        global.npz           full-domain 3D and 1D solution + their geometry
        subdomains/box_i_j_k.npz
        done                 written last; its presence marks a finished case

Each box file holds, over the box's local dofs (all boxes have the same size,
LENGTH_SUB with DISCRETIZATION_POINT_SUB_DOMAIN cells per side):

    coords           dof coordinates relative to `origin`
    origin, ijk      where the box sits in the global cube
    l2g              local -> global dof index (into global.npz arrays)
    exterior         True where the dof is outside the domain (u = 0 there)
    rhs              local load from the vessel network
    u_star           global solution restricted to the box   (the target)
    u_raw            isolated local solve, no communication
    f_out            Robin flux this box extracts, eq. (4)
    f_in             Robin flux this box receives, the rhs term of eq. (5)
    interface        P^Gamma selector on the artificial cut planes
    interface_mass   m_i, the test-function mass on those planes
    u_robin          local solve with the Robin flux, eq. (5)

Boxes lying entirely outside the domain are not written (every field is
zero there); they are listed in params.json under "empty_boxes".

Several processes can run at once, on one machine or many: give each its own
-start/-n_cases range so they never work on the same case.

Run:

    python generate_dataset.py -n_cases 50
    python generate_dataset.py -start 50 -n_cases 50          # next chunk
    python generate_dataset.py -n_cases 10 -shape sphere
"""

import argparse
import json
import os
import random
import time
import traceback

import numpy as np

from dolfin import BoxMesh, Point, dof_to_vertex_map, vertex_to_dof_map

from Analytic_Domain import (
    Domain,
    REFERENCE_POINT_DENSITY,
    points_per_layer_at_density,
)
from Boundary import (
    CylinderBoundary,
    CylinderSphereBoundary,
    SphereBoundary,
    random_cylinder_points,
    random_cylinder_sphere_points,
    random_sphere_points,
)
from Decompose_Domain_Analytic_sphere import (
    DISCRETIZATION_POINT_SUB_DOMAIN,
    LENGTH_SUB,
    decomposeDomain,
)
from Robin_residual_sphere import apply_robin_residual
from Solver_full_domain import Solver3D1D

SHAPES = ("sphere", "cylinder", "cylinder_sphere")
AXES = ("x", "y", "z")

# Inlet/outlet points are kept this far from the wall, as in the drivers.
MIN_DIST_TO_BOUNDARY = 0.06


# =============================================================================
# 1-4. sampling
# =============================================================================

def sample_params(rng, shape=None):
    """Draw every random input of one case. Plain floats/ints, JSON-ready.

    `rng` is a numpy Generator used ONLY here; the network construction draws
    from the global np.random / random state, which main() seeds per case.
    """
    p = {}

    # 1. shape
    p["shape"] = shape if shape is not None else str(rng.choice(SHAPES))

    # 2. dimensions. Ranges keep the enclosing cube at <= ~7 subdomains per
    # axis, i.e. at most ~350 boxes and ~3M 3D dofs per case.
    if p["shape"] == "sphere":
        p["radius"] = float(rng.uniform(1.0, 2.5))
    else:
        p["radius"] = float(rng.uniform(1.0, 2.0))
        p["height"] = float(rng.uniform(1.5, 3.5))
        p["axis"] = str(rng.choice(AXES))
    if p["shape"] == "cylinder_sphere":
        p["sphere_radius"] = float(rng.uniform(0.8, 2.0))
        # Position of the sphere along the cylinder axis. Must stay below
        # h/2 + sphere_radius or the two do not overlap (the boundary class
        # rejects it); the upper bound leaves at least half the ball biting
        # into the tube.
        half = 0.5 * p["height"]
        p["sphere_offset"] = float(
            rng.uniform(0.5 * half, half + 0.5 * p["sphere_radius"]))

    # 3. inlets/outlets. Same count for both: Domain pairs them with zip(), so
    # a longer list would be silently truncated.
    p["n_terminals"] = int(rng.integers(10, 61))
    p["n_vasi"] = int(rng.integers(5, 16))
    p["n_ramifications"] = int(rng.integers(5, 16))
    p["min_offset"] = float(rng.uniform(0.1, 0.4))

    # 4. network. Std as a fraction of the mean, so the normal draw does not
    # spend most of its mass clipped at radius_min.
    p["radius_mean"] = float(rng.uniform(0.005, 0.03))
    p["radius_std"] = float(rng.uniform(0.1, 0.5) * p["radius_mean"])
    p["radius_min"] = 0.001
    p["radius_max"] = 0.05
    p["density_scale"] = float(rng.uniform(0.7, 1.3))
    return p


def build_boundary(p):
    """Boundary object with its inlet/outlet points, from sampled params."""
    n, kw = p["n_terminals"], dict(min_dist_to_boundary=MIN_DIST_TO_BOUNDARY)
    if p["shape"] == "sphere":
        def pts(sign):
            return random_sphere_points(
                n, x_sign=sign, min_x=p["min_offset"], radius=p["radius"], **kw)
        return SphereBoundary(
            radius=p["radius"], inlet_points=pts(-1), outlet_points=pts(+1),
            border_eps=10e-1)

    if p["shape"] == "cylinder":
        def pts(sign):
            return random_cylinder_points(
                n, sign=sign, radius=p["radius"], height=p["height"],
                axis=p["axis"], min_offset=p["min_offset"], **kw)
        return CylinderBoundary(
            radius=p["radius"], height=p["height"], axis=p["axis"],
            inlet_points=pts(-1), outlet_points=pts(+1), border_eps=10e-1)

    def pts(sign):
        return random_cylinder_sphere_points(
            n, sign=sign, radius=p["radius"], height=p["height"],
            sphere_radius=p["sphere_radius"], sphere_offset=p["sphere_offset"],
            axis=p["axis"], min_offset=p["min_offset"], **kw)
    return CylinderSphereBoundary(
        radius=p["radius"], height=p["height"],
        sphere_radius=p["sphere_radius"], sphere_offset=p["sphere_offset"],
        axis=p["axis"], inlet_points=pts(-1), outlet_points=pts(+1),
        border_eps=10e-1)


def enclosing_cube(bbox):
    """(n_min, n_max) of the tightest cube about a per-axis bbox.

    Same construction as Decompose_Domain_Analytic_cylinder_sphere's, applied
    to boundary._bbox so one helper serves all three shapes.
    """
    side = max(hi - lo for lo, hi in bbox)
    mids = [0.5 * (lo + hi) for lo, hi in bbox]
    return (min(m - 0.5 * side for m in mids),
            max(m + 0.5 * side for m in mids))


# =============================================================================
# 6. saving
# =============================================================================

def save_global(path, solver):
    """3D and 1D solution of the whole domain, in dof order, with geometry."""
    V, Q = solver.W
    v_coords = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    exterior = np.zeros(V.dim(), dtype=bool)
    if solver.ext_dofs is not None:
        exterior[np.asarray(solver.ext_dofs, dtype=int)] = True

    # 1D fields are vertex-indexed MeshFunctions; bring them to dof order so
    # they line up with u1d, and express the edges in dof indices too.
    q_coords = Q.tabulate_dof_coordinates().reshape((Q.dim(), -1))
    d2v = dof_to_vertex_map(Q)
    v2d = vertex_to_dof_map(Q)

    np.savez_compressed(
        path,
        u3d=solver.u3d.vector().get_local(),
        v_coords=v_coords,
        v_exterior=exterior,
        u1d=solver.u1d.vector().get_local(),
        q_coords=q_coords,
        q_edges=v2d[np.asarray(solver.meshQ.cells(), dtype=int)],
        q_radii=np.asarray(solver.Q_radii.array(), dtype=float)[d2v],
        q_markers=np.asarray(solver.Q_markers.array(), dtype=int)[d2v],
    )


def save_box(path, sd):
    ps = sd["partition_solver"]
    l2g = np.asarray(sd["local_to_global_dof"], dtype=int)
    coords = ps.V.tabulate_dof_coordinates().reshape((ps.V.dim(), -1))
    origin = coords.min(axis=0)

    exterior = np.zeros(ps.V.dim(), dtype=bool)
    if getattr(ps, "ext_dofs", None) is not None:
        exterior[np.asarray(ps.ext_dofs, dtype=int)] = True

    # f_in exactly as eq. (5) uses it: received load, exterior dofs zeroed
    # (solve_local_robin does the same before the solve).
    f_in = np.asarray(sd["_f_incoming"][l2g], dtype=float).copy()
    f_in[exterior] = 0.0

    np.savez_compressed(
        path,
        ijk=np.asarray(sd["ijk"], dtype=int),
        origin=origin,
        coords=(coords - origin).astype(np.float32),
        l2g=l2g,
        exterior=exterior,
        rhs=np.asarray(ps.rhs, dtype=float),
        u_star=sd["sol_3d_sub"].vector().get_local(),
        u_raw=sd["_u_raw"],
        f_out=np.asarray(sd["f_star_minus"][l2g], dtype=float),
        f_in=f_in,
        interface=np.asarray(sd["_sel"], dtype=bool),
        interface_mass=np.asarray(sd["_mass"], dtype=float),
        u_robin=sd["u3d_robin"].vector().get_local(),
        rel_err_raw=float(sd["u3d_partition_error_local_raw_rel_l2"]),
        rel_err_robin=float(sd["u3d_robin_rel_l2"]),
    )


# =============================================================================
# 5. one case
# =============================================================================

def run_case(case_dir, seed, args):
    t_start = time.time()
    os.makedirs(case_dir, exist_ok=True)

    # Seed EVERY source the pipeline draws from. Boundary.py seeds np.random
    # once at import, but the vessel paths use Python's `random`, which was
    # never seeded -- that alone made two runs of the same case differ.
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    p = sample_params(rng, None if args.shape == "random" else args.shape)
    p.update(seed=seed, sigma3d=args.sigma3d, sigma1d=args.sigma1d,
             kappa=args.kappa, rho=args.rho, cross=args.cross,
             length_sub=LENGTH_SUB, n_sub=DISCRETIZATION_POINT_SUB_DOMAIN)
    # Written before the solve, so a failed case still records what it was.
    params_path = os.path.join(case_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(p, f, indent=2)
    print(f"\n{'=' * 70}\n{case_dir}  seed={seed}\n{json.dumps(p)}\n{'=' * 70}")

    boundary = build_boundary(p)
    n_min, n_max = enclosing_cube(boundary._bbox)

    # Network over the same cube as the domain, at the sampled density.
    n_per_layer = points_per_layer_at_density(
        n_min, n_max, density=p["density_scale"] * REFERENCE_POINT_DENSITY)
    net_dir = os.path.join(case_dir, "network")
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, "net")
    domain = Domain(
        name=name_stem,
        n_vasi=p["n_vasi"],
        n_ramifications=p["n_ramifications"],
        n_min=n_min,
        n_max=n_max,
        n_points_per_layer=n_per_layer,
        radius_mean=p["radius_mean"],
        radius_std=p["radius_std"],
        radius_min=p["radius_min"],
        radius_max=p["radius_max"],
        boundary=boundary,
    ).build()
    domain.export_xdmf()

    # Background cube with the usual clearance. Its resolution is irrelevant:
    # lenght_sub_domain makes the solver re-mesh it at n_sub cells per box.
    eps = Solver3D1D.BBOX_EPS
    lo, hi = n_min - eps, n_max + eps
    solver = Solver3D1D(
        path_to_1D_mesh=f"{name_stem}_",
        full_domain_mesh=BoxMesh(Point(lo, lo, lo), Point(hi, hi, hi), 2, 2, 2),
        boundary=boundary,
        sigma3d=args.sigma3d, sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet",
        lenght_sub_domain=LENGTH_SUB,
        n_sub=DISCRETIZATION_POINT_SUB_DOMAIN,
    ).build().solve()
    save_global(os.path.join(case_dir, "global.npz"), solver)

    subdomains = decomposeDomain(
        solver, boundary,
        x_ROM_lenght=LENGTH_SUB, y_ROM_lenght=LENGTH_SUB,
        z_ROM_lenght=LENGTH_SUB,
        restrict_global_C=args.cross)
    # Raw local fields must be captured BEFORE apply_robin_residual.
    for sd in subdomains:
        sd["_u_raw"] = sd["partition_solver"].u3d.vector().get_local().copy()
    result = apply_robin_residual(
        subdomains, solver, rho_robin=args.rho, cross=args.cross)

    box_dir = os.path.join(case_dir, "subdomains")
    os.makedirs(box_dir, exist_ok=True)
    empty = []
    for sd in subdomains:
        ps = sd["partition_solver"]
        ext = getattr(ps, "ext_dofs", None)
        n_ext = 0 if ext is None else np.size(ext)
        if not args.keep_empty and n_ext == ps.V.dim():
            empty.append(list(sd["ijk"]))
            continue
        i, j, k = sd["ijk"]
        save_box(os.path.join(box_dir, f"box_{i}_{j}_{k}.npz"), sd)

    V, Q = solver.W
    p.update(
        n_min=n_min, n_max=n_max,
        n_points_per_layer=n_per_layer,
        n_3d_dofs=int(V.dim()), n_1d_dofs=int(Q.dim()),
        n_1d_edges=int(solver.meshQ.num_cells()),
        n_boxes=len(subdomains), n_boxes_saved=len(subdomains) - len(empty),
        empty_boxes=empty,
        solver_iters=int(solver.niters), solver_time=float(solver.solve_time),
        rel_global_raw=float(
            subdomains[0]["u3d_partition_full_error_raw_rel_l2"]),
        rel_global_robin=float(result["rel_global"]),
        rel_local_robin=float(result["rel_local"]),
        iface_fraction=float(result["iface_fraction"]),
        rho_used=float(result["rho"]),
        wall_time=time.time() - t_start,
    )
    with open(params_path, "w") as f:
        json.dump(p, f, indent=2)
    open(os.path.join(case_dir, "done"), "w").close()
    print(f"case done in {p['wall_time']:.0f}s: {p['n_boxes_saved']} boxes, "
          f"global rel err raw {p['rel_global_raw']:.3e} -> "
          f"robin {p['rel_global_robin']:.3e}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate random 3D-1D cases for the surrogate dataset.")
    parser.add_argument("-out", type=str, default="dataset")
    parser.add_argument("-start", type=int, default=0,
                        help="index of the first case")
    parser.add_argument("-n_cases", type=int, default=1)
    parser.add_argument("-seed", type=int, default=0,
                        help="case i uses seed + i, so a case is reproducible "
                             "from its index alone")
    parser.add_argument("-shape", choices=("random",) + SHAPES,
                        default="random")
    parser.add_argument("-sigma1d", type=float, default=1.0)
    parser.add_argument("-sigma3d", type=float, default=1e-3)
    parser.add_argument("-kappa", type=float, default=1.0)
    parser.add_argument("-rho", type=float, default=1.0,
                        help="Robin penalty for eqs. (4)-(5)")
    parser.add_argument("-cross", action="store_true",
                        help="add the 1D cross term (forces restricted C)")
    parser.add_argument("-keep_empty", action="store_true",
                        help="also save boxes lying entirely outside the domain")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for idx in range(args.start, args.start + args.n_cases):
        case_dir = os.path.join(args.out, f"case_{idx:05d}")
        if os.path.isfile(os.path.join(case_dir, "done")):
            print(f"skip {case_dir}: already done")
            continue
        try:
            run_case(case_dir, args.seed + idx, args)
        except Exception:
            # One bad geometry must not kill a whole batch. The traceback is
            # kept next to the params that produced it.
            tb = traceback.format_exc()
            print(tb)
            os.makedirs(case_dir, exist_ok=True)
            with open(os.path.join(case_dir, "failed.txt"), "w") as f:
                f.write(tb)


if __name__ == "__main__":
    main()
