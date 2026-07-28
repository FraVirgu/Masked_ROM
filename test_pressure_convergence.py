import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfin import Function, FunctionSpace, Measure, assemble, interpolate, dx

from Boundary import boundary
from Domain import Domain
from Solver import Solver3D1D


def build_domain(case_name, boundary_fn, radius_mode, radius_value=0.01):
    """Build the 1D domain and export the mesh/radius files."""
    net_dir = os.path.join("nets", case_name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, case_name)
    mesh_prefix = f"{name_stem}_"

    domain = Domain(
        name=name_stem,
        n_vasi=4,
        n_ramifications=4,
        boundary=boundary_fn,
    ).build()

    if radius_mode == "fixed":
        for i in range(domain.vaso.num_vertices()):
            domain.vaso_radii[i] = float(radius_value)
    else:
        # Keep the current normal-sampled radii from Domain._fun.
        pass

    domain.export_xdmf()
    return mesh_prefix


def solve_case(case_name, boundary_fn, radius_mode, n_values, radius_value=0.01):
    """Run the solver for every n and return the 3D pressure errors against an extra n=64 reference."""
    mesh_prefix = build_domain(case_name, boundary_fn, radius_mode, radius_value=radius_value)

    all_ns = list(n_values) + [64]
    solutions = []
    for n in all_ns:
        print(f"\n=== {case_name} | n={n} ===")
        solver = Solver3D1D(
            path_to_1D_mesh=mesh_prefix,
            boundary=boundary_fn,
            n=n,
            sigma3d=1e-3,
            sigma1d=1.0,
            kappa=1.0,
            exterior="dirichlet",
        ).build().solve()
        solutions.append(solver)

    ref_solver = solutions[-1]
    V_ref = ref_solver.W[0]
    ref_u = ref_solver.u3d
    dx_ref = Measure("dx", domain=ref_solver.meshV)

    errors = []
    for solver in solutions[:-1]:
        u_interp = interpolate(solver.u3d, V_ref)
        diff = ref_u - u_interp
        err = np.sqrt(assemble(diff * diff * dx_ref))
        errors.append(float(err))

    return errors, ref_solver


def make_plot_fixed(n_values, errors_boundary_fixed, errors_full_fixed, output_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(n_values, errors_boundary_fixed, marker="o", linewidth=2, label="boundary fixed radii")
    plt.plot(n_values, errors_full_fixed, marker="^", linewidth=2, label="full domain fixed radii")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("3D mesh resolution n")
    plt.ylabel("L2 error vs finest solve")
    plt.title("3D pressure convergence: fixed radii")
    plt.grid(axis="y", which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_plot_random(n_values, errors_boundary_random, errors_full_random, output_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(n_values, errors_boundary_random, marker="s", linewidth=2, label="boundary random radii")
    plt.plot(n_values, errors_full_random, marker="D", linewidth=2, label="full domain random radii")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("3D mesh resolution n")
    plt.ylabel("L2 error vs finest solve")
    plt.title("3D pressure convergence: random radii")
    plt.grid(axis="y", which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    n_values = [ 4,8,16,32]
    np.random.seed(42)

    errors_fixed, _ = solve_case("pressure_fixed", boundary, "fixed", n_values, radius_value=0.01)
    errors_random, _ = solve_case("pressure_random", boundary, "random", n_values)
    errors_noboundary_fixed, _ = solve_case("pressure_noboundary_fixed", None, "fixed", n_values, radius_value=0.01)
    errors_noboundary_random, _ = solve_case("pressure_noboundary_random", None, "random", n_values)

    print("\nSummary")
    print("-" * 60)
    print("n | boundary fixed | boundary random | full-domain fixed | full-domain random")
    for n, e_fixed, e_rand, e_nb_f, e_nb_r in zip(
        n_values, errors_fixed, errors_random,
        errors_noboundary_fixed, errors_noboundary_random
    ):
        print(
            f"n={n:>2} | {e_fixed:.6e} | {e_rand:.6e} "
            f"| {e_nb_f:.6e} | {e_nb_r:.6e}"
        )

    plot_dir = os.path.join("solution")
    os.makedirs(plot_dir, exist_ok=True)

    output_fixed = os.path.join(plot_dir, "pressure_convergence_fixed_radii.png")
    make_plot_fixed(
        n_values,
        errors_fixed,
        errors_noboundary_fixed,
        output_fixed,
    )
    print(f"Saved fixed radii plot to {output_fixed}")

    output_random = os.path.join(plot_dir, "pressure_convergence_random_radii.png")
    make_plot_random(
        n_values,
        errors_random,
        errors_noboundary_random,
        output_random,
    )
    print(f"Saved random radii plot to {output_random}")


if __name__ == "__main__":
    main()
