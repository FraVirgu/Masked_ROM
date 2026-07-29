import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dolfin import Measure, assemble, interpolate, dx

from Analytic_Domain import Domain
from Solver_analytic import Solver3D1D


def build_full_domain(case_name, radius_value=0.01):
    net_dir = os.path.join("nets", case_name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, case_name)
    mesh_prefix = f"{name_stem}_"

    domain = Domain(
        name=name_stem,
        n_vasi=4,
        n_ramifications=4,
        boundary=None,
    ).build()

    for i in range(domain.vaso.num_vertices()):
        domain.vaso_radii[i] = float(radius_value)

    domain.export_xdmf()
    return mesh_prefix


def solve_variant(mesh_prefix, n, exterior, penalty=1.0):
    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix,
        boundary=None,
        n=n,
        sigma3d=1e-3,
        sigma1d=1.0,
        kappa=1.0,
        exterior=exterior,
        penalty=penalty,
    ).build().solve()
    return solver


def l2_error(u_a, u_b, V):
    diff = u_a - u_b
    dx_ref = Measure("dx", domain=V.mesh())
    return float(np.sqrt(assemble(diff * diff * dx_ref)))


def make_plot(n_values, diffs, output_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(n_values, diffs, marker="^", linewidth=2, color="green", label="dirichlet vs penalty")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("3D mesh resolution n")
    plt.ylabel("L2 error")
    plt.title("Full-domain: Dirichlet vs Penalty difference")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    n_values = [8, 16, 32]
    np.random.seed(42)

    mesh_prefix = build_full_domain("full_domain_no_boundary", radius_value=0.01)

    dir_solutions = [solve_variant(mesh_prefix, n, exterior="dirichlet") for n in n_values]
    pen_solutions = [solve_variant(mesh_prefix, n, exterior="penalty", penalty=1e2) for n in n_values]

    ref_dir = dir_solutions[-1]
    ref_pen = pen_solutions[-1]

    errors_dirichlet = [l2_error(sol.u3d, ref_dir.u3d, ref_dir.W[0]) for sol in dir_solutions]
    errors_penalty = [l2_error(sol.u3d, ref_pen.u3d, ref_pen.W[0]) for sol in pen_solutions]

    diffs = []
    for dir_sol, pen_sol in zip(dir_solutions, pen_solutions):
        V = dir_sol.W[0]
        pen_interp = interpolate(pen_sol.u3d, V)
        diffs.append(l2_error(dir_sol.u3d, pen_interp, V))

    print("\nSummary")
    print("-" * 70)
    print("n | dirichlet vs fine | penalty vs fine | dirichlet vs penalty")
    for n, e_dir, e_pen, diff in zip(n_values, errors_dirichlet, errors_penalty, diffs):
        print(f"{n:>2} | {e_dir:.6e} | {e_pen:.6e} | {diff:.6e}")

    output_path = os.path.join("solution", "full_domain_dirichlet_vs_penalty.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    make_plot(n_values, diffs, output_path)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
