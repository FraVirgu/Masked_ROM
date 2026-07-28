"""
Compare the three ways of handling the exterior (111) region of the background
box, against exterior='restrict' as the reference:

  restrict  : V on the SubMesh of the 222 cells. Exterior dofs do not exist.
              No parameter. REFERENCE.
  dirichlet : V on the full box; the dofs with NO support in any 222 cell are
              eliminated exactly (zero row/col, 1 on diag, 0 rhs). A dof pinned
              to zero contributes nothing to any interior equation, so this must
              reproduce 'restrict' to SOLVER TOLERANCE (~1e-10).
  penalty   : V on the full box; exterior pinned by penalty*inner(u,v)*dx(111).
              The integral also hits the interface vertices shared with the 222
              cells, so it contaminates the interior. Only an APPROXIMATION —
              expected to differ, and to have no good value of `penalty`.

Run:
    python test_exterior.py -name Prova_14_07 -n 40
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolfin import vertex_to_dof_map

from Boundary import boundary
from Domain import Domain
from Solver import Solver3D1D


def interior_map(s_restrict, s_full):
    """
    (dofs_sub, dofs_full): indices into the two 3D solution vectors addressing
    the SAME physical vertices.

    SubMesh records, for each of its vertices, the parent-mesh vertex it came
    from. Both spaces are CG1, so a vertex owns exactly one dof, via
    vertex_to_dof_map.
    """
    v2d_sub  = vertex_to_dof_map(s_restrict.W[0])
    v2d_full = vertex_to_dof_map(s_full.W[0])
    parent   = s_restrict.parent_vertex

    return v2d_sub[np.arange(len(parent))], v2d_full[parent]


def compare(label, s_restrict, s_full, tol):
    """Compare a full-box run against the restricted reference."""
    u_ref  = s_restrict.u3d.vector().get_local()
    u_full = s_full.u3d.vector().get_local()

    d_sub, d_full = interior_map(s_restrict, s_full)
    a = u_ref[d_sub]      # interior, reference
    b = u_full[d_full]    # same vertices, full-box run

    rel_l2 = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)
    max_ad = np.abs(a - b).max()

    # exterior dofs = full-box dofs not shared with the submesh
    mask = np.ones(s_full.W[0].dim(), dtype=bool)
    mask[d_full] = False
    ext = u_full[np.flatnonzero(mask)]

    p_ref  = s_restrict.u1d.vector().get_local()
    p_full = s_full.u1d.vector().get_local()
    rel_1d = np.linalg.norm(p_ref - p_full) / max(np.linalg.norm(p_ref), 1e-30)

    ok = (rel_l2 < tol) and (rel_1d < tol)

    m = dict(
        label       = label,
        n_interior  = int(len(a)),
        n_exterior  = int(mask.sum()),
        ndofs       = int(s_full.W[0].dim()),
        niters      = s_full.niters,
        rel_l2_3d   = float(rel_l2),
        max_diff_3d = float(max_ad),
        max_ext     = float(np.abs(ext).max()) if len(ext) else 0.0,
        rel_l2_1d   = float(rel_1d),
        passed      = bool(ok),
    )

    print(f"\n--- {label}  vs  restrict ---")
    print(f"  u3d interior  rel L2 diff      = {rel_l2:.6e}")
    print(f"  u3d interior  max |diff|       = {max_ad:.6e}")
    print(f"  u3d exterior  max |u|          = {m['max_ext']:.6e}  (want ~0)")
    print(f"  u1d           rel L2 diff      = {rel_1d:.6e}")
    print(f"  => {'PASS' if ok else 'FAIL'}  (tol {tol:g})")
    return m


def verdict(m_dir, m_pen, penalty, tol):
    """Derive the conclusions from the numbers, so file and stdout agree."""
    v = []

    if m_dir["passed"]:
        v.append(
            f"DIRICHLET == RESTRICT — PASS (rel L2 {m_dir['rel_l2_3d']:.2e} on the "
            f"3D interior, {m_dir['rel_l2_1d']:.2e} on the 1D field). The exact "
            f"elimination reproduces the restricted solution to solver tolerance, "
            f"and the exterior is identically zero "
            f"(max|u| = {m_dir['max_ext']:.2e}) because those dofs are removed "
            f"rather than penalised. Two independently implemented formulations "
            f"agreeing at this level is strong evidence neither has a bug."
        )
    else:
        v.append(
            f"DIRICHLET != RESTRICT — FAIL (rel L2 {m_dir['rel_l2_3d']:.2e}). These "
            f"MUST agree: a dof pinned to zero contributes nothing to any interior "
            f"equation. The elimination or the dof/vertex mapping is wrong."
        )

    if m_pen["passed"]:
        v.append(
            f"PENALTY — matched to {tol:g} at penalty={penalty:g}. Note this does "
            f"not make the penalty formulation sound; check that the exterior is "
            f"actually pinned (max|u| = {m_pen['max_ext']:.2e}, want ~0). A penalty "
            f"weak enough to leave the interior alone generally fails to constrain "
            f"the exterior at all."
        )
    else:
        v.append(
            f"PENALTY — differs from the reference by {m_pen['rel_l2_3d']:.2e} on "
            f"the interior at penalty={penalty:g}, and leaves the exterior at "
            f"max|u| = {m_pen['max_ext']:.2e} (want ~0). This is EXPECTED and is "
            f"why the penalty formulation was abandoned: the dx(111) integral also "
            f"hits the interface vertices shared with the 222 cells, so a penalty "
            f"strong enough to pin the exterior contaminates the interior, and one "
            f"weak enough to spare the interior does not pin the exterior. There is "
            f"no good value; below ~1e-6 the matrix becomes singular (UMFPACK -5)."
        )

    v.append(
        f"USE 'restrict' ({m_dir['n_interior']} dofs) unless u3d is needed on the "
        f"full background box; then use 'dirichlet' ({m_dir['ndofs']} dofs, same "
        f"answer). Both are exact and parameter-free. Do not use 'penalty'."
    )
    return v


def write_report(path, args, m_dir, m_pen, dims, iters):
    """Write the comparison record: readable .txt + raw .json."""
    payload = {
        "generated" : datetime.now().isoformat(timespec="seconds"),
        "case"      : args.name,
        "n"         : args.n,
        "penalty"   : args.penalty,
        "tol"       : args.tol,
        "dims"      : dims,
        "iters"     : iters,
        "dirichlet" : m_dir,
        "penalty_run": m_pen,
        "verdict"   : verdict(m_dir, m_pen, args.penalty, args.tol),
    }
    with open(path + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    L = []
    L.append("=" * 78)
    L.append(f"EXTERIOR-HANDLING COMPARISON — {args.name}")
    L.append(f"generated {payload['generated']}")
    L.append("=" * 78)
    L.append("")
    L.append("How should the exterior (111) region of the background box be handled?")
    L.append("")
    L.append("  restrict  : V on the SubMesh of the 222 cells. Exterior dofs do not")
    L.append("              exist. No parameter. THE REFERENCE.")
    L.append("  dirichlet : V on the full box; dofs with NO support in any 222 cell")
    L.append("              are eliminated exactly (zero row/col, 1 on the diagonal,")
    L.append("              0 on the rhs). Must reproduce 'restrict' exactly.")
    L.append("  penalty   : V on the full box; exterior pinned by")
    L.append("              penalty*inner(u,v)*dx(111). An APPROXIMATION only.")
    L.append("")
    L.append(f"  n = {args.n}    penalty = {args.penalty:g}    tol = {args.tol:g}")
    L.append("")
    L.append("-" * 78)
    L.append("SIZES")
    L.append("-" * 78)
    L.append(f"  dim(V):  restrict {dims['restrict']}   dirichlet {dims['dirichlet']}"
             f"   penalty {dims['penalty']}")
    L.append(f"  iters :  restrict {iters['restrict']}   dirichlet {iters['dirichlet']}"
             f"   penalty {iters['penalty']}")
    L.append(f"  interior dofs compared : {m_dir['n_interior']}")
    L.append(f"  exterior dofs          : {m_dir['n_exterior']}")
    L.append("")

    for m in (m_dir, m_pen):
        L.append("-" * 78)
        L.append(f"{m['label'].upper()}  vs  restrict"
                 f"   =>  {'PASS' if m['passed'] else 'FAIL'}")
        L.append("-" * 78)
        L.append(f"  u3d interior  rel L2 diff : {m['rel_l2_3d']:.6e}")
        L.append(f"  u3d interior  max |diff|  : {m['max_diff_3d']:.6e}")
        L.append(f"  u3d exterior  max |u|     : {m['max_ext']:.6e}   (want ~0)")
        L.append(f"  u1d           rel L2 diff : {m['rel_l2_1d']:.6e}")
        L.append("")

    L.append("=" * 78)
    L.append("VERDICT")
    L.append("=" * 78)
    for i, line in enumerate(payload["verdict"], 1):
        L.append("")
        words, cur, out = line.split(), "", []
        for w in words:
            if len(cur) + len(w) + 1 > 74:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        out.append(cur)
        L.append(f"{i}. {out[0]}")
        for o in out[1:]:
            L.append(f"   {o}")
    L.append("")
    L.append("=" * 78)

    text = "\n".join(L)
    with open(path + ".txt", "w") as f:
        f.write(text + "\n")

    print("\n" + text)
    print(f"\nReport written:\n  {path}.txt\n  {path}.json")


def build_domain_case(case_name, inlet, outlet, boundary_fn, radius_mode, radius_value=0.01):
    net_dir = os.path.join("nets", case_name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, case_name)
    mesh_prefix = f"{name_stem}_"

    domain = Domain(
        name=name_stem,
        n_vasi=inlet,
        n_ramifications=outlet,
        boundary=boundary_fn,
    ).build()

    if radius_mode == "fixed":
        for i in range(domain.vaso.num_vertices()):
            domain.vaso_radii[i] = float(radius_value)

    domain.export_xdmf()
    return mesh_prefix


def plot_exterior_comparison(case_label, m_dir, m_pen, output_path):
    labels = ["dirichlet", "penalty"]
    rel_3d = [m_dir["rel_l2_3d"], m_pen["rel_l2_3d"]]
    rel_1d = [m_dir["rel_l2_1d"], m_pen["rel_l2_1d"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, rel_3d, width, label="3D rel L2")
    ax.bar(x + width / 2, rel_1d, width, label="1D rel L2")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Exterior handling")
    ax.set_ylabel("Relative L2 error vs restrict")
    ax.set_title(f"Exterior comparison — {case_label} radii")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_exterior_case(args, case_name, radius_mode, radius_value=0.01):
    mesh_prefix = build_domain_case(
        f"{args.name}_{case_name}",
        args.inlet,
        args.outlet,
        boundary,
        radius_mode,
        radius_value=radius_value,
    )

    common = dict(
        path_to_1D_mesh=mesh_prefix,
        boundary=boundary,
        n=args.n,
        sigma3d=1e-3,
        sigma1d=1.0,
        kappa=1.0,
    )

    print("\n" + "=" * 70)
    print(f"RUN 1/3 — {case_name} fixed? {radius_mode == 'fixed'} — exterior='restrict'   (REFERENCE: interior only)")
    print("=" * 70)
    s_res = Solver3D1D(**common, exterior="restrict").build().solve()

    print("\n" + "=" * 70)
    print("RUN 2/3 — exterior='dirichlet'  (full box, exterior eliminated)")
    print("=" * 70)
    s_dir = Solver3D1D(**common, exterior="dirichlet").build().solve()

    print("\n" + "=" * 70)
    print(f"RUN 3/3 — exterior='penalty'    (full box, penalty={args.penalty})")
    print("=" * 70)
    s_pen = Solver3D1D(**common, exterior="penalty",
                       penalty=args.penalty).build().solve()

    print("\n" + "=" * 70)
    print("COMPARISON (on the interior dofs shared with the reference)")
    print("=" * 70)
    print(f"dim(V): restrict={s_res.W[0].dim()}  "
          f"dirichlet={s_dir.W[0].dim()}  penalty={s_pen.W[0].dim()}")
    print(f"iters : restrict={s_res.niters}  "
          f"dirichlet={s_dir.niters}  penalty={s_pen.niters}")

    m_dir = compare("dirichlet", s_res, s_dir, args.tol)
    m_pen = compare("penalty", s_res, s_pen, args.tol)

    dims = dict(restrict=s_res.W[0].dim(),
                 dirichlet=s_dir.W[0].dim(),
                 penalty=s_pen.W[0].dim())
    iters = dict(restrict=s_res.niters,
                 dirichlet=s_dir.niters,
                 penalty=s_pen.niters)

    out_path = os.path.join("test_solution", f"{args.name}_{case_name}_exterior")
    write_report(out_path, args, m_dir, m_pen, dims, iters)

    return m_dir, m_pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-name",    type=str,   default="Prova_14_07")
    ap.add_argument("-inlet",   type=int,   default=4,
                    help="number of inflow vessels to create in the 1D network")
    ap.add_argument("-outlet",  type=int,   default=4,
                    help="number of outlet ramifications per inlet")
    ap.add_argument("-n",       type=int,   default=40,
                    help="3D background mesh resolution")
    ap.add_argument("-penalty", type=float, default=1e-4)
    ap.add_argument("-tol",     type=float, default=1e-9,
                    help="tolerance for dirichlet vs restrict (should be exact)")
    ap.add_argument("-out",     type=str,   default=None,
                    help="path prefix for the result files (.txt + .json). "
                         "Default: ./test_solution/<name>_exterior")
    args = ap.parse_args()

    plot_dir = os.path.join("solution")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join("test_solution"), exist_ok=True)

    print("\nRunning fixed radii exterior comparison\n")
    m_dir_fixed, m_pen_fixed = run_exterior_case(args, "fixed", "fixed", radius_value=0.01)
    fixed_plot_path = os.path.join(plot_dir, f"{args.name}_exterior_fixed_radii.png")
    plot_exterior_comparison("fixed", m_dir_fixed, m_pen_fixed, fixed_plot_path)
    print(f"Saved fixed-radii exterior comparison plot to {fixed_plot_path}")

    print("\nRunning random radii exterior comparison\n")
    m_dir_random, m_pen_random = run_exterior_case(args, "random", "random")
    random_plot_path = os.path.join(plot_dir, f"{args.name}_exterior_random_radii.png")
    plot_exterior_comparison("random", m_dir_random, m_pen_random, random_plot_path)
    print(f"Saved random-radii exterior comparison plot to {random_plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
