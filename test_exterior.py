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
import numpy as np
from dolfin import vertex_to_dof_map

from CCO_Domain import CCOVascularMesh, boundary_from_obj
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

    print(f"\n--- {label}  vs  restrict ---")
    print(f"  u3d interior  rel L2 diff      = {rel_l2:.6e}")
    print(f"  u3d interior  max |diff|       = {max_ad:.6e}")
    print(f"  u3d exterior  max |u|          = {np.abs(ext).max():.6e}  (want ~0)")
    print(f"  u1d           rel L2 diff      = {rel_1d:.6e}")
    print(f"  => {'PASS' if ok else 'FAIL'}  (tol {tol:g})")
    return ok, rel_l2, rel_1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-name",    type=str,   default="Prova_14_07")
    ap.add_argument("-graph",   type=str,   default="graphExport")
    ap.add_argument("-obj",     type=str,   default="graphExport/domain.obj")
    ap.add_argument("-n",       type=int,   default=40)
    ap.add_argument("-penalty", type=float, default=1e-4)
    ap.add_argument("-tol",     type=float, default=1e-9,
                    help="tolerance for dirichlet vs restrict (should be exact)")
    args = ap.parse_args()

    cco = CCOVascularMesh(graph_folder=args.graph, obj_path=args.obj,
                          name=args.name)
    cco.load().build().export_xdmf()
    boundary = boundary_from_obj(obj_path=args.obj, scale=cco.scale,
                                 center=cco.center)

    common = dict(
        path_to_1D_mesh = f"./nets/{args.name}/{args.name}_",
        boundary        = boundary,
        n               = args.n,
        sigma3d         = 1e-3,
        sigma1d         = 1.0,
        kappa           = 1.0,
    )

    print("\n" + "=" * 70)
    print("RUN 1/3 — exterior='restrict'   (REFERENCE: interior only)")
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

    ok_dir, _, _ = compare("dirichlet", s_res, s_dir, args.tol)
    ok_pen, r_pen, _ = compare("penalty", s_res, s_pen, args.tol)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if ok_dir:
        print("PASS  dirichlet == restrict — exact elimination reproduces the")
        print("      restricted solution, and keeps u3d on the full box mesh.")
    else:
        print("FAIL  dirichlet != restrict — these MUST agree; the elimination")
        print("      or the dof/vertex mapping is wrong.")

    if not ok_pen:
        print(f"      penalty differs by {r_pen:.2e} on the interior, as expected:")
        print( "      the dx(111) integral also hits the interface vertices, so it")
        print( "      cannot pin the exterior without perturbing the interior.")
    else:
        print(f"      penalty also matched to {args.tol:g} at this value.")

    # only the dirichlet agreement is a correctness requirement
    return 0 if ok_dir else 1


if __name__ == "__main__":
    raise SystemExit(main())
