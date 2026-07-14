"""
Isolate the averaging operator C.

C is a circle-average: (C u)(s) = (1/|circle|) * int_circle u. Applied to a
polynomial that CG1 can represent exactly, it must reproduce the exact average
on every row, for ANY mesh and ANY radius. Three probes, in increasing strength:

  1. u = 1              -> C u must be 1        (partition of unity)
  2. u = x, y, z        -> C u must be the centerline coordinate, because the
                           circle is centred there and is symmetric.
  3. row sum vs #quadrature points actually used

Probe 3 is the diagnostic: it shows whether rows are short because quadrature
points are being SKIPPED (`if c >= limit: continue`) while curve_measure still
divides by the FULL weight sum.

Run:
    python test_averaging.py -n 20
    python test_averaging.py -n 80
"""
import argparse
import numpy as np
from dolfin import (BoxMesh, Point, FunctionSpace, Function, Mesh, XDMFFile,
                    MeshFunction, interpolate, Expression, cells)
from scipy.sparse import csr_matrix

from Solver import average_matrix_diff_radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-name", type=str, default="Prova_14_07")
    ap.add_argument("-n",    type=int, default=40)
    args = ap.parse_args()

    # background box — identical to Solver._load_meshes
    meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), args.n, args.n, args.n)
    V     = FunctionSpace(meshV, "CG", 1)

    # 1D mesh + radii
    meshQ = Mesh()
    with XDMFFile(f"./nets/{args.name}/{args.name}_marked_mesh.xdmf") as f:
        f.read(meshQ)
    Q      = FunctionSpace(meshQ, "CG", 1)
    radii  = MeshFunction("double", meshQ, 0)
    with XDMFFile(f"./nets/{args.name}/{args.name}_radii.xdmf") as f:
        f.read(radii)

    C_petsc = average_matrix_diff_radii(V, Q, radii)
    indptr, idx, data = C_petsc.getValuesCSR()
    C = csr_matrix((data, idx, indptr), shape=(Q.dim(), V.dim()))

    h = meshV.hmax()
    r = np.array(radii.array())
    print("\n" + "=" * 70)
    print(f"AVERAGING OPERATOR C   (n={args.n}, hmax={h:.5f}, "
          f"R=[{r.min():.5f}, {r.max():.5f}], h/Rmin={h/r.min():.1f})")
    print("=" * 70)

    # --- probe 1: partition of unity ---
    row_sum = np.asarray(C @ np.ones(V.dim())).ravel()
    err1    = np.abs(row_sum - 1.0)
    print("\n1. PARTITION OF UNITY   C @ 1 == 1")
    print(f"   row sum : min={row_sum.min():.6f}  max={row_sum.max():.6f}")
    print(f"   max err : {err1.max():.6e}")
    bad = np.flatnonzero(err1 > 1e-8)
    print(f"   rows off by >1e-8 : {len(bad)} / {C.shape[0]}")
    if len(bad):
        print(f"   worst rows (row, sum): "
              f"{[(int(i), round(float(row_sum[i]), 4)) for i in bad[:6]]}")

    # --- probe 2: linear reproduction ---
    # The circle is centred on the centerline and symmetric, so averaging a
    # linear function returns its value AT THE CENTRE.
    print("\n2. LINEAR REPRODUCTION   C @ x == x_centerline")
    dofx = Q.tabulate_dof_coordinates().reshape((Q.dim(), -1))
    for k, comp in enumerate(["x[0]", "x[1]", "x[2]"]):
        f_ = interpolate(Expression(comp, degree=1), V).vector().get_local()
        got = np.asarray(C @ f_).ravel()
        exact = dofx[:, k]
        e = np.abs(got - exact).max()
        print(f"   {comp}: max |C@u - u_center| = {e:.6e}")

    # --- probe 3: what do the bad rows have in common? ---
    print("\n3. DIAGNOSIS")
    if err1.max() < 1e-8:
        print("   C is a valid partition of unity at this resolution.")
        print("=" * 70)
        return

    excess = row_sum - 1.0
    print(f"   Rows sum to MORE than 1 (max {row_sum.max():.4f}), never less.")
    print( "   => nothing is being dropped; mass is being COUNTED TWICE.")
    print(f"   mean excess over bad rows : {excess[bad].mean():+.4f}")

    # Map each Q dof to its vertex radius, and see whether the excess tracks R/h.
    # A vertex whose circle is small compared to h has all its quadrature points
    # inside ONE cell; a vertex whose circle spans several cells has points on
    # shared faces. Which correlates?
    from dolfin import vertex_to_dof_map
    v2d   = vertex_to_dof_map(Q)
    d2v   = np.zeros(Q.dim(), dtype=int)
    d2v[v2d] = np.arange(Q.dim())
    R_of_row = r[d2v]                    # radius at the vertex owning each row

    print("\n   row | rowsum  |   R     |  R/h   | excess")
    print("   " + "-" * 46)
    order = np.argsort(-excess)
    for i in order[:10]:
        print(f"   {i:>3} | {row_sum[i]:>7.4f} | {R_of_row[i]:.5f} | "
              f"{R_of_row[i]/h:>6.3f} | {excess[i]:+.4f}")

    ok_rows = np.flatnonzero(err1 <= 1e-8)
    if len(ok_rows) and len(bad):
        print(f"\n   mean R/h, GOOD rows : {(R_of_row[ok_rows]/h).mean():.3f}")
        print(f"   mean R/h, BAD  rows : {(R_of_row[bad]/h).mean():.3f}")
        c = np.corrcoef(R_of_row / h, excess)[0, 1]
        print(f"   corr(R/h, excess)   : {c:+.3f}")
        if c > 0.5:
            print("\n   The excess grows with R/h: rows whose circle spans MULTIPLE")
            print("   cells are the broken ones. Quadrature points landing on a")
            print("   shared face are attributed to a cell, and the basis functions")
            print("   of BOTH cells sharing that vertex pick up the weight — the")
            print("   contribution is added more than once.")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
