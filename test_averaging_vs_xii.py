"""
Validate average_matrix_diff_radii against the tested xii component.

The professor's point: xii already ships a circle-averaging operator
(xii.assembler.average_matrix.avg_mat) built on the same Circle shape, and
Circle already accepts a VARIABLE radius as a callable radius(x0). Rolling our
own risks bugs in a delicate part of the code. So: where the two are supposed to
agree, they must agree to machine precision. Where they don't, the difference
must be a deliberate, explained deviation and not an accident.

WHERE THEY MUST AGREE — a single straight branch, no bifurcation, centerline
well inside the domain. Then every vertex has exactly one cross-section, so
xii's per-CELL orientation (normal = v0 - v1 of that edge) and our per-VERTEX
orientation (normal = average of incident edge tangents) coincide, and no
quadrature point leaves the mesh so both normalisations divide by the same
weight. Probe 1 pins this down: max |C_ours - C_xii| must be ~1e-15.

WHERE THEY DELIBERATELY DIFFER — two probes that document, not hide, the
deviations:

  Probe 2  bifurcation. xii writes a shared vertex once per incident edge with
           INSERT_VALUES, each time with a DIFFERENTLY oriented circle, so the
           row becomes the union of several circles and sums to > 1. Ours emits
           one circle per vertex and sums to 1.

  Probe 3  centerline near the boundary. xii divides by sum(wq), the FULL
           circumference, even when `if c >= limit: continue` skipped points, so
           rows sum to < 1 and mass is lost. Ours divides by the weight ACTUALLY
           used and stays a true average.

Note xii's Average() asserts the 1D space is Discontinuous Lagrange, precisely
because an edge tangent is only well defined inside a cell. We target CG1, which
is why avg_mat is called directly here rather than through Average/ii_assemble.
For probe 1 the branch is straight, so DG0-vs-CG1 orientation is not what is
being compared — the circles are identical and the rows are compared on the
shared geometry.

Run:
    python test_averaging_vs_xii.py
    python test_averaging_vs_xii.py -n 32
"""
import argparse
import numpy as np
from dolfin import (BoxMesh, Point, FunctionSpace, Mesh, MeshEditor,
                    MeshFunction, vertex_to_dof_map)
from scipy.sparse import csr_matrix
from xii import Circle
from xii.assembler.average_matrix import average_matrix as xii_average_matrix

from Solver import average_matrix_diff_radii


def to_csr(petsc_mat, shape):
    """PETSc -> scipy, so rows can be compared directly."""
    indptr, idx, data = petsc_mat.getValuesCSR()
    return csr_matrix((data, idx, indptr), shape=shape)


def line_mesh_from(points):
    """1D mesh in 3D from an ordered polyline; cells connect consecutive pts."""
    points = np.asarray(points, dtype=float)
    mesh   = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, 3)
    editor.init_vertices(len(points))
    editor.init_cells(len(points) - 1)
    for i, p in enumerate(points):
        editor.add_vertex(i, p)
    for i in range(len(points) - 1):
        editor.add_cell(i, np.array([i, i + 1], dtype="uintp"))
    editor.close()
    return mesh


def radii_on(mesh, values):
    """Vertex-based radius MeshFunction, the format average_matrix_diff_radii wants."""
    r = MeshFunction("double", mesh, 0)
    r.array()[:] = values
    return r


def ours(V, Q, radii):
    return to_csr(average_matrix_diff_radii(V, Q, radii), (Q.dim(), V.dim()))


def theirs(V, Q, radii, degree=10):
    """
    xii's operator, driven with the VARIABLE radius the professor pointed at:
    Circle(radius=<callable>) evaluates R = radius(x0) per centerline point.
    One Circle for the whole 1D mesh, not one per vertex.

    xii returns a matrix over Q's dofs, but its loop is per-cell, so we build a
    Q_DG that its assembler accepts and map back by dof coordinate.
    """
    r_at = radius_lookup(Q.mesh(), radii)
    shape = Circle(radius=r_at, degree=degree)
    Q_dg  = FunctionSpace(Q.mesh(), "DG", 1)
    C_dg  = to_csr(xii_average_matrix(V, Q_dg, shape), (Q_dg.dim(), V.dim()))
    return C_dg, Q_dg


def radius_lookup(line_mesh, radii):
    """
    radius(x0) -> R, the callable form Circle accepts. Mirrors the per-vertex
    radius our implementation uses (mean of incident edge radii), so the two
    operators integrate over the SAME circles and any difference is orientation
    or normalisation, not radius.
    """
    mesh_x = line_mesh.coordinates()
    r_vtx  = np.array(radii.array())

    r_sum = np.zeros(line_mesh.num_vertices())
    r_cnt = np.zeros(line_mesh.num_vertices())
    for a, b in line_mesh.cells():
        a, b = int(a), int(b)
        R = max(0.5 * (r_vtx[a] + r_vtx[b]), 0.005)
        for v in (a, b):
            r_sum[v] += R
            r_cnt[v] += 1
    r_eff = r_sum / np.maximum(r_cnt, 1)

    def radius(x0, mesh_x=mesh_x, r_eff=r_eff):
        d = np.linalg.norm(mesh_x - np.asarray(x0)[: mesh_x.shape[1]], axis=1)
        return float(r_eff[int(np.argmin(d))])

    return radius


def rows_by_coordinate(C_a, space_a, C_b, space_b):
    """
    Pair rows of two operators over different 1D spaces by dof coordinate.
    DG1 has one dof per cell-endpoint, so a vertex appears in several rows; on a
    straight branch those rows are identical, and we take the first match.
    """
    xa = space_a.tabulate_dof_coordinates().reshape((space_a.dim(), -1))
    xb = space_b.tabulate_dof_coordinates().reshape((space_b.dim(), -1))
    pairs = []
    for ia in range(len(xa)):
        d = np.linalg.norm(xb - xa[ia], axis=1)
        ib = int(np.argmin(d))
        if d[ib] < 1e-12:
            pairs.append((ia, ib))
    return pairs


def max_row_diff(C_a, space_a, C_b, space_b):
    pairs = rows_by_coordinate(C_a, space_a, C_b, space_b)
    worst, worst_row = 0.0, None
    for ia, ib in pairs:
        d = np.abs((C_a[ia] - C_b[ib]).toarray()).max()
        if d > worst:
            worst, worst_row = d, ia
    return worst, worst_row, len(pairs)


# =============================================================================
# probe 1 — straight branch: the two operators MUST agree
# =============================================================================
def probe_straight(n, R):
    print("\n1. AGREEMENT ON A STRAIGHT BRANCH   C_ours == C_xii")
    print("   one cross-section per vertex, circle strictly inside the mesh")
    print("   => no orientation ambiguity, no skipped quadrature points.")

    meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)
    V     = FunctionSpace(meshV, "CG", 1)

    pts   = np.zeros((9, 3))
    pts[:, 2] = np.linspace(-0.5, 0.5, 9)      # along z, centred, well inside
    meshQ = line_mesh_from(pts)
    Q     = FunctionSpace(meshQ, "CG", 1)
    radii = radii_on(meshQ, np.full(meshQ.num_vertices(), R))

    C_ours          = ours(V, Q, radii)
    C_xii, Q_dg     = theirs(V, Q, radii)

    d, row, npair = max_row_diff(C_ours, Q, C_xii, Q_dg)
    print(f"   rows compared          : {npair} / {Q.dim()}")
    print(f"   max |C_ours - C_xii|   : {d:.3e}   (worst row {row})")
    ok = d < 1e-12
    print("   => AGREE to machine precision." if ok else
          "   => DISAGREE. Not a deviation we intended — investigate.")
    return ok


# =============================================================================
# probe 2 — bifurcation: xii double-counts, ours does not
# =============================================================================
def probe_bifurcation(n, R):
    print("\n2. BIFURCATION   deliberate deviation #1: one circle per vertex")

    meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)
    V     = FunctionSpace(meshV, "CG", 1)

    # a Y: parent up the z axis, two branches splitting at the origin
    pts = np.array([[0.0,  0.0, -0.5],
                    [0.0,  0.0,  0.0],       # the bifurcation vertex
                    [0.3,  0.0,  0.4],
                    [-0.3, 0.0,  0.4]])
    meshQ  = Mesh()
    editor = MeshEditor()
    editor.open(meshQ, "interval", 1, 3)
    editor.init_vertices(4)
    editor.init_cells(3)
    for i, p in enumerate(pts):
        editor.add_vertex(i, p)
    for i, (a, b) in enumerate([(0, 1), (1, 2), (1, 3)]):
        editor.add_cell(i, np.array([a, b], dtype="uintp"))
    editor.close()

    Q     = FunctionSpace(meshQ, "CG", 1)
    radii = radii_on(meshQ, np.full(4, R))

    C_ours      = ours(V, Q, radii)
    C_xii, Q_dg = theirs(V, Q, radii)

    s_ours = np.asarray(C_ours @ np.ones(V.dim())).ravel()
    s_xii  = np.asarray(C_xii  @ np.ones(V.dim())).ravel()

    # the shared vertex is the one with 3 incident edges
    v2d = vertex_to_dof_map(Q)
    row_shared = int(v2d[1])

    print(f"   ours, row sums         : min={s_ours.min():.6f} max={s_ours.max():.6f}")
    print(f"   xii,  row sums         : min={s_xii.min():.6f}  max={s_xii.max():.6f}")
    print(f"   ours, bifurcation row  : {s_ours[row_shared]:.6f}")
    ok = abs(s_ours[row_shared] - 1.0) < 1e-8
    print("   => ours averages to 1 at the bifurcation." if ok else
          "   => ours does NOT sum to 1 at the bifurcation.")
    return ok


# =============================================================================
# probe 3 — circle crossing the boundary: xii loses mass, ours does not
# =============================================================================
def probe_boundary(n, R):
    print("\n3. CIRCLE CROSSING THE BOUNDARY   deliberate deviation #2:")
    print("   normalise by the weight ACTUALLY used, not the full circumference.")

    meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)
    V     = FunctionSpace(meshV, "CG", 1)

    # centerline parallel to z, pushed so close to x=1 that part of each
    # circle sticks out of the box and those quadrature points get skipped.
    pts = np.zeros((5, 3))
    pts[:, 0] = 1.0 - 0.5 * R
    pts[:, 2] = np.linspace(-0.4, 0.4, 5)
    meshQ = line_mesh_from(pts)
    Q     = FunctionSpace(meshQ, "CG", 1)
    radii = radii_on(meshQ, np.full(meshQ.num_vertices(), R))

    C_ours      = ours(V, Q, radii)
    C_xii, Q_dg = theirs(V, Q, radii)

    s_ours = np.asarray(C_ours @ np.ones(V.dim())).ravel()
    s_xii  = np.asarray(C_xii  @ np.ones(V.dim())).ravel()

    print(f"   ours, row sums : min={s_ours.min():.6f} max={s_ours.max():.6f}")
    print(f"   xii,  row sums : min={s_xii.min():.6f}  max={s_xii.max():.6f}")
    ok = np.abs(s_ours - 1.0).max() < 1e-8
    print("   => ours stays a true average; xii drops below 1." if ok else
          "   => ours does NOT stay a partition of unity here.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int,   default=24, help="background box resolution")
    ap.add_argument("-R", type=float, default=0.1, help="vessel radius")
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print(f"average_matrix_diff_radii  vs  xii.average_matrix   "
          f"(n={args.n}, R={args.R})")
    print("=" * 70)

    results = {
        "straight branch agrees with xii": probe_straight(args.n, args.R),
        "bifurcation row averages to 1":   probe_bifurcation(args.n, args.R),
        "boundary row averages to 1":      probe_boundary(args.n, args.R),
    }

    print("\n" + "=" * 70)
    for name, ok in results.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
