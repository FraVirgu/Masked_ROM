# =============================================================================
# Entry point
# =============================================================================

import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from dolfin import (
    BoxMesh,
    Point,
    FunctionSpace,
    Function,
    MeshFunction,
    Measure,
    cells,
    facets,
    TrialFunction,
    TestFunction,
    inner,
    assemble,
    as_backend_type,
)

from Solver_full_domain import Solver3D1D
from Analytic_Domain import Domain
from Boundary import SphereBoundary, random_sphere_points
from Decompose_Domain_Analytic_sphere import decomposeDomain, check_sphere_domain_consistency


from dolfin import vertices, edges as mesh_edges, vertex_to_dof_map

def subdomain_to_graph(subdomain, jump=None):
    import networkx as nx
    meshV_sub = subdomain["meshV"]
    V_sub     = subdomain["V_sub"]
    coords    = V_sub.tabulate_dof_coordinates().reshape((V_sub.dim(), -1))
    v2d       = vertex_to_dof_map(V_sub)          # vertex index -> dof

    meshV_sub.init(1)                              # build edge connectivity
    meshV_sub.init(1, 0)

    # Inside/outside per NODE, derived from the cell markers rather than from a
    # point test on the node itself. The solver's notion of "inside" is the
    # 222/111 cell marking (set at cell midpoints in decomposeDomain), and a
    # node sitting just outside the analytic surface can still belong to a 222
    # cell -- those boundary-layer nodes carry a nonzero solution, so a plain
    # point test would mislabel exactly the ones that matter.
    #
    #   inside  = touches only marked cells   (strict interior)
    #   outside = touches no marked cell      (never enters the PDE)
    #   the rest straddle the surface -> marked as boundary
    markers = subdomain.get("V_cell_markers")
    boundary = subdomain.get("boundary")
    n_marked = {}
    n_cells = {}
    if markers is not None:
        meshV_sub.init(3, 0)
        for cell in cells(meshV_sub):
            is_in = int(markers[cell]) == 222
            for vi in cell.entities(0):
                d = int(v2d[int(vi)])
                n_cells[d] = n_cells.get(d, 0) + 1
                n_marked[d] = n_marked.get(d, 0) + (1 if is_in else 0)

    G = nx.Graph()
    for v in vertices(meshV_sub):
        d = int(v2d[v.index()])
        attrs = dict(
            pos=coords[d],
            global_dof=int(subdomain["local_to_global_dof"][d]),
            value=float(subdomain["sol_3d_sub"].vector()[d]),
        )
        if markers is not None:
            nm = n_marked.get(d, 0)
            nc = n_cells.get(d, 0)
            if nm == 0:
                region = "outside"
            elif nm == nc:
                region = "inside"
            else:
                region = "boundary"
            attrs["region"] = region
            attrs["inside"] = region != "outside"      # enters the PDE at all
            attrs["marked_cell_fraction"] = float(nm) / nc if nc else 0.0
        if boundary is not None:
            # The raw analytic predicate, kept separate: it answers "is this
            # point in the domain", not "does this dof enter the local solve".
            attrs["inside_analytic"] = bool(boundary(list(map(float, coords[d]))))
        G.add_node(d, **attrs)

    for e in mesh_edges(meshV_sub):
        a, b = e.entities(0)
        da, db = int(v2d[int(a)]), int(v2d[int(b)])
        G.add_edge(da, db, length=float(np.linalg.norm(coords[da] - coords[db])))

    if jump is not None:
        G = add_jump_edges(G, jump=jump)
    return G


def add_jump_edges(G, jump=1, keep_hop=True):
    """Densify a mesh graph by connecting every node to its k-hop neighbours.

    jump=1 returns the graph unchanged (direct neighbours are already edges);
    jump=2 adds neighbour-of-neighbour shortcuts, jump=3 goes one hop further,
    and so on. This widens the receptive field of a single message-passing
    layer without stacking more layers -- on a BoxMesh, where an interior node
    already has up to 14 tet neighbours, jump=2 typically pushes the degree
    into the hundreds, so raise it carefully.

    The original mesh edges keep hop=1. Added edges carry the hop count they
    were discovered at, plus the straight-line distance between endpoints --
    NOT the path length through the mesh, since a shortcut is a chord, not a
    walk. Filter on `hop` later to recover the original connectivity.

    Returns a new graph; G is left untouched.
    """
    import networkx as nx

    if jump is None or jump < 1:
        raise ValueError(f"jump must be >= 1, got {jump!r}")

    H = G.copy()
    if keep_hop:
        for a, b in H.edges:
            H.edges[a, b]["hop"] = 1
    if jump == 1:
        return H

    pos = nx.get_node_attributes(G, "pos")

    # Breadth-first from each node out to `jump` hops, on the ORIGINAL graph:
    # expanding on H instead would let edges added for one node inflate the
    # neighbourhood seen by the next, so the result would depend on node order.
    for src in G.nodes:
        # Start the walk already one hop out: the direct neighbours are hop 1
        # and are present as real mesh edges, so the first set discovered by
        # the loop below sits at hop 2.
        frontier = set(G.neighbors(src))
        seen = {src} | frontier
        for hop in range(2, jump + 1):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
            for dst in nxt:
                if H.has_edge(src, dst):
                    continue
                attrs = {}
                if src in pos and dst in pos:
                    attrs["length"] = float(
                        np.linalg.norm(np.asarray(pos[src]) - np.asarray(pos[dst]))
                    )
                if keep_hop:
                    attrs["hop"] = hop
                H.add_edge(src, dst, **attrs)
    return H


def subdomain_to_graph_1d(subdomain, solver=None, clip_to_box=True):
    """Build the 1D vessel-centerline graph seen by one subdomain.

    Nodes are 1D mesh vertices, edges are the interval cells. The 1D mesh is
    the GLOBAL centerline (decomposeDomain stores the same meshQ object in
    every subdomain), so with clip_to_box=True the graph is restricted to the
    vertices whose coordinates fall inside this box's extents -- the per-box
    vessel subgraph that the decomposition itself never builds.

    Node ids are global 1D vertex indices, so graphs of neighbouring boxes
    share ids wherever a vessel crosses a cut plane and nx.compose stitches
    them without an explicit join.

    Radii and inlet/outlet markers live on the solver, not in the subdomain
    dict; pass `solver` to attach them as node attributes.
    """
    import networkx as nx

    meshQ = subdomain["meshQ"]
    coords = meshQ.coordinates()

    # 1D dof values: vertex_to_dof_map is valid because Q is CG1.
    sol_1d = subdomain["sol_1d"]
    v2d_Q = vertex_to_dof_map(sol_1d.function_space())
    values = sol_1d.vector().get_local()[v2d_Q]

    radii = None
    markers = None
    if solver is not None:
        if getattr(solver, "Q_radii", None) is not None:
            radii = np.asarray(solver.Q_radii.array(), dtype=float)
        if getattr(solver, "Q_markers", None) is not None:
            markers = np.asarray(solver.Q_markers.array(), dtype=int)

    # Box extents. decomposeDomain snaps these to the structured grid, so the
    # test is a plain inclusive bounding-box membership.
    if clip_to_box:
        pos = subdomain["V_sub"].tabulate_dof_coordinates().reshape(
            (subdomain["V_sub"].dim(), -1)
        )
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        tol = 1e-10

        def inside(c):
            return bool(np.all(c >= lo - tol) and np.all(c <= hi + tol))
    else:
        def inside(c):
            return True

    G = nx.Graph()
    for v in range(meshQ.num_vertices()):
        c = coords[v]
        if not inside(c):
            continue
        attrs = {"pos": c, "value": float(values[v])}
        if radii is not None:
            attrs["radius"] = float(radii[v])
        if markers is not None:
            attrs["marker"] = int(markers[v])
        G.add_node(int(v), **attrs)

    # Keep an edge only when BOTH endpoints survived the clip, so the graph
    # stays a subgraph of the global centerline rather than growing stubs.
    for a, b in meshQ.cells():
        a, b = int(a), int(b)
        if a in G and b in G:
            G.add_edge(a, b,
                       length=float(np.linalg.norm(coords[a] - coords[b])))
    return G



def add_coupling_edges(
    G3d,
    G1d,
    subdomain,
    solver=None,
    degree=10,
    min_radius=0.005,
    weight_tol=0.0,
    prefix_3d="v",
    prefix_1d="q",
):
    """Join the 3D and 1D graphs with the edges of the averaging operator.

    This mirrors average_matrix_diff_radii in Solver_full_domain.py: around
    every 1D vertex it lays a circle of radius R(s) in the plane normal to the
    centerline, evaluates the 3D basis functions at the circle's quadrature
    points, and connects that 1D node to each 3D dof whose basis function is
    nonzero there. The edge weight is exactly the matrix entry C[q, v] -- the
    quadrature-weighted average of the 3D basis over the vessel wall -- so the
    coupling appears in the graph as the discrete operator, not as an invented
    proximity rule.

    Node ids are prefixed because the two graphs number from 0 independently:
    3D nodes become ("v", local_dof) and 1D nodes ("q", global_vertex).

    Quadrature points landing outside the LOCAL box are dropped and the row is
    renormalised by the weight that stayed inside, matching the used_measure
    logic of the global assembly. A 1D vertex whose circle misses the box
    entirely simply gets no coupling edges -- for a subdomain that is ordinary,
    whereas the global assembly raises there.

    Returns a new graph; G3d and G1d are untouched.
    """
    import networkx as nx
    from dolfin import Cell, Point as _Point
    from xii import Circle

    V_sub = subdomain["V_sub"]
    meshV_sub = V_sub.mesh()
    meshQ = subdomain["meshQ"]
    mesh_x = meshQ.coordinates()

    radii = None
    if solver is not None and getattr(solver, "Q_radii", None) is not None:
        radii = np.asarray(solver.Q_radii.array(), dtype=float)
    if radii is None:
        raise ValueError(
            "add_coupling_edges needs the vessel radii; pass solver=... so "
            "solver.Q_radii is available (the subdomain dict does not carry it)."
        )

    tree = meshV_sub.bounding_box_tree()
    limit = meshV_sub.num_cells()
    Vel = V_sub.element()
    V_dm = V_sub.dofmap()
    basis_values = np.zeros(Vel.space_dimension())

    # Per-vertex tangent and radius, accumulated edge by edge exactly as the
    # assembly does: each incident edge direction is flipped when needed so the
    # contributions do not cancel at a vertex where the centerline reverses.
    n_vtx = meshQ.num_vertices()
    tangent = np.zeros((n_vtx, 3))
    r_sum = np.zeros(n_vtx)
    r_cnt = np.zeros(n_vtx)
    for a, b in meshQ.cells():
        a, b = int(a), int(b)
        e = mesh_x[a] - mesh_x[b]
        nrm = np.linalg.norm(e)
        if nrm > 0:
            e = e / nrm
        for v in (a, b):
            tangent[v] += e if np.dot(tangent[v], e) >= 0 else -e
        R = max(0.5 * (radii[a] + radii[b]), min_radius)
        for v in (a, b):
            r_sum[v] += R
            r_cnt[v] += 1

    H = nx.Graph()
    H.add_nodes_from(((prefix_3d, n), d) for n, d in G3d.nodes(data=True))
    H.add_nodes_from(((prefix_1d, n), d) for n, d in G1d.nodes(data=True))
    for a, b, d in G3d.edges(data=True):
        H.add_edge((prefix_3d, a), (prefix_3d, b), kind="mesh3d", **d)
    for a, b, d in G1d.edges(data=True):
        H.add_edge((prefix_1d, a), (prefix_1d, b), kind="mesh1d", **d)

    n_coupled = 0
    for v in G1d.nodes:
        v = int(v)
        nrm = np.linalg.norm(tangent[v])
        if nrm < 1e-14:
            a, b = meshQ.cells()[0]
            normal = mesh_x[int(a)] - mesh_x[int(b)]
        else:
            normal = tangent[v] / nrm

        Ri = r_sum[v] / max(r_cnt[v], 1)
        quadrature = Circle(radius=Ri, degree=degree).quadrature(mesh_x[v], normal)

        data = {}
        used_measure = 0.0
        for ip, wq in zip(quadrature.points, quadrature.weights):
            c = tree.compute_first_entity_collision(_Point(*ip))
            if c >= limit:
                continue
            used_measure += wq
            Vcell = Cell(meshV_sub, c)
            basis_values[:] = Vel.evaluate_basis_all(
                ip, Vcell.get_vertex_coordinates(), Vcell.orientation()
            )
            for col, val in zip(V_dm.cell_dofs(c), basis_values * wq):
                data[int(col)] = data.get(int(col), 0.0) + float(val)

        if used_measure <= 0.0:
            continue

        n_coupled += 1
        for col, val in data.items():
            w = val / used_measure
            if abs(w) <= weight_tol:
                continue
            if (prefix_3d, col) not in H:
                continue
            H.add_edge((prefix_1d, v), (prefix_3d, col), weight=w, kind="coupling")

    H.graph["n_coupled_1d_nodes"] = n_coupled
    H.graph["n_1d_nodes"] = G1d.number_of_nodes()
    return H


class SurrogateModel():
    """Turn the decomposed 3D-1D problem into graphs for a surrogate.

    Per subdomain it builds three things and stores them back in the dict:

        graphV -- the 3D mesh graph (nodes = local dofs)
        graphQ -- the 1D vessel graph seen by this box (nodes = global 1D vertices)
        graph  -- the two joined by the averaging operator's edges

    `graph` is the per-box training sample. `global_graph` composes every box
    into one graph, stitching neighbours on their shared cut planes.
    """

    def __init__(self, subdomains, solver=None, jump=None, couple=True):
        self.subdomains = subdomains
        self.solver = solver
        for subdomain in subdomains:
            subdomain["graphV"] = subdomain_to_graph(subdomain, jump=jump)
            subdomain["graphQ"] = subdomain_to_graph_1d(subdomain, solver=solver)
            if couple and solver is not None:
                subdomain["graph"] = add_coupling_edges(
                    subdomain["graphV"], subdomain["graphQ"],
                    subdomain, solver=solver,
                )
            else:
                # No radii available -> the coupling edges cannot be built, so
                # `graph` is just the two meshes side by side, unconnected.
                import networkx as nx
                subdomain["graph"] = nx.union(
                    nx.relabel_nodes(subdomain["graphV"], lambda n: ("v", n)),
                    nx.relabel_nodes(subdomain["graphQ"], lambda n: ("q", n)),
                )

    def global_graph(self):
        """Compose the per-box graphs into one, stitched on the cut planes.

        3D nodes are relabelled from local dofs to ("v", global_dof) so the two
        copies of a shared cut-plane node collapse into one; 1D nodes already
        carry global vertex ids and merge on their own. What this does NOT
        recover are the tet edges that crossed a cut in the original global
        mesh: each box is tetrahedralised independently, so the interface has
        the right nodes but a thinner edge set than the true global graph.
        """
        import networkx as nx

        H = nx.Graph()
        for sd in self.subdomains:
            g = sd["graph"]
            l2g = sd["local_to_global_dof"]
            H = nx.compose(
                H,
                nx.relabel_nodes(
                    g,
                    {n: (("v", int(l2g[n[1]])) if n[0] == "v" else n) for n in g},
                ),
            )
        return H

    def interface_nodes(self, subdomain):
        """Global dofs of the 3D nodes on this box's artificial cut planes.

        Read off the P projector built by decomposeDomain: P is diagonal, so
        its stored indices ARE the selected interface dofs.
        """
        return set(subdomain["P"].indices.tolist())



if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a simple analytic (non-OpenCCO) vascular domain and "
                    "solve the 3D-1D problem with the no-penalty solver.",
    )
    parser.add_argument("-name",   type=str, required=True,
                        help="subfolder name inside nets/ (e.g. sphere01)")
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
                        help="radius of the spherical boundary")
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

    

    boundary = SphereBoundary(
        radius=args.radius,
        inlet_points=random_sphere_points(
            40,
            x_sign=-1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        outlet_points=random_sphere_points(
            40,
            x_sign=+1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        border_eps=10e-1,
    )

    check_sphere_domain_consistency(
        boundary=boundary,
        n_min=-args.radius,
        n_max=args.radius,
    )

    # --- 1. build the analytic-boundary domain (unit sphere from Boundary.py) ---
    domain = Domain(
        name            = name_stem,
        n_vasi          = args.inlet,
        n_ramifications = args.outlet,
        boundary        = boundary,
        radius_mean = 0.005, 
        radius_std = 0.001,
        radius_min = 0.001,
        radius_max = 0.01,
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
    solver  = Solver3D1D(
        path_to_1D_mesh = mesh_prefix,
        boundary        = boundary,
        n               = args.n,
        sigma3d         = args.sigma3d,
        sigma1d         = args.sigma1d,
        kappa           = args.kappa,
        exterior        = "dirichlet",
    ).build().solve()

    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")



    subdomains = decomposeDomain(solver, boundary)