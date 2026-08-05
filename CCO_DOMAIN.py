import os
import argparse
import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction, XDMFFile, MPI, BoxMesh, Point, cells
from scipy.ndimage import binary_fill_holes
from Boundary import Boundary
from Solver_full_domain import Solver3D1D


# =============================================================================
# Boundary from OBJ
# =============================================================================

def boundary_from_obj(obj_path: str, center: np.ndarray | None = None) -> Boundary:
    """
    Build a Boundary from an OBJ file without rescaling the geometry.

    The coordinates are translated so the OBJ centroid lies at the origin.
    The resulting bbox is expressed in the centered coordinate system.
    """

    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = map(float, line.split()[1:4])
                verts.append([x, y, z])
    verts = np.array(verts)

    if center is None:
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0

    verts_centered = verts - center

    vmin_vox = np.floor(verts_centered.min(axis=0)).astype(int)
    vmax_vox = np.floor(verts_centered.max(axis=0)).astype(int)
    dims = vmax_vox - vmin_vox + 2

    mask = np.zeros(dims, dtype=bool)
    for v in verts_centered:
        ix = np.clip(int(np.floor(v[0])) - vmin_vox[0], 0, dims[0] - 1)
        iy = np.clip(int(np.floor(v[1])) - vmin_vox[1], 0, dims[1] - 1)
        iz = np.clip(int(np.floor(v[2])) - vmin_vox[2], 0, dims[2] - 1)
        mask[ix, iy, iz] = True

    mask = binary_fill_holes(mask)
    print(f"Mask: {mask.shape}, {mask.sum()} inside voxels / {mask.size} total")

    p_min = vmin_vox.astype(float)
    p_max = (vmax_vox + 1).astype(float)
    bbox = (
        (float(p_min[0]), float(p_max[0])),
        (float(p_min[1]), float(p_max[1])),
        (float(p_min[2]), float(p_max[2])),
    )

    print("Boundary bbox (centered):")
    print(f"  x: [{bbox[0][0]:.4f}, {bbox[0][1]:.4f}]  size={bbox[0][1]-bbox[0][0]:.4f}")
    print(f"  y: [{bbox[1][0]:.4f}, {bbox[1][1]:.4f}]  size={bbox[1][1]-bbox[1][0]:.4f}")
    print(f"  z: [{bbox[2][0]:.4f}, {bbox[2][1]:.4f}]  size={bbox[2][1]-bbox[2][0]:.4f}")

    return Boundary(source=mask, bbox=bbox, inlet_points=None, outlet_points=None)


# =============================================================================
# Path helpers
# =============================================================================

def resolve_graph_paths(graph_folder: str, obj_path: str):
    """Resolve graph and OBJ paths to an existing dataset on disk."""

    candidates = []
    if graph_folder:
        candidates.append(graph_folder)
    candidates.extend(["graph/liver_toy", "graphExport"])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        if not os.path.isdir(candidate):
            continue

        required_files = [
            os.path.join(candidate, "vertex.dat"),
            os.path.join(candidate, "edges.dat"),
            os.path.join(candidate, "radius.dat"),
        ]
        if all(os.path.exists(path) for path in required_files):
            resolved_obj = obj_path
            if not resolved_obj or not os.path.exists(resolved_obj):
                alt_obj = os.path.join(candidate, "domain.obj")
                if os.path.exists(alt_obj):
                    resolved_obj = alt_obj
            return candidate, resolved_obj

    return graph_folder, obj_path


# =============================================================================
# CCOVascularMesh (translation-only)
# =============================================================================

class CCOVascularMesh:
    """
    Loads a CCO vascular graph, translates it so the OBJ center lies at the
    origin, builds a 1D FEniCS mesh with inlet/outlet markers and per-vertex
    radii, and exports to XDMF for the full-domain solver.
    """

    def __init__(self, graph_folder: str, obj_path: str, name: str = "cco"):
        self.graph_folder, self.obj_path = resolve_graph_paths(graph_folder, obj_path)
        self.name = name
        self.output_dir = os.path.join("nets", name)

        self.vertices = None
        self.edges = None
        self.radii = None
        self.center = None

        self.mesh1 = None
        self.inlet = None
        self.leaves = None
        self.markers = None
        self.vertex_radii = None
        self.vaso = None
        self.vaso_markers = None
        self.vaso_radii = None

    def load(self):
        self._load_and_center()
        return self

    def build(self):
        self._require("vertices", "edges", "radii")
        self._build_mesh()
        self._mark_vertices()
        self._trace_paths()
        self._transfer_radii()
        return self

    def build_full_domain(self, n: int = 20):
        """Build the full 3D domain mesh and the associated cell markers."""
        self._require("center")
        boundary = boundary_from_obj(self.obj_path, center=self.center)

        bbox = boundary._bbox
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
        meshV = BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), n, n, n)

        cell_markers = MeshFunction("size_t", meshV, 3, 111)
        inside_count = 0
        for cell in cells(meshV):
            mp = cell.midpoint()
            if boundary([mp.x(), mp.y(), mp.z()]):
                cell_markers[cell] = 222
                inside_count += 1

        total = meshV.num_cells()
        print(f"3D mesh: {total} cells | inside domain: {inside_count} ({inside_count / total * 100:.1f}%)")
        return meshV, cell_markers, boundary

    def export_xdmf(self):
        self._require("vaso", "vaso_markers", "vaso_radii")
        os.makedirs(self.output_dir, exist_ok=True)

        def path(suffix):
            return os.path.join(self.output_dir, f"{suffix}.xdmf")

        for fname, obj, rename in [
            ("marked_mesh", self.vaso, None),
            ("markers", self.vaso_markers, None),
            ("radii", self.vaso_radii, ("radius", "vessel radius")),
        ]:
            f = XDMFFile(MPI.comm_world, path(fname))
            f.parameters["flush_output"] = True
            if rename:
                obj.rename(*rename)
            f.write(obj)
            f.close()

        print(f"XDMF written → {self.output_dir}/")
        print(f"  {self.name}_marked_mesh.xdmf")
        print(f"  {self.name}_markers.xdmf")
        print(f"  {self.name}_radii.xdmf")
        return self

    def _load_and_center(self):
        vertices = {}
        with open(f"{self.graph_folder}/vertex.dat") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y, z = map(float, line.split())
                vertices[i] = np.array([x, y, z])

        edges = []
        with open(f"{self.graph_folder}/edges.dat") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                edges.append((int(parts[0]), int(parts[1])))

        radii = {}
        with open(f"{self.graph_folder}/radius.dat") as f:
            i = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                radii[i] = float(line)
                i += 1

        invalid = {i: r for i, r in radii.items() if r <= 0}
        if invalid:
            raise ValueError(
                f"{len(invalid)} vertices with radius <= 0:\n"
                + "\n".join(f"  vertex {i}: r={r}" for i, r in invalid.items())
            )

        obj_verts = []
        with open(self.obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = map(float, line.split()[1:4])
                    obj_verts.append([x, y, z])
        obj_verts = np.array(obj_verts)

        self.center = (obj_verts.min(axis=0) + obj_verts.max(axis=0)) / 2.0
        self.vertices = {i: v - self.center for i, v in vertices.items()}
        self.radii = radii
        self.edges = edges

        g = np.array(list(self.vertices.values()))
        print(f"Loaded {len(vertices)} vertices, {len(edges)} edges")
        print(f"Center at origin: {np.round(self.center, 3)}")
        print("Graph range after centering:")
        print(f"  x=[{g[:,0].min():.4f}, {g[:,0].max():.4f}]  "
              f"y=[{g[:,1].min():.4f}, {g[:,1].max():.4f}]  "
              f"z=[{g[:,2].min():.4f}, {g[:,2].max():.4f}]")
        print(f"Radii: min={min(self.radii.values()):.6f}  max={max(self.radii.values()):.6f}")

    def _build_mesh(self):
        mesh1 = Mesh()
        me = MeshEditor()
        me.open(mesh1, "interval", 1, 3)
        me.init_vertices(len(self.vertices))
        me.init_cells(len(self.edges))

        for i, v in self.vertices.items():
            me.add_vertex(i, v)
        for k, (a, b) in enumerate(self.edges):
            me.add_cell(k, np.array([a, b], dtype=np.uintp))

        me.close()
        mesh1.init()
        self.mesh1 = mesh1

        vertex_radii = MeshFunction("double", mesh1, 0, 0.0)
        for i, r in self.radii.items():
            vertex_radii[i] = r
        self.vertex_radii = vertex_radii

    def _mark_vertices(self):
        degree = {i: 0 for i in self.vertices}
        for a, b in self.edges:
            degree[a] += 1
            degree[b] += 1

        leaves = [i for i in self.vertices if degree[i] == 1]
        inlet = max(leaves, key=lambda i: self.radii[i])
        outlets = [i for i in leaves if i != inlet]

        markers = MeshFunction("size_t", self.mesh1, 0, 0)
        for i in range(self.mesh1.num_vertices()):
            if i == inlet:
                markers[i] = 111
            elif i in outlets:
                markers[i] = 999
            else:
                markers[i] = 555

        self.inlet = inlet
        self.leaves = outlets
        self.markers = markers

        print(f"Inlet: node {inlet} (r={self.radii[inlet]:.4f}) | "
              f"Outlets: {len(outlets)} | "
              f"Interior: {self.mesh1.num_vertices() - 1 - len(outlets)}")

    def _trace_paths(self):
        import networkx as nx
        from xii import EmbeddedMesh, transfer_markers

        G, edge_indices = nx.Graph(), {}
        for k, (a, b) in enumerate(self.edges):
            w = float(np.linalg.norm(self.vertices[a] - self.vertices[b]))
            G.add_edge(a, b, weight=w)
            edge_indices[tuple(sorted((a, b)))] = k

        facet_f = MeshFunction("size_t", self.mesh1, 1, 0)
        for out in self.leaves:
            path = nx.shortest_path(G, source=self.inlet, target=out, weight="weight")
            for a, b in zip(path[:-1], path[1:]):
                facet_f[edge_indices[tuple(sorted((a, b)))]] = 1

        self.vaso = EmbeddedMesh(facet_f, 1)
        self.vaso_markers = transfer_markers(self.vaso, self.markers)
        print(f"Vaso mesh: {self.vaso.num_vertices()} vertices, {self.vaso.num_cells()} edges")

    def _transfer_radii(self):
        coord_to_radius = {
            tuple(np.round(self.mesh1.coordinates()[i], 10)): self.vertex_radii[i]
            for i in range(self.mesh1.num_vertices())
        }

        vaso_radii = MeshFunction("double", self.vaso, 0, 0.0)
        fallback_count = 0
        for i in range(self.vaso.num_vertices()):
            key = tuple(np.round(self.vaso.coordinates()[i], 10))
            r = coord_to_radius.get(key, None)
            if r is None or r <= 0:
                fallback_count += 1
                vaso_radii[i] = 1e-3
            else:
                vaso_radii[i] = r

        if fallback_count > 0:
            print(f"Warning: {fallback_count} vaso vertices → fallback radius 1e-3")
        else:
            r_arr = vaso_radii.array()
            print(f"Vaso radii OK: min={r_arr.min():.6f}  max={r_arr.max():.6f}")

        self.vaso_radii = vaso_radii

    def _require(self, *attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(f"'{attr}' not available — call load()/build() first.")

    def __repr__(self):
        return (
            f"CCOVascularMesh(graph='{self.graph_folder}', obj='{self.obj_path}', "
            f"name='{self.name}', built={self.vaso is not None})"
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-name", type=str, required=True, help="subfolder name inside nets/ (e.g. liver05)")
    parser.add_argument("-graph", type=str, default="graph/liver_toy", help="folder with vertex.dat / edges.dat / radius.dat")
    parser.add_argument("-obj", type=str, default="graph/liver_toy/domain.obj", help="path to liver domain OBJ file")
    parser.add_argument("-n", type=int, default=100, help="3D mesh resolution")
    parser.add_argument("-rad", type=float, default=0.05, help="unused legacy argument kept for compatibility")
    parser.add_argument("-sigma1d", type=float, default=10.0, help="1D conductivity (geometry-scaled starting value)")
    parser.add_argument("-sigma3d", type=float, default=1e-3, help="3D conductivity (geometry-scaled starting value)")
    parser.add_argument("-kappa", type=float, default=1.0, help="coupling coefficient (geometry-scaled starting value)")
    parser.add_argument("-beta", type=float, default=50.0, help="Nitsche inlet penalty (beta)")
    args = parser.parse_args()

    cco = CCOVascularMesh(graph_folder=args.graph, obj_path=args.obj, name=args.name)
    cco.load().build().export_xdmf()

    meshV, cell_markers, boundary = cco.build_full_domain(n=args.n)
    solver = Solver3D1D(
        path_to_1D_mesh=os.path.join("nets", args.name) + os.sep,
        full_domain_mesh=meshV,
        full_domain_markers=cell_markers,
        boundary=boundary,
        n=args.n,
        sigma3d=args.sigma3d,
        sigma1d=args.sigma1d,
        kappa=args.kappa,
        beta_nitsche=args.beta,
    )
    solver.build()
    solver.solve()
    solver.save_paraview(os.path.join("solution", args.name))
