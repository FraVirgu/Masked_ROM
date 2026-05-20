import os
import argparse
import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction, XDMFFile, MPI
from scipy.ndimage import binary_fill_holes
from Boundary import Boundary
from Solver import Solver3D1D


# =============================================================================
# Boundary from OBJ
# =============================================================================

def boundary_from_obj(obj_path: str, scale: float, center: np.ndarray) -> Boundary:
    """
    Build a Boundary from a voxelized OBJ (liver05Domain.obj).

    Uses the SAME scale/center as CCOVascularMesh so coordinates are
    consistent with the graph already in [-1,1]^3.
    """

    # --- 1. read OBJ vertices ---
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = map(float, line.split()[1:4])
                verts.append([x, y, z])
    verts = np.array(verts)

    # --- 2. voxel indices from half-integer coords ---
    vmin_vox = np.floor(verts.min(axis=0)).astype(int)
    vmax_vox = np.floor(verts.max(axis=0)).astype(int)
    dims     = vmax_vox - vmin_vox + 2   # +2 safe border

    # --- 3. mark surface voxels ---
    mask = np.zeros(dims, dtype=bool)
    for v in verts:
        ix = np.clip(int(np.floor(v[0])) - vmin_vox[0], 0, dims[0] - 1)
        iy = np.clip(int(np.floor(v[1])) - vmin_vox[1], 0, dims[1] - 1)
        iz = np.clip(int(np.floor(v[2])) - vmin_vox[2], 0, dims[2] - 1)
        mask[ix, iy, iz] = True

    # --- 4. fill interior ---
    mask = binary_fill_holes(mask)
    print(f"Mask: {mask.shape}, {mask.sum()} inside voxels / {mask.size} total")

    # --- 5. bbox: voxel corners → scaled coords (same transform as graph) ---
    p_min = (vmin_vox.astype(float)       - center) * scale
    p_max = ((vmax_vox + 1).astype(float) - center) * scale
    bbox  = (
        (float(p_min[0]), float(p_max[0])),
        (float(p_min[1]), float(p_max[1])),
        (float(p_min[2]), float(p_max[2])),
    )

    print(f"Boundary bbox (scaled):")
    print(f"  x: [{bbox[0][0]:.4f}, {bbox[0][1]:.4f}]  size={bbox[0][1]-bbox[0][0]:.4f}")
    print(f"  y: [{bbox[1][0]:.4f}, {bbox[1][1]:.4f}]  size={bbox[1][1]-bbox[1][0]:.4f}")
    print(f"  z: [{bbox[2][0]:.4f}, {bbox[2][1]:.4f}]  size={bbox[2][1]-bbox[2][0]:.4f}")

    return Boundary(source=mask, bbox=bbox, inlet_points=None, outlet_points=None)


# =============================================================================
# CCOVascularMesh
# =============================================================================

class CCOVascularMesh:
    """
    Loads a CCO vascular graph (vertex.dat, edges.dat, radius.dat),
    rescales it to [-1,1]^3 using the DOMAIN (OBJ) bounding box,
    builds a 1D FEniCS mesh with inlet/outlet markers and per-vertex
    radii, and exports to XDMF for Solver3D1D.
    """

    def __init__(self, graph_folder: str, obj_path: str, name: str = "cco"):
        self.graph_folder = graph_folder
        self.obj_path     = obj_path        # liver domain OBJ → drives scaling
        self.name         = name
        self.output_dir   = os.path.join("nets", name)

        # populated by load()
        self.vertices     = None
        self.edges        = None
        self.radii        = None
        self.scale        = None
        self.center       = None

        # populated by build()
        self.mesh1        = None
        self.inlet        = None
        self.leaves       = None
        self.markers      = None
        self.vertex_radii = None
        self.vaso         = None
        self.vaso_markers = None
        self.vaso_radii   = None

    # =========================================================================
    # Public API
    # =========================================================================

    def load(self):
        """Load and rescale graph data. Returns self."""
        self._load_and_rescale()
        return self

    def build(self):
        """Build 1D mesh, markers, radii, extract vascular sub-mesh. Returns self."""
        self._require("vertices", "edges", "radii")
        self._build_mesh()
        self._mark_vertices()
        self._trace_paths()
        self._transfer_radii()
        return self

    def export_xdmf(self):
        """Write marked_mesh, markers, radii to nets/{name}/. Returns self."""
        self._require("vaso", "vaso_markers", "vaso_radii")
        os.makedirs(self.output_dir, exist_ok=True)

        def path(suffix):
            return os.path.join(self.output_dir, f"{self.name}_{suffix}.xdmf")

        for fname, obj, rename in [
            ("marked_mesh", self.vaso,         None),
            ("markers",     self.vaso_markers, None),
            ("radii",       self.vaso_radii,   ("radius", "vessel radius")),
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

    # =========================================================================
    # Private steps
    # =========================================================================

    def _load_and_rescale(self):
        """Load graph .dat files and rescale using DOMAIN (OBJ) bbox → [-1,1]^3."""

        # --- graph data ---
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

        # --- scale/center from DOMAIN bbox (not graph bbox) ---
        # the domain is always larger than the graph by the CCO border margin
        obj_verts = []
        with open(self.obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = map(float, line.split()[1:4])
                    obj_verts.append([x, y, z])
        obj_verts = np.array(obj_verts)

        vmin   = obj_verts.min(axis=0)
        vmax   = obj_verts.max(axis=0)
        center = (vmin + vmax) / 2.0
        scale  = 2.0 / (vmax - vmin).max()   # domain fits in [-1,1]^3

        self.vertices = {i: (v - center) * scale for i, v in vertices.items()}
        self.radii    = {i: r * scale             for i, r in radii.items()}
        self.edges    = edges
        self.scale    = scale
        self.center   = center

        # verify graph is strictly inside [-1,1]^3
        g = np.array(list(self.vertices.values()))
        print(f"Loaded {len(vertices)} vertices, {len(edges)} edges")
        print(f"Domain scale={scale:.6f}, center={np.round(center, 3)}")
        print(f"Graph range after scaling:")
        print(f"  x=[{g[:,0].min():.4f}, {g[:,0].max():.4f}]  "
              f"y=[{g[:,1].min():.4f}, {g[:,1].max():.4f}]  "
              f"z=[{g[:,2].min():.4f}, {g[:,2].max():.4f}]")
        print(f"Radii: min={min(self.radii.values()):.6f}  "
              f"max={max(self.radii.values()):.6f}")
        assert (g >= -1.0 - 1e-6).all() and (g <= 1.0 + 1e-6).all(), \
            "Graph vertices outside [-1,1]^3 — check OBJ bbox"

    def _build_mesh(self):
        mesh1 = Mesh()
        me    = MeshEditor()
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

        leaves  = [i for i in self.vertices if degree[i] == 1]
        inlet   = max(leaves, key=lambda i: self.radii[i])
        outlets = [i for i in leaves if i != inlet]

        markers = MeshFunction("size_t", self.mesh1, 0, 0)
        for i in range(self.mesh1.num_vertices()):
            if i == inlet:
                markers[i] = 111
            elif i in outlets:
                markers[i] = 999
            else:
                markers[i] = 555

        self.inlet   = inlet
        self.leaves  = outlets
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

        self.vaso         = EmbeddedMesh(facet_f, 1)
        self.vaso_markers = transfer_markers(self.vaso, self.markers)
        print(f"Vaso mesh: {self.vaso.num_vertices()} vertices, "
              f"{self.vaso.num_cells()} edges")

    def _transfer_radii(self):
        coord_to_radius = {
            tuple(np.round(self.mesh1.coordinates()[i], 10)): self.vertex_radii[i]
            for i in range(self.mesh1.num_vertices())
        }

        vaso_radii     = MeshFunction("double", self.vaso, 0, 0.0)
        fallback_count = 0
        for i in range(self.vaso.num_vertices()):
            key = tuple(np.round(self.vaso.coordinates()[i], 10))
            r   = coord_to_radius.get(key, None)
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

    # =========================================================================
    # Utilities
    # =========================================================================

    def _require(self, *attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(
                    f"'{attr}' not available — call load()/build() first."
                )

    def __repr__(self):
        return (
            f"CCOVascularMesh(graph='{self.graph_folder}', "
            f"obj='{self.obj_path}', "
            f"name='{self.name}', "
            f"built={self.vaso is not None})"
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-name",  type=str,   required=True,
                        help="subfolder name inside nets/ (e.g. liver05)")
    parser.add_argument("-graph", type=str,   default="graphExport",
                        help="folder with vertex.dat / edges.dat / radius.dat")
    parser.add_argument("-obj",   type=str,   default="graphExport/domain.obj",
                        help="path to liver domain OBJ file")
    parser.add_argument("-n",     type=int,   default=40,
                        help="3D mesh resolution")
    parser.add_argument("-rad",   type=float, default=0.05,
                        help="coupling radius")
    args = parser.parse_args()

    # --- 1. build vascular mesh (domain OBJ drives scaling) ---
    cco = CCOVascularMesh(
        graph_folder = args.graph,
        obj_path     = args.obj,
        name         = args.name,
    )
    cco.load().build().export_xdmf()

    # --- 2. build boundary (same scale/center as graph) ---
    boundary_cco = boundary_from_obj(
        obj_path = args.obj,
        scale    = cco.scale,
        center   = cco.center,
    )

    # --- 3. run solver ---
    out_dir = f"./solution/{args.name}_rad{args.rad}_n{args.n}"
    solver  = Solver3D1D(
        path_to_1D_mesh = f"./nets/{args.name}/{args.name}_",
        boundary        = boundary_cco,
        n               = args.n,
        sigma3d         = 1e-3,
        sigma1d         = 1.0,
        kappa           = 1.0,
        
    ).build().solve()

    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")
    