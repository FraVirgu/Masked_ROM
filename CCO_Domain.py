import os
import argparse
import numpy as np
from dolfin import (
    Mesh, MeshEditor, MeshFunction, XDMFFile, MPI
)


class CCOVascularMesh:

    def __init__(self, graph_folder: str, name: str = "cco"):
        self.graph_folder = graph_folder
        self.name         = name
        self.output_dir   = os.path.join("nets", name)   # nets/{name}/

        # --- populated by load() ---
        self.vertices     = None
        self.edges        = None
        self.radii        = None
        self.scale        = None

        # --- populated by build() ---
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
        self._load_and_rescale()
        return self

    def build(self):
        self._require("vertices", "edges", "radii")
        self._build_mesh()
        self._mark_vertices()
        self._trace_paths()
        self._transfer_radii()
        return self

    def export_xdmf(self):
        """Write marked_mesh, markers and radii to nets/{name}/."""
        self._require("vaso", "vaso_markers", "vaso_radii")

        # create nets/{name}/ if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        def xdmf_path(suffix):
            return os.path.join(self.output_dir, f"{self.name}_{suffix}.xdmf")

        mesh_file = XDMFFile(MPI.comm_world, xdmf_path("marked_mesh"))
        mesh_file.parameters["flush_output"] = True
        mesh_file.write(self.vaso)
        mesh_file.close()

        marker_file = XDMFFile(MPI.comm_world, xdmf_path("markers"))
        marker_file.parameters["flush_output"] = True
        marker_file.write(self.vaso_markers)
        marker_file.close()

        radius_file = XDMFFile(MPI.comm_world, xdmf_path("radii"))
        radius_file.parameters["flush_output"] = True
        self.vaso_radii.rename("radius", "vessel radius")
        radius_file.write(self.vaso_radii)
        radius_file.close()

        print(f"XDMF written → {self.output_dir}/")
        print(f"  {self.name}_marked_mesh.xdmf")
        print(f"  {self.name}_markers.xdmf")
        print(f"  {self.name}_radii.xdmf")
        return self

    # =========================================================================
    # Private steps
    # =========================================================================

    def _load_and_rescale(self):
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
                a, b = int(parts[0]), int(parts[1])
                edges.append((a, b))

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
                f"Found {len(invalid)} vertices with radius <= 0:\n"
                + "\n".join(f"  vertex {i}: r={r}" for i, r in invalid.items())
            )

        coords = np.array(list(vertices.values()))
        vmin   = coords.min(axis=0)
        vmax   = coords.max(axis=0)
        center = (vmin + vmax) / 2.0
        scale  = 2.0 / (vmax - vmin).max()

        self.vertices = {i: (v - center) * scale for i, v in vertices.items()}
        self.radii    = {i: r * scale             for i, r in radii.items()}
        self.edges    = edges
        self.scale    = scale

        print(f"Loaded {len(self.vertices)} vertices, {len(self.edges)} edges")
        print(f"Scale: {scale:.6f} | "
              f"Radii: min={min(self.radii.values()):.6f}  max={max(self.radii.values()):.6f}")

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

        print(f"Inlet: {inlet} | Outlets: {len(outlets)} | "
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

    def _transfer_radii(self):
        coord_to_radius = {
            tuple(np.round(self.mesh1.coordinates()[i], 10)): self.vertex_radii[i]
            for i in range(self.mesh1.num_vertices())
        }

        vaso_radii, fallback_count = MeshFunction("double", self.vaso, 0, 0.0), 0
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
                raise RuntimeError(f"'{attr}' not available — call load()/build() first.")

    def __repr__(self):
        return (
            f"CCOVascularMesh(folder='{self.graph_folder}', "
            f"name='{self.name}', "
            f"output='{self.output_dir}', "
            f"built={self.vaso is not None})"
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, required=True,
                        help="subfolder name inside nets/ (e.g. liver05)")
    parser.add_argument("-graph", type=str, default="graphExport",
                        help="folder containing vertex.dat / edges.dat / radius.dat")
    args = parser.parse_args()

    cco = CCOVascularMesh(graph_folder=args.graph, name=args.name)
    cco.load().build().export_xdmf()