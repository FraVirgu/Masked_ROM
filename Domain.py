import numpy as np
import scipy.spatial as sptl
import ufl
import random
from dolfin import (
    Mesh, MeshEditor, MeshFunction, File, Point, BoxMesh,
    XDMFFile, MPI, cells
)
from Boundary import Boundary, boundary  # noqa: local import

class Domain:
    """
    Represents an n-layer vascular network domain.
    Encapsulates point cloud generation, obstacle definition, Voronoi
    tessellation, 1D mesh creation, boundary projection, and network extraction.
    """

    def __init__(
        self,
        name: str,
        n_vasi: int,
        n_ramifications: int,
        n_min: float = -1.0,
        n_max: float = 1.0,
        n_layers: int = 3,
        n_points_per_layer: int | list[int] = 1690,
        num_points_in: int = 560,
        num_points_out: int = 560,
        boundary: Boundary | None = None
    ):
        # --- identity & bounds ---
        self.name     = name
        self.n_min    = n_min
        self.n_max    = n_max
        self.n_layers = n_layers
        self.p_min    = [n_min, n_min, n_min]   # min point (x,y,z)
        self.p_max    = [n_max, n_max, n_max]  # max point (x,y,z)

        # --- layer geometry, how high should each layer be ---
        self.delta = (n_max - n_min) / n_layers
        self.h = [
            [n_min + i * self.delta, n_min + (i + 1) * self.delta]
            for i in range(n_layers)
        ]

        # allow a single int (same for all layers) or one value per layer
        if isinstance(n_points_per_layer, int):
            self.n_points_per_layer = [n_points_per_layer] * n_layers
        else:
            if len(n_points_per_layer) != n_layers:
                raise ValueError(
                    f"n_points_per_layer has {len(n_points_per_layer)} entries "
                    f"but n_layers={n_layers}."
                )
            self.n_points_per_layer = list(n_points_per_layer)

        # --- point counts ---
        self.num_points_in  = num_points_in
        self.num_points_out = num_points_out

        # --- vessel topology ---
        self.n_vasi          = n_vasi
        self.n_ramifications = n_ramifications

        # --- derived (populated by build()) ---
        self.base_pts     = None
        self.obstacle     = None  # dict: rot_mat, scal_mat, A, A_inv
        self.vor          = None
        self.shape_box    = None
        self.box_mesh     = None
        self.planes_in    = None
        self.planes_out   = None
        self.edges        = None
        self.vert_list    = None
        self.mesh1D       = None
        self.markers      = None
        self.vaso         = None
        self.vaso_markers = None

        self._check_warnings()
        self.boundary = boundary

    # =========================================================================
    # Public API
    # =========================================================================

    def build(self):
        """Run all build steps in order. Returns self for chaining."""
        self._build_point_cloud()
        self._build_obstacle()
        self._build_voronoi()
        self._build_box_and_planes()
        self._build_edges()
        self._build_voronoi_mesh()
        self._build_network()
        return self

    def export_box(self):
        """Write the bounding box mesh to a .pvd file."""
        self._require("box_mesh")
        print("\n...printing box vtk mesh")
        File(f"{self.name}_box.pvd") << self.box_mesh

    def export_reticolo(self):
        """Write the raw 1D Voronoi mesh to a .pvd file."""
        self._require("mesh1D")
        print("\n...printing base voronoi vtk mesh")
        File(f"{self.name}_reticolo.pvd") << self.mesh1D

    def export_vaso(self):
        """Write the extracted vascular network to a .pvd file."""
        self._require("vaso")
        print("\n...printing ultimate vaso vtk mesh")
        File(f"{self.name}_vaso.pvd") << self.vaso

    def export_xdmf(self):
        """Write the marked vascular mesh, markers, and per-edge radii to .xdmf files."""
        self._require("vaso", "vaso_markers", "vaso_radii")

        mesh_file = XDMFFile(MPI.comm_world, f"{self.name}_marked_mesh.xdmf")
        mesh_file.parameters["flush_output"] = True
        mesh_file.write(self.vaso)
        mesh_file.close()

        marker_file = XDMFFile(MPI.comm_world, f"{self.name}_markers.xdmf")
        marker_file.parameters["flush_output"] = True
        marker_file.write(self.vaso_markers)
        marker_file.close()

        radius_file = XDMFFile(MPI.comm_world, f"{self.name}_radii.xdmf")
        radius_file.parameters["flush_output"] = True
        self.vaso_radii.rename("radius", "vessel radius")
        radius_file.write(self.vaso_radii)
        radius_file.close()

        print(f"...xdmf files written → {self.name}_marked_mesh.xdmf / _markers.xdmf / _radii.xdmf")


   
    # =========================================================================
    # Private build steps
    # =========================================================================

    def _check_warnings(self):
        if self.num_points_in >= self.n_points_per_layer[0]:
            print("WARNING: possibly unnatural straight inflow channels")
        if self.num_points_out >= self.n_points_per_layer[-1]:
            print("WARNING: possibly unnatural straight outflow channels")
        if self.n_vasi >= self.num_points_in:
            print("WARNING: number of vasi greater than number of inflows")
        if self.n_ramifications * self.n_vasi >= self.num_points_out:
            print("WARNING: number of total vasi outflow greater than number of outflows")

    def _build_point_cloud(self):
        """Scatter random points inside each layer slab and merge."""
        xy_min = [self.p_min[0], self.p_min[1]]
        xy_max = [self.p_max[0], self.p_max[1]]
        clouds = [
            self._random_rectangle(xy_min, xy_max, h, n)
            for h, n in zip(self.h, self.n_points_per_layer)
        ]
        self.base_pts = np.vstack(clouds)

    def _build_obstacle(self):
        """Randomly generate the ellipsoidal obstacle transform."""
        theta      = np.random.uniform(0.0, 2 * np.pi)
        ax, ay, az = np.random.uniform(0.6, 0.9, 3)
        scal_mat   = np.diag([ax, ay, az]).astype(np.float64)
        rot_axis   = np.random.randint(0, 3)
        c, s = np.cos(theta), np.sin(theta)

        if rot_axis == 0:
            rot_mat = np.array([[1, 0,  0],
                                [0, c, -s],
                                [0, s,  c]], dtype=np.float64)
        elif rot_axis == 1:
            rot_mat = np.array([[ c, 0, s],
                                [ 0, 1, 0],
                                [-s, 0, c]], dtype=np.float64)
        else:
            rot_mat = np.array([[c, -s, 0],
                                [s,  c, 0],
                                [0,  0, 1]], dtype=np.float64)

        A     = np.dot(scal_mat, rot_mat)
        A_inv = np.linalg.inv(A)

        self.obstacle = {
            "theta": theta, "rot_axis": rot_axis,
            "scal_mat": scal_mat, "rot_mat": rot_mat,
            "A": A, "A_inv": A_inv,
        }

    def _build_voronoi(self):
        """Compute the Voronoi tessellation on the merged point cloud."""
        self._require("base_pts")
        self.vor = sptl.Voronoi(points=self.base_pts)

    def _build_box_and_planes(self):
        """Derive the bounding box mesh and define inflow/outflow planes."""
        self._require("base_pts")
        self.shape_box = self._find_min_max_points(self.base_pts)

        (xmin, xmax) = self.shape_box[0]
        (ymin, ymax) = self.shape_box[1]
        (zmin, zmax) = self.shape_box[2]

        self.box_mesh = BoxMesh(
            Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), 1, 1, 1
        )
        self.planes_in  = [[0,0,1,zmin], [1,0,0,xmin], [0,1,0,ymin]]
        self.planes_out = [[0,0,1,zmax], [1,0,0,xmax], [0,1,0,ymax]]

    def _build_edges(self):
        """Extract Voronoi edges that lie inside the bounding box and outside the obstacle."""
        self._require("vor", "shape_box", "obstacle")
        vertices = self.vor.vertices
        A_inv    = self.obstacle["A_inv"]
        self.edges     = self._get_vor_edges(self.vor, vertices, self.shape_box, A_inv)
        self.vert_list = list({v for edge in self.edges for v in edge})

    def _build_voronoi_mesh(self):
        """Assemble the 1D FEniCS interval mesh from Voronoi edges, then project boundary vertices."""
        self._require("vor", "edges", "vert_list", "planes_in", "planes_out")
        vertices = self.vor.vertices
        raw_mesh = self._create_voronoi_mesh(vertices, self.vert_list, self.edges)
        self.mesh1D, self.markers = self._plane_new_verts(
            raw_mesh, self.planes_in, self.num_points_in,
            self.planes_out, self.num_points_out,
        )

    def _build_network(self):
        """Extract the vascular network via shortest paths between inlet/outlet markers."""
        self._require("mesh1D", "markers")
        self.vaso, self.vaso_markers, self.vaso_radii = self._fun(
            self.mesh1D, self.n_vasi, self.n_ramifications, self.markers
        )
        print(f"N DOF vaso: {self.vaso.num_edges()}")

    # =========================================================================
    # Geometry helpers (static)
    # =========================================================================

    @staticmethod
    def _random_rectangle(min_b, max_b, h, num_points):
        x_min, y_min = min_b
        x_max, y_max = max_b
        h_min, h_max = h
        x = np.random.uniform(x_min, x_max, num_points)
        y = np.random.uniform(y_min, y_max, num_points)
        z = np.random.uniform(h_min, h_max, num_points)
        return np.column_stack((x, y, z))

    @staticmethod
    def _find_min_max_points(point_cloud):
        x_min = y_min = z_min =  float("inf")
        x_max = y_max = z_max = -float("inf")
        for x, y, z in point_cloud:
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
            z_min, z_max = min(z_min, z), max(z_max, z)
        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    @staticmethod
    def _box2sph(point):
        """Map a point from [-1,1]^3 cube to the unit sphere surface."""
        x, y, z = point
        xx = x * np.sqrt(1 - y**2/2 - z**2/2 + y**2*z**2/3)
        yy = y * np.sqrt(1 - z**2/2 - x**2/2 + x**2*z**2/3)
        zz = z * np.sqrt(1 - x**2/2 - y**2/2 + y**2*x**2/3)
        return np.array([xx, yy, zz])

    @staticmethod
    def _is_point_outside_domain(point, domain):
        x, y, z = point
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain
        return (x < x_min or x > x_max or
                y < y_min or y > y_max or
                z < z_min or z > z_max)

    @staticmethod
    def _is_inside_elli(vertices, edge, elli):
        """Return True if either endpoint of edge lies inside the ellipsoid."""
        v0 = np.dot(elli, vertices[edge[0]])
        v1 = np.dot(elli, vertices[edge[1]])
        return np.linalg.norm(v0) <= 1.0 or np.linalg.norm(v1) <= 1.0

    @staticmethod
    def _dist(mesh, i0, i1):
        v = mesh.coordinates()
        return np.linalg.norm(v[i0] - v[i1])

    @staticmethod
    def _get_3D_vector(arr):
        arr = np.array(arr)
        return arr.reshape(3, 1) if arr.shape != (3, 1) else arr

    @staticmethod
    def _again_np_array(arr):
        return arr.reshape(1, 3)[0]

    @staticmethod
    def _get_homo_vector(arr):
        arr = np.array(arr)
        if arr.shape == (1, 3):
            return np.append(arr, 1).reshape(4, 1)
        return np.hstack((arr, [1]))

    # =========================================================================
    # Mesh construction helpers
    # =========================================================================

    @staticmethod
    def _point_plane_proj(point_coords, plane):
        """Project point_coords onto the plane Ax+By+Cz=D."""
        A, B, C, D = plane
        point_coords = Domain._get_homo_vector(point_coords)

        def _build_proj(normal, shift_idx, shift_val):
            n = Domain._get_3D_vector(normal / np.linalg.norm(normal))
            P = np.identity(3) - np.dot(n, n.T)
            P = np.vstack((P, np.zeros(3)))
            P = np.hstack((P, np.array([0, 0, 0, 1]).reshape(-1, 1)))
            t_neg = np.eye(4); t_neg[shift_idx, 3] = -shift_val
            t_pos = np.eye(4); t_pos[shift_idx, 3] =  shift_val
            return np.dot(t_pos, np.dot(P, t_neg))

        if A != 0:
            u = np.array([-C/A, 0, 1])
            v = np.array([-B/A, 1, 0])
            P = _build_proj(np.cross(u, v), 0, D/A)
        elif B != 0:
            u = np.array([1, -A/B, 0])
            v = np.array([0, -C/B, 1])
            P = _build_proj(np.cross(u, v), 1, D/B)
        elif C != 0:
            u = np.array([1, 0, -A/C])
            v = np.array([0, 1, -B/C])
            P = _build_proj(np.cross(u, v), 2, D/C)
        else:
            print("plane equation error"); return None

        return np.dot(P, point_coords)[:3]

    @staticmethod
    def _find_nearest_points_plane(mesh, plane_coeffs, n):
        vertices  = mesh.coordinates()
        A, B, C, D = plane_coeffs
        distances = (np.abs(np.dot(vertices, [A, B, C]) - D)
                     / np.linalg.norm([A, B, C]))
        closest_indices = np.argsort(distances)[:n]
        return vertices[closest_indices], closest_indices
    

    @staticmethod
    def _find_nearest_points_vertex(mesh, vertex):
        vertices = mesh.coordinates()
        distances = np.linalg.norm(vertices - vertex, axis=1)
        closest_indices = np.argsort(distances)[:1]   # ← slice, keeps array shape
        return vertices[closest_indices], closest_indices

    def _get_vor_edges(self, vor, vertices, shape_box, elli):
        """Build and filter Voronoi ridge edges."""
        edges = []
        for facet in vor.ridge_vertices:
            for a, b in zip(facet[:-1] + [facet[-1]], facet[1:] + [facet[0]]):
                edges.append([a, b])
        edges = np.array(edges)
        edges = edges[~np.any(edges == -1, axis=1)]  # remove infinity edges

        # remove out-of-box edges
        mask = [self._is_point_outside_domain(vertices[e[0]], shape_box) or
                self._is_point_outside_domain(vertices[e[1]], shape_box)
                for e in edges]
        edges = edges[~np.array(mask)]

        # keep edge only if BOTH endpoints are inside the boundary
        keep = np.array([
            self.boundary(vertices[e[0]]) and self.boundary(vertices[e[1]])
            for e in edges
        ])
        edges = edges[keep]

        # (optionally) remove edges inside ellipsoid — uncomment to enable
        # mask = [self._is_inside_elli(vertices, e, elli) for e in edges]
        # edges = edges[~np.array(mask)]

        edges = np.sort(edges, axis=1)
        edges = edges[:, 0] + 1j * edges[:, 1]
        edges = np.unique(edges)
        edges = np.vstack((np.real(edges), np.imag(edges))).T
        return np.array(edges, dtype=int)

    @staticmethod
    def _create_voronoi_mesh(vert_coord, vert_list, edges):
        """Assemble a FEniCS interval mesh from Voronoi vertices and edges."""
        gdim     = len(vert_coord[0])
        ufl_cell = ufl.Cell("interval", gdim)
        assert 1 == ufl_cell.topological_dimension()

        mesh = Mesh()
        me   = MeshEditor()
        me.open(mesh, "interval", 1, gdim)

        nv = max(max(vert_list), len(vert_list)) + 1
        me.init_vertices(nv)
        for i in range(nv):
            coord = vert_coord[i] if i in vert_list else [-666, -666, -666]
            me.add_vertex(i, coord)

        me.init_cells(len(edges))
        for i, edge in enumerate(edges):
            me.add_cell(i, (int(edge[0]), int(edge[1])))
        me.close()
        return mesh

    def _plane_new_verts(self, mesh, planes_in, n_in, planes_out, n_out):
        """Project boundary vertices onto inlet/outlet planes and rebuild mesh with markers."""
        from xii import EmbeddedMesh, transfer_markers  # noqa: local import
        
        vertices = mesh.coordinates()
        nv_old   = len(vertices)

        if self.boundary.is_inlet_empty():
            print("WARNING: no inlet points provided, creating based on planes_in")
            li, lo   = len(planes_in), len(planes_out)

            proj_in  = [None] * li
            proj_out = [None] * lo

            for pl, (pin, pout) in enumerate(zip(planes_in, planes_out)):
                coords_in,  idx_in  = self._find_nearest_points_plane(mesh, pin,  n_in)
                coords_out, idx_out = self._find_nearest_points_plane(mesh, pout, n_out)
                proj_in[pl]  = (list(idx_in),
                                [self._point_plane_proj(c, pin)  for c in coords_in])
                proj_out[pl] = (list(idx_out),
                                [self._point_plane_proj(c, pout) for c in coords_out])

        else:
            print("INFO: using boundary.inlet / boundary.outlet points")
            li = len(self.boundary.inlet)
            lo = len(self.boundary.outlet)

            proj_in  = [None] * li
            proj_out = [None] * lo

            for i, (vin, vout) in enumerate(zip(self.boundary.inlet, self.boundary.outlet)):
                # find closest mesh vertex to each inlet/outlet point
                _, idx_in  = self._find_nearest_points_vertex(mesh, vin)
                _, idx_out = self._find_nearest_points_vertex(mesh, vout)
                # the new vertex IS the inlet/outlet point itself (no plane projection)
                proj_in[i]  = ([int(idx_in[0])],  [np.array(vin)])
                proj_out[i] = ([int(idx_out[0])], [np.array(vout)])

        nv_new_in  = [len(proj_in[pl][1])  for pl in range(li)]
        nv_new_out = [len(proj_out[pl][1]) for pl in range(lo)]

        # --- build new mesh --- (unchanged below)
        ufl_cell = ufl.Cell("interval", len(vertices[0]))
        assert 1 == ufl_cell.topological_dimension()

        mesh1 = Mesh()
        me    = MeshEditor()
        me.open(mesh1, "interval", 1, len(vertices[0]))
        me.init_vertices(nv_old + sum(nv_new_in) + sum(nv_new_out))

        for i, coord in enumerate(vertices):
            me.add_vertex(i, coord)

        for pl in range(li):
            base = nv_old + sum(nv_new_in[:pl])
            for i, coord in enumerate(proj_in[pl][1]):
                me.add_vertex(i + base, self._again_np_array(coord))

        for pl in range(lo):
            base = nv_old + sum(nv_new_in) + sum(nv_new_out[:pl])
            for i, coord in enumerate(proj_out[pl][1]):
                me.add_vertex(i + base, self._again_np_array(coord))

        ne_old = mesh.num_cells()
        me.init_cells(ne_old + sum(nv_new_in) + sum(nv_new_out))

        for cell in cells(mesh):
            me.add_cell(cell.index(), cell.entities(0))

        for pl in range(li):
            base_v = nv_old + sum(nv_new_in[:pl])
            base_e = ne_old + sum(nv_new_in[:pl])
            for i, idx in enumerate(proj_in[pl][0]):
                me.add_cell(i + base_e, (i + base_v, idx))

        for pl in range(lo):
            base_v = nv_old + sum(nv_new_in) + sum(nv_new_out[:pl])
            base_e = ne_old + sum(nv_new_in) + sum(nv_new_out[:pl])
            for i, idx in enumerate(proj_out[pl][0]):
                me.add_cell(i + base_e, (i + base_v, idx))

        me.close()

        # --- mark vertices ---
        eps     = 1e-4
        markers = MeshFunction("size_t", mesh1, 0, 0)

        for i, coord in enumerate(vertices):
            if not (min(coord) <= -666 + eps and min(coord) >= -666 - eps):
                markers[i] = 555  # bulk

        tot_in  = [c for pl in range(li)  for c in proj_in[pl][1]]
        tot_out = [c for pl in range(lo) for c in proj_out[pl][1]]

        for i, coord in enumerate(tot_in):
            markers[i + nv_old] = 111  # inflow

        for i, coord in enumerate(tot_out):
            markers[i + nv_old + sum(nv_new_in)] = 999  # outflow

        return mesh1, markers

    def _fun(self, mesh, n_departures, n_arrivals, markers):
        import networkx as nx
        from xii import EmbeddedMesh, transfer_markers

        facet_f = MeshFunction("size_t", mesh, 1, 0)
        mesh.init(1, 0)

        # per-vertex radius on the parent mesh (dim=0)
        vertex_radii = MeshFunction("double", mesh, 0, 1.0)
        
        G             = nx.Graph()
        edge_indices  = {}
        vertex_markers = markers.array()

        for cell in cells(mesh):
            idx  = cell.index()
            conn = cell.entities(0).tolist()
            w    = self._dist(mesh, conn[0], conn[1])
            G.add_edge(conn[0], conn[1], weight=w)
            edge_indices[tuple(sorted(conn))] = idx

        departures = [i for i in range(mesh.num_vertices()) if vertex_markers[i] == 111]
        arrivals   = [i for i in range(mesh.num_vertices()) if vertex_markers[i] == 999]

        for _ in range(n_departures):
            v0 = random.choice(departures)
            for _ in range(n_arrivals):
                v1   = random.choice(arrivals)
                path = nx.shortest_path(G, source=v0, target=v1, weight="weight")
                for a, b in zip(path[:-1], path[1:]):
                    key = tuple(sorted((a, b)))
                    facet_f[edge_indices[key]] = 1

        vaso         = EmbeddedMesh(facet_f, 1)
        vaso_markers = transfer_markers(vaso, markers)
        # transfer_markers doesn't work for double MeshFunction
        # instead: vaso vertices are a subset of parent vertices, same coordinates
        # build coordinate → radius lookup on parent mesh
        coord_to_radius = {}
        parent_coords = mesh.coordinates()
        for i in range(mesh.num_vertices()):
            key = tuple(np.round(parent_coords[i], 10))
            coord_to_radius[key] = vertex_radii[i]

        # assign to vaso vertices by coordinate match
        vaso_radii = MeshFunction("double", vaso, 0, 0.0)
        vaso_coords = vaso.coordinates()
        for i in range(vaso.num_vertices()):
            key = tuple(np.round(vaso_coords[i], 10))
            vaso_radii[i] = coord_to_radius.get(key, 1.0)  # fallback to 1.0

        return vaso, vaso_markers, vaso_radii
    # =========================================================================
    # Utilities
    # =========================================================================

    def _require(self, *attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(
                    f"'{attr}' is not yet built. Call build() first "
                    f"(or the relevant _build_* step)."
                )

    def __repr__(self):
        return (
            f"Domain(name={self.name!r}, n_layers={self.n_layers}, "
            f"points_per_layer={self.n_points_per_layer}, "
            f"n_vasi={self.n_vasi}, n_ramifications={self.n_ramifications}, "
            f"built={self.base_pts is not None})"
        )
    




import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name",   type=str)
parser.add_argument("-test",   type=str)
parser.add_argument("-inlet",  type=int)
parser.add_argument("-outlet", type=int)
args, _ = parser.parse_known_args()



domain = Domain(
    name            = f"./nets/{args.test}/{args.name}_",
    n_vasi          = args.inlet,
    n_ramifications = args.outlet,
    boundary        = boundary
).build()

domain.export_box()
domain.export_reticolo()
domain.export_vaso()
domain.export_xdmf()