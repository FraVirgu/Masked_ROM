import numpy as np




class Boundary:
    def __init__(self, source, bbox=None, inlet_points=None, outlet_points=None, border_eps=1e-2):
        if callable(source):
            self._mode   = "implicit"
            self._fn     = source
        elif isinstance(source, np.ndarray) and source.ndim == 3:
            self._mode   = "mask"
            self._mask   = source.astype(bool)
            self._shape  = source.shape
            if bbox is None:
                raise ValueError("bbox required when source is a voxel mask.")
            self._bbox   = bbox
        else:
            raise TypeError("source must be a callable or a 3D numpy array.")

        self.inlet  = np.array(inlet_points)  if inlet_points  is not None else np.empty((0, 3))
        self.outlet = np.array(outlet_points) if outlet_points is not None else np.empty((0, 3))
        if not self.is_inlet_empty() and not self.is_outlet_empty():
            self._border_eps = border_eps
            self._assert_interior_or_border(self.inlet,  "inlet")
            self._assert_interior_or_border(self.outlet, "outlet")

    
    def __call__(self, point) -> bool:
        """Return True if point lies inside the domain."""
        x, y, z = point
        if self._mode == "implicit":
            return bool(self._fn(x, y, z))
        else:
            return self._query_mask(x, y, z)

    def is_inside(self, point) -> bool:
        return self.__call__(point)

    def filter_points(self, points: np.ndarray) -> np.ndarray:
        return np.array([p for p in points if self(p)])

    def _query_mask(self, x, y, z) -> bool:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._bbox
        Nx, Ny, Nz = self._shape
        ix = int((x - xmin) / (xmax - xmin) * (Nx - 1))
        iy = int((y - ymin) / (ymax - ymin) * (Ny - 1))
        iz = int((z - zmin) / (zmax - zmin) * (Nz - 1))
        ix = max(0, min(ix, Nx - 1))
        iy = max(0, min(iy, Ny - 1))
        iz = max(0, min(iz, Nz - 1))
        return bool(self._mask[ix, iy, iz])

    def _is_on_border(self, point) -> bool:
        x, y, z = point
        eps = self._border_eps
        if self._mode == "implicit":
            inside = lambda p: bool(self._fn(p[0], p[1], p[2]))
        else:
            inside = lambda p: self._query_mask(p[0], p[1], p[2])
        if not inside(point):
            return False
        neighbors = [
            [x+eps, y,   z  ], [x-eps, y,   z  ],
            [x,   y+eps, z  ], [x,   y-eps, z  ],
            [x,   y,   z+eps], [x,   y,   z-eps],
        ]
        return any(not inside(n) for n in neighbors)

    def _assert_interior_or_border(self, points, label):
        for i, p in enumerate(points):
            assert self(p), (
                f"{label}[{i}] = {p} is not inside the domain."
            )

    def is_inlet_empty(self):
        return self.inlet.shape[0] == 0

    def is_outlet_empty(self):
        return self.outlet.shape[0] == 0

    def _assert_on_border(self, points, label):
        for i, p in enumerate(points):
            assert self._is_on_border(p), (
                f"{label}[{i}] = {p} is not on the boundary surface "
                f"(eps={self._border_eps}). "
            )

    def __repr__(self):
        return f"Boundary(mode={self._mode!r})"


class SphereBoundary(Boundary):
    def __init__(
        self,
        radius=1.0,
        center=(0.0, 0.0, 0.0),
        inlet_points=None,
        outlet_points=None,
        border_eps=1e-2,
    ):
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive for SphereBoundary.")

        cx, cy, cz = self.center
        r = self.radius
        # Keep compatibility with code paths that expect boundary._bbox.
        self._bbox = ((cx - r, cx + r), (cy - r, cy + r), (cz - r, cz + r))

        super().__init__(
            source=lambda x, y, z: (
                (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius ** 2
            ),
            bbox=None,
            inlet_points=inlet_points,
            outlet_points=outlet_points,
            border_eps=border_eps,
        )

    def __repr__(self):
        return (
            f"SphereBoundary(radius={self.radius!r}, "
            f"center={tuple(self.center)!r})"
        )



class CylinderBoundary(Boundary):
    """Finite circular cylinder, axis-aligned along `axis`.

    Inside is the intersection of the infinite cylinder of radius `radius`
    about the axis through `center` with the slab of total extent `height`
    along that axis, i.e. a closed solid capped by two flat discs.
    """

    _AXES = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        radius=1.0,
        height=2.0,
        center=(0.0, 0.0, 0.0),
        axis="z",
        inlet_points=None,
        outlet_points=None,
        border_eps=1e-2,
    ):
        self.radius = float(radius)
        self.height = float(height)
        self.center = np.asarray(center, dtype=float)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive for CylinderBoundary.")
        if self.height <= 0.0:
            raise ValueError("height must be positive for CylinderBoundary.")
        if axis not in self._AXES:
            raise ValueError(f"axis must be one of {tuple(self._AXES)}.")
        self.axis = axis
        ai = self._AXES[axis]
        self._axis_index = ai

        r, h = self.radius, self.height
        half = 0.5 * h
        # The bbox is tight: `radius` in the two transverse directions, half the
        # height along the axis. Kept for code paths that expect boundary._bbox.
        ext = [r, r, r]
        ext[ai] = half
        self._bbox = tuple(
            (self.center[d] - ext[d], self.center[d] + ext[d]) for d in range(3)
        )

        # The two transverse directions -- the ones the radial test applies to.
        t0, t1 = [d for d in (0, 1, 2) if d != ai]
        c = self.center

        def inside(x, y, z):
            p = (x, y, z)
            radial = (p[t0] - c[t0]) ** 2 + (p[t1] - c[t1]) ** 2
            return (radial <= r ** 2) and (abs(p[ai] - c[ai]) <= half)

        super().__init__(
            source=inside,
            bbox=None,
            inlet_points=inlet_points,
            outlet_points=outlet_points,
            border_eps=border_eps,
        )

    def __repr__(self):
        return (
            f"CylinderBoundary(radius={self.radius!r}, "
            f"height={self.height!r}, axis={self.axis!r}, "
            f"center={tuple(self.center)!r})"
        )


np.random.seed(42)

def random_sphere_points(
    n,
    x_sign,
    min_x=0.2,
    min_dist_to_boundary=0.06,
    radius=1.0,
):
    pts = []
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive when sampling sphere points.")
    max_radius = radius - min_dist_to_boundary
    if max_radius <= 0.0:
        raise ValueError(
            "min_dist_to_boundary must be smaller than the sphere radius."
        )
    while len(pts) < n:
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        v *= np.random.uniform(0.0, max_radius)
        if x_sign * v[0] > min_x * radius:
            pts.append(v.tolist())
    return pts


def random_cylinder_points(
    n,
    sign,
    radius=1.0,
    height=2.0,
    axis="z",
    split_axis=None,
    min_offset=0.2,
    min_dist_to_boundary=0.06,
    center=(0.0, 0.0, 0.0),
):
    """Sample `n` points strictly inside a CylinderBoundary.

    Mirrors random_sphere_points: `sign` (+1/-1) with `min_offset` keeps the
    points on one side, so an inlet and an outlet set can be drawn separately.
    By default the split is along the cylinder axis, giving points near one cap
    or the other; pass `split_axis` to split transversally instead.

    `min_dist_to_boundary` is honoured on the curved wall AND on both caps, so
    every point stays off the surface -- inlets sitting exactly on the boundary
    make the border checks in Boundary ambiguous.
    """
    axes = CylinderBoundary._AXES
    if axis not in axes:
        raise ValueError(f"axis must be one of {tuple(axes)}.")
    ai = axes[axis]
    if split_axis is None:
        si = ai
    else:
        if split_axis not in axes:
            raise ValueError(f"split_axis must be one of {tuple(axes)}.")
        si = axes[split_axis]

    radius, height = float(radius), float(height)
    if radius <= 0.0:
        raise ValueError("radius must be positive when sampling cylinder points.")
    if height <= 0.0:
        raise ValueError("height must be positive when sampling cylinder points.")

    max_radius = radius - min_dist_to_boundary
    half = 0.5 * height - min_dist_to_boundary
    if max_radius <= 0.0 or half <= 0.0:
        raise ValueError(
            "min_dist_to_boundary must be smaller than the cylinder radius "
            "and than half its height."
        )

    # The half-extent along whichever direction the sign test uses.
    span = half if si == ai else max_radius
    center = np.asarray(center, dtype=float)
    t0, t1 = [d for d in (0, 1, 2) if d != ai]

    pts = []
    while len(pts) < n:
        # Uniform over the disc: sqrt keeps the density flat in area.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        rr = max_radius * np.sqrt(np.random.uniform(0.0, 1.0))
        v = np.empty(3)
        v[t0] = rr * np.cos(theta)
        v[t1] = rr * np.sin(theta)
        v[ai] = np.random.uniform(-half, half)
        if sign * v[si] > min_offset * span:
            pts.append((center + v).tolist())
    return pts


