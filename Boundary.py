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
        if(not self.is_inlet_empty() and not self.is_outlet_empty()):
            self._border_eps = border_eps
            self._assert_on_border(self.inlet,  "inlet")
            self._assert_on_border(self.outlet, "outlet")

    
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


boundary = Boundary(
    source = lambda x, y, z: (x**2 + y**2 + z**2 <= 1.0),
    bbox   = None,
    inlet_points = [
        [-1.0,  0.0,  0.0],   # left pole
    ],
    outlet_points = [
        [ 1.0,  0.0,  0.0],   # right pole
    ],
    border_eps = 1e-2
)