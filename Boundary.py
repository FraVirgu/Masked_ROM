import numpy as np


class Boundary:
    """
    A callable domain boundary compatible with OpenCCO's DomainController interface.

    OpenCCO defines a domain via a binary mask (3D array, 1=inside 0=outside).
    This class unifies:
      - implicit functions  f(x,y,z) -> bool  (analytical boundaries)
      - voxel masks         np.ndarray of shape (Nx, Ny, Nz)  (OpenCCO output)

    In both cases, calling the object on a point returns True if the point
    is INSIDE the domain (i.e. should be kept in the network).
    """

    def __init__(self, source, bbox=None):
        """
        Parameters
        ----------
        source : callable or np.ndarray
            - callable : f(x, y, z) -> bool   (your analytical boundary)
            - np.ndarray of shape (Nx, Ny, Nz) : binary voxel mask from OpenCCO
        bbox : tuple of pairs, optional
            ((xmin,xmax),(ymin,ymax),(zmin,zmax)) — required when source is a mask,
            used to map world coordinates → voxel indices.
        """
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

    # --- main interface ---------------------------------------------------

    def __call__(self, point) -> bool:
        """Return True if point lies inside the domain."""
        x, y, z = point
        if self._mode == "implicit":
            return bool(self._fn(x, y, z))
        else:
            return self._query_mask(x, y, z)

    def is_inside(self, point) -> bool:
        """Alias — mirrors OpenCCO DomainController naming."""
        return self.__call__(point)

    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorised filter: keep only points inside the domain."""
        return np.array([p for p in points if self(p)])

    # --- mask helper ------------------------------------------------------

    def _query_mask(self, x, y, z) -> bool:
        """Map world coords to voxel index and look up the mask."""
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._bbox
        Nx, Ny, Nz = self._shape

        ix = int((x - xmin) / (xmax - xmin) * (Nx - 1))
        iy = int((y - ymin) / (ymax - ymin) * (Ny - 1))
        iz = int((z - zmin) / (zmax - zmin) * (Nz - 1))

        # clamp to valid range
        ix = max(0, min(ix, Nx - 1))
        iy = max(0, min(iy, Ny - 1))
        iz = max(0, min(iz, Nz - 1))

        return bool(self._mask[ix, iy, iz])

    def __repr__(self):
        return f"Boundary(mode={self._mode!r})"