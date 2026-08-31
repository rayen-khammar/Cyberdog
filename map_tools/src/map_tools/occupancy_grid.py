"""
occupancy_grid.py — Load mesh/point cloud → 2D Occupancy Grid

Converts 3D spatial data (point clouds, meshes) into a 2D occupancy grid
compatible with ROS map_server format (.pgm + .yaml).

Usage:
    grid = OccupancyGrid.from_config('config/map_config.yaml')
    grid.from_numpy(obstacle_array, resolution=0.05)
    grid.save('data/my_map')
    grid = OccupancyGrid.load('data/my_map')
"""

import os
import numpy as np
import yaml
from PIL import Image
from typing import Tuple, Optional, Dict, Any


class OccupancyGrid:
    """2D occupancy grid for navigation planning.
    
    Grid values:
        0   = free space
        100 = occupied
        -1  = unknown
    
    Coordinate system:
        - Grid origin is at bottom-left in world coordinates
        - Cell (i, j) maps to world (origin_x + j * resolution, origin_y + i * resolution)
    """

    def __init__(
        self,
        grid: np.ndarray,
        resolution: float = 0.05,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        occupied_thresh: float = 0.65,
        free_thresh: float = 0.196,
    ):
        """
        Args:
            grid: 2D numpy array (height x width) with values in {-1, 0, 100}.
            resolution: Meters per cell.
            origin: (x, y, yaw) of the bottom-left corner in world frame.
            occupied_thresh: Probability above this = occupied.
            free_thresh: Probability below this = free.
        """
        self.grid = grid.astype(np.int8)
        self.resolution = resolution
        self.origin = origin
        self.occupied_thresh = occupied_thresh
        self.free_thresh = free_thresh

    @property
    def height(self) -> int:
        """Grid height in cells."""
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        """Grid width in cells."""
        return self.grid.shape[1]

    @property
    def width_m(self) -> float:
        """Grid width in meters."""
        return self.width * self.resolution

    @property
    def height_m(self) -> float:
        """Grid height in meters."""
        return self.height * self.resolution

    # ──────────────────────────────────────────────
    #  Construction methods
    # ──────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str) -> 'OccupancyGrid':
        """Create an empty grid from a YAML config file.
        
        Args:
            config_path: Path to map_config.yaml.
        
        Returns:
            Empty OccupancyGrid with dimensions from config.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        map_cfg = config['map']
        resolution = map_cfg['resolution']
        width_cells = int(map_cfg['width_m'] / resolution)
        height_cells = int(map_cfg['height_m'] / resolution)
        origin = tuple(map_cfg.get('origin', [0.0, 0.0, 0.0]))

        # Start with all free space
        grid = np.zeros((height_cells, width_cells), dtype=np.int8)

        return cls(
            grid=grid,
            resolution=resolution,
            origin=origin,
            occupied_thresh=map_cfg.get('occupied_thresh', 0.65),
            free_thresh=map_cfg.get('free_thresh', 0.196),
        )

    @classmethod
    def from_numpy(
        cls,
        obstacle_mask: np.ndarray,
        resolution: float = 0.05,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> 'OccupancyGrid':
        """Create a grid from a boolean obstacle mask.
        
        Args:
            obstacle_mask: 2D bool array (True = occupied).
            resolution: Meters per cell.
            origin: World origin of bottom-left corner.
        
        Returns:
            OccupancyGrid with occupied/free cells.
        """
        grid = np.zeros(obstacle_mask.shape, dtype=np.int8)
        grid[obstacle_mask] = 100
        return cls(grid=grid, resolution=resolution, origin=origin)

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        resolution: float = 0.05,
        ground_z_min: float = -0.3,
        ground_z_max: float = 0.1,
        obstacle_z_max: float = 2.5,
        padding: float = 1.0,
    ) -> 'OccupancyGrid':
        """Create occupancy grid from a 3D point cloud.
        
        Projects points onto the XY plane, classifying them as ground
        or obstacle based on Z height thresholds.
        
        Args:
            points: Nx3 numpy array of (x, y, z) points.
            resolution: Meters per cell.
            ground_z_min: Points with z < this are below ground (ignored).
            ground_z_max: Points with z between ground_z_min and this are ground.
            obstacle_z_max: Points with z > this are above obstacles (ignored).
            padding: Extra meters around the bounding box.
        
        Returns:
            OccupancyGrid with obstacle cells from elevated points.
        """
        if points.shape[1] < 3:
            raise ValueError(f"Points must have at least 3 columns (x,y,z), got {points.shape[1]}")

        # Filter valid height range
        valid = (points[:, 2] >= ground_z_min) & (points[:, 2] <= obstacle_z_max)
        pts = points[valid]

        # Bounding box
        x_min, y_min = pts[:, 0].min() - padding, pts[:, 1].min() - padding
        x_max, y_max = pts[:, 0].max() + padding, pts[:, 1].max() + padding

        width_cells = int(np.ceil((x_max - x_min) / resolution))
        height_cells = int(np.ceil((y_max - y_min) / resolution))

        grid = np.zeros((height_cells, width_cells), dtype=np.int8)

        # Obstacle points: above ground but below obstacle ceiling
        obstacles = pts[(pts[:, 2] > ground_z_max) & (pts[:, 2] <= obstacle_z_max)]

        if len(obstacles) > 0:
            # Rasterize obstacle points into grid cells
            col = ((obstacles[:, 0] - x_min) / resolution).astype(int)
            row = ((obstacles[:, 1] - y_min) / resolution).astype(int)

            # Clip to grid bounds
            col = np.clip(col, 0, width_cells - 1)
            row = np.clip(row, 0, height_cells - 1)

            grid[row, col] = 100

        origin = (x_min, y_min, 0.0)
        return cls(grid=grid, resolution=resolution, origin=origin)

    # ──────────────────────────────────────────────
    #  Coordinate transforms
    # ──────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices.
        
        Args:
            x: World X coordinate (meters).
            y: World Y coordinate (meters).
        
        Returns:
            (row, col) grid cell indices.
        """
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center).
        
        Args:
            row: Grid row index.
            col: Grid column index.
        
        Returns:
            (x, y) world coordinates at cell center.
        """
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return (x, y)

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if a grid cell is within bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        """Check if a grid cell is free (not occupied, not unknown)."""
        if not self.is_in_bounds(row, col):
            return False
        return self.grid[row, col] == 0

    def is_occupied(self, row: int, col: int) -> bool:
        """Check if a grid cell is occupied."""
        if not self.is_in_bounds(row, col):
            return True  # Out of bounds treated as occupied
        return self.grid[row, col] == 100

    # ──────────────────────────────────────────────
    #  Modification
    # ──────────────────────────────────────────────

    def set_occupied(self, row: int, col: int) -> None:
        """Mark a cell as occupied."""
        if self.is_in_bounds(row, col):
            self.grid[row, col] = 100

    def set_free(self, row: int, col: int) -> None:
        """Mark a cell as free."""
        if self.is_in_bounds(row, col):
            self.grid[row, col] = 0

    def set_rect_occupied(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        """Mark a rectangular region in world coords as occupied."""
        r_min, c_min = self.world_to_grid(x_min, y_min)
        r_max, c_max = self.world_to_grid(x_max, y_max)
        r_min, r_max = max(0, min(r_min, r_max)), min(self.height - 1, max(r_min, r_max))
        c_min, c_max = max(0, min(c_min, c_max)), min(self.width - 1, max(c_min, c_max))
        self.grid[r_min:r_max + 1, c_min:c_max + 1] = 100

    def set_rect_free(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        """Mark a rectangular region in world coords as free."""
        r_min, c_min = self.world_to_grid(x_min, y_min)
        r_max, c_max = self.world_to_grid(x_max, y_max)
        r_min, r_max = max(0, min(r_min, r_max)), min(self.height - 1, max(r_min, r_max))
        c_min, c_max = max(0, min(c_min, c_max)), min(self.width - 1, max(c_min, c_max))
        self.grid[r_min:r_max + 1, c_min:c_max + 1] = 0

    # ──────────────────────────────────────────────
    #  I/O — ROS map_server compatible (.pgm + .yaml)
    # ──────────────────────────────────────────────

    def save(self, path_prefix: str) -> Tuple[str, str]:
        """Save grid as .pgm image + .yaml metadata (ROS map_server format).
        
        Args:
            path_prefix: Base path without extension (e.g., 'data/my_map').
        
        Returns:
            (pgm_path, yaml_path) tuple of saved file paths.
        """
        pgm_path = path_prefix + '.pgm'
        yaml_path = path_prefix + '.yaml'

        os.makedirs(os.path.dirname(pgm_path) or '.', exist_ok=True)

        # ROS convention: 254 = free, 0 = occupied, 205 = unknown
        img = np.full(self.grid.shape, 205, dtype=np.uint8)
        img[self.grid == 0] = 254     # free
        img[self.grid == 100] = 0     # occupied

        # Flip vertically because PGM origin is top-left, ROS origin is bottom-left
        img = np.flipud(img)
        Image.fromarray(img, mode='L').save(pgm_path)

        # YAML metadata
        metadata = {
            'image': os.path.basename(pgm_path),
            'resolution': float(self.resolution),
            'origin': [float(v) for v in self.origin],
            'negate': 0,
            'occupied_thresh': float(self.occupied_thresh),
            'free_thresh': float(self.free_thresh),
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        return (pgm_path, yaml_path)

    @classmethod
    def load(cls, path_prefix: str) -> 'OccupancyGrid':
        """Load grid from .pgm + .yaml files.
        
        Args:
            path_prefix: Base path without extension, or path to .yaml file.
        
        Returns:
            OccupancyGrid loaded from disk.
        """
        if path_prefix.endswith('.yaml'):
            yaml_path = path_prefix
        else:
            yaml_path = path_prefix + '.yaml'

        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

        pgm_dir = os.path.dirname(yaml_path)
        pgm_path = os.path.join(pgm_dir, metadata['image'])

        img = np.array(Image.open(pgm_path), dtype=np.uint8)
        img = np.flipud(img)  # Undo the vertical flip from save

        # Convert pixel values back to grid values
        grid = np.full(img.shape, -1, dtype=np.int8)  # default unknown
        grid[img >= 250] = 0      # free (254 in saved format)
        grid[img <= 10] = 100     # occupied (0 in saved format)

        return cls(
            grid=grid,
            resolution=metadata['resolution'],
            origin=tuple(metadata['origin']),
            occupied_thresh=metadata.get('occupied_thresh', 0.65),
            free_thresh=metadata.get('free_thresh', 0.196),
        )

    # ──────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get grid statistics."""
        total = self.grid.size
        free = int(np.sum(self.grid == 0))
        occupied = int(np.sum(self.grid == 100))
        unknown = int(np.sum(self.grid == -1))
        return {
            'total_cells': total,
            'free_cells': free,
            'occupied_cells': occupied,
            'unknown_cells': unknown,
            'free_pct': round(100.0 * free / total, 1),
            'occupied_pct': round(100.0 * occupied / total, 1),
            'dimensions': f'{self.width}x{self.height} cells ({self.width_m:.1f}x{self.height_m:.1f} m)',
            'resolution': self.resolution,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"OccupancyGrid({stats['dimensions']}, "
            f"res={self.resolution}m, "
            f"free={stats['free_pct']}%, "
            f"occupied={stats['occupied_pct']}%)"
        )
