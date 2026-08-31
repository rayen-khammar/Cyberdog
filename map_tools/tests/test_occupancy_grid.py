"""Tests for occupancy_grid.py"""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_tools.occupancy_grid import OccupancyGrid


class TestOccupancyGridConstruction:
    """Test grid construction methods."""

    def test_from_numpy_basic(self):
        """Grid from a boolean obstacle mask."""
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:20, 30:50] = True  # obstacle block

        grid = OccupancyGrid.from_numpy(mask, resolution=0.1)

        assert grid.width == 200
        assert grid.height == 100
        assert grid.resolution == 0.1
        assert grid.is_occupied(15, 40)
        assert grid.is_free(0, 0)

    def test_from_numpy_dimensions(self):
        """Grid dimensions match input."""
        mask = np.zeros((50, 80), dtype=bool)
        grid = OccupancyGrid.from_numpy(mask, resolution=0.05)

        assert grid.width == 80
        assert grid.height == 50
        assert grid.width_m == pytest.approx(4.0)
        assert grid.height_m == pytest.approx(2.5)

    def test_from_point_cloud(self):
        """Grid from 3D point cloud."""
        # Create a simple point cloud with ground and one obstacle
        ground = np.column_stack([
            np.random.uniform(0, 10, 1000),
            np.random.uniform(0, 10, 1000),
            np.random.uniform(-0.1, 0.05, 1000),
        ])
        obstacle = np.column_stack([
            np.random.uniform(4, 6, 200),
            np.random.uniform(4, 6, 200),
            np.random.uniform(0.5, 1.5, 200),
        ])
        points = np.vstack([ground, obstacle])

        grid = OccupancyGrid.from_point_cloud(points, resolution=0.1)

        # Grid should exist and have reasonable dimensions
        assert grid.width > 0
        assert grid.height > 0

        # Center area should have some occupied cells (obstacle)
        center_r, center_c = grid.world_to_grid(5.0, 5.0)
        # Check a small neighborhood for occupied cells
        occupied_count = 0
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if grid.is_occupied(center_r + dr, center_c + dc):
                    occupied_count += 1
        assert occupied_count > 0, "Obstacle area should have occupied cells"

    def test_from_config(self):
        """Grid from config file."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'map_config.yaml'
        )
        if os.path.exists(config_path):
            grid = OccupancyGrid.from_config(config_path)
            assert grid.width > 0
            assert grid.height > 0
            assert grid.resolution == 0.05


class TestOccupancyGridCoordinates:
    """Test coordinate transforms."""

    def test_world_to_grid_origin(self):
        """Origin maps to (0, 0)."""
        grid = OccupancyGrid(np.zeros((100, 100)), resolution=0.1)
        row, col = grid.world_to_grid(0.0, 0.0)
        assert row == 0
        assert col == 0

    def test_world_to_grid_offset(self):
        """Positive coordinates map to positive indices."""
        grid = OccupancyGrid(np.zeros((100, 100)), resolution=0.1)
        row, col = grid.world_to_grid(5.0, 3.0)
        assert col == 50
        assert row == 30

    def test_grid_to_world_roundtrip(self):
        """grid_to_world ∘ world_to_grid ≈ identity (within resolution)."""
        grid = OccupancyGrid(np.zeros((100, 100)), resolution=0.1)
        x_orig, y_orig = 3.7, 5.2
        row, col = grid.world_to_grid(x_orig, y_orig)
        x_back, y_back = grid.grid_to_world(row, col)

        # Should be within one cell of original
        assert abs(x_back - x_orig) < grid.resolution * 1.5
        assert abs(y_back - y_orig) < grid.resolution * 1.5

    def test_is_in_bounds(self):
        """Bounds checking."""
        grid = OccupancyGrid(np.zeros((100, 200)), resolution=0.1)
        assert grid.is_in_bounds(0, 0)
        assert grid.is_in_bounds(99, 199)
        assert not grid.is_in_bounds(-1, 0)
        assert not grid.is_in_bounds(100, 0)
        assert not grid.is_in_bounds(0, 200)


class TestOccupancyGridModification:
    """Test grid modification methods."""

    def test_set_occupied_free(self):
        """Set individual cells."""
        grid = OccupancyGrid(np.zeros((100, 100)), resolution=0.1)
        assert grid.is_free(50, 50)

        grid.set_occupied(50, 50)
        assert grid.is_occupied(50, 50)

        grid.set_free(50, 50)
        assert grid.is_free(50, 50)

    def test_set_rect_occupied(self):
        """Set rectangular region."""
        grid = OccupancyGrid(np.zeros((100, 100)), resolution=0.1)
        grid.set_rect_occupied(2.0, 3.0, 5.0, 6.0)

        # Center of rectangle should be occupied
        r, c = grid.world_to_grid(3.5, 4.5)
        assert grid.is_occupied(r, c)

        # Outside should be free
        r, c = grid.world_to_grid(0.5, 0.5)
        assert grid.is_free(r, c)


class TestOccupancyGridIO:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self):
        """Grid survives save → load."""
        grid = OccupancyGrid(np.zeros((100, 200), dtype=np.int8), resolution=0.1)
        grid.set_rect_occupied(2.0, 3.0, 5.0, 6.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'test_grid')
            grid.save(prefix)

            loaded = OccupancyGrid.load(prefix)

            assert loaded.width == grid.width
            assert loaded.height == grid.height
            assert loaded.resolution == grid.resolution
            assert np.array_equal(loaded.grid, grid.grid)

    def test_save_creates_files(self):
        """Save creates both .pgm and .yaml files."""
        grid = OccupancyGrid(np.zeros((50, 50), dtype=np.int8), resolution=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'test')
            pgm, yml = grid.save(prefix)

            assert os.path.exists(pgm)
            assert os.path.exists(yml)
            assert pgm.endswith('.pgm')
            assert yml.endswith('.yaml')


class TestOccupancyGridStats:
    """Test utility methods."""

    def test_get_stats(self):
        """Stats report correct counts."""
        data = np.zeros((100, 100), dtype=np.int8)
        data[0:10, :] = 100  # 1000 occupied
        grid = OccupancyGrid(data, resolution=0.1)

        stats = grid.get_stats()
        assert stats['total_cells'] == 10000
        assert stats['occupied_cells'] == 1000
        assert stats['free_cells'] == 9000

    def test_repr(self):
        """String representation."""
        grid = OccupancyGrid(np.zeros((100, 200)), resolution=0.1)
        s = repr(grid)
        assert '200x100' in s
        assert '0.1' in s
