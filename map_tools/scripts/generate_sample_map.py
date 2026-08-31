#!/usr/bin/env python3
"""
generate_sample_map.py — Generate a synthetic campus occupancy grid.

Creates a simple campus layout with:
  - Buildings (occupied blocks)
  - Sidewalks (free paths)
  - A road crossing
  - Open quad area

Saves as .pgm + .yaml for use with the map pipeline.

Usage:
    python scripts/generate_sample_map.py
"""

import os
import sys
import numpy as np

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_tools.occupancy_grid import OccupancyGrid


def generate_campus_map() -> OccupancyGrid:
    """Generate a synthetic campus occupancy grid.
    
    Layout (100m x 80m at 0.05m resolution = 2000x1600 cells):
    
        Y=80 ┌──────────────────────────────────────┐
             │     North Sidewalk + Library          │
        Y=65 ├──────── ROAD ────────────────────────┤
        Y=55 ├──────────────────────────────────────┤
             │                                      │
             │  Quad    │ Science │     Park         │
             │  (grass) │ Bldg   │                  │
             │          │        │                  │
        Y=15 ├──────── MAIN SIDEWALK ───────────────┤
        Y=5  │                                      │
        Y=0  └──────────────────────────────────────┘
            X=0                                    X=100
    """
    # Create empty grid: 100m x 80m at 0.05m/cell
    resolution = 0.05
    width_m = 100.0
    height_m = 80.0
    width_cells = int(width_m / resolution)
    height_cells = int(height_m / resolution)

    # Start with all free space
    grid_data = np.zeros((height_cells, width_cells), dtype=np.int8)
    grid = OccupancyGrid(
        grid=grid_data,
        resolution=resolution,
        origin=(0.0, 0.0, 0.0),
    )

    # ── Add buildings as occupied regions ──

    # Science Building (center-right area)
    grid.set_rect_occupied(55, 25, 65, 52)

    # Admin Building (left side)
    grid.set_rect_occupied(2, 20, 15, 52)

    # Library Building (north, right of center)
    grid.set_rect_occupied(68, 68, 85, 78)

    # Student Center (north, left of center)
    grid.set_rect_occupied(15, 68, 35, 78)

    # Small structure near main sidewalk
    grid.set_rect_occupied(80, 18, 92, 30)

    # ── Add road as occupied (vehicles, not walkable) ──
    # Road runs east-west, but the crosswalk area is kept free
    # Road Y=55 to Y=65
    grid.set_rect_occupied(0, 55, 100, 65)

    # Clear the crosswalk (free zone in the road)
    grid.set_rect_free(40, 55, 50, 65)

    # ── Clear sidewalk paths ──
    # Main sidewalk (south)
    grid.set_rect_free(5, 5, 95, 18)

    # North sidewalk
    grid.set_rect_free(5, 65, 95, 68)

    # Path from main sidewalk to quad
    grid.set_rect_free(16, 15, 19, 55)

    # Path around science building (east side)
    grid.set_rect_free(66, 15, 70, 55)

    # Path to library (north of crosswalk)
    grid.set_rect_free(40, 65, 50, 75)

    # Clear quad area (navigable grass)
    grid.set_rect_free(20, 20, 50, 52)

    print(f"Generated campus map: {grid}")
    stats = grid.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return grid


def main():
    """Generate and save the sample campus map."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    grid = generate_campus_map()

    # Save as .pgm + .yaml
    output_prefix = os.path.join(data_dir, 'sample_grid')
    pgm_path, yaml_path = grid.save(output_prefix)
    print(f"\nSaved:")
    print(f"  Grid: {pgm_path}")
    print(f"  Meta: {yaml_path}")

    # Verify round-trip
    loaded = OccupancyGrid.load(output_prefix)
    assert loaded.width == grid.width, "Width mismatch after load"
    assert loaded.height == grid.height, "Height mismatch after load"
    assert np.array_equal(loaded.grid, grid.grid), "Grid data mismatch after load"
    print(f"\n✓ Round-trip verification passed: {loaded}")


if __name__ == '__main__':
    main()
