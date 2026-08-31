#!/usr/bin/env python3
"""
visualize_map.py — Render occupancy grid + zones + A* route + checkpoints.

Acceptance criteria from CyberDogSkill.md:
  "occupancy grid + A* route + checkpoints render correctly in RViz
   for ≥3 route queries"

This script renders in matplotlib (no RViz required for PoC).

Usage:
    python scripts/visualize_map.py
    python scripts/visualize_map.py --routes 5
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_tools.occupancy_grid import OccupancyGrid
from map_tools.behavior_layer import BehaviorLayer
from map_tools.astar_planner import AStarPlanner


# Zone type → color mapping
ZONE_COLORS = {
    'grass': '#4CAF50',
    'stairs': '#F44336',
    'crosswalk': '#FFC107',
    'entrance_zone': '#2196F3',
    'sidewalk': '#9E9E9E',
    'road': '#424242',
}


def plot_map_with_routes(
    grid: OccupancyGrid,
    behavior_layer: BehaviorLayer,
    planner: AStarPlanner,
    route_queries: list,
    save_path: str = None,
):
    """Plot the full map with zones and multiple A* routes.
    
    Args:
        grid: Occupancy grid.
        behavior_layer: Behavior layer with zones.
        planner: A* planner instance.
        route_queries: List of (start, goal, label) tuples.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax_idx, ax in enumerate(axes):
        # ── Plot occupancy grid ──
        # Convert grid values to display: free=white, occupied=black, unknown=gray
        display = np.ones((*grid.grid.shape, 3))  # white background
        display[grid.grid == 100] = [0.15, 0.15, 0.15]  # occupied = dark gray
        display[grid.grid == -1] = [0.7, 0.7, 0.7]      # unknown = light gray

        # Extent maps grid indices to world coordinates
        extent = [
            grid.origin[0],
            grid.origin[0] + grid.width * grid.resolution,
            grid.origin[1],
            grid.origin[1] + grid.height * grid.resolution,
        ]
        ax.imshow(display, origin='lower', extent=extent, aspect='equal', alpha=0.8)

        # ── Plot behavior zones ──
        for zone in behavior_layer.zones:
            poly = plt.Polygon(
                zone.polygon,
                alpha=0.25,
                facecolor=ZONE_COLORS.get(zone.zone_type, '#888888'),
                edgecolor=ZONE_COLORS.get(zone.zone_type, '#888888'),
                linewidth=2,
                label=zone.zone_type if ax_idx == 0 else None,
            )
            ax.add_patch(poly)

            # Label zone
            cx, cy = zone.centroid()
            name = zone.metadata.get('name', zone.zone_id)
            ax.annotate(
                name, (cx, cy),
                fontsize=7, ha='center', va='center',
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5),
            )

    # ── Left panel: overview with all routes ──
    ax_left = axes[0]
    ax_left.set_title('Campus Map — All Routes', fontsize=14, fontweight='bold')

    route_colors = plt.cm.tab10(np.linspace(0, 1, max(len(route_queries), 1)))

    for i, (start, goal, label) in enumerate(route_queries):
        color = route_colors[i % len(route_colors)]
        route = planner.plan_with_checkpoints(grid, behavior_layer, start, goal)

        if route is None:
            print(f"  ✗ Route '{label}': No path found from {start} to {goal}")
            continue

        print(f"  ✓ Route '{label}': {route}")

        # Plot polyline
        poly_x = [p[0] for p in route.polyline]
        poly_y = [p[1] for p in route.polyline]
        ax_left.plot(poly_x, poly_y, '-', color=color, linewidth=2.5, label=label, zorder=5)

        # Plot checkpoints
        for cp in route.checkpoints:
            marker = 'o'
            size = 60
            if cp.checkpoint_type == 'turn':
                marker = '^'
                size = 100
            elif cp.checkpoint_type == 'zone_entry':
                marker = 's'
                size = 80
            elif cp.checkpoint_type == 'zone_exit':
                marker = 'D'
                size = 80

            ax_left.scatter(
                cp.position[0], cp.position[1],
                marker=marker, s=size, color=color,
                edgecolors='black', linewidth=1, zorder=10,
            )

        # Start/goal markers
        ax_left.scatter(*start, marker='*', s=200, color=color, edgecolors='black',
                       linewidth=1.5, zorder=15)
        ax_left.scatter(*goal, marker='X', s=200, color=color, edgecolors='black',
                       linewidth=1.5, zorder=15)

    ax_left.legend(loc='upper left', fontsize=9)
    ax_left.set_xlabel('X (meters)')
    ax_left.set_ylabel('Y (meters)')
    ax_left.grid(True, alpha=0.3)

    # ── Right panel: first route detail with annotations ──
    ax_right = axes[1]
    ax_right.set_title('Route Detail — Checkpoints', fontsize=14, fontweight='bold')

    if route_queries:
        start, goal, label = route_queries[0]
        route = planner.plan_with_checkpoints(grid, behavior_layer, start, goal)

        if route:
            poly_x = [p[0] for p in route.polyline]
            poly_y = [p[1] for p in route.polyline]
            ax_right.plot(poly_x, poly_y, '-', color='#E91E63', linewidth=3, zorder=5)

            for cp in route.checkpoints:
                marker = 'o'
                size = 100
                color = '#E91E63'
                if cp.checkpoint_type == 'turn':
                    marker = '^'
                    size = 150
                    color = '#FF5722'
                elif cp.checkpoint_type == 'zone_entry':
                    marker = 's'
                    size = 120
                    color = '#4CAF50'
                elif cp.checkpoint_type == 'zone_exit':
                    marker = 'D'
                    size = 120
                    color = '#2196F3'

                ax_right.scatter(
                    cp.position[0], cp.position[1],
                    marker=marker, s=size, color=color,
                    edgecolors='black', linewidth=1.5, zorder=10,
                )

                # Annotate checkpoints
                annotation = cp.checkpoint_type
                if cp.announcements:
                    annotation = cp.announcements[0][:30]
                ax_right.annotate(
                    annotation,
                    (cp.position[0], cp.position[1]),
                    textcoords='offset points',
                    xytext=(10, 10),
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                )

            # Legend for checkpoint types
            ax_right.scatter([], [], marker='^', s=100, color='#FF5722', label='Turn')
            ax_right.scatter([], [], marker='s', s=80, color='#4CAF50', label='Zone Entry')
            ax_right.scatter([], [], marker='D', s=80, color='#2196F3', label='Zone Exit')
            ax_right.scatter([], [], marker='o', s=60, color='#E91E63', label='Waypoint')
            ax_right.legend(loc='upper left', fontsize=9)

            # Print route details
            print(f"\n  Route detail: {route.to_dict()}")

    ax_right.set_xlabel('X (meters)')
    ax_right.set_ylabel('Y (meters)')
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize campus map with routes')
    parser.add_argument('--routes', type=int, default=3, help='Number of route queries')
    parser.add_argument('--save', type=str, default=None, help='Save figure path')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(base_dir, 'data')
    config_dir = os.path.join(base_dir, 'config')

    # Load map
    grid_path = os.path.join(data_dir, 'sample_grid')
    if not os.path.exists(grid_path + '.yaml'):
        print("Sample grid not found. Run generate_sample_map.py first.")
        print("  python scripts/generate_sample_map.py")
        sys.exit(1)

    print("Loading map...")
    grid = OccupancyGrid.load(grid_path)
    print(f"  Grid: {grid}")

    # Load behavior layer
    zones_path = os.path.join(data_dir, 'sample_zones.json')
    config_path = os.path.join(config_dir, 'map_config.yaml')
    behavior_layer = BehaviorLayer.from_config(config_path, zones_path)
    print(f"  Zones: {behavior_layer}")

    # Create planner
    planner = AStarPlanner.from_config(config_path)
    print(f"  Planner: {planner}")

    # Define route queries (≥3 as per acceptance criteria)
    route_queries = [
        ((12, 10), (75, 70), "Main Gate → Library"),
        ((12, 10), (60, 30), "Main Gate → Science Bldg"),
        ((60, 30), (75, 70), "Science Bldg → Library"),
    ]

    # Add more routes if requested
    extra_routes = [
        ((85, 10), (20, 70), "East Lot → Student Center"),
        ((20, 35), (85, 70), "Quad → East Campus"),
    ]
    while len(route_queries) < args.routes and extra_routes:
        route_queries.append(extra_routes.pop(0))

    print(f"\nPlanning {len(route_queries)} routes...")
    plot_map_with_routes(
        grid, behavior_layer, planner, route_queries,
        save_path=args.save or os.path.join(data_dir, 'map_visualization.png'),
    )


if __name__ == '__main__':
    main()
