"""
map_tools — Ground Truth Map + Behavior Layer
Layer 1 of the CyberDog Blind Navigation PoC.

Provides:
  - OccupancyGrid: load mesh/pcd → 2D grid
  - BehaviorLayer: semantic zone annotations + queries
  - AStarPlanner: pathfinding + checkpoint extraction
"""

from map_tools.occupancy_grid import OccupancyGrid
from map_tools.behavior_layer import BehaviorLayer, BehaviorZone
from map_tools.astar_planner import AStarPlanner

__version__ = '0.1.0'

__all__ = [
    'OccupancyGrid',
    'BehaviorLayer',
    'BehaviorZone',
    'AStarPlanner',
]
