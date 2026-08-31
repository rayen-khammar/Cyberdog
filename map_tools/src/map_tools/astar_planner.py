"""
astar_planner.py — A* Pathfinding + Checkpoint Extraction

A* search over the occupancy grid with:
  - Diagonal movement support
  - Ramer-Douglas-Peucker polyline simplification
  - Checkpoint extraction based on turns and zone boundaries

From the CyberDog Skill spec:
  Implement A* over occupancy grid; simplify polyline with RDP →
  checkpoints (turn points double as voice-announcement triggers).

Usage:
    planner = AStarPlanner.from_config('config/map_config.yaml')
    route = planner.plan(grid, start=(5.0, 10.0), goal=(80.0, 60.0))
    route = planner.plan_with_checkpoints(grid, behavior_layer, start, goal)
"""

import heapq
import math
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from map_tools.occupancy_grid import OccupancyGrid
from map_tools.behavior_layer import BehaviorLayer


@dataclass
class Checkpoint:
    """A navigation checkpoint along a route.
    
    Checkpoints are significant points where the robot should
    announce terrain changes, turns, or zone transitions.
    """
    index: int                      # index in the simplified polyline
    position: Tuple[float, float]   # (x, y) in world coords
    checkpoint_type: str            # 'turn', 'zone_entry', 'zone_exit', 'waypoint'
    turn_angle: float = 0.0        # degrees, for turn-type checkpoints
    zone_types: List[str] = field(default_factory=list)
    announcements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'position': list(self.position),
            'checkpoint_type': self.checkpoint_type,
            'turn_angle': round(self.turn_angle, 1),
            'zone_types': self.zone_types,
            'announcements': self.announcements,
        }


@dataclass
class Route:
    """A planned navigation route with polyline and checkpoints.
    
    Attributes:
        raw_path: Full-resolution A* path in world coordinates.
        polyline: RDP-simplified path in world coordinates.
        checkpoints: Ordered list of navigation checkpoints.
        total_distance: Total route distance in meters.
        grid_cells_explored: Number of cells explored by A*.
    """
    raw_path: List[Tuple[float, float]]
    polyline: List[Tuple[float, float]]
    checkpoints: List[Checkpoint]
    total_distance: float
    grid_cells_explored: int = 0

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'polyline': [list(p) for p in self.polyline],
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'total_distance': round(self.total_distance, 2),
            'num_points_raw': len(self.raw_path),
            'num_points_simplified': len(self.polyline),
            'num_checkpoints': self.num_checkpoints,
            'grid_cells_explored': self.grid_cells_explored,
        }

    def __repr__(self) -> str:
        return (
            f"Route(dist={self.total_distance:.1f}m, "
            f"points={len(self.polyline)}, "
            f"checkpoints={self.num_checkpoints})"
        )


class AStarPlanner:
    """A* pathfinding over an occupancy grid with checkpoint extraction.
    
    Features:
      - 8-connected grid search (diagonal movement)
      - Euclidean heuristic
      - Ramer-Douglas-Peucker polyline simplification
      - Checkpoint generation from turns and zone boundaries
    """

    def __init__(
        self,
        diagonal_movement: bool = True,
        diagonal_cost: float = 1.414,
        cardinal_cost: float = 1.0,
        rdp_epsilon: float = 0.3,
        min_checkpoint_spacing: float = 2.0,
        max_checkpoint_spacing: float = 8.0,
        turn_angle_threshold: float = 30.0,
    ):
        """
        Args:
            diagonal_movement: Allow 8-connected movement.
            diagonal_cost: Cost of diagonal steps (sqrt(2) ≈ 1.414).
            cardinal_cost: Cost of cardinal steps.
            rdp_epsilon: RDP simplification tolerance in meters.
            min_checkpoint_spacing: Min distance between checkpoints (meters).
            max_checkpoint_spacing: Max distance between checkpoints (meters).
            turn_angle_threshold: Turns sharper than this (degrees) become checkpoints.
        """
        self.diagonal_movement = diagonal_movement
        self.diagonal_cost = diagonal_cost
        self.cardinal_cost = cardinal_cost
        self.rdp_epsilon = rdp_epsilon
        self.min_checkpoint_spacing = min_checkpoint_spacing
        self.max_checkpoint_spacing = max_checkpoint_spacing
        self.turn_angle_threshold = turn_angle_threshold

    @classmethod
    def from_config(cls, config_path: str) -> 'AStarPlanner':
        """Create planner from YAML config."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        astar_cfg = config.get('astar', {})
        return cls(
            diagonal_movement=astar_cfg.get('diagonal_movement', True),
            diagonal_cost=astar_cfg.get('diagonal_cost', 1.414),
            cardinal_cost=astar_cfg.get('cardinal_cost', 1.0),
            rdp_epsilon=astar_cfg.get('rdp_epsilon', 0.3),
            min_checkpoint_spacing=astar_cfg.get('min_checkpoint_spacing', 2.0),
            max_checkpoint_spacing=astar_cfg.get('max_checkpoint_spacing', 8.0),
            turn_angle_threshold=astar_cfg.get('turn_angle_threshold', 30.0),
        )

    # ──────────────────────────────────────────────
    #  A* Search
    # ──────────────────────────────────────────────

    def _neighbors(self, row: int, col: int) -> List[Tuple[int, int, float]]:
        """Get valid neighbors with movement costs.
        
        Returns:
            List of (row, col, cost) tuples.
        """
        cardinal = [
            (row - 1, col, self.cardinal_cost),
            (row + 1, col, self.cardinal_cost),
            (row, col - 1, self.cardinal_cost),
            (row, col + 1, self.cardinal_cost),
        ]
        if self.diagonal_movement:
            diagonal = [
                (row - 1, col - 1, self.diagonal_cost),
                (row - 1, col + 1, self.diagonal_cost),
                (row + 1, col - 1, self.diagonal_cost),
                (row + 1, col + 1, self.diagonal_cost),
            ]
            return cardinal + diagonal
        return cardinal

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def search(
        self, grid: OccupancyGrid, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]
    ) -> Optional[Tuple[List[Tuple[int, int]], int]]:
        """Run A* search on the grid.
        
        Args:
            grid: The occupancy grid to search.
            start_rc: Start cell (row, col).
            goal_rc: Goal cell (row, col).
        
        Returns:
            (path, cells_explored) tuple where path is list of (row, col),
            or None if no path found.
        """
        if not grid.is_free(*start_rc):
            return None
        if not grid.is_free(*goal_rc):
            return None

        # Priority queue: (f_cost, counter, row, col)
        counter = 0
        open_set = [(0.0, counter, start_rc[0], start_rc[1])]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_rc: 0.0}
        closed_set = set()
        cells_explored = 0

        while open_set:
            _, _, cr, cc = heapq.heappop(open_set)
            current = (cr, cc)

            if current == goal_rc:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return (path, cells_explored)

            if current in closed_set:
                continue
            closed_set.add(current)
            cells_explored += 1

            for nr, nc, cost in self._neighbors(cr, cc):
                neighbor = (nr, nc)
                if neighbor in closed_set:
                    continue
                if not grid.is_in_bounds(nr, nc):
                    continue
                if not grid.is_free(nr, nc):
                    continue

                tentative_g = g_score[current] + cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_rc)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nr, nc))

        return None  # No path found

    # ──────────────────────────────────────────────
    #  Ramer-Douglas-Peucker Simplification
    # ──────────────────────────────────────────────

    @staticmethod
    def _perpendicular_distance(
        point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) -> float:
        """Perpendicular distance from a point to a line segment."""
        if np.allclose(line_start, line_end):
            return float(np.linalg.norm(point - line_start))

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.dot(line_vec, line_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = line_start + t * line_vec
        return float(np.linalg.norm(point - projection))

    def rdp_simplify(
        self, points: List[Tuple[float, float]], epsilon: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Ramer-Douglas-Peucker polyline simplification.
        
        Args:
            points: List of (x, y) points.
            epsilon: Tolerance in meters. Uses self.rdp_epsilon if None.
        
        Returns:
            Simplified list of (x, y) points.
        """
        if epsilon is None:
            epsilon = self.rdp_epsilon

        if len(points) <= 2:
            return list(points)

        pts = np.array(points)

        # Find the point with the greatest distance from the line start→end
        start = pts[0]
        end = pts[-1]
        distances = np.array([
            self._perpendicular_distance(pts[i], start, end)
            for i in range(1, len(pts) - 1)
        ])

        max_dist = distances.max()
        max_idx = distances.argmax() + 1

        if max_dist > epsilon:
            # Recurse on both halves
            left = self.rdp_simplify(
                [tuple(p) for p in pts[:max_idx + 1]], epsilon
            )
            right = self.rdp_simplify(
                [tuple(p) for p in pts[max_idx:]], epsilon
            )
            return left[:-1] + right
        else:
            return [tuple(start), tuple(end)]

    # ──────────────────────────────────────────────
    #  Checkpoint Extraction
    # ──────────────────────────────────────────────

    @staticmethod
    def _angle_between(
        p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
    ) -> float:
        """Compute the turn angle at p2 (in degrees).
        
        Returns the angle between vectors (p1→p2) and (p2→p3).
        0° = straight ahead, 180° = U-turn.
        """
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def extract_checkpoints(
        self,
        polyline: List[Tuple[float, float]],
        behavior_layer: Optional[BehaviorLayer] = None,
    ) -> List[Checkpoint]:
        """Extract navigation checkpoints from a simplified polyline.
        
        Checkpoints are placed at:
          1. Significant turns (angle > threshold)
          2. Zone boundary crossings (if behavior_layer provided)
          3. Regular intervals (max_checkpoint_spacing)
          4. Start and end points
        
        Args:
            polyline: Simplified route in world coordinates.
            behavior_layer: Optional behavior layer for zone queries.
        
        Returns:
            Ordered list of Checkpoint objects.
        """
        if len(polyline) < 2:
            return []

        checkpoints = []
        accumulated_dist = 0.0
        last_checkpoint_dist = 0.0
        prev_zones: List[str] = []

        # Always add start point
        start_zones = []
        start_announcements = []
        if behavior_layer:
            actions = behavior_layer.query_actions(*polyline[0])
            start_zones = actions['zone_types']
            start_announcements = actions['announcements']
            prev_zones = start_zones

        checkpoints.append(Checkpoint(
            index=0,
            position=polyline[0],
            checkpoint_type='waypoint',
            zone_types=start_zones,
            announcements=start_announcements,
        ))

        for i in range(1, len(polyline)):
            # Accumulate distance
            dx = polyline[i][0] - polyline[i - 1][0]
            dy = polyline[i][1] - polyline[i - 1][1]
            seg_dist = math.sqrt(dx * dx + dy * dy)
            accumulated_dist += seg_dist

            is_checkpoint = False
            cp_type = 'waypoint'
            turn_angle = 0.0
            announcements = []

            # Check for significant turn
            if 0 < i < len(polyline) - 1:
                angle = self._angle_between(
                    polyline[i - 1], polyline[i], polyline[i + 1]
                )
                if angle > self.turn_angle_threshold:
                    is_checkpoint = True
                    cp_type = 'turn'
                    turn_angle = angle

                    # Determine turn direction for announcement
                    v1 = np.array([polyline[i][0] - polyline[i-1][0],
                                   polyline[i][1] - polyline[i-1][1]])
                    v2 = np.array([polyline[i+1][0] - polyline[i][0],
                                   polyline[i+1][1] - polyline[i][1]])
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    direction = "left" if cross > 0 else "right"
                    announcements.append(f"Turn {direction} ahead")

            # Check for zone transitions
            current_zones = []
            if behavior_layer:
                actions = behavior_layer.query_actions(*polyline[i])
                current_zones = actions['zone_types']
                zone_announcements = actions['announcements']

                # Zone entry
                new_zones = set(current_zones) - set(prev_zones)
                if new_zones:
                    is_checkpoint = True
                    if cp_type == 'waypoint':
                        cp_type = 'zone_entry'
                    announcements.extend(zone_announcements)

                # Zone exit
                left_zones = set(prev_zones) - set(current_zones)
                if left_zones and not new_zones:
                    is_checkpoint = True
                    if cp_type == 'waypoint':
                        cp_type = 'zone_exit'

                prev_zones = current_zones

            # Check spacing — force checkpoint if too far from last
            dist_since_last = accumulated_dist - last_checkpoint_dist
            if dist_since_last >= self.max_checkpoint_spacing:
                is_checkpoint = True

            # Skip if too close to last checkpoint
            if is_checkpoint and dist_since_last < self.min_checkpoint_spacing:
                # Only skip if not the last point
                if i < len(polyline) - 1:
                    is_checkpoint = False

            if is_checkpoint:
                checkpoints.append(Checkpoint(
                    index=i,
                    position=polyline[i],
                    checkpoint_type=cp_type,
                    turn_angle=turn_angle,
                    zone_types=current_zones,
                    announcements=announcements,
                ))
                last_checkpoint_dist = accumulated_dist

        # Always add end point (if not already a checkpoint)
        if checkpoints[-1].index != len(polyline) - 1:
            end_zones = []
            end_announcements = ["Destination reached"]
            if behavior_layer:
                actions = behavior_layer.query_actions(*polyline[-1])
                end_zones = actions['zone_types']
                end_announcements += actions['announcements']

            checkpoints.append(Checkpoint(
                index=len(polyline) - 1,
                position=polyline[-1],
                checkpoint_type='waypoint',
                zone_types=end_zones,
                announcements=end_announcements,
            ))

        return checkpoints

    # ──────────────────────────────────────────────
    #  High-level planning API
    # ──────────────────────────────────────────────

    def plan(
        self,
        grid: OccupancyGrid,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[Route]:
        """Plan a route from start to goal.
        
        Args:
            grid: Occupancy grid for pathfinding.
            start: (x, y) start position in world coordinates.
            goal: (x, y) goal position in world coordinates.
        
        Returns:
            Route object, or None if no path found.
        """
        start_rc = grid.world_to_grid(*start)
        goal_rc = grid.world_to_grid(*goal)

        result = self.search(grid, start_rc, goal_rc)
        if result is None:
            return None

        grid_path, cells_explored = result

        # Convert grid path to world coordinates
        raw_path = [grid.grid_to_world(r, c) for r, c in grid_path]

        # Simplify
        polyline = self.rdp_simplify(raw_path)

        # Compute total distance
        total_dist = 0.0
        for i in range(1, len(polyline)):
            dx = polyline[i][0] - polyline[i - 1][0]
            dy = polyline[i][1] - polyline[i - 1][1]
            total_dist += math.sqrt(dx * dx + dy * dy)

        # Basic checkpoints (no behavior layer)
        checkpoints = self.extract_checkpoints(polyline)

        return Route(
            raw_path=raw_path,
            polyline=polyline,
            checkpoints=checkpoints,
            total_distance=total_dist,
            grid_cells_explored=cells_explored,
        )

    def plan_with_checkpoints(
        self,
        grid: OccupancyGrid,
        behavior_layer: BehaviorLayer,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[Route]:
        """Plan a route with behavior-layer-aware checkpoints.
        
        Like plan(), but checkpoints include zone transitions and announcements.
        
        Args:
            grid: Occupancy grid for pathfinding.
            behavior_layer: Behavior layer for zone queries.
            start: (x, y) start position in world coordinates.
            goal: (x, y) goal position in world coordinates.
        
        Returns:
            Route object with zone-aware checkpoints, or None if no path found.
        """
        start_rc = grid.world_to_grid(*start)
        goal_rc = grid.world_to_grid(*goal)

        result = self.search(grid, start_rc, goal_rc)
        if result is None:
            return None

        grid_path, cells_explored = result

        # Convert grid path to world coordinates
        raw_path = [grid.grid_to_world(r, c) for r, c in grid_path]

        # Simplify
        polyline = self.rdp_simplify(raw_path)

        # Compute total distance
        total_dist = 0.0
        for i in range(1, len(polyline)):
            dx = polyline[i][0] - polyline[i - 1][0]
            dy = polyline[i][1] - polyline[i - 1][1]
            total_dist += math.sqrt(dx * dx + dy * dy)

        # Zone-aware checkpoints
        checkpoints = self.extract_checkpoints(polyline, behavior_layer)

        return Route(
            raw_path=raw_path,
            polyline=polyline,
            checkpoints=checkpoints,
            total_distance=total_dist,
            grid_cells_explored=cells_explored,
        )

    def __repr__(self) -> str:
        return (
            f"AStarPlanner(diagonal={self.diagonal_movement}, "
            f"rdp_eps={self.rdp_epsilon}m, "
            f"turn_thresh={self.turn_angle_threshold}°)"
        )
