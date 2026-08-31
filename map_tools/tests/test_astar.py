"""Tests for astar_planner.py"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_tools.occupancy_grid import OccupancyGrid
from map_tools.behavior_layer import BehaviorLayer, BehaviorZone, ActionRules
from map_tools.astar_planner import AStarPlanner, Route, Checkpoint


class TestAStarSearch:
    """Test raw A* search on grids."""

    def _make_simple_grid(self, width=100, height=100, resolution=0.1):
        """Create a simple free grid."""
        data = np.zeros((height, width), dtype=np.int8)
        return OccupancyGrid(data, resolution=resolution)

    def _make_grid_with_wall(self):
        """Create a grid with a wall that requires routing around."""
        data = np.zeros((100, 100), dtype=np.int8)
        # Vertical wall from row 10 to 90, at column 50
        data[10:90, 50] = 100
        return OccupancyGrid(data, resolution=0.1)

    def test_straight_path(self):
        """A* finds a path in an open grid."""
        grid = self._make_simple_grid()
        planner = AStarPlanner()

        result = planner.search(grid, (10, 10), (90, 90))
        assert result is not None
        path, cells = result
        assert len(path) > 0
        assert path[0] == (10, 10)
        assert path[-1] == (90, 90)

    def test_path_around_wall(self):
        """A* routes around a wall."""
        grid = self._make_grid_with_wall()
        planner = AStarPlanner()

        result = planner.search(grid, (50, 20), (50, 80))
        assert result is not None
        path, _ = result
        assert path[0] == (50, 20)
        assert path[-1] == (50, 80)

        # Path should go around the wall, not through it
        for r, c in path:
            assert grid.is_free(r, c), f"Path goes through occupied cell ({r}, {c})"

    def test_no_path(self):
        """A* returns None when no path exists."""
        data = np.zeros((100, 100), dtype=np.int8)
        # Complete horizontal wall
        data[50, :] = 100
        grid = OccupancyGrid(data, resolution=0.1)
        planner = AStarPlanner()

        result = planner.search(grid, (20, 20), (80, 80))
        assert result is None

    def test_start_occupied(self):
        """A* returns None when start is occupied."""
        grid = self._make_simple_grid()
        grid.set_occupied(10, 10)
        planner = AStarPlanner()

        result = planner.search(grid, (10, 10), (90, 90))
        assert result is None

    def test_goal_occupied(self):
        """A* returns None when goal is occupied."""
        grid = self._make_simple_grid()
        grid.set_occupied(90, 90)
        planner = AStarPlanner()

        result = planner.search(grid, (10, 10), (90, 90))
        assert result is None

    def test_same_start_goal(self):
        """A* handles start == goal."""
        grid = self._make_simple_grid()
        planner = AStarPlanner()

        result = planner.search(grid, (50, 50), (50, 50))
        assert result is not None
        path, _ = result
        assert len(path) == 1
        assert path[0] == (50, 50)

    def test_cardinal_only(self):
        """A* with diagonal movement disabled."""
        grid = self._make_simple_grid()
        planner = AStarPlanner(diagonal_movement=False)

        result = planner.search(grid, (10, 10), (20, 20))
        assert result is not None
        path, _ = result

        # With cardinal-only, consecutive steps should differ by exactly 1 in row OR col
        for i in range(1, len(path)):
            dr = abs(path[i][0] - path[i-1][0])
            dc = abs(path[i][1] - path[i-1][1])
            assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1), \
                f"Non-cardinal step: {path[i-1]} → {path[i]}"


class TestRDPSimplification:
    """Test Ramer-Douglas-Peucker simplification."""

    def test_straight_line(self):
        """Straight line simplifies to 2 points."""
        planner = AStarPlanner(rdp_epsilon=0.1)
        points = [(float(i), 0.0) for i in range(100)]
        simplified = planner.rdp_simplify(points)
        assert len(simplified) == 2
        assert simplified[0] == (0.0, 0.0)
        assert simplified[-1] == (99.0, 0.0)

    def test_l_shape(self):
        """L-shaped path keeps the corner."""
        planner = AStarPlanner(rdp_epsilon=0.1)
        points = [(float(i), 0.0) for i in range(50)]
        points += [(49.0, float(i)) for i in range(1, 50)]
        simplified = planner.rdp_simplify(points)

        # Should keep start, corner, and end (3 points)
        assert len(simplified) >= 3
        # Corner should be approximately (49, 0)
        assert any(abs(p[0] - 49.0) < 1 and abs(p[1]) < 1 for p in simplified)

    def test_preserves_endpoints(self):
        """RDP always preserves start and end points."""
        planner = AStarPlanner(rdp_epsilon=1.0)
        points = [(0.0, 0.0), (5.0, 2.0), (10.0, 0.0)]
        simplified = planner.rdp_simplify(points)
        assert simplified[0] == (0.0, 0.0)
        assert simplified[-1] == (10.0, 0.0)

    def test_two_points(self):
        """Two-point input returns as-is."""
        planner = AStarPlanner()
        points = [(0.0, 0.0), (10.0, 10.0)]
        simplified = planner.rdp_simplify(points)
        assert len(simplified) == 2

    def test_epsilon_effect(self):
        """Larger epsilon = more simplification."""
        planner_tight = AStarPlanner(rdp_epsilon=0.01)
        planner_loose = AStarPlanner(rdp_epsilon=5.0)

        # Noisy path
        points = [(float(i), float(np.sin(i * 0.1) * 2)) for i in range(100)]

        tight = planner_tight.rdp_simplify(points)
        loose = planner_loose.rdp_simplify(points)

        assert len(tight) >= len(loose)


class TestCheckpointExtraction:
    """Test checkpoint generation."""

    def test_basic_checkpoints(self):
        """Basic checkpoints on a straight path."""
        planner = AStarPlanner(max_checkpoint_spacing=5.0)
        polyline = [(0, 0), (10, 0), (20, 0), (30, 0)]
        checkpoints = planner.extract_checkpoints(polyline)

        # Should have start and end at minimum
        assert len(checkpoints) >= 2
        assert checkpoints[0].position == (0, 0)
        assert checkpoints[-1].position == (30, 0)

    def test_turn_checkpoint(self):
        """Sharp turn generates a checkpoint."""
        planner = AStarPlanner(turn_angle_threshold=20.0, min_checkpoint_spacing=0.0)
        # 90-degree turn
        polyline = [(0, 0), (10, 0), (10, 10)]
        checkpoints = planner.extract_checkpoints(polyline)

        # Should have a turn checkpoint at (10, 0)
        turn_cps = [c for c in checkpoints if c.checkpoint_type == 'turn']
        assert len(turn_cps) >= 1

    def test_zone_transition_checkpoint(self):
        """Zone boundary crossing generates a checkpoint."""
        planner = AStarPlanner(min_checkpoint_spacing=0.0)

        # Create a behavior layer with a zone
        layer = BehaviorLayer()
        layer.add_zone(BehaviorZone(
            zone_id='grass1',
            zone_type='grass',
            polygon=[(5, -5), (15, -5), (15, 5), (5, 5)],
            action_rules=ActionRules(
                speed_modifier=0.5, announce=True,
                announcement='Grass ahead',
            ),
        ))

        # Path crosses into the zone
        polyline = [(0, 0), (3, 0), (6, 0), (10, 0), (16, 0), (20, 0)]
        checkpoints = planner.extract_checkpoints(polyline, layer)

        # Should have zone entry/exit checkpoints
        zone_cps = [c for c in checkpoints
                    if c.checkpoint_type in ('zone_entry', 'zone_exit')]
        assert len(zone_cps) >= 1

    def test_checkpoint_spacing(self):
        """Checkpoints respect min spacing."""
        planner = AStarPlanner(
            min_checkpoint_spacing=5.0,
            turn_angle_threshold=10.0,
        )
        # Many sharp turns close together
        polyline = [(i, (i % 2) * 2.0) for i in range(20)]
        checkpoints = planner.extract_checkpoints(polyline)

        # Check spacing between consecutive checkpoints
        for i in range(1, len(checkpoints) - 1):  # skip start/end
            p1 = checkpoints[i - 1].position
            p2 = checkpoints[i].position
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            # Allow some tolerance — min spacing is enforced but not exact
            # The last checkpoint (end) can violate spacing
            if i < len(checkpoints) - 1:
                assert dist >= planner.min_checkpoint_spacing * 0.9

    def test_checkpoint_serialization(self):
        """Checkpoint serializes to dict."""
        cp = Checkpoint(
            index=5,
            position=(10.0, 20.0),
            checkpoint_type='turn',
            turn_angle=90.0,
            zone_types=['grass'],
            announcements=['Turn right ahead'],
        )
        d = cp.to_dict()
        assert d['index'] == 5
        assert d['checkpoint_type'] == 'turn'
        assert d['turn_angle'] == 90.0


class TestHighLevelPlanning:
    """Test the plan() and plan_with_checkpoints() API."""

    def _make_test_grid(self):
        """Create a test grid with a gap to route through."""
        data = np.zeros((200, 200), dtype=np.int8)
        # Add some buildings
        data[80:120, 80:85] = 100  # vertical wall with gap
        data[80:120, 115:120] = 100
        return OccupancyGrid(data, resolution=0.1)

    def test_plan_returns_route(self):
        """plan() returns a Route object."""
        grid = self._make_test_grid()
        planner = AStarPlanner()

        route = planner.plan(grid, start=(2.0, 2.0), goal=(18.0, 18.0))
        assert route is not None
        assert isinstance(route, Route)
        assert route.total_distance > 0
        assert len(route.polyline) >= 2

    def test_plan_with_checkpoints(self):
        """plan_with_checkpoints() returns zone-aware checkpoints."""
        grid = self._make_test_grid()
        planner = AStarPlanner()

        layer = BehaviorLayer()
        layer.add_zone(BehaviorZone(
            zone_id='grass1',
            zone_type='grass',
            polygon=[(8, 8), (12, 8), (12, 12), (8, 12)],
            action_rules=ActionRules(speed_modifier=0.5, announce=True,
                                    announcement='Grass area'),
        ))

        route = planner.plan_with_checkpoints(
            grid, layer, start=(2.0, 2.0), goal=(18.0, 18.0)
        )
        assert route is not None
        assert route.num_checkpoints >= 2

    def test_plan_no_path(self):
        """plan() returns None when no path exists."""
        data = np.zeros((100, 100), dtype=np.int8)
        data[50, :] = 100  # complete wall
        grid = OccupancyGrid(data, resolution=0.1)
        planner = AStarPlanner()

        route = planner.plan(start=(2.0, 2.0), goal=(8.0, 8.0), grid=grid)
        assert route is None

    def test_route_dict(self):
        """Route serializes to dict."""
        grid = self._make_test_grid()
        planner = AStarPlanner()

        route = planner.plan(grid, start=(2.0, 2.0), goal=(18.0, 18.0))
        assert route is not None

        d = route.to_dict()
        assert 'polyline' in d
        assert 'checkpoints' in d
        assert 'total_distance' in d
        assert d['total_distance'] > 0

    def test_route_repr(self):
        """Route has a readable string representation."""
        grid = self._make_test_grid()
        planner = AStarPlanner()
        route = planner.plan(grid, start=(2.0, 2.0), goal=(18.0, 18.0))
        assert route is not None
        s = repr(route)
        assert 'dist=' in s


class TestPlannerConfig:
    """Test planner configuration."""

    def test_from_config(self):
        """Create planner from config file."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'map_config.yaml'
        )
        if os.path.exists(config_path):
            planner = AStarPlanner.from_config(config_path)
            assert planner.diagonal_movement is True
            assert planner.rdp_epsilon == 0.3
            assert planner.turn_angle_threshold == 30.0

    def test_repr(self):
        """Planner string representation."""
        planner = AStarPlanner()
        s = repr(planner)
        assert 'diagonal=' in s
