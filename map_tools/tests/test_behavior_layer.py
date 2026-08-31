"""Tests for behavior_layer.py"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from map_tools.behavior_layer import BehaviorLayer, BehaviorZone, ActionRules


class TestBehaviorZone:
    """Test individual zone operations."""

    def _make_square_zone(self, x=0, y=0, size=10, zone_type='grass'):
        """Helper: create a square zone."""
        return BehaviorZone(
            zone_id=f'test_{zone_type}',
            zone_type=zone_type,
            polygon=[
                (x, y), (x + size, y),
                (x + size, y + size), (x, y + size),
            ],
            action_rules=ActionRules(
                speed_modifier=0.5,
                announce=True,
                announcement=f'{zone_type} area',
            ),
        )

    def test_contains_point_inside(self):
        """Point inside square zone."""
        zone = self._make_square_zone(0, 0, 10)
        assert zone.contains_point(5, 5)

    def test_contains_point_outside(self):
        """Point outside square zone."""
        zone = self._make_square_zone(0, 0, 10)
        assert not zone.contains_point(15, 15)

    def test_contains_point_on_edge(self):
        """Point on zone boundary — implementation-dependent but shouldn't crash."""
        zone = self._make_square_zone(0, 0, 10)
        # Just verify it doesn't raise
        zone.contains_point(0, 5)
        zone.contains_point(10, 5)

    def test_contains_point_triangle(self):
        """Point inside a triangular zone."""
        zone = BehaviorZone(
            zone_id='triangle',
            zone_type='crosswalk',
            polygon=[(0, 0), (10, 0), (5, 10)],
        )
        assert zone.contains_point(5, 3)     # inside
        assert not zone.contains_point(0, 10)  # outside

    def test_distance_to_boundary(self):
        """Distance to zone boundary."""
        zone = self._make_square_zone(0, 0, 10)

        # Point at center: distance to nearest edge = 5
        dist = zone.distance_to_boundary(5, 5)
        assert dist == pytest.approx(5.0, abs=0.1)

        # Point on edge: distance ≈ 0
        dist = zone.distance_to_boundary(0, 5)
        assert dist == pytest.approx(0.0, abs=0.1)

    def test_centroid(self):
        """Zone centroid calculation."""
        zone = self._make_square_zone(0, 0, 10)
        cx, cy = zone.centroid()
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_serialization(self):
        """Zone serializes and deserializes correctly."""
        zone = self._make_square_zone(10, 20, 5, 'stairs')
        zone.metadata = {'name': 'Test Stairs'}

        d = zone.to_dict()
        restored = BehaviorZone.from_dict(d)

        assert restored.zone_id == zone.zone_id
        assert restored.zone_type == zone.zone_type
        assert len(restored.polygon) == len(zone.polygon)
        assert restored.action_rules.speed_modifier == zone.action_rules.speed_modifier
        assert restored.metadata['name'] == 'Test Stairs'

    def test_too_few_vertices(self):
        """Zone with <3 vertices can't contain any point."""
        zone = BehaviorZone(
            zone_id='line',
            zone_type='grass',
            polygon=[(0, 0), (10, 0)],
        )
        assert not zone.contains_point(5, 0)


class TestBehaviorLayer:
    """Test the behavior layer collection."""

    def _make_test_layer(self):
        """Create a test layer with multiple zones."""
        layer = BehaviorLayer()
        layer.add_zone(BehaviorZone(
            zone_id='grass1',
            zone_type='grass',
            polygon=[(0, 0), (20, 0), (20, 20), (0, 20)],
            action_rules=ActionRules(speed_modifier=0.5, announce=True,
                                    announcement='Grass area'),
            metadata={'name': 'Main Quad'},
        ))
        layer.add_zone(BehaviorZone(
            zone_id='stairs1',
            zone_type='stairs',
            polygon=[(10, 10), (15, 10), (15, 12), (10, 12)],
            action_rules=ActionRules(speed_modifier=0.0, announce=True,
                                    announcement='Stairs ahead',
                                    requires_confirmation=True),
            metadata={'name': 'Building Stairs'},
        ))
        layer.add_zone(BehaviorZone(
            zone_id='sidewalk1',
            zone_type='sidewalk',
            polygon=[(0, 25), (30, 25), (30, 28), (0, 28)],
            action_rules=ActionRules(speed_modifier=1.0),
            metadata={'name': 'Main Path'},
        ))
        return layer

    def test_add_and_get_zone(self):
        """Add zones and retrieve by ID."""
        layer = self._make_test_layer()
        assert len(layer.zones) == 3
        assert layer.get_zone('grass1') is not None
        assert layer.get_zone('nonexistent') is None

    def test_remove_zone(self):
        """Remove zone by ID."""
        layer = self._make_test_layer()
        assert layer.remove_zone('grass1')
        assert len(layer.zones) == 2
        assert not layer.remove_zone('grass1')  # already removed

    def test_query_point_single_zone(self):
        """Query point in a single zone."""
        layer = self._make_test_layer()
        zones = layer.query_point(5, 5)
        assert len(zones) == 1
        assert zones[0].zone_type == 'grass'

    def test_query_point_overlapping_zones(self):
        """Query point in overlapping zones (stairs inside grass)."""
        layer = self._make_test_layer()
        zones = layer.query_point(12, 11)
        zone_types = [z.zone_type for z in zones]
        assert 'grass' in zone_types
        assert 'stairs' in zone_types

    def test_query_point_no_zone(self):
        """Query point outside all zones."""
        layer = self._make_test_layer()
        zones = layer.query_point(50, 50)
        assert len(zones) == 0

    def test_query_actions_overlapping(self):
        """Combined actions at overlapping zones."""
        layer = self._make_test_layer()

        # Point in stairs (which is inside grass): most restrictive
        actions = layer.query_actions(12, 11)
        assert actions['speed_modifier'] == 0.0  # stairs = stop
        assert len(actions['announcements']) >= 1
        assert actions['requires_confirmation'] is True

    def test_query_actions_free_space(self):
        """Actions in free space."""
        layer = self._make_test_layer()
        actions = layer.query_actions(50, 50)
        assert actions['speed_modifier'] == 1.0
        assert len(actions['announcements']) == 0

    def test_zone_types(self):
        """List unique zone types."""
        layer = self._make_test_layer()
        types = layer.zone_types
        assert 'grass' in types
        assert 'stairs' in types
        assert 'sidewalk' in types

    def test_get_zones_by_type(self):
        """Filter zones by type."""
        layer = self._make_test_layer()
        grass_zones = layer.get_zones_by_type('grass')
        assert len(grass_zones) == 1

    def test_nearest_zone(self):
        """Find nearest zone to a point."""
        layer = self._make_test_layer()
        result = layer.nearest_zone(25, 5, zone_type='grass')
        assert result is not None
        zone, dist = result
        assert zone.zone_type == 'grass'
        assert dist > 0

    def test_destination_nodes(self):
        """Destination nodes for intent parsing."""
        layer = self._make_test_layer()
        nodes = layer.destination_nodes
        assert len(nodes) == 3
        names = [n['name'] for n in nodes]
        assert 'Main Quad' in names

    def test_save_load_roundtrip(self):
        """Layer survives save → load."""
        layer = self._make_test_layer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'zones.json')
            layer.save(path)

            loaded = BehaviorLayer.load(path)
            assert len(loaded.zones) == len(layer.zones)
            assert loaded.zones[0].zone_id == layer.zones[0].zone_id
            assert loaded.zones[1].action_rules.speed_modifier == 0.0

    def test_load_sample_zones(self):
        """Load the sample_zones.json data file."""
        zones_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'sample_zones.json'
        )
        if os.path.exists(zones_path):
            layer = BehaviorLayer.load(zones_path)
            assert len(layer.zones) > 0
            assert len(layer.zone_types) >= 3

    def test_stats(self):
        """Layer statistics."""
        layer = self._make_test_layer()
        stats = layer.get_stats()
        assert stats['total_zones'] == 3
        assert 'grass' in stats['zone_types']
