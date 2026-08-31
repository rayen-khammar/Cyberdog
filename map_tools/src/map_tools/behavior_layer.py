"""
behavior_layer.py — Semantic Zone Annotations + Queries

Manages Behavior Layer zones: polygonal regions with semantic tags
(grass, stairs, crosswalk, etc.) and associated action rules.

From the CyberDog Skill spec:
  Annotate Behavior Layer: polygons with tags
  {grass: slow+announce, stairs: STOP+announce, crosswalk, entrance_zone}

Usage:
    bl = BehaviorLayer.load('data/sample_zones.json')
    zones = bl.query_point(x=15.0, y=22.0)
    for zone in zones:
        print(f"{zone.zone_type}: {zone.action_rules}")
"""

import json
import os
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class ActionRules:
    """Action rules triggered when entering a behavior zone."""
    speed_modifier: float = 1.0       # 1.0 = full speed, 0.0 = stop
    announce: bool = False            # whether to trigger voice announcement
    announcement: str = ""            # TTS text
    requires_confirmation: bool = False  # whether user must confirm to proceed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'speed_modifier': self.speed_modifier,
            'announce': self.announce,
            'announcement': self.announcement,
            'requires_confirmation': self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ActionRules':
        return cls(
            speed_modifier=d.get('speed_modifier', 1.0),
            announce=d.get('announce', False),
            announcement=d.get('announcement', ''),
            requires_confirmation=d.get('requires_confirmation', False),
        )


@dataclass
class BehaviorZone:
    """A semantic zone with polygon boundary and action rules.
    
    Attributes:
        zone_id: Unique identifier for the zone.
        zone_type: Semantic type (grass, stairs, crosswalk, etc.).
        polygon: List of (x, y) vertices defining the zone boundary.
        action_rules: Rules triggered when entering this zone.
        metadata: Optional extra data (e.g., floor level, building name).
    """
    zone_id: str
    zone_type: str
    polygon: List[Tuple[float, float]]
    action_rules: ActionRules = field(default_factory=ActionRules)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a world point (x, y) is inside this zone's polygon.
        
        Uses the ray-casting algorithm for point-in-polygon test.
        
        Args:
            x: World X coordinate.
            y: World Y coordinate.
        
        Returns:
            True if the point is inside the polygon.
        """
        n = len(self.polygon)
        if n < 3:
            return False

        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Compute minimum distance from point to the polygon boundary.
        
        Args:
            x: World X coordinate.
            y: World Y coordinate.
        
        Returns:
            Distance in meters to nearest polygon edge.
        """
        min_dist = float('inf')
        n = len(self.polygon)
        point = np.array([x, y])

        for i in range(n):
            a = np.array(self.polygon[i])
            b = np.array(self.polygon[(i + 1) % n])

            # Project point onto line segment
            ab = b - a
            ap = point - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10), 0.0, 1.0)
            closest = a + t * ab
            dist = np.linalg.norm(point - closest)
            min_dist = min(min_dist, dist)

        return float(min_dist)

    def centroid(self) -> Tuple[float, float]:
        """Compute the centroid of the polygon."""
        pts = np.array(self.polygon)
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize zone to dict."""
        d = {
            'zone_id': self.zone_id,
            'zone_type': self.zone_type,
            'polygon': [list(p) for p in self.polygon],
            'action_rules': self.action_rules.to_dict(),
        }
        if self.metadata:
            d['metadata'] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BehaviorZone':
        """Deserialize zone from dict."""
        return cls(
            zone_id=d['zone_id'],
            zone_type=d['zone_type'],
            polygon=[tuple(p) for p in d['polygon']],
            action_rules=ActionRules.from_dict(d.get('action_rules', {})),
            metadata=d.get('metadata', {}),
        )


class BehaviorLayer:
    """Manages a collection of semantic zones for navigation behavior control.
    
    Provides spatial queries to determine which zones contain a given point,
    and what actions should be triggered.
    """

    def __init__(self, zones: Optional[List[BehaviorZone]] = None):
        """
        Args:
            zones: Initial list of behavior zones.
        """
        self.zones: List[BehaviorZone] = zones or []
        self._zone_index: Dict[str, BehaviorZone] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the zone ID lookup index."""
        self._zone_index = {z.zone_id: z for z in self.zones}

    # ──────────────────────────────────────────────
    #  Zone management
    # ──────────────────────────────────────────────

    def add_zone(self, zone: BehaviorZone) -> None:
        """Add a zone to the layer."""
        self.zones.append(zone)
        self._zone_index[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID. Returns True if found and removed."""
        if zone_id in self._zone_index:
            zone = self._zone_index.pop(zone_id)
            self.zones.remove(zone)
            return True
        return False

    def get_zone(self, zone_id: str) -> Optional[BehaviorZone]:
        """Get a zone by ID."""
        return self._zone_index.get(zone_id)

    def get_zones_by_type(self, zone_type: str) -> List[BehaviorZone]:
        """Get all zones of a given type."""
        return [z for z in self.zones if z.zone_type == zone_type]

    @property
    def zone_types(self) -> List[str]:
        """List of unique zone types present."""
        return list(set(z.zone_type for z in self.zones))

    @property
    def destination_nodes(self) -> List[Dict[str, Any]]:
        """List of named destinations for intent parsing.
        
        Returns zones that have a 'name' in their metadata,
        formatted for use by the Task Planner's intent parser.
        """
        nodes = []
        for z in self.zones:
            name = z.metadata.get('name', z.zone_id)
            cx, cy = z.centroid()
            nodes.append({
                'zone_id': z.zone_id,
                'name': name,
                'zone_type': z.zone_type,
                'centroid': (cx, cy),
            })
        return nodes

    # ──────────────────────────────────────────────
    #  Spatial queries
    # ──────────────────────────────────────────────

    def query_point(self, x: float, y: float) -> List[BehaviorZone]:
        """Find all zones containing a world point.
        
        Args:
            x: World X coordinate.
            y: World Y coordinate.
        
        Returns:
            List of BehaviorZone objects containing the point.
        """
        return [z for z in self.zones if z.contains_point(x, y)]

    def query_actions(self, x: float, y: float) -> Dict[str, Any]:
        """Get the combined action rules at a world point.
        
        When multiple zones overlap, the most restrictive rules apply:
        - Speed: minimum of all zone speeds
        - Announce: any zone that announces triggers announcement
        - Confirmation: any zone requiring confirmation triggers it
        
        Args:
            x: World X coordinate.
            y: World Y coordinate.
        
        Returns:
            Combined action dict with keys: speed_modifier, announcements, 
            requires_confirmation, zone_types.
        """
        zones = self.query_point(x, y)
        
        if not zones:
            return {
                'speed_modifier': 1.0,
                'announcements': [],
                'requires_confirmation': False,
                'zone_types': [],
            }

        speed = min(z.action_rules.speed_modifier for z in zones)
        announcements = [
            z.action_rules.announcement
            for z in zones
            if z.action_rules.announce and z.action_rules.announcement
        ]
        requires_conf = any(z.action_rules.requires_confirmation for z in zones)
        zone_types = [z.zone_type for z in zones]

        return {
            'speed_modifier': speed,
            'announcements': announcements,
            'requires_confirmation': requires_conf,
            'zone_types': zone_types,
        }

    def nearest_zone(
        self, x: float, y: float, zone_type: Optional[str] = None
    ) -> Optional[Tuple[BehaviorZone, float]]:
        """Find the nearest zone boundary to a point.
        
        Args:
            x: World X coordinate.
            y: World Y coordinate.
            zone_type: If provided, only consider zones of this type.
        
        Returns:
            (zone, distance) tuple, or None if no zones exist.
        """
        candidates = self.zones
        if zone_type:
            candidates = [z for z in candidates if z.zone_type == zone_type]

        if not candidates:
            return None

        best_zone = None
        best_dist = float('inf')
        for z in candidates:
            d = z.distance_to_boundary(x, y)
            if d < best_dist:
                best_dist = d
                best_zone = z

        return (best_zone, best_dist)

    # ──────────────────────────────────────────────
    #  I/O
    # ──────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Save zones to a JSON file.
        
        Args:
            path: Output file path.
        
        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'version': '1.0',
            'zones': [z.to_dict() for z in self.zones],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> 'BehaviorLayer':
        """Load zones from a JSON file.
        
        Args:
            path: Path to zones JSON file.
        
        Returns:
            BehaviorLayer with loaded zones.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        zones = [BehaviorZone.from_dict(d) for d in data.get('zones', [])]
        return cls(zones=zones)

    @classmethod
    def from_config(
        cls, config_path: str, zones_path: Optional[str] = None
    ) -> 'BehaviorLayer':
        """Create a BehaviorLayer with action rules from config.
        
        If zones_path is provided, loads zones and applies config defaults.
        Otherwise returns empty layer with config loaded.
        
        Args:
            config_path: Path to map_config.yaml.
            zones_path: Optional path to zones JSON file.
        
        Returns:
            BehaviorLayer instance.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        zone_actions = config.get('behavior_layer', {}).get('zone_actions', {})

        if zones_path:
            layer = cls.load(zones_path)
            # Apply config defaults to zones that don't have explicit rules
            for zone in layer.zones:
                if zone.zone_type in zone_actions:
                    defaults = zone_actions[zone.zone_type]
                    rules = zone.action_rules
                    if rules.speed_modifier == 1.0 and 'speed_modifier' in defaults:
                        rules.speed_modifier = defaults['speed_modifier']
                    if not rules.announce and defaults.get('announce', False):
                        rules.announce = True
                        rules.announcement = defaults.get('announcement', '')
                    if not rules.requires_confirmation and defaults.get('requires_confirmation', False):
                        rules.requires_confirmation = True
            return layer

        return cls()

    # ──────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        type_counts = {}
        for z in self.zones:
            type_counts[z.zone_type] = type_counts.get(z.zone_type, 0) + 1
        return {
            'total_zones': len(self.zones),
            'zone_types': type_counts,
            'destination_count': len(self.destination_nodes),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"BehaviorLayer({stats['total_zones']} zones: {stats['zone_types']})"
