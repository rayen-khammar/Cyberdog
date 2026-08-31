# map_tools — Ground Truth Map + Behavior Layer

**Layer 1** of the CyberDog Blind Navigation PoC.

## Purpose

The deterministic **"WHERE"** source: localization prior, global routes, semantic zone triggers.
No hardware required. No AI. Pure geometry + data structures.

## Components

| Module | Description |
|---|---|
| `occupancy_grid.py` | Load mesh/point cloud → 2D occupancy grid (ROS map_server compatible) |
| `behavior_layer.py` | Semantic zone annotations (grass, stairs, crosswalk) with point-in-polygon queries |
| `astar_planner.py` | A* pathfinding + Ramer–Douglas–Peucker simplification → checkpoints |

## Architecture Reference

From the CyberDog Skill spec:
- **Map decides WHERE** (3D map + Behavior Layer + A*)
- Output feeds → Task Planner (Layer 3)
- Acceptance: occupancy grid + A* route + checkpoints render correctly for ≥3 route queries

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a synthetic sample map
python scripts/generate_sample_map.py

# Visualize the map + zones + route
python scripts/visualize_map.py

# Run tests
python -m pytest tests/ -v
```

## Data Formats

- **Occupancy Grid**: `.pgm` + `.yaml` (ROS `map_server` format)
- **Behavior Zones**: JSON with polygon vertices + semantic tags
- **Route Output**: list of `(x, y)` waypoints + checkpoint metadata

## Directory Structure

```
map_tools/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── map_config.yaml
├── data/
│   └── sample_zones.json
├── scripts/
│   ├── generate_sample_map.py
│   └── visualize_map.py
├── src/
│   └── map_tools/
│       ├── __init__.py
│       ├── occupancy_grid.py
│       ├── behavior_layer.py
│       └── astar_planner.py
└── tests/
    ├── test_occupancy_grid.py
    ├── test_behavior_layer.py
    └── test_astar.py
```
