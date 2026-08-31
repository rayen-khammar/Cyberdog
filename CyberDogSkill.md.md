# Cyberdog Blind Navigation — PoC Development Skill
## Map-Grounded VAMOS + RDog + CE-RRT* | Zero-Hardware Proof of Concept

---

## 0. How to Use This File

- **Audience:** coding agent / developer starting the PoC.
- **Golden rule:** build every layer against **mocks first**, then swap in real modules one at a time. Never debug two untested layers at once.
- **Scope (PoC):** a closed-loop *"go to the library"* demo running in a **digital twin** (synthetic camera/LiDAR/pose), with the real VAMOS model in the loop. **No robot hardware required.**
- **Out of scope (later phases):** real Cyberdog locomotion wiring, real LiDAR rig, physical Smart Handle, blind-user trials.
- **Design principle:** *AI proposes, simple code disposes.* All safety-critical decisions are deterministic rules; all open-world understanding lives in AI modules.

---

## 1. Context

**Goal:** navigation system on a Xiaomi Cyberdog (Unitree Go2 kinematics) that guides blind students across campus — a robotic guide dog.

**Core papers:**
| Paper | Role in our architecture |
|---|---|
| **VAMOS** (arXiv:2510.20818) | The "brain": hierarchical VLA. VLM planner (PaliGemma 2 3B) outputs 5 candidate 2D paths; affordance MLP re-ranks by traversability. 90% real-robot success. |
| **RDog** (CHI 2024, 10.1145/3613904.3642227) | The "body interface + map": FAST-LIO 3D map + Behavior Layer (semantic zones), rigid Smart Handle (load cell + joystick), preemptive voice feedback. |
| **CrossTracer / NaviTrace** (arXiv:2608.06688 / 2510.26909) | The benchmark to beat (CrossTracer score 45.68). NaviTrace val split = fine-tuning signal + diagnostic. |

**Finalized architectural decisions:**
1. **Map decides WHERE** (3D map + Behavior Layer + A*), **VLM decides HOW** (VAMOS local paths), **rules decide WHEN TO STOP**.
2. **VAMOS requires a goal pixel pointer** → solved by the **Checkpoint Projector** (pinhole projection of a "carrot" 2–4 m ahead on the map route). Pure geometry, zero AI.
3. **One PaliGemma 2 3B backbone, multiple LoRA adapters:** `nav` (VAMOS paths), `goal` (instruction → endpoint pixel), `intent` (free-form voice → slots), `describe` (optional terrain captions).
4. **Affordance MLP** trained in Isaac Lab (elevation + position + heading → traversability prob); **confidence gate rejects < 0.5**.
5. **CE-RRT\*** (chance-constrained, P_coll < 0.01, warm-started) for dynamic obstacle avoidance.
6. **Task Planner = deterministic finite state machine.** Not an LLM. It calls AI modules as subroutines.
7. **ROS split:** VAMOS runs ROS 1 (Noetic) in a container; everything new runs ROS 2 (Humble); `ros1_bridge` connects them.

**VAMOS repo facts (`github.com/vamos-vla/vamos-deployment`):** inference/deployment ONLY (ROS 1 workspace, demo scripts, VLM server). **No training code.** We reuse its inference server + message definitions; we re-implement all training and all other layers.

---

## 2. Architecture Overview

```
User Interface (voice, Smart Handle mock, haptic mock)
        ↓
Ground Truth Map (3D mesh/.pcd → occupancy grid → Behavior Layer zones → A* route + checkpoints)
        ↓
Sensor Simulator (digital twin: virtual RGB @0.5m, virtual LiDAR, GT pose)   [real sensors later]
        ↓
Task Planner (FSM: intent parser → A* checkpoints → CHECKPOINT PROJECTOR → goal pixel pointer)
        ↓
Enhanced VAMOS Core (VLM planner: image+pixel+voice → 5 paths → Affordance MLP re-rank)
        ↓
Shared Control & Safety (handle override · confidence gate · CE-RRT*)
        ↓
Low-Level Execution (PoC: kinematic executor | later: Cyberdog native locomotion RHC 2Hz)
        ↓  + Preemptive Voice Engine (Behavior Layer triggers)
Cyberdog Executes → haptic/audio feedback loop
```

---

## 3. Layer-by-Layer Development (Step by Step)

### LAYER 1 — Ground Truth Map + Behavior Layer (no hardware)
**Purpose:** the deterministic "WHERE" source: localization prior, global routes, semantic zone triggers.

**Data sources (pick one):**
- Phone photogrammetry: walk campus filming → **Scaniverse / Polycam / Luma AI / RealityScan** → export mesh (.obj/.ply) or point cloud. *(Best: real campus.)*
- **Google Photorealistic 3D Tiles** (if city covered) → convert to mesh.
- Public stand-ins for pipeline testing: **NCLT** (campus LiDAR), **HM3D / Stanford 2D-3D-S** (indoor).

**Steps:**
1. Clean mesh (decimate, fill ground holes) in CloudCompare/MeshLab.
2. Convert to `.pcd`; project ground plane → rasterize → **2D occupancy grid** (`nav_msgs/OccupancyGrid`).
3. Annotate **Behavior Layer**: polygons with tags `{grass: slow+announce, stairs: STOP+announce, crosswalk, entrance_zone}` → save as JSON.
4. Implement **A\*** over occupancy grid; simplify polyline with Ramer–Douglas–Peucker → **checkpoints** (turn points double as voice-announcement triggers).

**PoC stub:** none needed — this layer is fully doable now.
**Acceptance:** occupancy grid + A* route + checkpoints render correctly in RViz for ≥3 route queries.
**Feeds:** Task Planner (L3).

### LAYER 2 — Sensor Simulator / Digital Twin + Auto-Labeler
**Purpose:** replace all hardware with synthetic streams; generate free training labels.

**Steps:**
1. Import mesh into **Isaac Sim** (preferred; already needed for affordance) or Habitat-sim/Blender.
2. Add virtual sensors: RGB camera at **0.5 m** with known intrinsics `K`; LiDAR raycaster (Mid-360 pattern); ground-truth pose publisher.
3. Drive a virtual agent along A* polylines; publish `/camera/image_raw`, `/lidar/points`, `/odom` as rosbags/live topics.
4. **Auto-Labeler (killer feature):** using GT pose + `K`, project route waypoints into each rendered frame → dataset of `(image, goal_pixel, path_pixels, preference_text)` — LoRA training data with zero teleoperation.
5. Add 2–3 synthetic walking pedestrians for CE-RRT* testing.

**PoC stub:** kinematic agent motion is fine (no physics needed here).
**Acceptance:** ≥200 rendered dog-height frames; projected pixel overlay lands on intended target in ≥9/10 checked frames.
**Feeds:** VAMOS (L4), Projector (L3), CE-RRT* (L6), training pipeline.

### LAYER 3 — Task Planner + Checkpoint Projector (deterministic FSM)
**Purpose:** orchestration + the goal-pixel pointer VAMOS requires. **~90% of this is simple code.**

**States:** `IDLE → GLOBAL_NAV → ALIGN → LOCAL_NAV → ARRIVED` + interrupts `EMERGENCY_STOP`, `STAIRS_STOP`, `HOLD`.
**Sub-modules:**
1. **Intent parser:** lookup/fuzzy match of destination against Behavior Layer node list. (PaliGemma `intent` adapter optional later.)
2. **Route consumer:** reads A* polyline + checkpoints from L1.
3. **Checkpoint Projector:** sample carrot 2–4 m ahead; project:
   `P_cam = T_body→cam⁻¹ · T_map→body⁻¹ · P_map` ; `(u,v) = project(K, P_cam)`;
   valid if depth > 0 and inside FOV margin; else enter `ALIGN` (rotate until visible).
   **Fallback:** PaliGemma `goal` adapter for unmapped targets ("find a bench").
4. **Preference generator:** emits language steering from next checkpoint (e.g. *"turn right after the door, stay on dry pavement"* — include standing preferences: avoid puddles, stay on sidewalk).

**Interfaces out:** `/vamos/goal_pixel` @1–2 Hz, `/voice/feedback` triggers, `/route/polyline`.
**PoC stub:** voice input = text file/keyboard; handle = keyboard keys.
**Acceptance:** with replayed L2 streams, projector emits a smoothly moving in-frame pixel for a full route; FSM logs show correct transitions incl. ALIGN at a blind corner.
**Feeds:** VAMOS (L4).

### LAYER 4 — Enhanced VAMOS Core (repo reuse + LoRA)
**Purpose:** semantic local planner ("HOW").

**Steps:**
1. Clone `vamos-deployment`; audit `vamos_ws/src/` message schemas; run `server/` + `vamos_demo.py` on a lab RTX 4090 (ROS 1 Noetic container).
2. Set up `ros1_bridge`: `/camera/image_raw` ROS2→ROS1; `/vamos/candidate_paths` ROS1→ROS2.
3. Wire projector output into the VLM prompt: **(image + goal pixel + preference)** → receive **5 candidate 2D paths**.
4. **Zero-shot test first** on L2 rendered frames (pass criterion: ≥1 path stays on pavement toward pointer).
5. **LoRA fine-tune** (reimplement; repo has no training code): config per paper — targets `q,k,v,o,gate,up,down_proj`, **r=16, α=16, dropout=0.05**, start from HF checkpoint; data = L2 auto-labeled pairs + NaviTrace val (legged-robot subset) + rainy/puddle scenes.
6. Add `goal` adapter (instruction → endpoint pixel) as second LoRA on same backbone.

**PoC stub:** zero-shot checkpoint is acceptable for first closed-loop test; fine-tune is the upgrade.
**Acceptance:** end-to-end image+pixel → 5 paths at ≥0.5 Hz; paths track the moving carrot.
**Feeds:** Affordance (L5).

### LAYER 5 — Affordance Module (Isaac Lab)
**Purpose:** physics filter — knows stairs/wet/rough are not walkable even if the VLM likes them.

**Steps:**
1. Isaac Lab + Unitree Go2 URDF; procedural terrains (stairs, ramps, mounds, wet-flag patches).
2. Roll out **native locomotion policy**; log (local elevation map, position, heading, success/fail).
3. Train **MLP** from scratch, loss = binary cross-entropy, output traversability ∈ [0,1].
4. Runtime node: build local elevation from `/lidar/points`; score the 5 VAMOS candidates; **re-rank**; publish `/vamos/safe_path`; confidence gate rejects < 0.5.

**PoC stub:** elevation-heuristic scorer (flat=go, step>15cm=no-go) to unblock integration; swap trained MLP after.
**Acceptance:** held-out sim terrain accuracy > 90%; a stair-crossing candidate is never the top-ranked path.
**Feeds:** Safety (L6).

### LAYER 6 — Shared Control & Safety
**Purpose:** the differentiator. Human override + probabilistic dynamic avoidance.

**Sub-modules:**
1. **Handle mock → later ESP32:** `/handle/force` @50 Hz; rules: slight pull = decel, slight push = accel, **tug = EMERGENCY STOP** (hard override of everything).
2. **CE-RRT\* planner:** reference = `/vamos/safe_path`; obstacles = live `/lidar/points` (+ predicted pedestrian motion); chance constraint P_coll < 0.01; **warm-start** from previous tree; outputs `/cmd_vel_safe` @10–20 Hz.
3. **Safety mux:** merges CE-RRT* velocity with handle override + confidence gate + LiDAR hard-stop bubble → `/cmd_vel_final`.

**PoC stub:** plain RRT* or DWA first; upgrade to chance-constrained version second.
**Acceptance:** avoids a synthetic pedestrian without full stops; tug → zero velocity in < 100 ms; stair zone → STOP + voice.
**Feeds:** Execution (L7).

### LAYER 7 — Execution + Feedback (PoC: kinematic)
**Purpose:** close the loop; announce terrain.

**Steps:**
1. **Kinematic executor (PoC):** integrate `/cmd_vel_final` to move the virtual agent in the twin (this updates `/odom` and closes the loop). *Later phase replaces this with Cyberdog native locomotion (RHC 2 Hz) via ROS 2 driver.*
2. **Preemptive Voice Engine:** pose-in-zone checks against Behavior Layer → TTS strings ("grass ahead", "turning right", "mind the step — press button to continue").
3. **Haptic mock:** vibration command on turn/stop (keyboard LED / log now; vest/handle motor later).

**Acceptance:** full loop stable for a complete route; voice triggers fire at correct zones.

---

## 4. Connection Plan (ROS Graph & Interfaces)

| Topic | Type | Rate | Producer → Consumer |
|---|---|---|---|
| `/voice/command` | String | event | UI mock → task_planner |
| `/handle/force`, `/handle/joystick` | Float32 / Joy | 50 Hz | handle_mock → safety_mux / task_planner |
| `/map/occupancy`, `/map/zones` | OccupancyGrid / JSON | static | map_tools → task_planner, voice_engine |
| `/odom` | Odometry | 50 Hz | sim (later FAST-LIO) → projector, safety, planner |
| `/camera/image_raw` + `/camera/info` | Image / CameraInfo | 10–30 Hz | sim → VAMOS (bridge), projector debug |
| `/lidar/points` | PointCloud2 | 10 Hz | sim → elevation builder, ce_rrt |
| `/route/polyline` | Path | 1 Hz | task_planner → projector |
| `/vamos/goal_pixel` | PointStamped | 1–2 Hz | projector → VAMOS server (bridge) |
| `/vamos/candidate_paths` | PathList | 0.5–1 Hz | VAMOS (ROS1) → affordance (bridge) |
| `/vamos/safe_path` | Path | 0.5–1 Hz | affordance → ce_rrt |
| `/cmd_vel_safe` → `/cmd_vel_final` | Twist | 10–20 Hz | ce_rrt → safety_mux → executor |
| `/voice/feedback`, `/haptic/cmd` | String / UInt8 | event | voice_engine/safety → UI |

**Bridge:** `ros1_bridge` limited to the two VAMOS crossings above (keep bridge surface minimal).
**Integration order:** L1 → L2 → L3 → L4 → L5 → L6 → L7, testing each new node against the previous layer's *recorded* topics before going live.

---

## 5. Six-Week PoC Schedule & Exit Gates

| Week | Work | Exit gate |
|---|---|---|
| 1 | L1 map pipeline + L2 twin setup | Route + checkpoints in RViz; 200 rendered frames |
| 2 | L3 planner/projector + L4 VAMOS zero-shot wiring | Moving pixel pointer; VAMOS returns 5 paths on twin |
| 3 | L4 LoRA (auto-labeled data) + L5 affordance stub→MLP | Fine-tuned goal pixel error ↓; stair candidate never wins |
| 4 | L6 safety + CE-RRT* (+pedestrians) | Smooth avoidance; tug < 100 ms |
| 5 | L7 closed loop + voice; full-route runs | ≥80% route success over 10 runs; latency budget measured |
| 6 | Eval report + optional NaviTrace diagnostic | **Go/No-Go** for hardware phase |

**Latency budget to measure:** VAMOS ≤ 2–3 s/plan (known), projector 1–2 Hz, CE-RRT* ≥ 10 Hz, safety mux 50 Hz, handle stop < 100 ms.

---

## 6. Reuse vs Build

| From VAMOS repo (reuse) | We build (not in repo) |
|---|---|
| VLM inference server, ROS1 ws, demo scripts, msg defs | LoRA training (+ adapters), affordance training, task planner, projector, map pipeline, sensor twin, safety/CE-RRT*, handle/voice/haptic mocks, ros1_bridge config |

**Suggested repo layout:** `cyberdog-poc/{map_tools, sim_twin, task_planner, vamos_bridge, affordance, safety, execution, docs}`.

---

## 7. Definition of Done (PoC)

- [ ] Closed-loop *"go to the library"* completes in the digital twin with zero hardware.
- [ ] Every layer's failure is individually logged and interpretable (which layer rejected what).
- [ ] Latency + success metrics recorded; go/no-go report written.
- [ ] **Only then** buy hardware: Jetson Orin NX 16 GB, Livox Mid-360 (~$800), USB cam, ESP32 handle (~$40) — the brain rig phase.