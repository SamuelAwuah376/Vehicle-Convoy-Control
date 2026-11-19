"""Perception and the team belief (docs/perception_design.md).

Off by default (Scenario.perception.enabled False): the planner, the intentions and the policy then read the world's
true snapshot exactly as before. On, a sensor suite on each team car (Tier A: nohesi.perception.sim; Tier B: CARLA)
feeds per-sensor processing (nohesi.perception.processing) and one centralized IMM-UKF tracker
(nohesi.perception.tracking) whose Belief replaces the true traffic state for everything that plans or observes.
Collisions, scoring and the traffic's own behaviour stay on the true state.

Shared contract (every module codes against these):
- config.py: sensor specs, tracker settings, the Perception switch and the default suite (design section 4).
- types.py: raw sensor outputs (LidarScan, RadarScan, CameraFrame), EgoState, Detection (tracker input), Belief
  (tracker output) and TruthHistory (the true state interpolated to any recent time, for the simulators).
- geometry.py: frames (world = road frame on the straight road, car frame at the CG, sensor frames), sensor poses,
  car boxes, the pinhole camera model and ray casting against car boxes.
"""
