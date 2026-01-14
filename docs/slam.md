# SLAM Module

## Purpose
Estimate temporally consistent camera trajectories for driving scenes.

## Input
- Image sequence (monocular or stereo)
- Camera intrinsics
- (Optional) Dynamic object masks

## Output
- Camera poses in TUM format:
  [timestamp, tx, ty, tz, qx, qy, qz, qw]

## Notes
- SLAM is treated as a black-box module.
- Primary role is to provide temporal consistency.
- Long-term drift is acceptable and may be refined by SfM.
