# Project Pipeline

1. Input:
   - Multi-view images (real or simulated)
   - Camera poses (from SLAM / COLMAP)

2. Static Scene Reconstruction:
   - Use COLMAP or precomputed poses
   - Train static 3D Gaussian Splatting

3. Dynamic Object Handling:
   - Object masks / bounding boxes
   - Separate GS per object
   - Temporal consistency constraints

4. Rendering & Simulation:
   - Multi-view novel view synthesis
   - Autonomous driving simulation data generation
