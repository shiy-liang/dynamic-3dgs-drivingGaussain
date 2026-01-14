# Reconstruction Module

## Static Scene Reconstruction
- Use camera poses to initialize 3D Gaussian Splatting
- Dynamic regions are excluded during training

## Dynamic Object Modeling
- Dynamic objects are decomposed from the scene
- Each object can be represented by an independent Gaussian set

## Output
- Static background Gaussian model
- (Optional) Dynamic object Gaussian models

## Notes
- Focus on reconstruction stability rather than full SOTA performance
