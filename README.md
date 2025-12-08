# Sperm_Cell_Detection_Feature_Fusion_attention_blocks

This project focuses on developing an automated system capable of detecting sperm cells in microscopic videos and tracking their movement over time. The final goal is to extract reliable motion data that can support clinical analysis and help identify potential fertility issues.

To detect sperm cells — which are extremely small, fast-moving, and visually similar objects — a lightweight detection model was developed based on the YOLOv5 architecture.

## Key Modifications

### Improved small-object detection layers
Added and adjusted layers responsible for processing high-resolution spatial features to improve sensitivity to very small targets.

### Enhanced feature fusion
Implemented a better fusion strategy between low-level spatial features and high-level semantic features, improving object localization and reducing false detections.

### Domain-specific preprocessing pipeline
During data collection, strong variations appeared between patients (normal vs. affected cases).
To address this:

A custom foreground–background discrimination pipeline was created.

It keeps only the sperm cells in the frame and removes the background noise.

This allowed the model to learn only the relevant semantic features, significantly improving generalization.





## Results 
### Video 1 
https://github.com/user-attachments/assets/f462f1d3-e5f5-4633-a2cf-a9dc7a8cba21
### Video 2
https://github.com/user-attachments/assets/1120e80b-6ecd-4bf7-be9d-e403c6748860

