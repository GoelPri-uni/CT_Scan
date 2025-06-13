
## Run the file
## Ablation for sirt
python ablation_study_sirt_.py

## Overview

This project focuses on evaluating the **PDM-DART** algorithm for image reconstruction in cases with continuous and complex grey-level variations. We designed four distinct 2D **phantoms**—**Basic**, **Resolution**, **CT**, and **Filled**—to simulate medical imaging scenarios including soft tissue gradients, sharp edges, and overlapping anatomical structures. 

Each phantom is designed to test specific aspects of the reconstruction process:

- **Layered Phantom**: Tests segmentation accuracy for sharp edges and smooth grey-level transitions.
- **Resolution Phantom**: Adds small structures to test the algorithm’s ability to handle high-frequency detail.
- **CT Phantom**: Mimics real CT cross-sections of the human body with diverse anatomical shapes and intensity values.
- **Filled Phantom**: Contains overlapping and free-form shapes to challenge the method’s pixel classification under complexity.

All phantoms include Gaussian and/or Poisson noise to simulate realistic conditions.

---
