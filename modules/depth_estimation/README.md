# Depth Estimation

This directory contains scripts and utilities for performing depth estimation using the **DepthAnything V2** model.

## Overview

* `depth_estimator.py`: Implements a `DepthEstimator` class that wraps model loading and inference for DepthAnything V2.
* `depth_test.py`: Example script demonstrating how to use `DepthEstimator` on webcam or Tello videostream.

## Setup Instructions

1. **Clone the DepthAnything V2 repository**

   Inside this `depth_estimation` folder, run:

   ```bash
   git clone https://github.com/DepthAnything/Depth-Anything-V2
   ```

   This will create a `Depth-Anything-V2` subdirectory containing the model code and utilities.

2. **Create folders for model artifacts**

   ```bash
   mkdir checkpoints engines
   ```

   * `checkpoints/`: Place your downloaded or trained model checkpoint files here.
   * `engines/`: Store optimized inference engines (e.g., TensorRT `.engine` files) here.

3. **Run the test script**

   ```bash
   python depth_test.py
   ```

   Adjust any paths or flags in `depth_test.py` as needed.

## Citation

For more information on DepthAnything V2 model:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```
