# **EV-TTC: Event-Based Time-To-Collision Estimation**

## **Overview**
This repository contains a ROS 2 package for event-based Time-To-Collision (TTC) estimation using event camera data. The system processes event streams through multi-scale temporal filters and uses a TensorRT-based neural network to estimate optical flow and TTC.

---

## **Features**
- **Event Camera Integration:** Processes event streams using the `event_camera_msgs` and `event_camera_codecs` libraries.
- **Multi-Scale Temporal Filtering:** Applies exponential filters to event data for temporal feature extraction.
- **Neural Network Inference:** Uses TensorRT for high-performance inference on GPU.
- **Visualization:** Publishes filtered event data as images for debugging and analysis.
---

## **Repository Structure**
```
ev_ttc/
├── CMakeLists.txt          # Build configuration
├── package.xml             # ROS 2 package metadata
├── launch/
│   └── params.yaml         # Default parameters for the node
├── src/
│   ├── exp.cpp             # Main implementation of the EV-TTC node
│   ├── exp_node.cpp        # Entry point for the EV-TTC node
├── include/
│   ├── ev_ttc/
│   │   ├── config.h        # Configuration constants and parameter loading
│   │   ├── exp.h           # EV-TTC node header
│   │   ├── ev_processor.h  # Event processing and filtering logic
├── dep.repos               # Dependencies for the repository
```

---

## **Dependencies**
This package depends on several ROS 2 and third-party libraries. The required dependencies are listed in the dep.repos file. Please note, we asssume most of these system level dependencies are installed via Jetpack 6. We assume operation on at Jetson Orin NX 16GB.

### **Key Dependencies**
- **Event Camera Libraries:**
  - `event_camera_msgs`
  - `event_camera_codecs`
  - `metavision_driver`
- **TensorRT:** For GPU-based neural network inference.
- **CUDA:** Required for TensorRT and GPU operations.

---
## **Creating the TensorRT Engine**
```
trtexec --onnx=ttc.onnx \
        --saveEngine=ttc.engine \
        --exportProfile=ttc.json \
        --inputIOFormats=fp16:chw \
        --outputIOFormats=fp16:chw \
        --fp16 \
        --useSpinWait \
        --separateProfileRun \
        --profilingVerbosity=detailed \
        --best \
        --iterations=10
```

## **Installation**

### **1. Clone the Repository**
Clone this repository into your ROS 2 workspace:

### **2. Import Dependencies**
Use the dep.repos file to fetch the required dependencies:
```bash
vcs import < ev_ttc/dep.repos
```


### **3. Build the Workspace**
Build the workspace using `colcon`:
```bash
cd ~/ros2_ws
colcon build
```

---

## **Usage**

### **Launch File**
Run the EV-TTC node using the provided launch file:
```bash
ros2 launch ev_ttc exp_launch.py
```

### **Launch the Node**
You can pass a custom parameter file to the node:
```bash
ros2 run ev_ttc exp_node --ros-args --params-file src/ev_ttc/launch/params.yaml
```

---

## **Configuration**

### **Parameters**
The node supports the following parameters, which can be configured in the params.yaml file:

| Parameter         | Type    | Description                                      | Default Value                     |
|--------------------|---------|--------------------------------------------------|-----------------------------------|
| `camera.cx`       | `float` | Principal point x-coordinate                     | `602.498762579664`               |
| `camera.cy`       | `float` | Principal point y-coordinate                     | `360.850800481907`               |
| `camera.fx`       | `float` | Focal length in x                                | `1021.66382499730`               |
| `camera.fy`       | `float` | Focal length in y                                | `1017.70617933675`               |
| `camera.k1`       | `float` | Radial distortion coefficient k1                 | `0.0`                            |
| `camera.k2`       | `float` | Radial distortion coefficient k2                 | `0.0`                            |
| `camera.p1`       | `float` | Tangential distortion coefficient p1             | `0.0`                            |
| `camera.p2`       | `float` | Tangential distortion coefficient p2             | `0.0`                            |
| `event_topic`     | `string`| Topic for event camera data                      | `"/event_camera/events"`         |
| `profile`         | `bool`  | Enable profiling                                 | `true`                           |
| `imgs`            | `bool`  | Enable image output                              | `true`                           |
| `disp_num`        | `int`   | Display number for debugging                     | `5`                              |
| `alphas`          | `list`  | Alpha values for temporal filters                | `[0.1, 0.05, 0.025, ...]`        |
| `engine_path`     | `string`| Path to the TensorRT inference engine            | `"/home/anthony/rss/.../ttc.engine"` |

---


### **Code Structure**
- **`exp.cpp`:** Implements the main EV-TTC node, including ROS 2 communication and TensorRT inference.
- **`ev_processor.h`:** Handles event processing, filtering, and visualization.
- **`config.h`:** Defines configuration constants and parameter loading logic.


## **Acknowledgments**
- TensorRT integration inspired by [TensorRT C++ API](https://github.com/cyrusbehr/tensorrt-cpp-api).
- Event camera libraries from the [ROS Event Camera](https://github.com/ros-event-camera) project from Bernd Pfrommer

---