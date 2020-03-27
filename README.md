# Mask_RCNN
<!-- **Authors:** [Shi-Sheng Huang](https://cg.cs.tsinghua.edu.cn/people/~shisheng/) please contact with me via shishenghuang.net@gmail.com -->

This is a C++ wrapper Mask_RCNN, tested in Ubuntu 16.04


# 2. Prerequisites
We have tested the library in **16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## MaskRCNN
- (1) run `python_env.sh` to create a python3.5m virtual environment and install tensorflow 1.8 compatible with CUDA9.1 or CUDA10.1. **Note that you need first change to python3.5 and then install the 
virtualenv by using:**
```
sudo pip install virtualenv
```

- (2) download the mask_rcnn_coco.h5 model from this GitHub repository: `https://github.com/matterport/Mask_RCNN/releases`. And then put `mask_rcnn_coco.h5` to `${PROJECT_SOURCE_DIR}/Thirdparty/MaskRCNN/data/`


<!-- ## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin. -->


## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required OpenCV 3.1.1 or OpenCV 3.3.1 higher**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

<!-- ## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## OpenGL / GLUT (e.g. freeglut 2.8.0 or 3.0.0)
REQUIRED for the visualisation the library should run without available at http://freeglut.sourceforge.net/ -->

## CUDA (e.g. version 6.0 or 7.0 higher)
REQUIRED for all GPU accelerated code at least with cmake it is still possible to compile the CPU part without available at https://developer.nvidia.com/cuda-downloads


# 3. Compile

<!-- ## build the Thirdparty
```
cd Thirdparty
mkdir build
cd build
cmake ..
make -j8
```

## build the libDynaSeg.so and examples -->
```
mkdir build
cd build
cmake -DMASKRCNN_GPU=1 ..
make -j4
```

# 4. Usage
<!-- the executive file is generated in Examples/RGB-D/rgbd_tum_fusion, note the calib file: Examples/RGB-D/config/calib.txt is the calib file for the reconstruction as shown in InfiniTAM, you can change the file according to your dataset and sensor -->

## how to use
### 1. Run the example
```
cd bin
./run
```
<!-- the usage is:
```
Usage: ./runDynaSeg path_to_settings path_to_sequence path_to_association (see the source code for more details)
``` -->
<!-- 
### 2. Run the example

```
cd bin
./example [path_to_rgbd_files]
``` -->




