cmake_minimum_required(VERSION 2.8)
project(Mask_RCNN)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()



find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

# Call python for MaskRCNN
find_package(PythonLibs 3.5 EXACT REQUIRED)
message("PythonLibs Status: ")
message("=======")
message("Using python:  ${PYTHONLIBS_VERSION_STRING} ")
message("Using PYTHON_LIBRARIES:  ${PYTHON_LIBRARIES} ")
message("Using PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS} ")
message("=======")
# Copy file necessary for running

set(MASKRCNN_DIR ${CMAKE_SOURCE_DIR}/Thirdparty/MaskRCNN CACHE STRING "Location of MaskRCNN")
set(MASKRCNN_PYTHON_VE_PATH ${CMAKE_SOURCE_DIR}/python_env/ CACHE STRING "Python virtual environment to use")
set(MASKRCNN_MODEL_FILE "${MASKRCNN_DIR}/data/mask_rcnn_coco.h5" CACHE STRING "H5 MaskRCNN model weights")
set(MASKRCNN_GPU "0" CACHE STRING "Which GPU should MaskRCNN use")
configure_file("${MASKRCNN_DIR}/interface/MaskRCNN.py.in" "${CMAKE_SOURCE_DIR}/bin/MaskRCNN.py")
configure_file("${MASKRCNN_DIR}/interface/helpers.py" "${CMAKE_SOURCE_DIR}/bin/helpers.py")

#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules/FindGlog.cmake)
find_package(Eigen3 REQUIRED)
# find_package(Pangolin REQUIRED)
# FIND_PACKAGE(OpenGL REQUIRED)
# FIND_PACKAGE(GLUT REQUIRED)
#find_package(PCL REQUIRED)


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
# ${Pangolin_INCLUDE_DIRS}
# ${CUDA_INCLUDE_DIRS}
# ${GLUT_INCLUDE_DIR}
#/usr/include/glog
#${GLOG_INCLUDE_DIR}
# ${Boost_INCLUDE_DIRS}
# ${PROJECT_SOURCE_DIR}/Thirdparty/refusion/include
# ${PROJECT_SOURCE_DIR}/Thirdparty/SuperVoxel/
${PROJECT_SOURCE_DIR}/Thirdparty
/usr/include/python3.5m/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
# add_library(${PROJECT_NAME} SHARED
# src/System.cpp
# src/KeyFrame.cpp
# src/Map.cpp
# src/MapDrawer.cpp
# src/Frame.cpp
# src/Viewer.cpp
# src/Reconstruction.cpp
# src/ImageGenerator.cpp
# src/SemanticsUtil.cpp
# src/SemfileReader.cpp
# #src/Conversion.cpp
# include/misc.hpp
# )

# target_link_libraries(${PROJECT_NAME}
# ${OpenCV_LIBS}
# ${EIGEN3_LIBS}
# ${Pangolin_LIBRARIES}
# ${CUDA_LIBRARIES}
# #/usr/lib/x86_64-linux-gnu/libglog.so
# #${GLOG_LIBRARIES}
# libglut.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/refusion/lib/libeigen_wrapper.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/refusion/lib/librefusion.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/SuperVoxel/lib/libSuperVoxel.so
# ${PYTHON_LIBRARIES}
# /usr/local/lib/libpangolin.so
# )

# target_include_directories(${PROJECT_NAME} PUBLIC
#         ${PROJECT_SOURCE_DIR}/Thirdparty/refusion
#         ${PROJECT_SOURCE_DIR}/Thirdparty/SuperVoxel
#         ${PYTHON_INCLUDE_DIRS}
#         )

# Build examples
# ${GLUT_LIBRARY}

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(run
        app/main.cpp
        src/SemanticsUtil.cpp
)
target_link_libraries(run 
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PYTHON_LIBRARIES}
# ${GLUT_LIBRARY}
#  ${OPENGL_LIBRARY} ${CUDA_LIBRARIES}
)

# add_executable(refusion_test
#         app/example.cpp
#         app/fr_parser.cpp
# )
# target_link_libraries(refusion_test 
# ${PROJECT_NAME}
# ${GLUT_LIBRARY}
#  ${OPENGL_LIBRARY} ${CUDA_LIBRARIES} libglut.so /usr/local/lib/libpangolin.so
# )
