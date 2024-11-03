cmake_minimum_required(VERSION 3.10)
project(kinect_bridge)

# Find all required packages
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(kinect_bridge src/kinect_bridge.cpp)

# Include directories for the Python module
target_include_directories(kinect_bridge PRIVATE
    ${freenect2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries with the Python module
target_link_libraries(kinect_bridge PRIVATE
    ${freenect2_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# Set C++14 standard
set_target_properties(kinect_bridge PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)