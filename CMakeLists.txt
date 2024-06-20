cmake_minimum_required(VERSION 2.8)
project(yarvis)

# Add all source files in the src directory to the SRC_LIST variable
file(GLOB SRC_LIST src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_LIST})

# Find freenect and opencv, to set necessary variables
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories to get headers
include_directories(${freenect2_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Link libraries with the project
target_link_libraries(${PROJECT_NAME} ${freenect2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
