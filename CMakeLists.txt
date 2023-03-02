# cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_minimum_required(VERSION 2.8.12)

add_definitions(-std=c++11)

# set(CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(Autonomous_Driving)

# packages required
find_package(PCL 1.2 REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

# directories 
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

# MACRO(add_example name)
#   ADD_EXECUTABLE(${name} ${name}.cpp)
#   TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
# ENDMACRO()

# executable
add_executable (environment src/environment.cpp src/render/render.cpp src/processPointClouds.cpp)
target_link_libraries (environment ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(environment PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(environment PRIVATE ${OpenCV_LIBS})



