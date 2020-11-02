# /*
#  * @Author: Yuanwei Li, lywhbbj@126.com, TortoiseShell 
#  * @Date: 2020-11-02 15:09:43 
#  * @Last Modified by:   Yuanwei Li, lywhbbj@126.com, TortoiseShell 
#  * @Last Modified time: 2020-11-02 15:09:43 
#  */


cmake_minimum_required(VERSION 2.8.7)

# set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-std=c++11)

find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)
# find_package(Eigen3 REQUIRED)
message(STATUS ${OpenCV_LIBS})

include_directories(. ${OpenCV_INCLUDE_DIRS}
                        src/)

file(GLOB ANN_SRC  "ANN/*.cpp")

add_executable(fastGrtsOpticalFlow src/fastGrtsFlow.cpp ${ANN_SRC})
target_link_libraries(fastGrtsOpticalFlow ${OpenCV_LIBS})