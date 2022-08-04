cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME DFT) 

##LANGUAGES CXX CUDA)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS -fopenmp)

set(DACE_INCLUDE_DIR /home/mans/.local/lib/python3.8/site-packages/dace/runtime/include)

file(GLOB_RECURSE SRC .dacecache/*map*/*cpp)

list(FILTER SRC EXCLUDE REGEX "sample")
list(FILTER SRC EXCLUDE REGEX "build")
#foreach(S ${SRC})
	
#endforeach()

message(${SRC})
add_executable(DFT ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
		   ${SRC}
    )

target_include_directories(DFT PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(DFT PRIVATE ${DACE_INCLUDE_DIR})
