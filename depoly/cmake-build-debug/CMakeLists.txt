project(depoly)
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)


include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)
link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)

find_package(Torch REQUIRED PATHS "~/libtorch")

add_executable(depoly main.cpp)

target_link_libraries(depoly ${TORCH_LIBRARIES})



