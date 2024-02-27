project(helloworld)
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)
link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)

find_package(Torch REQUIRED PATHS "/usr/local/lib/libtorch")
target_link_libraries(digit ${TORCH_LIBRARIES})

add_executable(main main.cpp)

