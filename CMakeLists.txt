cmake_minimum_required(VERSION 3.25)
project(PP)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(PP main.cpp Overage1.cpp Random1.cpp Random2.cpp)

target_link_libraries(PP PRIVATE Threads::Threads OpenMP::OpenMP_CXX pthread)

set_source_files_properties(Overage1.cpp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

