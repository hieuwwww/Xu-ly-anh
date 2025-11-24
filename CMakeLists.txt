cmake_minimum_required(VERSION 3.10)
project(xulyanh)

set(CMAKE_CXX_STANDARD 17)

# ÉP DÙNG MINGW64 (có Qt, nhưng chạy ổn định)
set(OpenCV_DIR "C:/msys64/mingw64/lib/cmake/opencv4")
set(CMAKE_PREFIX_PATH "C:/msys64/mingw64")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(xulyanh ../main.cpp)
target_link_libraries(xulyanh ${OpenCV_LIBS})