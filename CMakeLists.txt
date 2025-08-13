cmake_minimum_required(VERSION 3.0)
project(ECBFA)

# setting C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# output directory build/Debug or build/Release
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")

# setting OpenCV path
set(OpenCV_DIR "D:/opencv-4.7.0/opencv/build/x64/vc16/lib")

# source file
set(SOURCES
    main.cpp
    utili.cpp
)

# header file
set(HEADERS
    utili.h
)

# find opencv
find_package(OpenCV REQUIRED)

# 
add_executable(AllBuild ${SOURCES} ${HEADERS})

# 
target_link_libraries(AllBuild ${OpenCV_LIBS})

# 
target_include_directories(AllBuild PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT AllBuild)

# 
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")