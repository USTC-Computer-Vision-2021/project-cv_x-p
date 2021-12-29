cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(project_name LookIntoPast)

project(${project_name})

set(OpenCV_DIR "D:/OpenCV/build/install")

find_package(OpenCV REQUIRED)

set(target_name lookintopast)

FILE (GLOB ALL_SOURCES "*.cpp")
FILE (GLOB ALL_INCLUDES "*.h")

SET (ALL_SRCS 
	 ${ALL_SOURCES}
	 ${ALL_INCLUDES}
	 ${ALL_UI}
)

include_directories("D:/Microsoft Visual Studio/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${target_name}  ${ALL_SRCS})
target_link_libraries(${target_name} ${OpenCV_LIBS})
