cmake_minimum_required(VERSION 3.20)

# Default compilers to Clang if not set by CLI or environment
if(NOT CMAKE_C_COMPILER)
	set(CMAKE_C_COMPILER clang)
endif()
if(NOT CMAKE_CXX_COMPILER)
	set(CMAKE_CXX_COMPILER clang++)
endif()

project(cpp_testing CXX C)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Automatically export compile_commands.json for VS Code IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Debug build if no build type is specified
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(glfw3 3.5 REQUIRED)

add_executable(${PROJECT_NAME}
src/main.cpp
src/gl.c
)

set(SHADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")
target_compile_definitions(${PROJECT_NAME} PRIVATE
	SHADERS_DIR="${SHADERS_DIR}"
	GRID_VERT_FILE="${SHADERS_DIR}/grid.vert"
	GRID_FRAG_FILE="${SHADERS_DIR}/grid.frag"
)

# Added ../shaders so #embed "shaders/..." works seamlessly across compilers & VS Code
target_include_directories(${PROJECT_NAME} PRIVATE 
	include
)

target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -Wextra -pedantic -Wno-c23-extensions>
	$<$<COMPILE_LANG_AND_ID:C,Clang>:-Wall -Wextra -pedantic>
)

target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
