cmake_minimum_required(VERSION 3.10)  # Use a more modern minimum version
project(fitpoints)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development) # Find Python and its development headers
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 not found")
endif()

# Find Eigen3
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
if (NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please make sure Eigen3 is installed and its pkg-config file is available.")
endif()
include_directories(${EIGEN3_INCLUDE_DIRS})

# Compiler flags (consider using target-specific flags instead of global)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive") # Avoid global flags, use target_compile_options
# Better approach for compiler flags (more robust and maintainable):
# target_compile_options(libransac PRIVATE -fpermissive)  # Only for libransac if needed

# Add pybind11 (as a subdirectory or find_package, subdirectory is simpler here)
add_subdirectory(pybind11)

# Source files
file(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/primitives/*.cpp")
file(GLOB MISC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/MiscLib/*.cpp")

# Create the library
add_library(libransac STATIC ${SRC} ${MISC_SRC})
set_target_properties(libransac PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Include directories for library:
target_include_directories(libransac
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}  # Use relative paths whenever possible
        ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
        ${Python3_INCLUDE_DIRS}       # Use the found Python include directories
)

# Create the Python module
# Don't explicitly set PYTHON_EXECUTABLE or PYBIND11_PYTHON_VERSION here.
# pybind11_add_module will use the Python3 found by find_package.
pybind11_add_module(fitpoints generate_points.cpp)
target_link_libraries(fitpoints PRIVATE libransac)
target_link_libraries(fitpoints PRIVATE Python3::Python)  # Link to the Python library

# Include directories for module:
target_include_directories(fitpoints
  PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
)
