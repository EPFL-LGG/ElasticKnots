################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.1)
project(ElasticKnots)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)  # eliminates 'add_library cannot create target' error, but gives The install of the tbb target requires changing an RPATH from the build tree, but this is not supported with the Ninja generator unless on an ELF-based or XCOFF-based platform

set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
    )

set(CMAKE_CXX_FLAGS_RELEASENATIVE        "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASENATIVE          "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE    "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELEASENATIVE
    CMAKE_C_FLAGS_RELEASENATIVE
    CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE
    CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE
    )

set(CMAKE_CXX_FLAGS_DEBUGNATIVE        "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_C_FLAGS_DEBUGNATIVE          "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE    "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUGNATIVE
    CMAKE_C_FLAGS_DEBUGNATIVE
    CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE
    )


if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Make sure warnings/errors are still colorized when using Ninja for building.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(-fdiagnostics-color=always)
endif()

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Common libraries
################################################################################
add_subdirectory(${THIRD_PARTY_DIR}/Tight-Inclusion)
add_subdirectory(${THIRD_PARTY_DIR}/ipc-toolkit)
add_library(tbb::tbb ALIAS tbb)  # ipc-toolkit uses TBB::tbb as tbb's alias, but elastic_rods uses tbb::tbb instead (if we do not define the right alias, the `tbb` target appears twice, generating an error)
add_subdirectory(${THIRD_PARTY_DIR}/ElasticRods)

################################################################################
# Elastic Knots Library
################################################################################
add_library(ElasticKnots SlidingProblem.cc)
target_link_libraries(ElasticKnots ElasticRods tight_inclusion)
target_link_libraries(ElasticKnots ipc::toolkit)
set_target_properties(ElasticKnots PROPERTIES CXX_STANDARD 14)
set_target_properties(ElasticKnots PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(ElasticKnots PUBLIC ${THIRD_PARTY_DIR})
target_include_directories(ElasticKnots PUBLIC ${THIRD_PARTY_DIR}/ElasticRods/3rdparty/MeshFEM/src/lib)

################################################################################
# Python Bindings
################################################################################
add_subdirectory(python_bindings)

################################################################################
# More binaries
################################################################################
# add_subdirectory(tests)  # TODO: write unit tests
