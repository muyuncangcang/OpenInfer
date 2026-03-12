# ============================================================================
# OpenInfer Core Library
# ============================================================================

# Collect all source files
file(GLOB_RECURSE OPEN_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/arch/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp
)

# Collect CUDA source files if CUDA is enabled
if(OPEN_BUILD_CUDA)
  file(GLOB_RECURSE OPEN_CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/kernels/cuda/*.cu
  )
  message(STATUS "CUDA sources found: ${OPEN_CUDA_SOURCES}")
endif()

# Create the main library
if(OPEN_BUILD_SHARED)
  add_library(open_core SHARED
    ${OPEN_CORE_SOURCES}
    ${OPEN_CUDA_SOURCES}
  )
else()
  add_library(open_core STATIC
    ${OPEN_CORE_SOURCES}
    ${OPEN_CUDA_SOURCES}
  )
endif()

# Link dependencies
target_link_libraries(open_core
  PUBLIC
    glog::glog
    Threads::Threads
)

# Link Eigen if enabled
if(OPEN_USE_EIGEN)
  target_link_libraries(open_core
    PUBLIC
      Eigen3::Eigen
  )
endif()

if(OPEN_BUILD_CUDA)
  target_link_libraries(open_core
    PUBLIC
      CUDA::cudart
      CUDA::cublas
  )
endif()

# Set target properties
set_target_properties(open_core PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(open_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compile definitions
if(OPEN_BUILD_CUDA)
  target_compile_definitions(open_core PUBLIC OPEN_USE_CUDA)
endif()

# Install library
install(TARGETS open_core
  EXPORT OpenInferTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
