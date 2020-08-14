set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

set(DEF_INSTALL_CMAKE_DIR lib/cmake/neodlr)

set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

set(NEODLR_HEADERS include/dlr.h)

set_target_properties(neodlr PROPERTIES PUBLIC_HEADER "include/dlr.h;include/dlr_tvm.h;include/dlr_treelite.h")

install(FILES ${NEODLR_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR})

install(TARGETS 
    treelite_runtime_static
    EXPORT neodlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    tvm_runtime_static
    EXPORT neodlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    dmlc
    EXPORT neodlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    neodlr
    EXPORT neodlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

export(TARGETS neodlr treelite_runtime_static tvm_runtime_static dmlc FILE "${PROJECT_BINARY_DIR}/neodlrTargets.cmake")

export(PACKAGE neodlr)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

configure_file("${PROJECT_SOURCE_DIR}/cmake/neodlrConfig.cmake.in" "${PROJECT_BINARY_DIR}/neodlrConfig.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "\${neodlr_CMAKE_DIR}/${REL_INCLUDE_DIR}")

configure_file("${PROJECT_SOURCE_DIR}/cmake/neodlrConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/neodlrConfig.cmake" @ONLY)

# ... for both
configure_file("${PROJECT_SOURCE_DIR}/cmake/neodlrConfigVersion.cmake.in" "${PROJECT_BINARY_DIR}/neodlrConfigVersion.cmake" @ONLY)

# Install the dlrConfig.cmake and dlrConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/neodlrConfig.cmake"
  "${PROJECT_BINARY_DIR}/neodlrConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree
install(EXPORT neodlrTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
