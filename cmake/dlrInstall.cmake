set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/dlr)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()


set(DLR_HEADERS 
    include/dlr.h)

set_target_properties(dlr PROPERTIES PUBLIC_HEADER "include/dlr.h")

install(FILES ${DLR_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR})

install(TARGETS 
    treelite_runtime_static
    EXPORT dlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    tvm_runtime_static
    EXPORT dlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    dmlc
    EXPORT dlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

install(TARGETS 
    dlr
    EXPORT dlrTargets
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev)

export(TARGETS dlr treelite_runtime_static tvm_runtime_static dmlc FILE "${PROJECT_BINARY_DIR}/dlrTargets.cmake")

export(PACKAGE dlr)

file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")

# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

configure_file("${PROJECT_SOURCE_DIR}/cmake/dlrConfig.cmake.in" "${PROJECT_BINARY_DIR}/dlrConfig.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "\${dlr_CMAKE_DIR}/${REL_INCLUDE_DIR}")

configure_file("${PROJECT_SOURCE_DIR}/cmake/dlrConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dlrConfig.cmake" @ONLY)

# ... for both
configure_file("${PROJECT_SOURCE_DIR}/cmake/dlrConfigVersion.cmake.in" "${PROJECT_BINARY_DIR}/dlrConfigVersion.cmake" @ONLY)

# Install the dlrConfig.cmake and dlrConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dlrConfig.cmake"
  "${PROJECT_BINARY_DIR}/dlrConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree
install(EXPORT dlrTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
