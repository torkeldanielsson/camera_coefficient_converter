cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(
    camera_coefficient_converter
    VERSION 1.0
    LANGUAGES CXX)

add_executable(${CMAKE_PROJECT_NAME}
    src/main.cpp)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    _CRT_SECURE_NO_WARNINGS UNICODE NOMINMAX WIN32_LEAN_AND_MEAN VC_EXTRALEAN)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    -W4 -WX -wd4201 -fp:fast -fp:except- -EHsc)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Weverything
    -Werror
    -Wfatal-errors
    -Wno-gnu-anonymous-struct
    -Wno-nested-anon-types
    -Wno-missing-braces
    -Wno-c++98-compat-pedantic
    -Wno-padded
    -Wno-missing-braces
    -Wno-covered-switch-default
    -Wno-gnu-anonymous-struct
    -Wno-nested-anon-types
    -Wno-weak-vtables
    -Wno-switch-enum
    -Wno-c++2a-compat
    -Wno-extra-semi-stmt
    -Wno-deprecated-declarations
    -Wno-unsafe-buffer-usage
    -ffast-math
    -fno-strict-aliasing
    -funsigned-char
    -ftrapv)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    -Wno-deprecated-declarations)
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PRIVATE
    ${PROJECT_ROOT}/thirdparty/include
    ${PROJECT_ROOT}/thirdparty/include/ceres/internal/miniglog)

find_library(CERES_LIB 
    NAMES ceres libceres.a libceres.so ceres.lib
    PATHS 
        ${PROJECT_ROOT}/thirdparty/lib
    NO_DEFAULT_PATH)

if(CERES_LIB)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${CERES_LIB})
else()
    message(FATAL_ERROR "Ceres library not found. Please ensure it's in thirdparty/lib/ or thirdparty/windows/ceres/lib/")
endif()

find_library(GLOG_LIB 
    NAMES glog libglog.a libglog.so glog.lib
    PATHS 
        ${PROJECT_ROOT}/thirdparty/lib
    NO_DEFAULT_PATH)

find_library(GFLAGS_LIB 
    NAMES gflags libgflags.a libgflags.so gflags.lib
    PATHS 
        ${PROJECT_ROOT}/thirdparty/lib
    NO_DEFAULT_PATH)

if(GLOG_LIB)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${GLOG_LIB})
endif()

if(GFLAGS_LIB)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${GFLAGS_LIB})
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE pthread dl)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})
    elseif(BLAS_FOUND)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
    endif()
endif()

if(WIN32)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/$<TARGET_FILE_NAME:${CMAKE_PROJECT_NAME}>
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:${CMAKE_PROJECT_NAME}>)
elseif(UNIX)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD COMMAND
    mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/bin/ &&
    cp ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:${CMAKE_PROJECT_NAME}>
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
endif()
