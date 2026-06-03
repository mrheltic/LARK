cmake_minimum_required(VERSION 3.16)
project(gr-lark)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Python-only OOT module — no C++ bindings, no SWIG
set(PYTHON_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/dist-packages/lark")

install(
    DIRECTORY python/
    DESTINATION "${PYTHON_DIR}"
    FILES_MATCHING PATTERN "*.py"
)

install(
    DIRECTORY grc/
    DESTINATION share/gnuradio/grc/blocks
    FILES_MATCHING PATTERN "*.block.yml"
)