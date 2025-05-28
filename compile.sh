export NUM_CMAKE_JOBS=20
cmake -B build
cmake --build build --config Release --parallel ${NUM_CMAKE_JOBS}