cmake_minimum_required(VERSION 3.18)
project(thinksound_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- GGML: vendored as a submodule, built in-tree. The backend is chosen at
# configure time with the upstream flags (-DGGML_CUDA=ON / -DGGML_VULKAN=ON);
# the corresponding ggml-<backend> target then exists and we link it. ----
# Colocate ggml's shared libs with our executables so ggml_backend_load_all()
# finds the backends at runtime (ggml otherwise puts .so under bin/).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(third_party/ggml)

set(TS_GGML_LIBS ggml ggml-base ggml-cpu)
if(TARGET ggml-cuda)
  list(APPEND TS_GGML_LIBS ggml-cuda)
  add_compile_definitions(TS_HAVE_CUDA=1)
endif()
if(TARGET ggml-vulkan)
  list(APPEND TS_GGML_LIBS ggml-vulkan)
  add_compile_definitions(TS_HAVE_VULKAN=1)
endif()
if(TARGET ggml-hip)
  list(APPEND TS_GGML_LIBS ggml-hip)
  add_compile_definitions(TS_HAVE_HIP=1)
endif()

# common static lib (loader, wav io, helpers, model graphs)
add_library(ts_common STATIC
  src/common/ts_model.cpp
  src/common/ts_wav.cpp
  src/vae_decoder.cpp
  src/mmdit.cpp
  src/clip_text.cpp
  src/t5_encoder.cpp
  src/tokenizer.cpp
)
target_include_directories(ts_common PUBLIC src)
target_link_libraries(ts_common PUBLIC ${TS_GGML_LIBS})

# Relocatable: find the colocated ggml .so next to the binary.
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# ---- tools ----
add_executable(ts-gguf-dump src/tools/gguf_dump.cpp)
target_link_libraries(ts-gguf-dump PRIVATE ts_common)

# (added incrementally: ts-vae-decode, ts-mmdit, ts-cli, ts-parity)
foreach(tool vae_decode parity dit_test dit_sample encoders_test generate cuda_smoke tok_test)
  if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/${tool}.cpp")
    add_executable(ts-${tool} src/tools/${tool}.cpp)
    target_link_libraries(ts-${tool} PRIVATE ts_common)
  endif()
endforeach()

# Resident HTTP server (keeps the model warm; wraps the generate pipeline).
find_package(Threads REQUIRED)
add_executable(ts-server src/tools/ts-server.cpp)
target_include_directories(ts-server PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib
  ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(ts-server PRIVATE ts_common Threads::Threads)
