cmake_minimum_required(VERSION 3.18)
project(thinksound_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- GGML: link against the prebuilt llama.cpp tree (commit 911b67a60) ----
set(GGML_DIR "/devel/tools/llama.cpp" CACHE PATH "llama.cpp/ggml source tree")
set(GGML_BUILD "${GGML_DIR}/build" CACHE PATH "prebuilt ggml build dir (CUDA)")
set(GGML_LIBDIR "${GGML_BUILD}/bin")

option(TS_CUDA "link the ggml CUDA backend" ON)

include_directories(${GGML_DIR}/ggml/include)

find_library(GGML_LIB        ggml        PATHS ${GGML_LIBDIR} REQUIRED)
find_library(GGML_BASE_LIB   ggml-base   PATHS ${GGML_LIBDIR} REQUIRED)
find_library(GGML_CPU_LIB    ggml-cpu    PATHS ${GGML_LIBDIR} REQUIRED)
set(TS_GGML_LIBS ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})
if(TS_CUDA)
  find_library(GGML_CUDA_LIB ggml-cuda PATHS ${GGML_LIBDIR})
  if(GGML_CUDA_LIB)
    list(APPEND TS_GGML_LIBS ${GGML_CUDA_LIB})
    add_compile_definitions(TS_HAVE_CUDA=1)
  endif()
endif()

option(TS_VULKAN "link the ggml Vulkan backend" ON)
if(TS_VULKAN)
  find_library(GGML_VULKAN_LIB ggml-vulkan PATHS ${GGML_LIBDIR})
  if(GGML_VULKAN_LIB)
    list(APPEND TS_GGML_LIBS ${GGML_VULKAN_LIB})
    add_compile_definitions(TS_HAVE_VULKAN=1)
  endif()
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

# RPATH so the prebuilt .so files are found at runtime
set(CMAKE_BUILD_RPATH "${GGML_LIBDIR}")
set(CMAKE_INSTALL_RPATH "${GGML_LIBDIR}")

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
