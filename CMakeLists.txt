cmake_minimum_required(VERSION 3.16)
project(gsvtts C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)   # engine-level serialization lock (pthread_mutex)

option(GSV_USE_CUDA "Link onnxruntime CUDA provider" ON)

# Japanese G2P (OpenJTalk 1.11 C frontend + MeCab C++ runtime), vendored in-tree
# under third_party/openjtalk/ so the build compiles it directly from source — no
# prebuilt .o required. The MeCab portion is C++, so when ja is enabled the final
# link uses the C++ driver + libstdc++. If the vendored tree is missing, fall back
# to a stub so zh/en-only builds still work.
set(GSV_JTALK_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openjtalk" CACHE PATH "vendored OpenJTalk 1.11 source root")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ORT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

file(GLOB GSV_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/*.c
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/*.c)

# ---- Japanese G2P (OpenJTalk) wiring ---------------------------------------
set(GSV_JTALK_OK FALSE)
if(EXISTS "${GSV_JTALK_SRC_DIR}/jpcommon/jpcommon.c" AND EXISTS "${GSV_JTALK_SRC_DIR}/mecab/src/mecab.cpp")
  set(GSV_JTALK_OK TRUE)
endif()

if(GSV_JTALK_OK)
  # Vendored OpenJTalk C frontend (single-threaded text->phonemes).
  set(_oj_c_sources
    ${GSV_JTALK_SRC_DIR}/njd/njd.c
    ${GSV_JTALK_SRC_DIR}/njd/njd_node.c
    ${GSV_JTALK_SRC_DIR}/jpcommon/jpcommon.c
    ${GSV_JTALK_SRC_DIR}/jpcommon/jpcommon_label.c
    ${GSV_JTALK_SRC_DIR}/jpcommon/jpcommon_node.c
    ${GSV_JTALK_SRC_DIR}/text2mecab/text2mecab.c
    ${GSV_JTALK_SRC_DIR}/mecab2njd/mecab2njd.c
    ${GSV_JTALK_SRC_DIR}/njd_set_pronunciation/njd_set_pronunciation.c
    ${GSV_JTALK_SRC_DIR}/njd_set_digit/njd_set_digit.c
    ${GSV_JTALK_SRC_DIR}/njd_set_accent_phrase/njd_set_accent_phrase.c
    ${GSV_JTALK_SRC_DIR}/njd_set_accent_type/njd_set_accent_type.c
    ${GSV_JTALK_SRC_DIR}/njd_set_unvoiced_vowel/njd_set_unvoiced_vowel.c
    ${GSV_JTALK_SRC_DIR}/njd_set_long_vowel/njd_set_long_vowel.c
    ${GSV_JTALK_SRC_DIR}/njd2jpcommon/njd2jpcommon.c)

  # MeCab C++ runtime: every mecab/src/*.cpp except standalone tools that carry a main().
  file(GLOB _oj_cpp_all ${GSV_JTALK_SRC_DIR}/mecab/src/*.cpp)
  list(REMOVE_ITEM _oj_cpp_all ${GSV_JTALK_SRC_DIR}/mecab/src/mecab-dict-index.cpp)

  # Static lib so its -O0 (see below) stays isolated from gsvtts' -O2.
  add_library(openjtalk STATIC ${_oj_c_sources} ${_oj_cpp_all})
  target_include_directories(openjtalk PUBLIC
    ${GSV_JTALK_SRC_DIR}/jpcommon ${GSV_JTALK_SRC_DIR}/njd ${GSV_JTALK_SRC_DIR}/text2mecab
    ${GSV_JTALK_SRC_DIR}/mecab2njd ${GSV_JTALK_SRC_DIR}/njd2jpcommon
    ${GSV_JTALK_SRC_DIR}/njd_set_pronunciation ${GSV_JTALK_SRC_DIR}/njd_set_digit
    ${GSV_JTALK_SRC_DIR}/njd_set_accent_phrase ${GSV_JTALK_SRC_DIR}/njd_set_accent_type
    ${GSV_JTALK_SRC_DIR}/njd_set_unvoiced_vowel ${GSV_JTALK_SRC_DIR}/njd_set_long_vowel
    ${GSV_JTALK_SRC_DIR}/mecab/src)
  target_compile_definitions(openjtalk PUBLIC CHARSET_UTF_8)
  # MeCab C++ needs the HAVE_* macros autotools would put in config.h (we don't run
  # configure). DIC_VERSION must equal the version_ in the shipped sys.dic header
  # (NAIST-JDic = 102) or Mecab_load's version check fails.
  set_source_files_properties(${_oj_cpp_all} PROPERTIES
    COMPILE_DEFINITIONS "HAVE_WINDOWS_H;HAVE_GCC_ATOMIC_OPS;HAVE_STDINT_H;DIC_VERSION=102;PACKAGE=\"MeCab\";VERSION=\"1.11\""
    COMPILE_OPTIONS "-fno-keep-inline-dllexport")
  # MeCab's Windows registry lookup (utils.cpp load_dictionary_resource) reads an
  # uninitialized HKEY that faults at -O2 during Mecab_load; -O0 keeps it benign.
  # G2P is CPU-light, so -O0 here costs nothing (heavy ONNX inference is separate).
  target_compile_options(openjtalk PRIVATE -O0 -Wall)

  list(APPEND GSV_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/ja/g2p_ja.c)
endif()
if(NOT GSV_JTALK_OK)
  message(FATAL_ERROR
    "gsvtts: vendored OpenJTalk not found under ${GSV_JTALK_SRC_DIR} — it is required for ja G2P")
endif()

add_library(gsvtts SHARED ${GSV_SOURCES})
set_target_properties(gsvtts PROPERTIES PREFIX "")
target_include_directories(gsvtts PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # src/g2p always on: g2p_ja.h (included by the dispatcher) uses bare "utf8.h"
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/g2p ${ORT_ROOT}/include)
target_compile_definitions(gsvtts PRIVATE GSVTTS_BUILD)

if(GSV_JTALK_OK)
  # OpenJTalk include dirs + CHARSET_UTF_8 propagate transitively from the openjtalk
  # static lib (PUBLIC interface), so g2p_ja.c resolves its bare "mecab.h"/"njd.h" etc.
  set_property(TARGET gsvtts PROPERTY LINKER_LANGUAGE CXX)   # MeCab is C++
  target_link_libraries(gsvtts PRIVATE openjtalk m)
endif()

# mingw can link directly against the DLL (no import lib needed).
# Only onnxruntime.dll is a load-time dependency; the CUDA EP plugin
# (onnxruntime_providers_cuda.dll) is loaded dynamically by ORT when the
# CUDA execution provider is registered, so it must merely sit next to
# onnxruntime.dll at runtime.
target_link_libraries(gsvtts PRIVATE ${ORT_ROOT}/lib/onnxruntime.dll)
target_link_libraries(gsvtts PRIVATE Threads::Threads)   # libwinpthread (engine lock)

target_compile_options(gsvtts PRIVATE -Wall -Wextra -O2)
# keep the DLL self-contained: no libgcc (and, with ja, no libstdc++) runtime dep
set(_gsv_static_flags -static-libgcc)
if(GSV_JTALK_OK)
  list(APPEND _gsv_static_flags -static-libstdc++)
endif()
set_target_properties(gsvtts PROPERTIES LINK_OPTIONS "${_gsv_static_flags}")

# copy runtime deps next to outputs for easy Windows-side runs
set(ORT_DLLS ${ORT_ROOT}/lib/onnxruntime.dll ${ORT_ROOT}/lib/onnxruntime_providers_shared.dll)
if(GSV_USE_CUDA)
  list(APPEND ORT_DLLS ${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll)
endif()
foreach(dll ${ORT_DLLS})
  add_custom_command(TARGET gsvtts POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll} $<TARGET_FILE_DIR:gsvtts>)
endforeach()
