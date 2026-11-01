# mingw-w64 cross toolchain (default install root: $HOME/opt/mingw; adjust to your setup)
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(_MINGW_ROOT "$ENV{HOME}/opt/mingw")
# -posix thread model: the vendored OpenJTalk/MeCab build (g++-posix) depends on it.
set(CMAKE_C_COMPILER   ${_MINGW_ROOT}/usr/bin/x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER ${_MINGW_ROOT}/usr/bin/x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER  ${_MINGW_ROOT}/usr/bin/x86_64-w64-mingw32-windres)

set(CMAKE_FIND_ROOT_PATH ${_MINGW_ROOT}/usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
