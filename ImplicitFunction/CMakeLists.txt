# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("ImplicitFunction")

include_directories("D:/SoftwareInstallation/Dependency/include/")
include_directories("D:/SoftwareInstallation/opencv/build/include/")
include_directories("D:/SoftwareInstallation/eigen-3.4.0/")
include_directories("ImplicitFunction/lib/imgui/")
include_directories("ImplicitFunction/algorithm/")
include_directories("ImplicitFunction/settings")

link_directories("D:/SoftwareInstallation/Dependency/lib/")
link_directories("D:/SoftwareInstallation/opencv/build/x64/vc16/lib/")
link_libraries("glfw3.lib")
link_libraries("opencv_world480d.lib")

add_subdirectory ("ImplicitFunction")
