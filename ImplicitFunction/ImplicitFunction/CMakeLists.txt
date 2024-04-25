# CMakeList.txt: ImplicitFunction 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

#"D:/SoftwareInstallation/imgui-1.90.5/imgui.cpp" 
#	"D:/SoftwareInstallation/imgui-1.90.5/imgui_draw.cpp" 
	#"D:/SoftwareInstallation/imgui-1.90.5/imgui_tables.cpp" 
	#"D:/SoftwareInstallation/imgui-1.90.5/imgui_widgets.cpp"
	#"D:/SoftwareInstallation/imgui-1.90.5/backends/imgui_impl_glfw.cpp"
	#"D:/SoftwareInstallation/imgui-1.90.5/backends/imgui_impl_opengl3.cpp"
# 将源代码添加到此项目的可执行文件。
file (GLOB IMGUI_SOURCES "lib/imgui/*.cpp")
file (GLOB VERTEX_SOURCES "resources/*.vert")
file (GLOB FRAGMENT_SOURCES "resources/*.frag")

add_executable (
	ImplicitFunction 
	"main.cpp" 
	"lib/glad.c" 
	${IMGUI_SOURCES} 
	${VERTEX_SOURCES} 
	${FRAGMENT_SOURCES} )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ImplicitFunction PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
