# CMakeList.txt: ImplicitFunction 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

# 将源代码添加到此项目的可执行文件。
add_executable (
	ImplicitFunction 
	"ImplicitFunction.cpp" 
	"ImplicitFunction.h" 
	"glad.c" 
	"algorithm/LinearSystem.hpp" 
	"algorithm/ConvexHull.hpp" 
	"settings/Camera.h" 
	"settings/Shader.h" 
	"settings/setting.hpp" 
	"resources/Point.vert" 
	"resources/Point.frag" 
	"resources/Line.vert" 
	"resources/Line.frag")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ImplicitFunction PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
