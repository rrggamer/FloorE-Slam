cmake_minimum_required(VERSION 3.8)
project(floore_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf launch rviz
  DESTINATION share/${PROJECT_NAME}
)

# Meshes live at the repo root (shared with the SolidWorks/CAD side), not
# duplicated under this package, so install them from there into our share dir.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
