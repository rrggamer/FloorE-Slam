cmake_minimum_required(VERSION 3.8)
project(floore_mapping)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  floore_mapping/record_data.py
  floore_mapping/build_ocgm.py
  floore_mapping/explore.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
