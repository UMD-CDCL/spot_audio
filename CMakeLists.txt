cmake_minimum_required(VERSION 3.8)
project(spot_audio)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cdcl_umd_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)


if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()


# Install the Python modules
ament_python_install_package(${PROJECT_NAME})


# install the Python executables
install(
    PROGRAMS ${PROJECT_NAME}/microphone_node.py
    DESTINATION lib/${PROJECT_NAME}
)
install(
        PROGRAMS ${PROJECT_NAME}/audio_classification_node.py
        DESTINATION lib/${PROJECT_NAME}
)
install(
        PROGRAMS ${PROJECT_NAME}/speaker_node.py
        DESTINATION lib/${PROJECT_NAME}
)

# install libraries
install(
    DIRECTORY ${PROJECT_NAME}/microphone
    DESTINATION lib/${PROJECT_NAME}
)
install(
    DIRECTORY ${PROJECT_NAME}/speaker
    DESTINATION lib/${PROJECT_NAME}
)


# install launch file
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)


ament_package()
