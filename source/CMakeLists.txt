cmake_minimum_required(VERSION 3.13.1)

if(NOT TARGET app)
     message(FATAL_ERROR "Please create a target named 'app' (ex: add_executable(app)) before adding this file")
endif()

include(edge-impulse-sdk/cmake/utils.cmake)
add_subdirectory(edge-impulse-sdk/cmake/zephyr)

RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "tflite-model" "*.cpp")
target_include_directories(app PRIVATE .)

# must match with pio filename and executable name from above
pico_generate_pio_header(app ${CMAKE_CURRENT_LIST_DIR}/hsync.pio)
pico_generate_pio_header(app ${CMAKE_CURRENT_LIST_DIR}/vsync.pio)
pico_generate_pio_header(app ${CMAKE_CURRENT_LIST_DIR}/rgb.pio)

# must match with executable name
target_link_libraries(app PRIVATE pico_stdlib pico_divider pico_multicore pico_bootsel_via_double_reset hardware_pio hardware_dma hardware_adc hardware_irq hardware_sync hardware_clocks hardware_pll)


# add all sources to the project
target_sources(app PRIVATE ${MODEL_SOURCE} vga_graphics.c)
