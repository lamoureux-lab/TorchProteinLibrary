add_library(GLFRAMEWORK SHARED GlutFramework.cpp Keyboard.cpp PerformanceTimer.cpp)
target_link_libraries(GLFRAMEWORK -lGL -lGLU -lglut -lm)
