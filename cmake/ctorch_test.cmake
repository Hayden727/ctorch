function(ctorch_add_test target)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE GTest::gtest_main ${ARG_LIBS})
  # Mirror src/CMakeLists.txt: scope CXX-only flags away from nvcc, and
  # forward warnings to the host compiler via -Xcompiler for .cu files.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CTORCH_WARNINGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CTORCH_CUDA_WARNINGS}>
  )
  include(GoogleTest)
  gtest_discover_tests(${target})
endfunction()
