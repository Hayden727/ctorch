if(CTORCH_CUDA)
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp
    LIBS    ctorch_core CUDA::cudart
  )
  set_source_files_properties(add_bench.cpp PROPERTIES LANGUAGE CUDA)
else()
  ctorch_add_test(add_bench
    SOURCES add_bench.cpp
    LIBS    ctorch_core
  )
endif()
