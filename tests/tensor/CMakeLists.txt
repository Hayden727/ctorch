ctorch_add_test(tensor_view_test
  SOURCES tensor_view_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(tensor_to_device_test
  SOURCES tensor_to_device_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(slice_test
  SOURCES slice_test.cpp
  LIBS    ctorch_core
)

ctorch_add_test(view_alias_test
  SOURCES view_alias_test.cpp
  LIBS    ctorch_core
)
