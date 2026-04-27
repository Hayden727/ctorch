set(CTORCH_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-parameter
)

if(CTORCH_WERROR)
  list(APPEND CTORCH_WARNINGS -Werror)
endif()
