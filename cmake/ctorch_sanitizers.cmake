# Sanitizer wiring. Only AddressSanitizer is exposed today; UBSan / TSan can
# follow the same pattern when needed. Applied at directory scope (i.e. to
# every target defined after this file is included from the root listfile)
# because both ctorch_core and the gtest-based tests need to be instrumented
# together for ASan to report cross-module issues.

if(CTORCH_ASAN)
  if(MSVC)
    message(WARNING "CTORCH_ASAN ignored: AddressSanitizer wiring for MSVC is not supported here.")
  else()
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
  endif()
endif()
