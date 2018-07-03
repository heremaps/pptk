
macro(remove_header x)
  string(REGEX REPLACE "^[^-]+" "" ${x} "${${x}}")
endmacro(remove_header)

macro(read_text x y)
  file(READ ${x} ${y})
  remove_header(${y})
endmacro(read_text)

macro(append_text x y)
  # reads text in file specified by x, removes header from text,
  # and then appends results to y
  # assumes x is a file
  read_text(${x} temp)
  string(CONCAT ${y} "${${y}}" "${temp}")
endmacro(append_text)

if(WIN32)
  read_text(LICENSE.win.txt added_text)
elseif(APPLE)
  read_text(LICENSE.mac.txt added_text)
  append_text(COPYING.gcc_rle31.txt added_text)
elseif(UNIX)
  read_text(LICENSE.linux.txt added_text)
endif()

append_text(COPYING.lgpl3.txt added_text)
append_text(COPYING.gpl3.txt added_text)
append_text(COPYING.apache2.txt added_text)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.append.txt "${added_text}")
