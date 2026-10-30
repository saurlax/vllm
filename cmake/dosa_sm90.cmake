# Research-only: compile unchanged planner/selection code for SM90. This does
# not enable a serving backend or qualify SM100 numerical guards for SM90.
cuda_archs_loose_intersection(DOSA_SM90_ARCHS "9.0a" "${CUDA_ARCHS}")
if(NOT DOSA_SM90_ARCHS)
  message(FATAL_ERROR "VLLM_DOSA_SM90_RESEARCH requires architecture 9.0a")
endif()

set(_dosa_sm90_root "${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/dosa")
set(_dosa_sm90_sources "${_dosa_sm90_root}/sm90/module.cpp")
set(_dosa_sm90_generated "${CMAKE_CURRENT_BINARY_DIR}/dosa_sm90")
file(MAKE_DIRECTORY "${_dosa_sm90_generated}")
# Keep the author's source untouched. Generate an auditable SM90 numerical
# specialization, failing configuration if the pinned source pattern changes.
file(READ "${_dosa_sm90_root}/planner.cu" _sm90_plan)
string(REGEX MATCHALL "0x1p-23" _dot_literals "${_sm90_plan}")
list(LENGTH _dot_literals _dot_literal_count)
string(FIND "${_sm90_plan}" "constexpr double DA_SCREEN = 0x1.5001690170d18p-16;" _direct_literal)
if(NOT _dot_literal_count EQUAL 3 OR _direct_literal LESS 0)
  message(FATAL_ERROR "Pinned DoSA planner numerical patterns changed; audit SM90 specialization")
endif()
# The FP8 WGMMA accumulator is below FP32 precision on Hopper. Use a deliberately
# conservative half-precision-size unit roundoff, not the old FP32 assumption.
string(REPLACE "0x1p-23" "0x1p-10" _sm90_plan "${_sm90_plan}")
string(REPLACE "constexpr double DA_SCREEN = 0x1.5001690170d18p-16;"
  "constexpr double DA_SCREEN = 0.25;" _sm90_plan "${_sm90_plan}")
file(GENERATE OUTPUT "${_dosa_sm90_generated}/planner_sm90.cuh" CONTENT "${_sm90_plan}")
foreach(_heads 32 64)
  foreach(_role planner direct_abs seed select post reference screen_reference hopper screen_fast partial refine_shared)
    if(_heads EQUAL 64 AND _role MATCHES "^(partial|refine_shared)$")
      continue()
    endif()
    set(_namespace "dosa_sm90_h${_heads}")
    if(_role STREQUAL "direct_abs")
      set(_namespace "dosa_sm90_h${_heads}_direct")
    endif()
    set(_unit "${_dosa_sm90_generated}/h${_heads}_${_role}.cu")
    set(_include "${_dosa_sm90_root}/${_role}.cu")
    set(_extra "")
    if(_role MATCHES "^(planner|direct_abs)$")
      set(_include "${_dosa_sm90_generated}/planner_sm90.cuh")
      if(_role STREQUAL "direct_abs")
        set(_extra "#define DOSA_DIRECT_ABS 1\n")
      endif()
    endif()
    if(_role MATCHES "^(reference|screen_reference|hopper|screen_fast|partial|refine_shared)$")
      set(_include "${_dosa_sm90_root}/sm90/${_role}.cu")
    endif()
    string(CONCAT _content
      "#define DOSA_SCORE_HEADS ${_heads}\n"
      "#define DOSA_NAMESPACE ${_namespace}\n"
      "#define DOSA_OP_PREFIX \"dosa_sm90_h${_heads}_${_role}_\"\n"
      "${_extra}"
      "#include \"${_include}\"\n")
    file(GENERATE OUTPUT "${_unit}" CONTENT "${_content}")
    set_gencode_flags_for_srcs(SRCS "${_unit}" CUDA_ARCHS "${DOSA_SM90_ARCHS}")
    set_source_files_properties("${_unit}" PROPERTIES INCLUDE_DIRECTORIES
      "${_dosa_sm90_root};${deepgemm_SOURCE_DIR}/deep_gemm/include;${deepgemm_SOURCE_DIR}/third-party/cutlass/include")
    set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_OPTIONS
      -O3 -lineinfo --expt-relaxed-constexpr -std=c++20
      --prec-div=true --prec-sqrt=true --ftz=false --fmad=true)
    if(_role MATCHES "^(direct_abs|screen_reference)$")
      set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_OPTIONS --fmad=false)
    endif()
    if(_role MATCHES "^(planner|direct_abs)$")
      set_property(SOURCE "${_unit}" APPEND PROPERTY COMPILE_DEFINITIONS
        DOSA_NEGATIVE_FILL_MINORITY=1)
    endif()
    list(APPEND _dosa_sm90_sources "${_unit}")
  endforeach()
endforeach()

set(_dosa_sm90_flags ${VLLM_GPU_FLAGS})
list(REMOVE_ITEM _dosa_sm90_flags --use_fast_math -use_fast_math)
define_extension_target(
  _dosa_sm90_C DESTINATION vllm LANGUAGE CUDA
  SOURCES ${_dosa_sm90_sources}
  COMPILE_FLAGS ${_dosa_sm90_flags}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3 WITH_SOABI)
target_compile_definitions(_dosa_sm90_C PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL
  USE_CUDA CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
