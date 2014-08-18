# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT(ImageProcessing)


# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

ADD_EXECUTABLE(ImageMutualInformation1 ImageMutualInformation1.cxx)
add_executable(DicomSeriesReadImageWrite2 DicomSeriesReadImageWrite2.cxx )


TARGET_LINK_LIBRARIES(ImageMutualInformation1 ${ITK_LIBRARIES})
target_link_libraries(DicomSeriesReadImageWrite2  ${ITK_LIBRARIES})
