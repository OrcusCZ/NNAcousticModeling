PROJECT(PhoneRecog)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

#INCLUDE_DIRECTORIES("source")

# Main project control.
ADD_EXECUTABLE(PhoneRecog
   source/main.cpp
   source/data.cpp
   source/mlf_io.cpp
   source/OL_FileList.cpp
   source/phone_recog.cpp 
)


