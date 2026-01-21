cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++  -DGPU_ARCH=sm_80

cmake --build build -v

export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_INFO=4
export LIBOMPTARGET_DEBUG=1

./build/bin/jcdp ./additionals/configs/config.in