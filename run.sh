cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++  -DGPU_ARCH=sm_80

cmake --build build -v

./build/bin/jcdp ./additionals/configs/config.in