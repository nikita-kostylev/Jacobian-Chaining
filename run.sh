cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++
cmake --build build

./build/bin/jcdp ./additionals/configs/config.in