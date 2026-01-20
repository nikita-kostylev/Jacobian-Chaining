cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++  \
-DCMAKE_CXX_FLAGS="-O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80" \
-DCMAKE_EXE_LINKER_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80" \

cmake --build build

./build/bin/jcdp ./additionals/configs/config.in