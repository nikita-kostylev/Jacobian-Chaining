module purge && module load intel/2023b && module load Clang/18.1.2-CUDA-12.3.0 && cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug && cmake --build build

LIBOMPTARGET_INFO=-1 CUDA_VISIBLE_DEVICES=3 ./build/bin/jcdp ./additionals/configs/config.in
CUDA_VISIBLE_DEVICES=3 ./build/bin/jcdp ./additionals/configs/config.in
