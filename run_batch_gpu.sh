#!/usr/bin/zsh

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=JCGPU
#SBATCH --output=JCGPU_out.txt

echo "Job nodes: ${SLURM_JOB_NODLIST}"
echo "Current machine: ${hostname}"
nvidia-smi

module purge
module load intel/2023b
module load Clang/18.1.2-CUDA-12.3.0

cmake -B build -S . -DCMAKE_CXX_COMPILER=clang++
cmake --build build

CUDA_VISIBLE_DEVICES=0 ./build/bin/jcdp ./additionals/configs/config.in