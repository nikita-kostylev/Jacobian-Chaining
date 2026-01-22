[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15373160.svg)](https://doi.org/10.5281/zenodo.15373160)

# Jacobian Chaining

This repository contains a GPU-compatible adaptation of https://github.com/STCE-at-RWTH/Jacobian-Chaining, a reference implementation which solves various versions of the Jacobian Chain Bracketing Problem.
The branch & bound algorithm for scheduling is implemented in a non-recursive version with a replacement of dynamic memory access structures with arrays.

The implementation is not complete, as the non-recursive version has a rare edgecase where it is stuck in a loop.
If the branch & bound optimizer is run with the GPU-version of the branch & bound scheduler the result are wrong, as the starting time function is not working correct.
## Build

### Requirements

- **C++23 Compiler / C++23 Standard Library**

   Tested on Linux with:
   - Clang 18.1.2 (libc++)

- **CMake >= 3.25.0**

   On Windows we need at least CMake 3.30.0 if OpenMP is enabled.


### Commands

```shell
mkdir build
cmake -B build -S . -DCMAKE_CXX_COMPILER=<replace-me>
cmake --build build

./build/bin/jcdp ./additionals/configs/config.in
```

### Options

```shell
cmake -DJCDP_USE_OPENMP=<ON|OFF>
export OMP_NUM_THREADS=<replace-me>
```

## Docker container

We also provide a Docker file which will create an Ubuntu 24.10 image with all necessary tools and automatically build the project. To build the Docker image run the following command from the root directory:

```shell
docker build --file additionals/docker/Dockerfile . --tag jcdp
```

In the docker image the executable are located at `/app/jcdp` and `/app/jcdp_batch`. The config files are in `/app/configs` and the plotting script is at `/app/generate_plots.py`. To directly run the solver, run:

```shell
docker run -it jcdp /app/jcdp /app/configs/config.in
```

Alternatively, just run `docker run -it jcdp` and work from inside the container.

## Config files

The config files are checked for the following key-value pairs:

- `length <q>`  
   Length of the Jacobian chains.

- `size_range <lower bound> <upper bound>`  
   Range for the input $n_i$ and output sizes $m_i$.

- `dag_size_range <lower bound> <upper bound>`  
   Range for the size of the elemental DAGs $|E_i|$.

- `available_threads <m>`  
   Number of available machines / threads for scheduling. $m=0$ indicates infinite threads (unlimited parallelism).

- `available_memory <M>`  
   Memory limit per machine. $\bar{M} = 0$ indicates infinite memory.

- `matrix_free <0/1>`  
   Flag that enables matrix-free variant of the Jacobian Chain Bracketing Problem.

- `time_to_solve <s>`  
   Time limit in seconds for the runtime of the Branch & Bound solvers.

- `seed <rng>`  
   Seed for the random number generator in the Jabobian chain generator for reproducibility.

- `amount <n>`  
   Number of chains to generate and solve. Only used by `jcdp_batch`.

## Statistical benchmarks

To run the statistical benchmarks, use for example the config file at `additionals/configs/config_batch_small.in`:

```shell
./build/bin/jcdp_batch ./additionals/configs/config_batch_small.in
```

After the program finish there should be three new files: `results3.csv`, `results4.csv`, and `results5.csv`. These can be passed to the plotting scripts to generate the boxplots:

```shell
./additionals/scripts/generate_plots.py ./results5.csv
```
