#pragma once
#include <curand_kernel.h>
#include "config.h"

__global__ void init_rand_kernel(curandState *states, unsigned long seed, int ant_count);

__global__ void ant_movement_kernel(
    CUDA_Ant *ants, bool *visited, double *pheromones, char *grid_map,
    int rows, int cols, curandState *states,
    double max_phero_cap, double alpha, double beta, int ant_count);