#pragma once
#include "config.h"
#include <curand_kernel.h>

// 在 aco_env.cuh 中修改 run_aco 的宣告

class ACO_Environment{
public:
    int env_id;
    unsigned long base_seed; // <--- 新增這行
    cudaStream_t stream;

    vector<vector<vector<double>>> pheromones;
    CUDA_Ant global_best_ants[TYPE_COUNT][MAX_ENDPOINTS];
    bool has_global_best[TYPE_COUNT][MAX_ENDPOINTS];
    double global_best_combined_score;

    double *d_pheromones;
    bool *d_visited;
    CUDA_Ant *d_ants;
    curandState *d_rand_states;

    ACO_Environment(int id, unsigned long seed);
    ~ACO_Environment();

    void drawSimulation(int iteration, const Chromosome &dna, int gen_num);
    double run_aco(const Chromosome &dna, bool visualize, int gen_num, bool use_gpu = true);
    // double run_aco(const Chromosome &dna, bool visualize, int gen_num);
};