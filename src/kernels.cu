#include "kernels.cuh"

__global__ void init_rand_kernel(curandState *states, unsigned long seed, int ant_count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < ant_count){
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void ant_movement_kernel(
    CUDA_Ant *ants, bool *visited, double *pheromones, char *grid_map,
    int rows, int cols, curandState *states,
    double max_phero_cap, double alpha, double beta, int ant_count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= ant_count) return;

    CUDA_Ant ant = ants[id];
    curandState local_rand = states[id];

    int visited_offset = id * rows * cols;
    visited[visited_offset + ant.pos_y * cols + ant.pos_x] = true;

    int dirs_x[4] = { 0, 0, -1, 1 };
    int dirs_y[4] = { -1, 1, 0, 0 };

    while(!ant.reached_end && !ant.stuck && ant.path_length < MAX_PATH_LEN - 1){
        int next_x[4], next_y[4];
        double probs[4];
        double prob_sum = 0.0;
        int valid_count = 0;

        for(int i = 0; i < 4; i++){
            int nx = ant.pos_x + dirs_x[i];
            int ny = ant.pos_y + dirs_y[i];

            if(nx >= 0 && nx < cols && ny >= 0 && ny < rows){
                char cell = grid_map[ny * cols + nx];
                bool is_wall = (cell == '1');
                bool is_other_terminal = false;
                if(ant.type == 0 && (cell == 'e' || cell == 'E')) is_other_terminal = true;
                if(ant.type == 1 && (cell == 'g' || cell == 'G')) is_other_terminal = true;

                if(!is_wall && !is_other_terminal && !visited[visited_offset + ny * cols + nx]){
                    double own_p = pheromones[(ant.type * rows + ny) * cols + nx];
                    double gamma = 1.0;

                    for(int t = 0; t < 2; t++){
                        if(t != ant.type){
                            double other_p = pheromones[(t * rows + ny) * cols + nx];
                            if(other_p > 0.1){
                                double intensity = fminf(1.0, other_p / max_phero_cap);
                                double modifier = 1.0 - 1.0 * intensity;
                                if(modifier < 0.01) modifier = 0.01;
                                gamma *= modifier;
                            }
                        }
                    }

                    double dist_target = sqrt(pow((double) (nx - ant.target_x), 2) + pow((double) (ny - ant.target_y), 2));
                    double heuristic = 1.0 / (dist_target + 1.0);
                    double p = pow(own_p * gamma, alpha) * pow(heuristic, beta);

                    next_x[valid_count] = nx;
                    next_y[valid_count] = ny;
                    probs[valid_count] = p;
                    prob_sum += p;
                    valid_count++;
                }
            }
        }

        if(valid_count == 0){
            ant.stuck = true;
        }
        else{
            int chosen_idx = valid_count - 1;
            if(curand_uniform_double(&local_rand) < 0.03){
                chosen_idx = (int) (curand_uniform_double(&local_rand) * valid_count);
                if(chosen_idx >= valid_count) chosen_idx = valid_count - 1;
            }
            else{
                double r = curand_uniform_double(&local_rand) * prob_sum;
                double cumulative = 0.0;
                for(int i = 0; i < valid_count; i++){
                    cumulative += probs[i];
                    if(cumulative >= r){
                        chosen_idx = i;
                        break;
                    }
                }
            }

            ant.last_dir_x = next_x[chosen_idx] - ant.pos_x;
            ant.last_dir_y = next_y[chosen_idx] - ant.pos_y;
            ant.pos_x = next_x[chosen_idx];
            ant.pos_y = next_y[chosen_idx];
            ant.path_x[ant.path_length] = ant.pos_x;
            ant.path_y[ant.path_length] = ant.pos_y;
            ant.path_length++;

            visited[visited_offset + ant.pos_y * cols + ant.pos_x] = true;
            if(ant.pos_x == ant.target_x && ant.pos_y == ant.target_y){
                ant.reached_end = true;
            }
        }
    }
    ants[id] = ant;
    states[id] = local_rand;
}