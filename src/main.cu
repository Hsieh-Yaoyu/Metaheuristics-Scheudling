#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <thread>
#include <mutex>
#include <fstream>
#include <filesystem> // 新增：用於建立資料夾
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>

using namespace std;
using namespace cv;
namespace fs = std::filesystem; // 檔案系統命名空間

// --- 演算法基本參數 ---
const int ANT_COUNT = 40;
const double MAX_PHERO = 100.0;
const int MAX_ITER = 1000;
const int CELL_SIZE = 25;
const int MAX_PATH_LEN = 800;

const int DIFFUSION_RADIUS = 2;
const int MAX_ENDPOINTS = 10;
const int SAFE_DISTANCE = 3;

const int POP_SIZE = 20;
const int GA_GENERATIONS = 10;

// --- 視覺化顏色設定 ---
const Scalar COLOR_BG = Scalar(255, 255, 255);
const Scalar COLOR_WALL = Scalar(50, 50, 50);
const Scalar COLOR_GRID = Scalar(200, 200, 200);
const Scalar COLOR_GAS = Scalar(0, 255, 255);
const Scalar COLOR_ELEC = Scalar(255, 225, 0);

enum PipeType{ GAS = 0, ELEC = 1 };
const int TYPE_COUNT = 2;

std::mutex print_mutex;

// --- 基因結構 (Chromosome) ---
struct Chromosome{
    double alpha;
    double beta;
    double rho;
    double Q;
    double turn_w;
    double clear_w;
    double fitness = -1.0;
};

double randDouble(double minVal, double maxVal){
    return minVal + (double) rand() / RAND_MAX * (maxVal - minVal);
}

// --- 多端點地圖定義 (全域唯讀共用) ---
const vector<string> grid_map_cpu = {
    "g00000000000000000000000",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "00000000000000E000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "G00000000000000000000000",
    "00000000000000000G000000",
    "000000000000000000000000",
    "E00000000000000000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "000000000000000000000000",
    "00000000000e000000000000"
};

int rows = grid_map_cpu.size();
int cols = grid_map_cpu[0].size();

vector<Point> start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];
char *d_grid_map_shared;

struct CUDA_Ant{
    int type;
    int pos_x, pos_y;
    int last_dir_x, last_dir_y;
    int target_x, target_y;
    int target_idx;
    int path_x[MAX_PATH_LEN];
    int path_y[MAX_PATH_LEN];
    int path_length;
    bool reached_end;
    bool stuck;
};

// --- CUDA Kernels ---
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
        int next_x[4];
        int next_y[4];
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
                                int pref = -1;
                                double modifier = 1.0 + pref * intensity;
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

// =====================================================================
// === 封裝：平行化環境類別 (每個基因擁有獨立的 GPU Stream 與記憶體) ===
// =====================================================================
class ACO_Environment{
public:
    int env_id;
    cudaStream_t stream;

    vector<vector<vector<double>>> pheromones;
    CUDA_Ant global_best_ants[TYPE_COUNT][MAX_ENDPOINTS];
    bool has_global_best[TYPE_COUNT][MAX_ENDPOINTS];
    double global_best_combined_score;

    double *d_pheromones;
    bool *d_visited;
    CUDA_Ant *d_ants;
    curandState *d_rand_states;

    ACO_Environment(int id, unsigned long seed) : env_id(id){
        cudaStreamCreate(&stream);
        cudaMalloc(&d_pheromones, TYPE_COUNT * rows * cols * sizeof(double));
        cudaMalloc(&d_visited, ANT_COUNT * rows * cols * sizeof(bool));
        cudaMalloc(&d_ants, ANT_COUNT * sizeof(CUDA_Ant));
        cudaMalloc(&d_rand_states, ANT_COUNT * sizeof(curandState));

        int blockSize = 256;
        int numBlocks = (ANT_COUNT + blockSize - 1) / blockSize;
        init_rand_kernel << <numBlocks, blockSize, 0, stream >> > (d_rand_states, seed, ANT_COUNT);
        cudaStreamSynchronize(stream);
    }

    ~ACO_Environment(){
        cudaFree(d_pheromones);
        cudaFree(d_visited);
        cudaFree(d_ants);
        cudaFree(d_rand_states);
        cudaStreamDestroy(stream);
    }

    void drawSimulation(int iteration, const Chromosome &dna, int gen_num);
    double run_aco(const Chromosome &dna, bool visualize, int gen_num);
};

void ACO_Environment::drawSimulation(int iteration, const Chromosome &dna, int gen_num){
    Mat img(rows * CELL_SIZE, cols * CELL_SIZE, CV_8UC3, COLOR_BG);
    double display_max[TYPE_COUNT] = { 0.1, 0.1 };
    double actual_max[TYPE_COUNT] = { 0.1, 0.1 };

    for(int t = 0; t < TYPE_COUNT; t++){
        for(int y = 0; y < rows; y++){
            for(int x = 0; x < cols; x++){
                if(pheromones[t][y][x] > actual_max[t]) actual_max[t] = pheromones[t][y][x];
            }
        }
        display_max[t] = min(actual_max[t], 100.0);
    }

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            Rect rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            if(grid_map_cpu[y][x] == '1'){
                rectangle(img, rect, COLOR_WALL, FILLED);
            }
            else{
                double gas_int = min(1.0, pheromones[GAS][y][x] / display_max[GAS]);
                double elec_int = min(1.0, pheromones[ELEC][y][x] / display_max[ELEC]);
                double max_int = max(gas_int, elec_int);

                if(max_int > 0.05){
                    double sum_int = gas_int + elec_int;
                    double w_gas = gas_int / sum_int;
                    double w_elec = elec_int / sum_int;
                    int final_b = COLOR_BG[0] * (1.0 - max_int) + (COLOR_GAS[0] * w_gas + COLOR_ELEC[0] * w_elec) * max_int;
                    int final_g = COLOR_BG[1] * (1.0 - max_int) + (COLOR_GAS[1] * w_gas + COLOR_ELEC[1] * w_elec) * max_int;
                    int final_r = COLOR_BG[2] * (1.0 - max_int) + (COLOR_GAS[2] * w_gas + COLOR_ELEC[2] * w_elec) * max_int;
                    rectangle(img, rect, Scalar(final_b, final_g, final_r), FILLED);
                }
                rectangle(img, rect, COLOR_GRID, 1);
            }
        }
    }

    for(int t = 0; t < TYPE_COUNT; t++){
        if(actual_max[t] > 5.0){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    if(pheromones[t][y][x] >= actual_max[t] * 0.6){
                        Rect rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                        Scalar base_c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
                        Scalar outline_c = Scalar(max(0.0, base_c[0] - 120), max(0.0, base_c[1] - 120), max(0.0, base_c[2] - 120));
                        rectangle(img, rect, outline_c, 2);
                    }
                }
            }
        }
    }

    for(int t = 0; t < TYPE_COUNT; t++){
        for(int e = 0; e < end_pos[t].size(); e++){
            if(has_global_best[t][e]){
                Scalar base_c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
                Scalar line_c = Scalar(max(0.0, base_c[0] - 140), max(0.0, base_c[1] - 140), max(0.0, base_c[2] - 140));
                for(int j = 0; j < global_best_ants[t][e].path_length - 1; j++){
                    Point p1(global_best_ants[t][e].path_x[j] * CELL_SIZE + CELL_SIZE / 2,
                        global_best_ants[t][e].path_y[j] * CELL_SIZE + CELL_SIZE / 2);
                    Point p2(global_best_ants[t][e].path_x[j + 1] * CELL_SIZE + CELL_SIZE / 2,
                        global_best_ants[t][e].path_y[j + 1] * CELL_SIZE + CELL_SIZE / 2);
                    line(img, p1, p2, line_c, 3, LINE_AA);
                }
            }
        }
    }

    for(int t = 0; t < TYPE_COUNT; t++){
        Scalar c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
        for(const auto &sp : start_pos[t]) rectangle(img, Rect(sp.x * CELL_SIZE, sp.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, FILLED);
        for(const auto &ep : end_pos[t]) rectangle(img, Rect(ep.x * CELL_SIZE, ep.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, 2);
    }

    char info[256];
    sprintf(info, "GA Gen: %d | Iter: %d | Score: %.1f", gen_num, iteration, (global_best_combined_score == 1e9 ? 0 : global_best_combined_score));
    putText(img, info, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);

    sprintf(info, "A:%.1f B:%.1f r:%.2f Q:%.0f Tw:%.1f Cw:%.1f", dna.alpha, dna.beta, dna.rho, dna.Q, dna.turn_w, dna.clear_w);
    putText(img, info, Point(10, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1);

    imshow("GA-Optimized ACO Routing", img);
    waitKey(1);

    // --- 新增：當視覺化播放到最後一格時，存檔並記錄在 img 資料夾 ---
    if(iteration == MAX_ITER){
        string filename = "img/iter_" + to_string(gen_num) + ".png";
        imwrite(filename, img);
    }
}

double ACO_Environment::run_aco(const Chromosome &dna, bool visualize, int gen_num){
    pheromones.assign(TYPE_COUNT, vector<vector<double>>(rows, vector<double>(cols, 0.1)));
    global_best_combined_score = 1e9;
    for(int t = 0; t < TYPE_COUNT; t++){
        for(int e = 0; e < MAX_ENDPOINTS; e++){
            has_global_best[t][e] = false;
        }
    }

    CUDA_Ant h_ants[ANT_COUNT];
    vector<double> flat_pheromones(TYPE_COUNT * rows * cols);
    int blockSize = 256;
    int numBlocks = (ANT_COUNT + blockSize - 1) / blockSize;

    // 強制統一隨機數種子，保證背景評估與視覺化的分數絕對一致！
    init_rand_kernel << <numBlocks, blockSize, 0, stream >> > (d_rand_states, 12345, ANT_COUNT);
    cudaStreamSynchronize(stream);

    // --- 偵錯計數器 ---
    int debug_valid_path_found = 0;
    int debug_stuck_ants_total = 0;
    int debug_step_limit_reached = 0;

    for(int iter = 1; iter <= MAX_ITER; iter++){
        for(int i = 0; i < ANT_COUNT; i++){
            CUDA_Ant ant;
            if(i % 2 == 0 && !start_pos[GAS].empty() && !end_pos[GAS].empty()){
                int gas_idx = i / 2;
                int num_starts = start_pos[GAS].size();
                int num_ends = end_pos[GAS].size();
                int s_idx = (gas_idx / num_ends) % num_starts;
                int e_idx = gas_idx % num_ends;
                ant.type = GAS;
                ant.pos_x = start_pos[GAS][s_idx].x; ant.pos_y = start_pos[GAS][s_idx].y;
                ant.target_x = end_pos[GAS][e_idx].x; ant.target_y = end_pos[GAS][e_idx].y;
                ant.target_idx = e_idx;
            }
            else if(!start_pos[ELEC].empty() && !end_pos[ELEC].empty()){
                int elec_idx = i / 2;
                int num_starts = start_pos[ELEC].size();
                int num_ends = end_pos[ELEC].size();
                int s_idx = (elec_idx / num_ends) % num_starts;
                int e_idx = elec_idx % num_ends;
                ant.type = ELEC;
                ant.pos_x = start_pos[ELEC][s_idx].x; ant.pos_y = start_pos[ELEC][s_idx].y;
                ant.target_x = end_pos[ELEC][e_idx].x; ant.target_y = end_pos[ELEC][e_idx].y;
                ant.target_idx = e_idx;
            }
            ant.last_dir_x = 0; ant.last_dir_y = 0;
            ant.path_x[0] = ant.pos_x; ant.path_y[0] = ant.pos_y;
            ant.path_length = 1;
            ant.reached_end = false;
            ant.stuck = false;
            h_ants[i] = ant;
        }

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    flat_pheromones[(t * rows + y) * cols + x] = pheromones[t][y][x];
                }
            }
        }

        cudaMemcpyAsync(d_ants, h_ants, ANT_COUNT * sizeof(CUDA_Ant), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pheromones, flat_pheromones.data(), TYPE_COUNT * rows * cols * sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(d_visited, 0, ANT_COUNT * rows * cols * sizeof(bool), stream);

        ant_movement_kernel << <numBlocks, blockSize, 0, stream >> > (
            d_ants, d_visited, d_pheromones, d_grid_map_shared,
            rows, cols, d_rand_states,
            MAX_PHERO, dna.alpha, dna.beta, ANT_COUNT
            );

        cudaMemcpyAsync(h_ants, d_ants, ANT_COUNT * sizeof(CUDA_Ant), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for(int i = 0; i < ANT_COUNT; i++){
            if(h_ants[i].stuck) debug_stuck_ants_total++;
            if(!h_ants[i].reached_end && !h_ants[i].stuck) debug_step_limit_reached++;
        }

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    pheromones[t][y][x] *= (1.0 - dna.rho);
                    if(pheromones[t][y][x] < 0.1) pheromones[t][y][x] = 0.1;
                }
            }
        }

        double best_score[TYPE_COUNT][MAX_ENDPOINTS];
        int best_ant_idx[TYPE_COUNT][MAX_ENDPOINTS];
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < MAX_ENDPOINTS; e++){
                best_score[t][e] = 1e9;
                best_ant_idx[t][e] = -1;
            }
        }

        for(int i = 0; i < ANT_COUNT; i++){
            CUDA_Ant &ant = h_ants[i];
            if(ant.reached_end){
                int turn_count = 0;
                if(ant.path_length > 2){
                    for(int j = 1; j < ant.path_length - 1; j++){
                        int dir1_x = ant.path_x[j] - ant.path_x[j - 1];
                        int dir1_y = ant.path_y[j] - ant.path_y[j - 1];
                        int dir2_x = ant.path_x[j + 1] - ant.path_x[j];
                        int dir2_y = ant.path_y[j + 1] - ant.path_y[j];
                        if(dir1_x != dir2_x || dir1_y != dir2_y) turn_count++;
                    }
                }

                double clearance_penalty = 0.0;
                for(int j = 0; j < ant.path_length; j++){
                    int px = ant.path_x[j], py = ant.path_y[j];
                    for(int t = 0; t < TYPE_COUNT; t++){
                        if(t != ant.type){
                            for(int dy = -SAFE_DISTANCE; dy <= SAFE_DISTANCE; dy++){
                                for(int dx = -SAFE_DISTANCE; dx <= SAFE_DISTANCE; dx++){
                                    int ny = py + dy, nx = px + dx;
                                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                                        double dist = sqrt(dx * dx + dy * dy);
                                        if(dist <= SAFE_DISTANCE){
                                            if(pheromones[t][ny][nx] > 1.0){
                                                double severity = (SAFE_DISTANCE - dist + 1.0);
                                                clearance_penalty += pheromones[t][ny][nx] * severity;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                double path_score = ant.path_length + (turn_count * dna.turn_w) + (clearance_penalty * dna.clear_w);

                if(path_score < best_score[ant.type][ant.target_idx]){
                    best_score[ant.type][ant.target_idx] = path_score;
                    best_ant_idx[ant.type][ant.target_idx] = i;
                }
            }
        }

        bool all_endpoints_reached = true;
        double current_combined_score = 0.0;
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < end_pos[t].size(); e++){
                if(best_ant_idx[t][e] == -1){
                    all_endpoints_reached = false;
                }
                else{
                    current_combined_score += best_score[t][e];
                }
            }
        }

        if(all_endpoints_reached){
            int overlap_count = 0;
            for(int ge = 0; ge < end_pos[GAS].size(); ge++){
                CUDA_Ant &gas_ant = h_ants[best_ant_idx[GAS][ge]];
                for(int ee = 0; ee < end_pos[ELEC].size(); ee++){
                    CUDA_Ant &elec_ant = h_ants[best_ant_idx[ELEC][ee]];
                    for(int j = 0; j < gas_ant.path_length; j++){
                        for(int k = 0; k < elec_ant.path_length; k++){
                            double dist = sqrt(pow((double) (gas_ant.path_x[j] - elec_ant.path_x[k]), 2) +
                                pow((double) (gas_ant.path_y[j] - elec_ant.path_y[k]), 2));
                            if(dist <= SAFE_DISTANCE){
                                overlap_count++;
                            }
                        }
                    }
                }
            }

            double evaluated_score = current_combined_score + (overlap_count * 1000.0);
            debug_valid_path_found++;

            if(evaluated_score < global_best_combined_score){
                global_best_combined_score = evaluated_score;
                for(int t = 0; t < TYPE_COUNT; t++){
                    for(int e = 0; e < end_pos[t].size(); e++){
                        global_best_ants[t][e] = h_ants[best_ant_idx[t][e]];
                        has_global_best[t][e] = true;
                    }
                }
            }
        }

        double elite_contribution_weight = 0.5;
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < end_pos[t].size(); e++){
                if(best_ant_idx[t][e] != -1){
                    CUDA_Ant &elite_ant = h_ants[best_ant_idx[t][e]];
                    double contribution = (dna.Q / best_score[t][e]) * elite_contribution_weight;

                    for(int j = 0; j < elite_ant.path_length; j++){
                        int px = elite_ant.path_x[j], py = elite_ant.path_y[j];
                        for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                            for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                                int ny = py + dy, nx = px + dx;
                                if(ny >= 0 && ny < rows && nx >= 0 && nx < cols && grid_map_cpu[ny][nx] != '1'){
                                    double dist_val = sqrt(dx * dx + dy * dy);
                                    if(dist_val <= DIFFUSION_RADIUS){
                                        double intensity = 1.0 - (dist_val / (DIFFUSION_RADIUS + 1.0));
                                        pheromones[t][ny][nx] += contribution * intensity;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < end_pos[t].size(); e++){
                if(has_global_best[t][e]){
                    CUDA_Ant &gb_ant = global_best_ants[t][e];
                    double total_ends = end_pos[GAS].size() + end_pos[ELEC].size();
                    double gb_score = (global_best_combined_score == 1e9) ? 100.0 : (global_best_combined_score / total_ends);
                    double contribution = (dna.Q / gb_score) * (1.0 - elite_contribution_weight);

                    for(int j = 0; j < gb_ant.path_length; j++){
                        int px = gb_ant.path_x[j], py = gb_ant.path_y[j];
                        for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                            for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                                int ny = py + dy, nx = px + dx;
                                if(ny >= 0 && ny < rows && nx >= 0 && nx < cols && grid_map_cpu[ny][nx] != '1'){
                                    double dist_val = sqrt(dx * dx + dy * dy);
                                    if(dist_val <= DIFFUSION_RADIUS){
                                        double intensity = 1.0 - (dist_val / (DIFFUSION_RADIUS + 1.0));
                                        pheromones[t][ny][nx] += contribution * intensity;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    if(pheromones[t][y][x] > MAX_PHERO) pheromones[t][y][x] = MAX_PHERO;
                }
            }
        }

        if(visualize && (iter % 100 == 0 || iter == MAX_ITER)){
            drawSimulation(iter, dna, gen_num);
        }
    }

    if(!visualize && global_best_combined_score == 1e9){
        std::lock_guard<std::mutex> lock(print_mutex);
        cout << "\n[偵錯] 基因 (Env ID: " << env_id << ") 發生 1e9 錯誤！參數(A:" << dna.alpha << ", B:" << dna.beta << ")" << endl;
        cout << "  - 包含重疊/扣分之成功路徑組合總次數: " << debug_valid_path_found << " / " << MAX_ITER << endl;
        cout << "  - 螞蟻卡死在死胡同總次數: " << debug_stuck_ants_total << endl;
        cout << "  - 螞蟻繞圈用盡 800 步的總次數: " << debug_step_limit_reached << endl;
    }

    return global_best_combined_score;
}

void init_shared_data(){
    for(int t = 0; t < TYPE_COUNT; t++){
        start_pos[t].clear(); end_pos[t].clear();
    }
    vector<char> flat_grid(rows * cols);
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            flat_grid[y * cols + x] = grid_map_cpu[y][x];
            if(grid_map_cpu[y][x] == 'g') start_pos[GAS].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'G') end_pos[GAS].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'e') start_pos[ELEC].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'E') end_pos[ELEC].push_back(Point(x, y));
        }
    }
    cudaMalloc(&d_grid_map_shared, rows * cols * sizeof(char));
    cudaMemcpy(d_grid_map_shared, flat_grid.data(), rows * cols * sizeof(char), cudaMemcpyHostToDevice);
}

// ===========================================
// === 主程式：執行多執行緒基因演算法 (GA) ===
// ===========================================
int main(){
    srand(time(NULL));
    init_shared_data();

    // --- 自動建立 img 目錄以儲存圖片 ---
    fs::create_directories("img");


    vector<ACO_Environment *> envs(POP_SIZE);
    for(int i = 0; i < POP_SIZE; i++){
        envs[i] = new ACO_Environment(i, time(NULL) + i);
    }

    vector<Chromosome> population(POP_SIZE);

    cout << "=== 初始化 GA 族群 ===" << endl;

    // --- 關鍵保底機制：植入已知能產生合法路徑的參數 ---
    population[0].alpha = 1.0;
    population[0].beta = 2.0;
    population[0].rho = 0.1;
    population[0].Q = 100.0;
    population[0].turn_w = 10.0;
    population[0].clear_w = 5.0;

    for(int i = 1; i < POP_SIZE; i++){
        population[i].alpha = randDouble(0.5, 2.5);
        population[i].beta = randDouble(1.5, 5.0);
        population[i].rho = randDouble(0.01, 0.2);
        population[i].Q = randDouble(50.0, 200.0);
        population[i].turn_w = randDouble(1.0, 15.0);
        population[i].clear_w = randDouble(1.0, 10.0);
    }

    // --- 準備 Gnuplot Pipe (清空歷史 Log) ---
    ofstream log_file("ga_log.txt", ios::trunc);
    log_file << "Gen Best Median\n";
    log_file.close();

    FILE *gp;
#ifdef _WIN32
    gp = _popen("gnuplot", "w"); // 移除 -persist，因為我們改用圖片輸出
#else
    // 只需清空基本路徑，並以背景模式執行
    gp = popen("env -u LD_LIBRARY_PATH gnuplot", "w");
#endif

    if(gp){
        // --- 核心修改：強制使用 PNG 後端，絕對不開啟任何 GUI 視窗 ---
        fprintf(gp, "set term pngcairo size 800,400 font 'sans,12'\n"); // 若報錯 pngcairo，可改成 set term png
        fprintf(gp, "set title 'GA Optimization Progress (Best & Median Score)'\n");
        fprintf(gp, "set xlabel 'Generation'\n");
        fprintf(gp, "set ylabel 'Path Score'\n");
        fprintf(gp, "set grid\n");
    }
    else{
        cout << "[警告] 無法呼叫 Gnuplot。\0" << endl;
    }

    for(int gen = 1; gen <= GA_GENERATIONS; gen++){
        cout << "\n[ 世代 " << gen << " / " << GA_GENERATIONS << " 開始 GPU 平行運算評估 ]" << endl;

        vector<thread> workers;
        for(int i = 0; i < POP_SIZE; i++){
            if(population[i].fitness < 0){
                workers.emplace_back([&, i](){
                    population[i].fitness = envs[i]->run_aco(population[i], false, gen);
                });
            }
            else{
                cout << "  保留菁英個體 " << i << " (成績: " << population[i].fitness << ")" << endl;
            }
        }

        for(auto &w : workers){
            w.join();
        }

        sort(population.begin(), population.end(), [](const Chromosome &a, const Chromosome &b){
            return a.fitness < b.fitness;
        });

        double best_score = population[0].fitness;
        double median_score = population[POP_SIZE / 2].fitness;
        cout << ">> 第 " << gen << " 代最佳適應度: " << best_score << " | 中位數: " << median_score << endl;

        // --- 將成績寫入 Log 並讓 Gnuplot 背景繪圖 ---
        log_file.open("ga_log.txt", ios::app);
        log_file << gen << " " << best_score << " " << median_score << "\n";
        log_file.close();

        if(gp){
            // 指定輸出到 img 資料夾
            fprintf(gp, "set output 'img/ga_progress.png'\n");
            fprintf(gp, "plot 'ga_log.txt' skip 1 using 1:2 with linespoints lw 2 lc rgb 'red' title 'Best Score', \\\n");
            fprintf(gp, "     '' skip 1 using 1:3 with linespoints lw 2 lc rgb 'blue' title 'Median Score'\n");
            fflush(gp); // 強制執行命令

            // 給 Gnuplot 一點微小的時間完成圖片寫入
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // --- 核心修改：讓 OpenCV 接管折線圖的顯示！ ---
            Mat plot_img = imread("img/ga_progress.png");
            if(!plot_img.empty()){
                imshow("GA Progress Chart", plot_img);
                waitKey(1); // 刷新視窗
            }
        }

        // --- 視覺化播放 ---
        cout << ">> 啟動視覺化：播放第 " << gen << " 代最佳基因之 ACO 搜尋過程..." << endl;
        envs[0]->run_aco(population[0], true, gen);

        if(gen < GA_GENERATIONS){
            for(int i = POP_SIZE / 2; i < POP_SIZE; i++){
                int p1 = rand() % (POP_SIZE / 2);
                int p2 = rand() % (POP_SIZE / 2);

                population[i].alpha = (population[p1].alpha + population[p2].alpha) / 2.0;
                population[i].beta = (population[p1].beta + population[p2].beta) / 2.0;
                population[i].rho = (population[p1].rho + population[p2].rho) / 2.0;
                population[i].Q = (population[p1].Q + population[p2].Q) / 2.0;
                population[i].turn_w = (population[p1].turn_w + population[p2].turn_w) / 2.0;
                population[i].clear_w = (population[p1].clear_w + population[p2].clear_w) / 2.0;

                if(randDouble(0, 1) < 0.2) population[i].alpha = randDouble(0.5, 2.5);
                if(randDouble(0, 1) < 0.2) population[i].beta = randDouble(1.5, 5.0);
                if(randDouble(0, 1) < 0.2) population[i].rho = randDouble(0.01, 0.2);
                if(randDouble(0, 1) < 0.2) population[i].Q = randDouble(50.0, 200.0);
                if(randDouble(0, 1) < 0.2) population[i].turn_w = randDouble(1.0, 15.0);
                if(randDouble(0, 1) < 0.2) population[i].clear_w = randDouble(1.0, 10.0);

                population[i].fitness = -1.0;
            }
        }
    }

    if(gp){
#ifdef _WIN32
        _pclose(gp);
#else
        pclose(gp);
#endif
    }

    cout << "\n=========================================" << endl;
    cout << "GA 訓練結束！歷史最佳參數組合：" << endl;
    cout << "ALPHA: " << population[0].alpha << ", BETA: " << population[0].beta << endl;
    cout << "rho: " << population[0].rho << ", Q: " << population[0].Q << endl;
    cout << "Turn Penalty: " << population[0].turn_w << ", Clearance Penalty: " << population[0].clear_w << endl;
    cout << "=========================================" << endl;

    // --- 駐留視窗 ---
    cout << "\n請在顯示的影像視窗中按下任意鍵，以關閉程式..." << endl;
    waitKey(0);

    for(int i = 0; i < POP_SIZE; i++) delete envs[i];
    cudaFree(d_grid_map_shared);

    return 0;
}