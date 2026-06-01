#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>

using namespace std;
using namespace cv;

// --- 演算法參數 (完全保留您的設定) ---
const double ALPHA = 1.0;
const double BETA = 2.0;
const double rho = 0.1;
const double Q = 100.0;
const double turn_penalty_weight = 10.0;
const double clearance_penalty_weight = 5.0; // 距離過近的嚴厲懲罰倍率


const int ANT_COUNT = 40;
const double MAX_PHERO = 100.0;
const int MAX_ITER = 15000;
const int CELL_SIZE = 25;
const int MAX_PATH_LEN = 800;

// 論文中固定的費洛蒙膨脹半徑
// 論文中固定的費洛蒙膨脹半徑
const int DIFFUSION_RADIUS = 2;
const int MAX_ENDPOINTS = 10;

// --- 新增：管線之間必須保持的「安全距離」(網格數) ---
const int SAFE_DISTANCE = 2; // 你可以隨意修改這個值 (例如 1, 2, 3)

// --- 視覺化顏色設定 ---
const Scalar COLOR_BG = Scalar(255, 255, 255);
const Scalar COLOR_WALL = Scalar(50, 50, 50);
const Scalar COLOR_GRID = Scalar(200, 200, 200);
const Scalar COLOR_GAS = Scalar(0, 255, 255);
const Scalar COLOR_ELEC = Scalar(255, 225, 0);

enum PipeType{ GAS = 0, ELEC = 1 };
const int TYPE_COUNT = 2;

// --- 多端點地圖定義 ---
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

// --- CPU 資料結構 ---
vector<Point> start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];
vector<vector<vector<double>>> pheromones;

// --- GPU 專用螞蟻結構 ---
struct CUDA_Ant{
    int type;
    int pos_x, pos_y;
    int last_dir_x, last_dir_y;
    int target_x, target_y;
    int target_idx; // 新增：記錄這隻螞蟻負責的是該管線的「第幾個」終點
    int path_x[MAX_PATH_LEN];
    int path_y[MAX_PATH_LEN];
    int path_length;
    bool reached_end;
    bool stuck;
};

// 擴充：支援多終點的「全域最佳解」陣列
CUDA_Ant global_best_ants[TYPE_COUNT][MAX_ENDPOINTS];
bool has_global_best[TYPE_COUNT][MAX_ENDPOINTS];
double global_best_combined_score = 1e9;

// --- GPU 端指標 ---
char *d_grid_map;
double *d_pheromones;
bool *d_visited;
CUDA_Ant *d_ants;
curandState *d_rand_states;

int *d_start_x; int *d_start_y; int *d_start_counts;
int *d_end_x;   int *d_end_y;   int *d_end_counts;

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
    double max_phero_cap, double alpha, double beta){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
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

            // 微小擾動 (Epsilon-Greedy)
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


// --- Host 函式 ---
void init(){
    pheromones.assign(TYPE_COUNT, vector<vector<double>>(rows, vector<double>(cols, 0.1)));

    for(int t = 0; t < TYPE_COUNT; t++){
        start_pos[t].clear(); end_pos[t].clear();
        for(int e = 0; e < MAX_ENDPOINTS; e++){
            has_global_best[t][e] = false;
        }
    }

    vector<char> flat_grid(rows * cols);
    int h_start_x[TYPE_COUNT * 10] = { 0 }, h_start_y[TYPE_COUNT * 10] = { 0 }, h_start_counts[TYPE_COUNT] = { 0 };
    int h_end_x[TYPE_COUNT * 10] = { 0 }, h_end_y[TYPE_COUNT * 10] = { 0 }, h_end_counts[TYPE_COUNT] = { 0 };

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            flat_grid[y * cols + x] = grid_map_cpu[y][x];
            if(grid_map_cpu[y][x] == 'g'){ h_start_x[GAS * 10 + h_start_counts[GAS]] = x; h_start_y[GAS * 10 + h_start_counts[GAS]++] = y; start_pos[GAS].push_back(Point(x, y)); }
            if(grid_map_cpu[y][x] == 'G'){ h_end_x[GAS * 10 + h_end_counts[GAS]] = x;     h_end_y[GAS * 10 + h_end_counts[GAS]++] = y;     end_pos[GAS].push_back(Point(x, y)); }
            if(grid_map_cpu[y][x] == 'e'){ h_start_x[ELEC * 10 + h_start_counts[ELEC]] = x; h_start_y[ELEC * 10 + h_start_counts[ELEC]++] = y; start_pos[ELEC].push_back(Point(x, y)); }
            if(grid_map_cpu[y][x] == 'E'){ h_end_x[ELEC * 10 + h_end_counts[ELEC]] = x;     h_end_y[ELEC * 10 + h_end_counts[ELEC]++] = y;     end_pos[ELEC].push_back(Point(x, y)); }
        }
    }

    cudaMalloc(&d_grid_map, rows * cols * sizeof(char));
    cudaMemcpy(d_grid_map, flat_grid.data(), rows * cols * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc(&d_start_x, TYPE_COUNT * 10 * sizeof(int)); cudaMemcpy(d_start_x, h_start_x, TYPE_COUNT * 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_start_y, TYPE_COUNT * 10 * sizeof(int)); cudaMemcpy(d_start_y, h_start_y, TYPE_COUNT * 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_start_counts, TYPE_COUNT * sizeof(int)); cudaMemcpy(d_start_counts, h_start_counts, TYPE_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_end_x, TYPE_COUNT * 10 * sizeof(int)); cudaMemcpy(d_end_x, h_end_x, TYPE_COUNT * 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_end_y, TYPE_COUNT * 10 * sizeof(int)); cudaMemcpy(d_end_y, h_end_y, TYPE_COUNT * 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_end_counts, TYPE_COUNT * sizeof(int)); cudaMemcpy(d_end_counts, h_end_counts, TYPE_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pheromones, TYPE_COUNT * rows * cols * sizeof(double));
    cudaMalloc(&d_visited, ANT_COUNT * rows * cols * sizeof(bool));
    cudaMalloc(&d_ants, ANT_COUNT * sizeof(CUDA_Ant));
    cudaMalloc(&d_rand_states, ANT_COUNT * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (ANT_COUNT + blockSize - 1) / blockSize;
    init_rand_kernel << <numBlocks, blockSize >> > (d_rand_states, 12345, ANT_COUNT);
    cudaDeviceSynchronize();
}

void drawSimulation(int iteration){
    Mat img(rows * CELL_SIZE, cols * CELL_SIZE, CV_8UC3, COLOR_BG);
    double display_max[TYPE_COUNT] = { 0.1, 0.1 };
    double actual_max[TYPE_COUNT] = { 0.1, 0.1 };

    for(int t = 0; t < TYPE_COUNT; t++){
        for(int y = 0; y < rows; y++){
            for(int x = 0; x < cols; x++){
                if(pheromones[t][y][x] > actual_max[t]){
                    actual_max[t] = pheromones[t][y][x];
                }
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

    // 繪製全域最佳解，現在會為每個成功尋獲的終點繪製線條
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

    putText(img, "Iter: " + to_string(iteration), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    imshow("CUDA ACO Routing (Thesis Baseline)", img);
    waitKey(1);
}

int main(){
    init();

    CUDA_Ant h_ants[ANT_COUNT];
    vector<double> flat_pheromones(TYPE_COUNT * rows * cols);

    int blockSize = 256;
    int numBlocks = (ANT_COUNT + blockSize - 1) / blockSize;

    for(int iter = 1; iter <= MAX_ITER; iter++){
        for(int i = 0; i < ANT_COUNT; i++){
            CUDA_Ant ant;

            // --- 核心修改 1：利用排列組合，讓起點與終點完美交錯平均分配 ---
            if(i % 2 == 0 && !start_pos[GAS].empty() && !end_pos[GAS].empty()){
                int gas_idx = i / 2;
                int num_starts = start_pos[GAS].size();
                int num_ends = end_pos[GAS].size();

                int s_idx = (gas_idx / num_ends) % num_starts; // 取商數找起點
                int e_idx = gas_idx % num_ends;                // 取餘數找終點

                ant.type = GAS;
                ant.pos_x = start_pos[GAS][s_idx].x;
                ant.pos_y = start_pos[GAS][s_idx].y;
                ant.target_x = end_pos[GAS][e_idx].x;
                ant.target_y = end_pos[GAS][e_idx].y;
                ant.target_idx = e_idx; // 記錄負責的終點
            }
            else if(!start_pos[ELEC].empty() && !end_pos[ELEC].empty()){
                int elec_idx = i / 2;
                int num_starts = start_pos[ELEC].size();
                int num_ends = end_pos[ELEC].size();

                int s_idx = (elec_idx / num_ends) % num_starts;
                int e_idx = elec_idx % num_ends;

                ant.type = ELEC;
                ant.pos_x = start_pos[ELEC][s_idx].x;
                ant.pos_y = start_pos[ELEC][s_idx].y;
                ant.target_x = end_pos[ELEC][e_idx].x;
                ant.target_y = end_pos[ELEC][e_idx].y;
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

        cudaMemcpy(d_ants, h_ants, ANT_COUNT * sizeof(CUDA_Ant), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pheromones, flat_pheromones.data(), TYPE_COUNT * rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_visited, 0, ANT_COUNT * rows * cols * sizeof(bool));

        ant_movement_kernel << <numBlocks, blockSize >> > (
            d_ants, d_visited, d_pheromones, d_grid_map,
            rows, cols, d_rand_states,
            MAX_PHERO, ALPHA, BETA
            );
        cudaDeviceSynchronize();

        cudaMemcpy(h_ants, d_ants, ANT_COUNT * sizeof(CUDA_Ant), cudaMemcpyDeviceToHost);

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    pheromones[t][y][x] *= (1.0 - rho);
                    if(pheromones[t][y][x] < 0.1) pheromones[t][y][x] = 0.1;
                }
            }
        }

        // 建立陣列以記錄「每個」終點在該回合的最佳成績
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

                // --- 修改 1：將單點踩踏懲罰升級為「安全範圍(AoE)掃描懲罰」 ---
                double clearance_penalty = 0.0;
                for(int j = 0; j < ant.path_length; j++){
                    int px = ant.path_x[j], py = ant.path_y[j];
                    for(int t = 0; t < TYPE_COUNT; t++){
                        if(t != ant.type){
                            // 掃描以螞蟻為中心的 SAFE_DISTANCE 範圍
                            for(int dy = -SAFE_DISTANCE; dy <= SAFE_DISTANCE; dy++){
                                for(int dx = -SAFE_DISTANCE; dx <= SAFE_DISTANCE; dx++){
                                    int ny = py + dy, nx = px + dx;
                                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                                        double dist = sqrt(dx * dx + dy * dy);

                                        if(dist <= SAFE_DISTANCE){
                                            // 如果該格子有敵方的實體軌跡或防護罩
                                            if(pheromones[t][ny][nx] > 1.0){
                                                // 距離越近，嚴重程度(severity)越高，懲罰分數暴增
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


                // 綜合路徑成本：長度 + 轉彎懲罰 + 安全距離懲罰
                double path_score = ant.path_length +
                    (turn_count * turn_penalty_weight) +
                    (clearance_penalty * clearance_penalty_weight);
                
                // 針對這隻螞蟻負責的終點進行成績登記
                if(path_score < best_score[ant.type][ant.target_idx]){
                    best_score[ant.type][ant.target_idx] = path_score;
                    best_ant_idx[ant.type][ant.target_idx] = i;
                }
            }
        }

        // --- 核心修改 2：聯合評估所有終點是否發生重疊 ---
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

        // 如果本回合「所有起終點的組合」都順利抵達，才進行嚴格的重疊檢查
        // --- 修改 2：全域最佳解審查升級，距離過近直接判定為衝突 ---
        if(all_endpoints_reached){
            bool is_overlapping = false;

            // 掃描任何一條瓦斯路徑，是否與任何一條電管路徑距離過近
            for(int ge = 0; ge < end_pos[GAS].size(); ge++){
                CUDA_Ant &gas_ant = h_ants[best_ant_idx[GAS][ge]];
                for(int ee = 0; ee < end_pos[ELEC].size(); ee++){
                    CUDA_Ant &elec_ant = h_ants[best_ant_idx[ELEC][ee]];

                    for(int j = 0; j < gas_ant.path_length; j++){
                        for(int k = 0; k < elec_ant.path_length; k++){

                            // 計算兩條管線節點之間的幾何距離
                            double dist = sqrt(pow((double) (gas_ant.path_x[j] - elec_ant.path_x[k]), 2) +
                                pow((double) (gas_ant.path_y[j] - elec_ant.path_y[k]), 2));

                            // 若小於等於安全距離，即視為無效組合 (拒絕更新為全域最佳)
                            if(dist <= SAFE_DISTANCE){
                                is_overlapping = true;
                                break;
                            }
                        }
                        if(is_overlapping) break;
                    }
                    if(is_overlapping) break;
                }
                if(is_overlapping) break;
            }

            // 如果保持完美安全距離且總分更低，覆寫歷史最佳紀錄
            if(!is_overlapping && current_combined_score < global_best_combined_score){
                global_best_combined_score = current_combined_score;
                for(int t = 0; t < TYPE_COUNT; t++){
                    for(int e = 0; e < end_pos[t].size(); e++){
                        global_best_ants[t][e] = h_ants[best_ant_idx[t][e]];
                        has_global_best[t][e] = true;
                    }
                }
            }
        }

        // 1. 本回合菁英螞蟻釋放 (探索力：30%)

        const double elite_contribution_weight = 0.5;

        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < end_pos[t].size(); e++){
                if(best_ant_idx[t][e] != -1){
                    CUDA_Ant &elite_ant = h_ants[best_ant_idx[t][e]];
                    double contribution = (Q / best_score[t][e]) * elite_contribution_weight;

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

        // 2. 全域歷史最佳螞蟻釋放 (收斂力：70%)
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int e = 0; e < end_pos[t].size(); e++){
                if(has_global_best[t][e]){
                    CUDA_Ant &gb_ant = global_best_ants[t][e];
                    double total_ends = end_pos[GAS].size() + end_pos[ELEC].size();
                    double gb_score = (global_best_combined_score == 1e9) ? 100.0 : (global_best_combined_score / total_ends);
                    double contribution = (Q / gb_score) * (1.0 - elite_contribution_weight);

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
                    if(pheromones[t][y][x] > MAX_PHERO){
                        pheromones[t][y][x] = MAX_PHERO;
                    }
                }
            }
        }

        drawSimulation(iter);
    }

    cout << "Simulation Finished." << endl;
    drawSimulation(MAX_ITER);
    waitKey(0);
    return 0;
}