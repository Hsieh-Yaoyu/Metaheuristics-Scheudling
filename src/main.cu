#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <curand_kernel.h>

using namespace std;
using namespace cv;

// --- 演算法參數 ---
const double ALPHA = 3.0;
const double BETA = 2.0;
const double rho = 0.05;
const double Q = 2000.0;
const double se_phero = 80;
const int ANT_COUNT = 40;
const double MAX_PHERO = 500.0;
const int MAX_ITER = 20000;
const int CELL_SIZE = 25;
const int MAX_PATH_LEN = 800;

// --- 修改 1：拆分防護罩與路徑擴散的參數 ---
const int TERMINAL_SHIELD_RADIUS = 2; // 起終點的保護防護罩 (永遠維持不變)
const int INITIAL_PATH_DIFFUSION = 3; // 螞蟻路徑的初始擴散範圍 (將隨退火遞減至0)

int last_diff = INITIAL_PATH_DIFFUSION;

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
    "00000000000000000g000000",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "00000000000000E000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "000000000000000000000000",
    "000000000000000000000000",
    "000000000000000000000000",
    "000000000000000000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "0000000000G0000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "000000000000000000000000",
    "e00000000000000000000000"
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
    int path_x[MAX_PATH_LEN];
    int path_y[MAX_PATH_LEN];
    int path_length;
    bool reached_end;
    bool stuck;
};

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
    int *start_x, int *start_y, int *start_counts,
    int *end_x, int *end_y, int *end_counts,
    double max_phero_cap, double alpha, double beta, int shield_radius) // 替換為 shield_radius
{
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

                // Home Zone 檢查 (使用固定的 shield_radius)
                bool in_home_zone = false;
                for(int s = 0; s < start_counts[ant.type]; s++){
                    int sx = start_x[ant.type * 10 + s];
                    int sy = start_y[ant.type * 10 + s];
                    if(sqrt(pow((double) (nx - sx), 2) + pow((double) (ny - sy), 2)) <= shield_radius){
                        in_home_zone = true; break;
                    }
                }
                if(!in_home_zone){
                    for(int e = 0; e < end_counts[ant.type]; e++){
                        int ex = end_x[ant.type * 10 + e];
                        int ey = end_y[ant.type * 10 + e];
                        if(sqrt(pow((double) (nx - ex), 2) + pow((double) (ny - ey), 2)) <= shield_radius){
                            in_home_zone = true; break;
                        }
                    }
                }

                bool is_force_field = false;
                if(!in_home_zone){
                    for(int t = 0; t < 2; t++){
                        if(t != ant.type){
                            double other_p = pheromones[(t * rows + ny) * cols + nx];
                            double own_p = pheromones[(ant.type * rows + ny) * cols + nx];
                            if(other_p > max_phero_cap * 0.4 && own_p < 2.0){
                                is_force_field = true;
                            }
                        }
                    }
                }

                if(!is_wall && !is_other_terminal && !is_force_field && !visited[visited_offset + ny * cols + nx]){
                    double gamma = 1.0;
                    for(int t = 0; t < 2; t++){
                        if(t != ant.type){
                            double other_p = pheromones[(t * rows + ny) * cols + nx];
                            double own_p = pheromones[(ant.type * rows + ny) * cols + nx];
                            if(other_p > 1.0 && other_p > own_p){
                                gamma *= exp(-(other_p - own_p) / 10.0);
                            }
                        }
                    }
                    if(gamma < 0.0001) gamma = 0.0001;

                    double bend_penalty = 1.0;
                    if(ant.last_dir_x != 0 || ant.last_dir_y != 0){
                        if(dirs_x[i] != ant.last_dir_x || dirs_y[i] != ant.last_dir_y){
                            bend_penalty = 0.5;
                        }
                    }

                    double own_p = pheromones[(ant.type * rows + ny) * cols + nx];
                    double perceived_own_p = own_p > 150.0 ? 150.0 : own_p;

                    double dist_target = sqrt(pow((double) (nx - ant.target_x), 2) + pow((double) (ny - ant.target_y), 2));
                    double heuristic = 1.0 / (dist_target + 1.0);

                    double p = pow(perceived_own_p * gamma, alpha) * pow(heuristic * bend_penalty, beta);

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
            double r = curand_uniform_double(&local_rand) * prob_sum;
            double cumulative = 0.0;
            int chosen_idx = valid_count - 1;
            for(int i = 0; i < valid_count; i++){
                cumulative += probs[i];
                if(cumulative >= r){
                    chosen_idx = i;
                    break;
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

    for(int t = 0; t < TYPE_COUNT; t++){
        vector<Point> all_terminals;
        all_terminals.insert(all_terminals.end(), start_pos[t].begin(), start_pos[t].end());
        all_terminals.insert(all_terminals.end(), end_pos[t].begin(), end_pos[t].end());

        for(const auto &p : all_terminals){
            pheromones[t][p.y][p.x] = se_phero;
            for(int dy = -TERMINAL_SHIELD_RADIUS; dy <= TERMINAL_SHIELD_RADIUS; dy++){
                for(int dx = -TERMINAL_SHIELD_RADIUS; dx <= TERMINAL_SHIELD_RADIUS; dx++){
                    int ny = p.y + dy, nx = p.x + dx;
                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                        double dist_val = sqrt(dx * dx + dy * dy);
                        if(dist_val <= TERMINAL_SHIELD_RADIUS && dist_val > 0){
                            pheromones[t][ny][nx] += se_phero / (dist_val + 1.0);
                        }
                    }
                }
            }
        }
    }
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

    for(int t = 0; t < TYPE_COUNT; t++){
        Scalar c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
        for(const auto &sp : start_pos[t]) rectangle(img, Rect(sp.x * CELL_SIZE, sp.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, FILLED);
        for(const auto &ep : end_pos[t]) rectangle(img, Rect(ep.x * CELL_SIZE, ep.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, 2);
    }

    putText(img, "Iter: " + to_string(iteration), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    imshow("CUDA ACO Routing", img);
    waitKey(1);
}

int main(){
    init();
    mt19937 rng(time(NULL));
    uniform_real_distribution<double> dist(0.0, 1.0);

    CUDA_Ant h_ants[ANT_COUNT];
    vector<double> flat_pheromones(TYPE_COUNT * rows * cols);

    int blockSize = 256;
    int numBlocks = (ANT_COUNT + blockSize - 1) / blockSize;

    for(int iter = 1; iter <= MAX_ITER; iter++){
        for(int i = 0; i < ANT_COUNT; i++){
            CUDA_Ant ant;
            if(i % 2 == 0 && !start_pos[GAS].empty()){
                ant.type = GAS;
                
                ant.pos_x = start_pos[GAS][(rng()) % start_pos[GAS].size()].x;
                ant.pos_y = start_pos[GAS][(rng()) % start_pos[GAS].size()].y;
                ant.target_x = end_pos[GAS][(rng()) % end_pos[GAS].size()].x;
                ant.target_y = end_pos[GAS][(rng()) % end_pos[GAS].size()].y;
            }
            else if(!start_pos[ELEC].empty()){
                ant.type = ELEC;
                ant.pos_x = start_pos[ELEC][(rng()) % start_pos[ELEC].size()].x;
                ant.pos_y = start_pos[ELEC][(rng()) % start_pos[ELEC].size()].y;
                ant.target_x = end_pos[ELEC][(rng()) % end_pos[ELEC].size()].x;
                ant.target_y = end_pos[ELEC][(rng()) % end_pos[ELEC].size()].y;
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
            d_start_x, d_start_y, d_start_counts,
            d_end_x, d_end_y, d_end_counts,
            MAX_PHERO, ALPHA, BETA, TERMINAL_SHIELD_RADIUS
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

        for(int t = 0; t < TYPE_COUNT; t++){
            vector<Point> all_terminals;
            all_terminals.insert(all_terminals.end(), start_pos[t].begin(), start_pos[t].end());
            all_terminals.insert(all_terminals.end(), end_pos[t].begin(), end_pos[t].end());

            for(const auto &p : all_terminals){
                pheromones[t][p.y][p.x] = max(se_phero, pheromones[t][p.y][p.x]);
                for(int dy = -TERMINAL_SHIELD_RADIUS; dy <= TERMINAL_SHIELD_RADIUS; dy++){
                    for(int dx = -TERMINAL_SHIELD_RADIUS; dx <= TERMINAL_SHIELD_RADIUS; dx++){
                        int ny = p.y + dy, nx = p.x + dx;
                        if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                            double dist_val = sqrt(dx * dx + dy * dy);
                            if(dist_val <= TERMINAL_SHIELD_RADIUS && dist_val > 0){
                                double shield_val = se_phero / (dist_val + 1.0);
                                pheromones[t][ny][nx] = max(pheromones[t][ny][nx], shield_val);
                            }
                        }
                    }
                }
            }
        }

        // --- 修改 2：動態退火計算路徑的擴散半徑 ---
        // 隨迭代次數線性降溫。前 70% 的時間就會將半徑從 2 降至 0
        double temperature = 1.0 - min(1.0, (double) iter / (MAX_ITER * 0.7));
        int current_path_diffusion = round(INITIAL_PATH_DIFFUSION * temperature);

        if(current_path_diffusion != last_diff){
            printf("Current Diffusion：%d", current_path_diffusion);
            last_diff = current_path_diffusion;
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

                double enemy_phero_penalty = 0.0;
                for(int j = 0; j < ant.path_length; j++){
                    int px = ant.path_x[j], py = ant.path_y[j];
                    for(int t = 0; t < TYPE_COUNT; t++){
                        if(t != ant.type && pheromones[t][py][px] > 1.0){
                            enemy_phero_penalty += pheromones[t][py][px];
                        }
                    }
                }

                const double turn_penalty_weight = 2.0;
                const double enemy_penalty_weight = 3.0;

                double ideal_steps = 15 + abs(ant.path_x[0] - ant.target_x) + abs(ant.path_y[0] - ant.target_y);
                if(ideal_steps < 1.0) ideal_steps = 1.0;

                double raw_path_score = ant.path_length * 5.0 + (turn_count * turn_penalty_weight) + (enemy_phero_penalty * enemy_penalty_weight);
                double distance_ratio = ideal_steps / 20.0;
                double normalized_path_score = raw_path_score / distance_ratio;
                double contribution = Q / pow(normalized_path_score, 1.5);

                // --- 修改 3：套用退火後的動態擴散半徑 ---
                for(int j = 0; j < ant.path_length; j++){
                    int px = ant.path_x[j], py = ant.path_y[j];

                    // 使用 current_path_diffusion 來限制暈染範圍
                    for(int dy = -current_path_diffusion; dy <= current_path_diffusion; dy++){
                        for(int dx = -current_path_diffusion; dx <= current_path_diffusion; dx++){
                            int ny = py + dy, nx = px + dx;
                            if(ny >= 0 && ny < rows && nx >= 0 && nx < cols && grid_map_cpu[ny][nx] != '1'){
                                double dist_val = sqrt(dx * dx + dy * dy);

                                if(dist_val <= current_path_diffusion){
                                    // 確保當 diffusion 退火到 0 時，自身這格獲得 100% (1.0) 的費洛蒙釋放
                                    double intensity = 1.0;
                                    if(current_path_diffusion > 0){
                                        intensity = 1.0 - (dist_val / (current_path_diffusion + 1.0));
                                    }
                                    pheromones[ant.type][ny][nx] += contribution * intensity;
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
    waitKey(0);
    return 0;
}