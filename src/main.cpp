#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// --- 演算法參數 ---
const double ALPHA = 5.0;
const double BETA = 1.0;
const double rho = 0.1;
const double Q = 200.0;
const double se_phero = 80;
const int ANT_COUNT = 40;
const double MAX_PHERO = 500.0; // --- 新增：費洛蒙天花板 ---
const int MAX_ITER = 500;
const int CELL_SIZE = 25;
const int DIFFUSION_RADIUS = 1;

// --- 視覺化顏色設定 ---
const Scalar COLOR_BG = Scalar(255, 255, 255);
const Scalar COLOR_WALL = Scalar(50, 50, 50);
const Scalar COLOR_GRID = Scalar(200, 200, 200);

const Scalar COLOR_GAS = Scalar(0, 255, 255);
const Scalar COLOR_ELEC = Scalar(255, 225, 0);

// --- 管線屬性定義 ---
enum PipeType{ GAS = 0, ELEC = 1 };
const int TYPE_COUNT = 2;
int preference[TYPE_COUNT][TYPE_COUNT] = {
    { 1, -1 },
    {-1,  1 }
};

// --- 多端點地圖定義 ---
// 現在支援多個起點(g, e)與終點(G, E)
const vector<string> grid_map = {
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

int rows = grid_map.size();
int cols = grid_map[0].size();

// --- 修改1：將起終點改為動態陣列 ---
vector<Point> start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];
vector<vector<vector<double>>> pheromones;

struct Ant{
    int type;
    Point pos;
    Point last_dir;
    Point target_pos;
    vector<Point> path;
    vector<vector<bool>> visited;
    bool reached_end;
    bool stuck;

    // --- 修改2：建構子現在接收特定的出生起點 ---
    Ant(int t, Point start_p, Point end_p){
        type = t;
        pos = start_p;
        last_dir = Point(0, 0);
        path.push_back(pos);
        visited.assign(rows, vector<bool>(cols, false));
        visited[pos.y][pos.x] = true;
        reached_end = false;
        target_pos = end_p;
        stuck = false;
    }
};

// --- 修改3：動態尋找最近的終點 ---
double getHeuristic(Point p, Point target){
    double dist = sqrt(pow(p.x - target.x, 2) + pow(p.y - target.y, 2));
    return 1.0 / (dist + 1.0);
}

void init(){
    pheromones.assign(TYPE_COUNT, vector<vector<double>>(rows, vector<double>(cols, 0.1)));

    // 清空並收集所有的起終點
    for(int t = 0; t < TYPE_COUNT; t++){
        start_pos[t].clear();
        end_pos[t].clear();
    }

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            if(grid_map[y][x] == 'g') start_pos[GAS].push_back(Point(x, y));
            if(grid_map[y][x] == 'G') end_pos[GAS].push_back(Point(x, y));
            if(grid_map[y][x] == 'e') start_pos[ELEC].push_back(Point(x, y));
            if(grid_map[y][x] == 'E') end_pos[ELEC].push_back(Point(x, y));
        }
    }

    // 為所有起點與終點建立「初始排斥防護罩」
    for(int t = 0; t < TYPE_COUNT; t++){
        vector<Point> all_terminals;
        all_terminals.insert(all_terminals.end(), start_pos[t].begin(), start_pos[t].end());
        all_terminals.insert(all_terminals.end(), end_pos[t].begin(), end_pos[t].end());

        for(const auto &p : all_terminals){
            pheromones[t][p.y][p.x] = se_phero;

            for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                    int ny = p.y + dy;
                    int nx = p.x + dx;
                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                        double dist_val = sqrt(dx * dx + dy * dy);
                        if(dist_val <= DIFFUSION_RADIUS && dist_val > 0){
                            pheromones[t][ny][nx] += se_phero / (dist_val + 1.0);
                        }
                    }
                }
            }
        }
    }
}

// 注意：函式簽名維持不變，以相容你的 main 呼叫
// 注意：函式簽名維持不變，以相容你的 main 呼叫
void drawSimulation(const vector<Ant> &ants, int iteration, double max_phero[]){
    Mat img(rows * CELL_SIZE, cols * CELL_SIZE, CV_8UC3, COLOR_BG);

    // --- 視覺化優化核心 ---
    double display_max[TYPE_COUNT] = { 0.1, 0.1 };
    double actual_max[TYPE_COUNT] = { 0.1, 0.1 }; // 新增：用於尋找真正的最高濃度路徑

    // 1. 掃描取得真正的最高濃度與視覺天花板
    for(int t = 0; t < TYPE_COUNT; t++){
        for(int y = 0; y < rows; y++){
            for(int x = 0; x < cols; x++){
                if(pheromones[t][y][x] > actual_max[t]){
                    actual_max[t] = pheromones[t][y][x];
                }
            }
        }
        // 視覺天花板依舊鎖定在 100，保證擴散邊緣的底色飽和度
        display_max[t] = min(actual_max[t], 100.0);
    }

    // 2. 繪製網格底色 (維持原有的混色邏輯)
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            Rect rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            if(grid_map[y][x] == '1'){
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

                    double target_b = COLOR_GAS[0] * w_gas + COLOR_ELEC[0] * w_elec;
                    double target_g = COLOR_GAS[1] * w_gas + COLOR_ELEC[1] * w_elec;
                    double target_r = COLOR_GAS[2] * w_gas + COLOR_ELEC[2] * w_elec;

                    int final_b = COLOR_BG[0] * (1.0 - max_int) + target_b * max_int;
                    int final_g = COLOR_BG[1] * (1.0 - max_int) + target_g * max_int;
                    int final_r = COLOR_BG[2] * (1.0 - max_int) + target_r * max_int;

                    rectangle(img, rect, Scalar(final_b, final_g, final_r), FILLED);
                }
                rectangle(img, rect, COLOR_GRID, 1);
            }
        }
    }

    // --- 新增：3. 繪製最高濃度路徑的外框 (管線實體化) ---
    for(int t = 0; t < TYPE_COUNT; t++){
        // 只有當該管線的最高濃度明顯大於起點防護罩(例如 >85.0)時，才代表真正有路徑成形
        if(actual_max[t] > 85.0){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){

                    // 閥值設定：濃度達到最高值的 60% 視為主幹線
                    if(pheromones[t][y][x] >= actual_max[t] * 0.6){
                        Rect rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

                        // 計算較深的外框顏色，讓管線看起來更有立體感 (原色減去 120)
                        Scalar base_c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
                        Scalar outline_c = Scalar(max(0.0, base_c[0] - 120), max(0.0, base_c[1] - 120), max(0.0, base_c[2] - 120));

                        // 繪製粗外框 (Thickness = 2)
                        rectangle(img, rect, outline_c, 2);
                    }
                }
            }
        }
    }

    // 4. 繪製所有的起終點
    for(int t = 0; t < TYPE_COUNT; t++){
        Scalar c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
        for(const auto &sp : start_pos[t]){
            rectangle(img, Rect(sp.x * CELL_SIZE, sp.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, FILLED);
        }
        for(const auto &ep : end_pos[t]){
            rectangle(img, Rect(ep.x * CELL_SIZE, ep.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, 2);
        }
    }

    // 5. 繪製螞蟻
    for(const auto &ant : ants){
        if(!ant.reached_end){
            Point center(ant.pos.x * CELL_SIZE + CELL_SIZE / 2, ant.pos.y * CELL_SIZE + CELL_SIZE / 2);
            Scalar ant_color = (ant.type == GAS) ? COLOR_GAS : COLOR_ELEC;
            Scalar draw_color = Scalar(max(0.0, ant_color[0] - 50), max(0.0, ant_color[1] - 50), max(0.0, ant_color[2] - 50));
            circle(img, center, CELL_SIZE / 4, draw_color, FILLED);
        }
    }

    putText(img, "Iter: " + to_string(iteration), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    imshow("Multi-Pipe ACO Routing", img);
    waitKey(1);
}

int main(){
    init();
    mt19937 rng(12345);
    uniform_real_distribution<double> dist(0.0, 1.0);
    Point dirs[4] = { Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0) };

    for(int iter = 1; iter <= MAX_ITER; iter++){
        vector<Ant> ants;

        // --- 修改5：將螞蟻平均分配給各個起點 ---
// --- 修改5：將螞蟻平均分配給各個起點，並同時分配專屬的目標終點 ---
        for(int i = 0; i < ANT_COUNT; i++){
            if(!start_pos[GAS].empty() && !end_pos[GAS].empty()){
                Point gas_start = start_pos[GAS][i % start_pos[GAS].size()];
                Point gas_end = end_pos[GAS][i % end_pos[GAS].size()]; // 依序分配目標終點
                ants.push_back(Ant(GAS, gas_start, gas_end));
            }
            if(!start_pos[ELEC].empty() && !end_pos[ELEC].empty()){
                Point elec_start = start_pos[ELEC][i % start_pos[ELEC].size()];
                Point elec_end = end_pos[ELEC][i % end_pos[ELEC].size()]; // 依序分配目標終點
                ants.push_back(Ant(ELEC, elec_start, elec_end));
            }
        }

        double max_phero[TYPE_COUNT] = { 0.1, 0.1 };
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    if(pheromones[t][y][x] > max_phero[t]) max_phero[t] = pheromones[t][y][x];
                }
            }
        }

        bool all_done = false;
        while(!all_done){
            all_done = true;
            for(auto &ant : ants){
                if(ant.reached_end || ant.stuck) continue;
                all_done = false;

                vector<Point> next_steps;
                vector<double> probabilities;
                double prob_sum = 0.0;

                for(int i = 0; i < 4; i++){
                    Point next_p = ant.pos + dirs[i];
                    if(next_p.x >= 0 && next_p.x < cols && next_p.y >= 0 && next_p.y < rows){

                        char cell = grid_map[next_p.y][next_p.x];
                        bool is_wall = (cell == '1');

                        bool is_other_terminal = false;
                        if(ant.type == GAS && (cell == 'e' || cell == 'E')) is_other_terminal = true;
                        if(ant.type == ELEC && (cell == 'g' || cell == 'G')) is_other_terminal = true;

                        // --- 修改6：檢查是否進入「任何一個」自己的起終點無敵領域 ---
                        bool in_home_zone = false;
                        for(const auto &sp : start_pos[ant.type]){
                            if(sqrt(pow(next_p.x - sp.x, 2) + pow(next_p.y - sp.y, 2)) <= DIFFUSION_RADIUS){
                                in_home_zone = true; break;
                            }
                        }
                        if(!in_home_zone){
                            for(const auto &ep : end_pos[ant.type]){
                                if(sqrt(pow(next_p.x - ep.x, 2) + pow(next_p.y - ep.y, 2)) <= DIFFUSION_RADIUS){
                                    in_home_zone = true; break;
                                }
                            }
                        }

                        // 在 main 迴圈檢查方向處
                        bool is_force_field = false;
                        if(!in_home_zone){
                            for(int t = 0; t < TYPE_COUNT; t++){
                                if(t != ant.type){
                                    double other_p = pheromones[t][next_p.y][next_p.x];
                                    double own_p = pheromones[ant.type][next_p.y][next_p.x];

                                    // 修改：只有對方濃度極高(>天花板的40%)，且我方幾乎沒走過時，才視為物理牆壁
                                    // 這讓螞蟻能依靠軟性排斥力(gamma)自然繞道，而不是在起點就被強制卡死
                                    if(other_p > MAX_PHERO * 0.4 && own_p < 2.0){
                                        is_force_field = true;
                                    }
                                }
                            }
                        }

                        if(!is_wall && !is_other_terminal && !is_force_field && !ant.visited[next_p.y][next_p.x]){

                            double gamma = 1.0;
                            for(int t = 0; t < TYPE_COUNT; t++){
                                if(t != ant.type){
                                    double other_p = pheromones[t][next_p.y][next_p.x];
                                    double own_p = pheromones[ant.type][next_p.y][next_p.x];
                                    if(other_p > 1.0 && other_p > own_p){
                                        gamma *= exp(-(other_p - own_p) / 10.0);
                                    }
                                }
                            }
                            gamma = max(0.0001, gamma);

                            double bend_penalty = 1.0;
                            if(ant.last_dir != Point(0, 0) && dirs[i] != ant.last_dir){
                                bend_penalty = 0.5;
                            }

                            next_steps.push_back(next_p);

                            // --- 核心修復：加入「自身費洛蒙感知天花板」 ---
                            // 限制螞蟻對自身費洛蒙的最大感知濃度 (例如 10.0)
                            // 這樣就能避免被自己起點高達 80 的防護罩引力給困住，讓啟發式(BETA)能發揮作用把螞蟻拉出門
                            double perceived_own_p = min(pheromones[ant.type][next_p.y][next_p.x], 10.0);

                            // 使用 perceived_own_p 來計算機率，取代原本的絕對濃度
                            double p = pow(perceived_own_p * gamma, ALPHA) * pow(getHeuristic(next_p, ant.target_pos) * bend_penalty, BETA);

                            probabilities.push_back(p);
                            prob_sum += p;
                        }
                    }
                }

                if(next_steps.empty()){
                    ant.stuck = true;
                }
                else{
                    double r = dist(rng) * prob_sum;
                    double cumulative = 0.0;
                    Point chosen_step = next_steps.back();

                    for(size_t i = 0; i < next_steps.size(); i++){
                        cumulative += probabilities[i];
                        if(cumulative >= r){
                            chosen_step = next_steps[i];
                            break;
                        }
                    }

                    ant.last_dir = Point(chosen_step.x - ant.pos.x, chosen_step.y - ant.pos.y);
                    ant.pos = chosen_step;
                    ant.path.push_back(ant.pos);
                    ant.visited[ant.pos.y][ant.pos.x] = true;

                    // --- 修改7：檢查是否抵達「任何一個」自己的終點 ---
                    if(ant.pos == ant.target_pos){
                        ant.reached_end = true;
                        break;
                    }
                }
            }
            drawSimulation(ants, iter, max_phero);
        }

        // 揮發
        for(int t = 0; t < TYPE_COUNT; t++){
            for(int y = 0; y < rows; y++){
                for(int x = 0; x < cols; x++){
                    pheromones[t][y][x] *= (1.0 - rho);
                    if(pheromones[t][y][x] < 0.1) pheromones[t][y][x] = 0.1;
                }
            }
        }

        // --- 修改8：刷新所有起終點的防護罩 ---
        for(int t = 0; t < TYPE_COUNT; t++){
            vector<Point> all_terminals;
            all_terminals.insert(all_terminals.end(), start_pos[t].begin(), start_pos[t].end());
            all_terminals.insert(all_terminals.end(), end_pos[t].begin(), end_pos[t].end());

            for(const auto &p : all_terminals){
                pheromones[t][p.y][p.x] = max(se_phero, pheromones[t][p.y][p.x]);

                for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                    for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                        int ny = p.y + dy;
                        int nx = p.x + dx;
                        if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                            double dist_val = sqrt(dx * dx + dy * dy);
                            if(dist_val <= DIFFUSION_RADIUS && dist_val > 0){
                                double shield_val = se_phero / (dist_val + 1.0);
                                pheromones[t][ny][nx] = max(pheromones[t][ny][nx], shield_val);
                            }
                        }
                    }
                }
            }
        }

        // 費洛蒙線性擴散 (Dilation) 與 轉彎/排斥 獎懲機制
        for(const auto &ant : ants){
            if(ant.reached_end){

                int turn_count = 0;
                if(ant.path.size() > 2){
                    for(size_t i = 1; i < ant.path.size() - 1; i++){
                        Point dir1 = Point(ant.path[i].x - ant.path[i - 1].x, ant.path[i].y - ant.path[i - 1].y);
                        Point dir2 = Point(ant.path[i + 1].x - ant.path[i].x, ant.path[i + 1].y - ant.path[i].y);
                        if(dir1 != dir2){
                            turn_count++;
                        }
                    }
                }

                double enemy_phero_penalty = 0.0;
                for(const auto &p : ant.path){
                    for(int t = 0; t < TYPE_COUNT; t++){
                        if(t != ant.type){
                            if(pheromones[t][p.y][p.x] > 1.0){
                                enemy_phero_penalty += pheromones[t][p.y][p.x];
                            }
                        }
                    }
                }

                const double turn_penalty_weight = 2.0;
                const double enemy_penalty_weight = 5.0;


                Point start_p = ant.path.front();
                double ideal_steps = abs(start_p.x - ant.target_pos.x) + abs(start_p.y - ant.target_pos.y);
                if(ideal_steps < 1.0) ideal_steps = 1.0; // 避免除以零

                double raw_path_score = ant.path.size() * 3 +
                    (turn_count * turn_penalty_weight) +
                    (enemy_phero_penalty * enemy_penalty_weight);

                double distance_ratio = ideal_steps / 20.0;

                double normalized_path_score = raw_path_score / distance_ratio;

                double contribution = Q / normalized_path_score;

                for(const auto &p : ant.path){
                    for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                        for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                            int ny = p.y + dy;
                            int nx = p.x + dx;
                            if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                                double dist_val = sqrt(dx * dx + dy * dy);
                                if(dist_val <= DIFFUSION_RADIUS){
                                    double intensity = 1.0 - (dist_val / (DIFFUSION_RADIUS + 1.0));
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
    }

    cout << "Simulation Finished." << endl;
    waitKey(0);
    return 0;
}