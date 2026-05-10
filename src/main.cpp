#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// --- 演算法參數 ---
const double ALPHA = 4.0;       // 提高費洛蒙(含排斥力)的影響力 (原為1.0)
const double BETA = 1.0;        // 稍微降低啟發式(最短距離)的絕對吸引力 (原為2.0)
const double rho = 0.1;
const double Q = 200.0;
const double se_phero = 80;
const int ANT_COUNT = 80;       // 每種管線的螞蟻數量
const int MAX_ITER = 500;       // 最大迭代次數
const int CELL_SIZE = 25;       // 視覺化網格大小
const int DIFFUSION_RADIUS = 2; // 費洛蒙擴散半徑 

// --- 視覺化顏色設定 (OpenCV 使用 BGR 格式) ---
const Scalar COLOR_BG = Scalar(255, 255, 255);     // 背景 (白)
const Scalar COLOR_WALL = Scalar(50, 50, 50);        // 牆壁 (深灰)
const Scalar COLOR_GRID = Scalar(200, 200, 200);     // 網格線 (淺灰)

const Scalar COLOR_GAS = Scalar(0, 255, 255);       // 瓦斯管 (橘色)
const Scalar COLOR_ELEC = Scalar(255, 225, 0);       // 電管 (黃色)

// --- 管線屬性定義  ---
enum PipeType{ GAS = 0, ELEC = 1 };
const int TYPE_COUNT = 2;
// 排斥矩陣：GAS遇GAS相吸(1), 遇ELEC排斥(-1)；ELEC同理
int preference[TYPE_COUNT][TYPE_COUNT] = {
    { 1, -1 },
    {-1,  1 }
};

// --- 地圖定義 ---
// g/G: 瓦斯管(Gas)起終點, e/E: 電管(Elec)起終點, 1: 牆壁, 0: 空氣
// --- 地圖定義 ---
const vector<string> grid_map = {
    "g00000000000000000000000",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "000E00000000000000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "000000000000000000000000",
    "000000000000000000000000",
    "000000000000000000000000",
    "000000000000000000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "0000000000000e0000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "000000000000000000000000",
    "0000000G0000000000000000"
};

int rows = grid_map.size();
int cols = grid_map[0].size();
Point start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];

// 費洛蒙場: pheromones[type][y][x]
vector<vector<vector<double>>> pheromones;

// 擴展後的螞蟻結構
struct Ant{
    int type;           // 管線屬性
    Point pos;
    Point last_dir;     // 記錄上一次的行進方向，用於轉彎評估 
    vector<Point> path;
    vector<vector<bool>> visited;
    bool reached_end;
    bool stuck;

    Ant(int t){
        type = t;
        pos = start_pos[t];
        last_dir = Point(0, 0);
        path.push_back(pos);
        visited.assign(rows, vector<bool>(cols, false));
        visited[pos.y][pos.x] = true;
        reached_end = false;
        stuck = false;
    }
};

double getHeuristic(Point p, int type){
    double dist = sqrt(pow(p.x - end_pos[type].x, 2) + pow(p.y - end_pos[type].y, 2));
    return 1.0 / (dist + 1.0);
}

void init(){
    pheromones.assign(TYPE_COUNT, vector<vector<double>>(rows, vector<double>(cols, 0.1)));
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            if(grid_map[y][x] == 'g') start_pos[GAS] = Point(x, y);
            if(grid_map[y][x] == 'G') end_pos[GAS] = Point(x, y);
            if(grid_map[y][x] == 'e') start_pos[ELEC] = Point(x, y);
            if(grid_map[y][x] == 'E') end_pos[ELEC] = Point(x, y);
        }
    }

    // --- 新增：為起點與終點建立「初始排斥防護罩」 ---
    // 這會與我們之前的排斥係數 gamma 完美配合
    for(int t = 0; t < TYPE_COUNT; t++){
        Point terminals[2] = { start_pos[t], end_pos[t] };

        for(int i = 0; i < 2; i++){
            Point p = terminals[i];
            pheromones[t][p.y][p.x] = se_phero; // 中心點給予極高濃度

            // 依據擴散半徑向外遞減，形成力場
            for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                    int ny = p.y + dy;
                    int nx = p.x + dx;
                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                        double dist_val = sqrt(dx * dx + dy * dy);
                        if(dist_val <= DIFFUSION_RADIUS && dist_val > 0){
                            // 距離越近，防護罩濃度越高
                            pheromones[t][ny][nx] += se_phero / (dist_val + 1.0);
                        }
                    }
                }
            }
        }
    }
}

void drawSimulation(const vector<Ant> &ants, int iteration, double max_phero[]){
    // 使用全域背景色初始化影像
    Mat img(rows * CELL_SIZE, cols * CELL_SIZE, CV_8UC3, COLOR_BG);

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            Rect rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            if(grid_map[y][x] == '1'){
                rectangle(img, rect, COLOR_WALL, FILLED);
            }
            else{
                double gas_int = pheromones[GAS][y][x] / max_phero[GAS];
                double elec_int = pheromones[ELEC][y][x] / max_phero[ELEC];
                double total_int = gas_int + elec_int;

                // --- 泛用型費洛蒙混色邏輯 ---
                if(total_int > 0.05){
                    // 計算兩種費洛蒙在當前格子的比例
                    double w_gas = gas_int / total_int;
                    double w_elec = elec_int / total_int;

                    // 依比例混和出目標管線顏色
                    double target_b = COLOR_GAS[0] * w_gas + COLOR_ELEC[0] * w_elec;
                    double target_g = COLOR_GAS[1] * w_gas + COLOR_ELEC[1] * w_elec;
                    double target_r = COLOR_GAS[2] * w_gas + COLOR_ELEC[2] * w_elec;

                    // 根據總濃度與背景色進行線性漸層混色
                    double blend = min(1.0, total_int); // 濃度最高不超過 1.0 (實色)
                    int final_b = COLOR_BG[0] * (1.0 - blend) + target_b * blend;
                    int final_g = COLOR_BG[1] * (1.0 - blend) + target_g * blend;
                    int final_r = COLOR_BG[2] * (1.0 - blend) + target_r * blend;

                    rectangle(img, rect, Scalar(final_b, final_g, final_r), FILLED);
                }

                rectangle(img, rect, COLOR_GRID, 1); // 繪製網格
            }
        }
    }

    // 繪製起終點
    for(int t = 0; t < TYPE_COUNT; t++){
        Scalar c = (t == GAS) ? COLOR_GAS : COLOR_ELEC;
        rectangle(img, Rect(start_pos[t].x * CELL_SIZE, start_pos[t].y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, FILLED);
        rectangle(img, Rect(end_pos[t].x * CELL_SIZE, end_pos[t].y * CELL_SIZE, CELL_SIZE, CELL_SIZE), c, 2);
    }

    // 繪製螞蟻
    for(const auto &ant : ants){
        if(!ant.reached_end){
            Point center(ant.pos.x * CELL_SIZE + CELL_SIZE / 2, ant.pos.y * CELL_SIZE + CELL_SIZE / 2);
            Scalar ant_color = (ant.type == GAS) ? COLOR_GAS : COLOR_ELEC;

            // 為了讓螞蟻在費洛蒙上更明顯，稍微把螞蟻畫深一點點 (可選)
            Scalar draw_color = Scalar(max(0.0, ant_color[0] - 30), max(0.0, ant_color[1] - 30), max(0.0, ant_color[2] - 30));

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
        for(int i = 0; i < ANT_COUNT; i++){
            ants.push_back(Ant(GAS));
            ants.push_back(Ant(ELEC));
        }

        // 正規化費洛蒙基準，用於計算擴散反應參數 gamma 
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

                        // --- 新增：判斷是否為「其他管線的起終點」 ---
                        bool is_other_terminal = false;
                        if(ant.type == GAS && (cell == 'e' || cell == 'E')) is_other_terminal = true;
                        if(ant.type == ELEC && (cell == 'g' || cell == 'G')) is_other_terminal = true;


                        if(!is_wall && !is_other_terminal && !ant.visited[next_p.y][next_p.x]){

                            // 1. 計算費洛蒙強度 gamma 
                            double gamma = 1.0;
                            for(int t = 0; t < TYPE_COUNT; t++){
                                if(t != ant.type){
                                    // 只有當對方的費洛蒙明顯高於底線(0.1)時，才視為有管線經過
                                    if(pheromones[t][next_p.y][next_p.x] > 0.15){
                                        double intensity = pheromones[t][next_p.y][next_p.x] / max_phero[t];

                                        // 乘上放大係數 (例如 5.0)，讓排斥力變得極度強烈
                                        gamma += preference[ant.type][t] * intensity * 10.0;
                                    }
                                }
                            }
                            // 若被強烈排斥，將機率降到極低 (原為0.01，改為0.0001讓螞蟻更不願意走)
                            gamma = max(0.0001, gamma);

                            // 2. 轉彎懲罰評估 
                            double bend_penalty = 1.0;
                            if(ant.last_dir != Point(0, 0) && dirs[i] != ant.last_dir){
                                bend_penalty = 0.5;
                            }

                            next_steps.push_back(next_p);
                            double p = pow(pheromones[ant.type][next_p.y][next_p.x] * gamma, ALPHA) * pow(getHeuristic(next_p, ant.type) * bend_penalty, BETA);
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

                    if(ant.pos == end_pos[ant.type]) ant.reached_end = true;
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

        // --- 改為以下這段「全範圍防護罩刷新」邏輯 ---
        // 確保起終點與其周圍的擴散力場永遠維持在一定的底線，不會被完全揮發
        for(int t = 0; t < TYPE_COUNT; t++){
            Point terminals[2] = { start_pos[t], end_pos[t] };
            for(int i = 0; i < 2; i++){
                Point p = terminals[i];

                // 1. 鎖定正中心點
                pheromones[t][p.y][p.x] = max(se_phero, pheromones[t][p.y][p.x]);

                // 2. 重新刷新周圍的擴散力場
                for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                    for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                        int ny = p.y + dy;
                        int nx = p.x + dx;
                        if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                            double dist_val = sqrt(dx * dx + dy * dy);
                            if(dist_val <= DIFFUSION_RADIUS && dist_val > 0){

                                // 計算該距離應有的基礎防護罩濃度 (數值可依需求微調)
                                double shield_val = (se_phero / 2.0) / (dist_val + 1.0);

                                // 使用 max 確保濃度不會低於防護罩底線
                                pheromones[t][ny][nx] = max(pheromones[t][ny][nx], shield_val);
                            }
                        }
                    }
                }
            }
        }

        // 費洛蒙線性擴散 (Dilation) 與 轉彎獎勵機制
        // 費洛蒙線性擴散 (Dilation) 與 轉彎/排斥 獎懲機制
        for(const auto &ant : ants){
            if(ant.reached_end){

                // 1. 計算該螞蟻最終路徑的轉彎次數
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

                // --- 新增：2. 計算路徑上踩到的「排斥性(其他管線)費洛蒙」總量 ---
                double enemy_phero_penalty = 0.0;
                for(const auto &p : ant.path){
                    for(int t = 0; t < TYPE_COUNT; t++){
                        // 檢查其他管線的費洛蒙
                        if(t != ant.type){
                            // 設定一個閥值 (例如 1.0)，只懲罰踩到對方「明顯軌跡」的行為，忽略極淡的擴散邊緣
                            if(pheromones[t][p.y][p.x] > 1.0){
                                enemy_phero_penalty += pheromones[t][p.y][p.x];
                            }
                        }
                    }
                }

                // 3. 綜合評分公式 (路徑長度 + 轉彎懲罰 + 排斥懲罰)
                const double turn_penalty_weight = 5.0;
                const double enemy_penalty_weight = 2.0; // 踩到對方費洛蒙的扣分權重 (可依需求微調)

                // path_score 越高代表路徑品質越差
                double path_score = ant.path.size() +
                    (turn_count * turn_penalty_weight) +
                    (enemy_phero_penalty * enemy_penalty_weight);

                // 4. 計算實際釋放的費洛蒙貢獻量 (分數越大，除出來的貢獻量就越低)
                double contribution = Q / path_score;

                // 進行擴散 (維持不變)
                for(const auto &p : ant.path){
                    for(int dy = -DIFFUSION_RADIUS; dy <= DIFFUSION_RADIUS; dy++){
                        for(int dx = -DIFFUSION_RADIUS; dx <= DIFFUSION_RADIUS; dx++){
                            int ny = p.y + dy;
                            int nx = p.x + dx;
                            if(ny >= 0 && ny < rows && nx >= 0 && nx < cols){
                                double dist_val = sqrt(dx * dx + dy * dy);
                                if(dist_val <= DIFFUSION_RADIUS){
                                    double intensity = 1.0 - (dist_val / (DIFFUSION_RADIUS + 1.0)); // 隨距離遞減 
                                    pheromones[ant.type][ny][nx] += contribution * intensity;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    cout << "Simulation Finished." << endl;
    waitKey(0);
    return 0;
}