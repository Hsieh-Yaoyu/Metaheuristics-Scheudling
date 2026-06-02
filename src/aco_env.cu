#include "aco_env.cuh"
#include "map_data.cuh"
#include "kernels.cuh"
#include <mutex>

static std::mutex print_mutex;

ACO_Environment::ACO_Environment(int id, unsigned long seed) : env_id(id){
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

ACO_Environment::~ACO_Environment(){
    cudaFree(d_pheromones);
    cudaFree(d_visited);
    cudaFree(d_ants);
    cudaFree(d_rand_states);
    cudaStreamDestroy(stream);
}

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

    // --- 修改：在畫面加上 Dr (Diffusion Radius) 顯示 ---
    sprintf(info, "A:%.1f B:%.1f r:%.2f Q:%.0f Tw:%.1f Cw:%.1f Dr:%d",
        dna.alpha, dna.beta, dna.rho, dna.Q, dna.turn_w, dna.clear_w, (int) round(dna.diffusion_rad));
    putText(img, info, Point(10, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50, 50, 50), 1);

    imshow("GA-Optimized ACO Routing", img);
    waitKey(1);

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

    init_rand_kernel << <numBlocks, blockSize, 0, stream >> > (d_rand_states, 12345, ANT_COUNT);
    cudaStreamSynchronize(stream);

    int debug_valid_path_found = 0;
    int debug_stuck_ants_total = 0;
    int debug_step_limit_reached = 0;

    // --- 修改：取得目前基因的擴散半徑 (限制最少為 1 格) ---
    int diff_rad = max(1, (int) round(dna.diffusion_rad));

    auto start_time = chrono::high_resolution_clock::now();
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

                    // --- 修改：套用擴散半徑基因 ---
                    for(int j = 0; j < elite_ant.path_length; j++){
                        int px = elite_ant.path_x[j], py = elite_ant.path_y[j];
                        for(int dy = -diff_rad; dy <= diff_rad; dy++){
                            for(int dx = -diff_rad; dx <= diff_rad; dx++){
                                int ny = py + dy, nx = px + dx;
                                if(ny >= 0 && ny < rows && nx >= 0 && nx < cols && grid_map_cpu[ny][nx] != '1'){
                                    double dist_val = sqrt(dx * dx + dy * dy);
                                    if(dist_val <= diff_rad){
                                        double intensity = 1.0 - (dist_val / (diff_rad + 1.0));
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

                    // --- 修改：套用擴散半徑基因 ---
                    for(int j = 0; j < gb_ant.path_length; j++){
                        int px = gb_ant.path_x[j], py = gb_ant.path_y[j];
                        for(int dy = -diff_rad; dy <= diff_rad; dy++){
                            for(int dx = -diff_rad; dx <= diff_rad; dx++){
                                int ny = py + dy, nx = px + dx;
                                if(ny >= 0 && ny < rows && nx >= 0 && nx < cols && grid_map_cpu[ny][nx] != '1'){
                                    double dist_val = sqrt(dx * dx + dy * dy);
                                    if(dist_val <= diff_rad){
                                        double intensity = 1.0 - (dist_val / (diff_rad + 1.0));
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

        if(visualize && (iter % 10 == 0 || iter == MAX_ITER)){
            drawSimulation(iter, dna, gen_num);
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    double elapsed_sec = chrono::duration<double>(end_time - start_time).count();
    // std::cout << "  - GPU 評估完成，耗時: " << elapsed_sec << " 秒" << std::endl;

    if(!visualize && global_best_combined_score == 1e9){
        std::lock_guard<std::mutex> lock(print_mutex);
        cout << "\n[偵錯] 基因 (Env ID: " << env_id << ") 發生 1e9 錯誤！參數(A:" << dna.alpha << ", B:" << dna.beta << ")" << endl;
        cout << "  - 包含重疊/扣分之成功路徑組合總次數: " << debug_valid_path_found << " / " << MAX_ITER << endl;
        cout << "  - 螞蟻卡死在死胡同總次數: " << debug_stuck_ants_total << endl;
        cout << "  - 螞蟻繞圈用盡 800 步的總次數: " << debug_step_limit_reached << endl;
    }

    return global_best_combined_score;
}
