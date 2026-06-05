#include "config.h"
#include "map_data.cuh"
#include "aco_env.cuh"
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

double randDouble(double minVal, double maxVal){
    return minVal + (double) rand() / RAND_MAX * (maxVal - minVal);
}

int main(){
    srand(time(NULL));

    // 建立必要的基礎資料夾
    fs::create_directories("data");
    fs::create_directories("img");

    // ========================================================
    // 準備三個統計輸出的 CSV 檔案
    // ========================================================
    // 1. 總計統計檔 (min, avg, max, sd)
    ofstream csv_stat("data/statistics.csv", ios::trunc);
    csv_stat << "map,min,avg,max,sd\n";

    // 2. 歷代收斂分數檔 (iter_1 ~ iter_N)
    ofstream csv_iter("data/iteration_scores.csv", ios::trunc);
    csv_iter << "data,test";
    for(int i = 1; i <= GA_GENERATIONS; i++){
        csv_iter << ",iter_" << i;
    }
    csv_iter << "\n";

    // 3. 最佳參數與分數檔 (alpha, beta, rho, Q, T_w, C_w, Dr, score)
    ofstream csv_param("data/best_parameters.csv", ios::trunc);
    csv_param << "data,alpha,beta,rho,Q,T_w,C_w,Dr,score\n";
    // ========================================================

    int map_idx = 0;
    const bool is_gpu = false; // 固定使用 GPU 進行批次測試

    // 不斷尋找下一個 mapX.txt 直到找不到為止
    while(true){
        string map_filename = "data/map" + to_string(map_idx) + ".txt";
        string map_name = "map" + to_string(map_idx);

        if(!fs::exists(map_filename)){
            if(map_idx == 0){
                cout << "[提示] 找不到 " << map_filename << "，請確認資料夾內有地圖檔案！" << endl;
            }
            else{
                cout << "\n==================================================" << endl;
                cout << ">> 所有地圖處理完畢！共完成 " << map_idx << " 張地圖的批次測試。" << endl;
                cout << ">> 統計結果已儲存於 data 資料夾下的 3 個 CSV 檔案中！" << endl;
                cout << "==================================================" << endl;
            }
            break; // 找不到下一張地圖，結束批次測試
        }

        cout << "\n==================================================" << endl;
        cout << "開始進行地圖批次測試: " << map_filename << endl;
        cout << "==================================================" << endl;

        // 建立專屬於這張地圖的輸出資料夾
        string map_dir = "data/" + map_name;
        fs::create_directories(map_dir);

        // 載入這張地圖到 GPU 全域共用記憶體
        init_shared_data(map_filename);

        vector<double> scores; // 儲存 20 次執行的最終 Best Score

        auto batch_start_time = chrono::high_resolution_clock::now();
        for(int test_idx = 0; test_idx < 20; ++test_idx){
            auto test_start_time = chrono::high_resolution_clock::now();
            cout << "  [執行 " << map_filename << " - Test " << test_idx << " / 19] ..." << endl;

            unsigned long current_test_seed = time(NULL) + test_idx * 1000;

            vector<ACO_Environment *> envs(POP_SIZE);
            for(int i = 0; i < POP_SIZE; i++){
                envs[i] = new ACO_Environment(i, current_test_seed);
            }

            vector<Chromosome> population(POP_SIZE);
            population[0] = { 1.0, 2.0, 0.1, 100.0, 10.0, 5.0, 2.0, -1.0 }; // 保底菁英
            for(int i = 1; i < POP_SIZE; i++){
                population[i] = {
                    randDouble(0.5, 2.5), randDouble(1.5, 5.0), randDouble(0.01, 0.2),
                    randDouble(50.0, 200.0), randDouble(1.0, 15.0), randDouble(1.0, 10.0),
                    randDouble(1.0, 5.0), -1.0
                };
            }

            ofstream log_file("ga_log.txt", ios::trunc);
            log_file << "Gen Best Median\n";
            log_file.close();

            FILE *gp;
#ifdef _WIN32
            gp = _popen("gnuplot", "w");
#else
            gp = popen("env -u LD_LIBRARY_PATH gnuplot", "w");
#endif

            if(gp){
                fprintf(gp, "set term pngcairo size 800,400 font 'sans,12'\n");
                fprintf(gp, "set title 'GA Optimization Progress (Map %d, Test %d)'\n", map_idx, test_idx);
                fprintf(gp, "set xlabel 'Generation'\n");
                fprintf(gp, "set ylabel 'Path Score'\n");
                fprintf(gp, "set grid\n");
            }

            // 用來儲存這個 Test 中，每一個 Generation 的 Best Score
            vector<double> gen_best_scores;

            // --- 開始 GA 演化迴圈 ---
            for(int gen = 1; gen <= GA_GENERATIONS; gen++){
                vector<thread> workers;
                auto start_time = chrono::high_resolution_clock::now();
                for(int i = 0; i < POP_SIZE; i++){
                    if(population[i].fitness < 0){
                        workers.emplace_back([&, i](){
                            population[i].fitness = envs[i]->run_aco(population[i], false, gen, is_gpu);
                        });
                    }
                }
                for(auto &w : workers) w.join();
                auto end_time = chrono::high_resolution_clock::now();
                chrono::duration<double> gen_duration = end_time - start_time;
                cout << ">> 世代 " << gen << " 評估完成，耗時: " << gen_duration.count() << " 秒" << endl;

                sort(population.begin(), population.end(), [](const Chromosome &a, const Chromosome &b){
                    return a.fitness < b.fitness;
                });

                double best_score = population[0].fitness;
                double median_score = population[POP_SIZE / 2].fitness;

                // 記錄這一代的最佳分數到陣列中
                gen_best_scores.push_back(best_score);

                log_file.open("ga_log.txt", ios::app);
                log_file << gen << " " << best_score << " " << median_score << "\n";
                log_file.close();

                if(gp){
                    string curve_path = map_dir + "/test" + to_string(test_idx) + "_curve.png";
                    fprintf(gp, "set output '%s'\n", curve_path.c_str());
                    fprintf(gp, "plot 'ga_log.txt' skip 1 using 1:2 with linespoints lw 2 lc rgb 'red' title 'Best Score', \\\n");
                    fprintf(gp, "     '' skip 1 using 1:3 with linespoints lw 2 lc rgb 'blue' title 'Median Score'\n");
                    fflush(gp);
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                bool is_last_gen = (gen == GA_GENERATIONS);
                if(is_last_gen){
                    // 確保使用相同的運算設備畫圖
                    envs[0]->run_aco(population[0], true, gen, is_gpu);

                    string src_img = "img/iter_" + to_string(gen) + ".png";
                    string dst_img = map_dir + "/test" + to_string(test_idx) + "_img.png";
                    if(fs::exists(src_img)){
                        fs::copy_file(src_img, dst_img, fs::copy_options::overwrite_existing);
                    }
                }

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
                        population[i].diffusion_rad = (population[p1].diffusion_rad + population[p2].diffusion_rad) / 2.0;

                        if(randDouble(0, 1) < 0.2) population[i].alpha = randDouble(0.5, 2.5);
                        if(randDouble(0, 1) < 0.2) population[i].beta = randDouble(1.5, 5.0);
                        if(randDouble(0, 1) < 0.2) population[i].rho = randDouble(0.01, 0.2);
                        if(randDouble(0, 1) < 0.2) population[i].Q = randDouble(50.0, 200.0);
                        if(randDouble(0, 1) < 0.2) population[i].turn_w = randDouble(1.0, 15.0);
                        if(randDouble(0, 1) < 0.2) population[i].clear_w = randDouble(1.0, 10.0);
                        if(randDouble(0, 1) < 0.2) population[i].diffusion_rad = randDouble(1.0, 5.0);

                        population[i].fitness = -1.0;
                    }
                }
            } // 結束單一 Test 的 GA 演化

            if(gp){
#ifdef _WIN32
                _pclose(gp);
#else
                pclose(gp);
#endif
            }

            scores.push_back(population[0].fitness);

            // ========================================================
            // 寫入 CSV 紀錄 (Iteration 分數與最佳參數)
            // ========================================================
            // 寫入 iteration_scores.csv
            csv_iter << map_name << "," << test_idx;
            for(double s : gen_best_scores){
                csv_iter << "," << s;
            }
            csv_iter << "\n";
            csv_iter.flush();

            // 寫入 best_parameters.csv
            csv_param << map_name << ","
                << population[0].alpha << ","
                << population[0].beta << ","
                << population[0].rho << ","
                << population[0].Q << ","
                << population[0].turn_w << ","
                << population[0].clear_w << ","
                << population[0].diffusion_rad << ","
                << population[0].fitness << "\n";
            csv_param.flush();
            // ========================================================

            for(int i = 0; i < POP_SIZE; i++) delete envs[i];
            auto test_end_time = chrono::high_resolution_clock::now();
            chrono::duration<double> test_duration = test_end_time - test_start_time;
            cout << "  -> Test " << test_idx << " 完成，Best Score: " << population[0].fitness << "，耗時: " << test_duration.count() << " 秒" << endl;
        } // 結束 20 次 Test

        auto batch_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> batch_duration = batch_end_time - batch_start_time;
        cout << "\n>> 地圖 " << map_filename << " 的 20 次測試全部完成，總耗時: " << batch_duration.count() << " 秒" << endl;

        // --- 統計並寫入總計 CSV (statistics.csv) ---
        double min_s = *min_element(scores.begin(), scores.end());
        double max_s = *max_element(scores.begin(), scores.end());
        double sum = accumulate(scores.begin(), scores.end(), 0.0);
        double avg = sum / scores.size();

        double variance = 0.0;
        for(double s : scores){
            variance += (s - avg) * (s - avg);
        }
        double sd = sqrt(variance / scores.size());

        csv_stat << map_name << "," << min_s << "," << avg << "," << max_s << "," << sd << "\n";
        csv_stat.flush();

        cout << "  -> 統計結果: Min=" << min_s << ", Avg=" << avg << ", Max=" << max_s << ", SD=" << sd << endl;

        free_shared_data();
        map_idx++;
    } // 結束 map_idx 迴圈

    csv_stat.close();
    csv_iter.close();
    csv_param.close();
    return 0;
}