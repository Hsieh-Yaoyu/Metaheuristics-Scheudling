#include "config.h"
#include "map_data.cuh"
#include "aco_env.cuh"
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char *argv[]){
    Chromosome dna;
    string map_filename = "data/map.txt";
    bool run_20_times = false; // 新增：是否連續執行 20 次的旗標

    // 檢查是否有透過指令列輸入參數
    // 現在需要至少 9 個參數: 程式名 + 7個參數 + true/false
    if(argc >= 9){
        dna.alpha = stod(argv[1]);
        dna.beta = stod(argv[2]);
        dna.rho = stod(argv[3]);
        dna.Q = stod(argv[4]);
        dna.turn_w = stod(argv[5]);
        dna.clear_w = stod(argv[6]);
        dna.diffusion_rad = stod(argv[7]);

        string flag = argv[8];
        if(flag == "true" || flag == "1") run_20_times = true;

        if(argc >= 10) map_filename = argv[9]; // 第9個參數允許輸入自訂地圖

        cout << ">> 載入自訂參數測試模式..." << endl;
    }
    else{
        cout << "使用方法: ./ACO_Runner <alpha> <beta> <rho> <Q> <turn_w> <clear_w> <diff_rad> <run_20(true/false)> [map_path]" << endl;
        cout << ">> 未偵測到完整輸入，使用預設保底參數測試模式..." << endl;
        dna = { 1.0, 2.0, 0.1, 100.0, 10.0, 5.0, 2.0, -1.0 };
    }

    cout << ">> 載入地圖: " << map_filename << endl;
    init_shared_data(map_filename);

    fs::create_directories("img");
    fs::create_directories("data"); // 確保 data 資料夾存在以寫入 CSV

    if(run_20_times){
        // ========================================================
        // 模式 A：連續執行 20 次測試，並輸出 CSV 與統計結果
        // ========================================================
        cout << "\n=========================================" << endl;
        cout << ">> 啟動 20 次重複測試模式 (使用 GPU)..." << endl;
        cout << "=========================================" << endl;

        ofstream csv_file("data/aco_20_tests.csv", ios::trunc);
        csv_file << "test_idx,score\n";

        vector<double> scores;
        auto batch_start = chrono::high_resolution_clock::now();

        for(int i = 0; i < 20; i++){
            // 每回合給予完全不同的隨機數種子，保證實驗獨立性
            ACO_Environment env(0, time(NULL) + i * 1000);

            // 執行 ACO，關閉單次的視覺化以加快速度 (傳入 use_gpu = true)
            double s = env.run_aco(dna, false, 0, true);
            scores.push_back(s);

            // 寫入 CSV 檔案
            csv_file << i << "," << s << "\n";
            csv_file.flush();
            cout << "   - Test " << i << " 完成，得分: " << s << endl;
        }

        auto batch_end = chrono::high_resolution_clock::now();
        double total_time = chrono::duration<double>(batch_end - batch_start).count();

        // --- 統計數據結算 ---
        double min_s = *min_element(scores.begin(), scores.end());
        double max_s = *max_element(scores.begin(), scores.end());
        double sum = accumulate(scores.begin(), scores.end(), 0.0);
        double avg = sum / scores.size();

        double variance = 0.0;
        for(double s : scores){
            variance += (s - avg) * (s - avg);
        }
        double sd = sqrt(variance / scores.size());

        cout << "\n=========================================" << endl;
        cout << ">> 20 次測試統計結果 (總耗時: " << total_time << " 秒)" << endl;
        cout << "   Min: " << min_s << " | Avg: " << avg << " | Max: " << max_s << " | SD: " << sd << endl;
        cout << ">> 詳細數據已儲存於 data/aco_20_tests.csv" << endl;
        cout << "=========================================" << endl;

        csv_file.close();

        // 結算完畢後，再跑一次並彈出視窗讓使用者觀看最終的走法
        cout << "\n>> 啟動最後一次視覺化展示..." << endl;
        ACO_Environment env_vis(0, time(NULL));
        env_vis.run_aco(dna, true, 0, true);
        waitKey(0);

    }
    else{
        // ========================================================
        // 模式 B：原本的 CPU 與 GPU 運算時間對比模式
        // ========================================================
        ACO_Environment env(0, time(NULL));

        cout << "=========================================" << endl;
        cout << ">> 1. 啟動 CPU 多核心運算測試 (MAX_ITER: " << MAX_ITER << ")..." << endl;
        auto start_cpu = chrono::high_resolution_clock::now();

        // 傳入 false 代表使用 CPU
        double cpu_score = env.run_aco(dna, false, 0, false);

        auto end_cpu = chrono::high_resolution_clock::now();
        double cpu_time = chrono::duration<double>(end_cpu - start_cpu).count();
        cout << "   - CPU 耗時: " << cpu_time << " 秒 (得分: " << cpu_score << ")" << endl;


        cout << "\n>> 2. 啟動 GPU CUDA 運算測試 (MAX_ITER: " << MAX_ITER << ")..." << endl;
        auto start_gpu = chrono::high_resolution_clock::now();

        // 傳入 true 代表使用 GPU
        double gpu_score = env.run_aco(dna, false, 0, true);

        auto end_gpu = chrono::high_resolution_clock::now();
        double gpu_time = chrono::duration<double>(end_gpu - start_gpu).count();
        cout << "   - GPU 耗時: " << gpu_time << " 秒 (得分: " << gpu_score << ")" << endl;
        cout << "=========================================" << endl;

        // 計算加速比
        cout << "💡 結論：GPU 比 CPU 快了 " << (cpu_time / gpu_time) << " 倍！" << endl;

        // 最後用 GPU 跑一次並顯示視覺化畫面
        cout << "\n>> 啟動視覺化..." << endl;
        env.run_aco(dna, true, 0, true);
        waitKey(0);
    }

    free_shared_data();
    return 0;
}