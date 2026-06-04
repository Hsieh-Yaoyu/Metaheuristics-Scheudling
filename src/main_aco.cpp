#include "config.h"
#include "map_data.cuh"
#include "aco_env.cuh"
#include <iostream>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char *argv[]){
    Chromosome dna;
    string map_filename = "data/map.txt";

    // 檢查是否有透過指令列輸入參數
    if(argc >= 8){
        dna.alpha = stod(argv[1]);
        dna.beta = stod(argv[2]);
        dna.rho = stod(argv[3]);
        dna.Q = stod(argv[4]);
        dna.turn_w = stod(argv[5]);
        dna.clear_w = stod(argv[6]);
        dna.diffusion_rad = stod(argv[7]);

        if(argc >= 9) map_filename = argv[8]; // 允許第8個參數為地圖路徑

        cout << ">> 載入自訂參數測試模式..." << endl;
    }
    else{
        cout << "使用方法: ./ACO_Runner <alpha> <beta> <rho> <Q> <turn_w> <clear_w> <diff_rad> [map_path]" << endl;
        cout << ">> 未偵測到輸入，使用預設保底參數測試模式..." << endl;
        dna = { 1.0, 2.0, 0.1, 100.0, 10.0, 5.0, 2.0, -1.0 };
    }

    cout << ">> 載入地圖: " << map_filename << endl;
    init_shared_data(map_filename);

    fs::create_directories("img");

    // 以下程式碼完全不變...
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

    // (選用) 最後用 GPU 跑一次並顯示視覺化畫面
    cout << "\n>> 啟動視覺化..." << endl;
    env.run_aco(dna, true, 0, true);
    waitKey(0);

    free_shared_data();
    return 0;
}