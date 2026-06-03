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
    cout << "開始執行 ACO 模擬 (MAX_ITER: " << MAX_ITER << ")..." << endl;
    double final_score = env.run_aco(dna, true, 0);

    cout << "=========================================" << endl;
    cout << "模擬結束！最終取得的適應度 (Score): " << final_score << endl;
    cout << "=========================================" << endl;

    cout << "\n請在顯示的影像視窗中按下任意鍵，以關閉程式..." << endl;
    waitKey(0);

    free_shared_data();
    return 0;
}