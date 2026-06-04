#include "config.h"
#include "map_data.cuh"
#include "aco_env.cuh"
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

double randDouble(double minVal, double maxVal){
    return minVal + (double) rand() / RAND_MAX * (maxVal - minVal);
}

// 支援透過指令列接收地圖檔案參數
int main(int argc, char *argv[]){

    string map_filename = "data/map.txt";
    if(argc > 1){
        map_filename = argv[1]; // 若有輸入則替換
    }

    srand(time(NULL));

    // 將檔案路徑傳入初始化
    cout << ">> 載入地圖: " << map_filename << endl;
    init_shared_data(map_filename);

    fs::create_directories("img");

    // 以下程式碼完全不變...
    vector<ACO_Environment *> envs(POP_SIZE);
    for(int i = 0; i < POP_SIZE; i++) envs[i] = new ACO_Environment(i, time(NULL) + i);

    // ... 下方的 GA 迴圈與 Gnuplot 呼叫皆無需改動，與前一版完全相同 ...
    vector<Chromosome> population(POP_SIZE);
    cout << "=== 初始化 GA 族群 ===" << endl;

    population[0] = { 1.0, 2.0, 0.1, 100.0, 10.0, 5.0, 2.0, -1.0 };

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
        fprintf(gp, "set title 'GA Optimization Progress (Best & Median Score)'\n");
        fprintf(gp, "set xlabel 'Generation'\n");
        fprintf(gp, "set ylabel 'Path Score'\n");
        fprintf(gp, "set grid\n");
    }

    bool is_gpu = true; // 預設使用 GPU 運算，若要測試 CPU 可改為 false
    for(int gen = 1; gen <= GA_GENERATIONS; gen++){
        cout << "\n[ 世代 " << gen << " / " << GA_GENERATIONS << " 開始 " <<
            (is_gpu ? "GPU" : "CPU") << " 平行運算評估 ]" << endl;
        auto start_time = chrono::high_resolution_clock::now();
        vector<thread> workers;
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
        cout << ">> 第 " << gen << " 代最佳適應度: " << best_score << " | 中位數: " << median_score << endl;

        log_file.open("ga_log.txt", ios::app);
        log_file << gen << " " << best_score << " " << median_score << "\n";
        log_file.close();

        if(gp){
            fprintf(gp, "set output 'img/ga_progress.png'\n");
            fprintf(gp, "plot 'ga_log.txt' skip 1 using 1:2 with linespoints lw 2 lc rgb 'red' title 'Best Score', \\\n");
            fprintf(gp, "     '' skip 1 using 1:3 with linespoints lw 2 lc rgb 'blue' title 'Median Score'\n");
            fflush(gp);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            Mat plot_img = imread("img/ga_progress.png");
            if(!plot_img.empty()){
                imshow("GA Progress Chart", plot_img);
                waitKey(1);
            }
        }

        envs[0]->run_aco(population[0], true, gen, is_gpu);

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
    cout << "Diffusion Radius: " << (int) round(population[0].diffusion_rad) << endl;
    cout << "=========================================" << endl;
    cout << "\n請在顯示的影像視窗中按下任意鍵，以關閉程式..." << endl;
    waitKey(0);

    for(int i = 0; i < POP_SIZE; i++) delete envs[i];
    free_shared_data();
    return 0;
}