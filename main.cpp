#include<iostream>
#include<vector>
#include<random>
#include<algorithm>
#include<cstdio>
#include <fstream>
#include <filesystem> // C++17 用於自動讀取資料夾
#include <string>

namespace fs = std::filesystem;

// 儲存解析後的測資資料
struct TestCase{
    int num_jobs;
    int num_machines;
    std::string instance_name;
    std::vector<std::vector<int>> job_mtx;
};

// 解析單一 Taillard 格式檔案
bool read_taillard_file(const std::string &filepath, TestCase &tc){
    std::ifstream file(filepath);
    if(!file.is_open()){
        std::cerr << "Error: 無法開啟檔案 " << filepath << "\n";
        return false;
    }

    // 讀取第一行：工作數、機器數、輸出圖表名稱 (測資名稱)
    file >> tc.num_jobs >> tc.num_machines >> tc.instance_name;

    // 調整二維陣列大小 [Machine][Job]
    tc.job_mtx.assign(tc.num_machines, std::vector<int>(tc.num_jobs));

    // 依序讀取每台機器的處理時間
    for(int m = 0; m < tc.num_machines; ++m){
        for(int j = 0; j < tc.num_jobs; ++j){
            file >> tc.job_mtx[m][j];
        }
    }

    file.close();
    return true;
}

class Scheduling{
    struct Machine{
        int job_id;
        int time;
    };
public:
    std::vector<std::vector<int>> job_mtx; //[machine][job]

    int run_scheduling(std::vector<int> order){
        if(job_mtx.empty() || order.empty()){
            return 0;
        }

        int num_machines = job_mtx.size();

        std::vector<int> machine_end_time(num_machines, 0);

        for(int job : order){
            int current_job_ready_time = 0;
            for(int m = 0; m < num_machines; ++m){
                int start_time = std::max(current_job_ready_time, machine_end_time[m]);

                int finish_time = start_time + job_mtx[m][job];

                machine_end_time[m] = finish_time;
                current_job_ready_time = finish_time;
            }
        }

        return machine_end_time.back();
    }
};

class MetaheuristicSolver{
private:
    Scheduling &evaluator;
    int num_jobs;
    std::mt19937 rng;
    std::vector<int> makespan_history, best_makespan_history;

    FILE *gnuplotPipe;
    bool live_plot_enabled;

    std::vector<int> generate_initial_solution(){
        std::vector<int> sol(num_jobs);
        best_makespan_history.clear();
        makespan_history.clear();
        for(int i = 0; i < num_jobs; ++i)
            sol[i] = i;
        std::shuffle(sol.begin(), sol.end(), rng);
        return sol;
    }

    std::vector<int> get_random_neighbor(const std::vector<int> &current){
        std::vector<int> neighbor = current;
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        while(idx1 == idx2){
            idx2 = dist(rng);
        }
        std::swap(neighbor[idx1], neighbor[idx2]);
        return neighbor;
    }

    void init_gnuplot(const std::string &title){
        if(!live_plot_enabled) return;

        gnuplotPipe = popen("gnuplot -persist", "w");
        if(gnuplotPipe){
            fprintf(gnuplotPipe, "set title '%s'\n", title.c_str());
            fprintf(gnuplotPipe, "set xlabel 'Iteration'\n");
            fprintf(gnuplotPipe, "set ylabel 'Makespan'\n");
        }
        else{
            std::cerr << "Error: Could not open Gnuplot pipe.\n";
            live_plot_enabled = false;
        }
    }

    void update_plot(){
        if(!live_plot_enabled || !gnuplotPipe || best_makespan_history.empty()) return;

        fprintf(gnuplotPipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', '-' with lines title 'Current Makespan' lc rgb 'red'\n");

        for(size_t i = 0; i < best_makespan_history.size(); ++i){
            fprintf(gnuplotPipe, "%zu %d\n", i, best_makespan_history[i]);
        }
        fprintf(gnuplotPipe, "e\n");

        for(size_t i = 0; i < makespan_history.size(); ++i){
            fprintf(gnuplotPipe, "%zu %d\n", i, makespan_history[i]);
        }
        fprintf(gnuplotPipe, "e\n");

        fflush(gnuplotPipe);
    }

    // --- 新增：儲存圖片輔助函式 ---
    void save_plot(const std::string &algo_prefix, const std::string &instance_name, int best_makespan){
        if(!gnuplotPipe || best_makespan_history.empty()) return;

        // 組裝檔名：[演算法]_[測資名稱]_[最佳解].jpg
        std::string filename = algo_prefix + "_" + instance_name + "_" + std::to_string(best_makespan) + ".jpg";

        // 更改 Gnuplot 終端機設定為 jpeg 並指定輸出檔名
        fprintf(gnuplotPipe, "set terminal jpeg size 800,600\n");
        fprintf(gnuplotPipe, "set output '%s'\n", filename.c_str());

        // 重新發送繪圖指令與資料 (寫入檔案)
        fprintf(gnuplotPipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', '-' with lines title 'Current Makespan' lc rgb 'red'\n");
        for(size_t i = 0; i < best_makespan_history.size(); ++i){
            fprintf(gnuplotPipe, "%zu %d\n", i, best_makespan_history[i]);
        }
        fprintf(gnuplotPipe, "e\n");
        for(size_t i = 0; i < makespan_history.size(); ++i){
            fprintf(gnuplotPipe, "%zu %d\n", i, makespan_history[i]);
        }
        fprintf(gnuplotPipe, "e\n");
        fflush(gnuplotPipe);

        std::cout << ">> 已儲存圖表: " << filename << "\n";
    }

    void close_gnuplot(){
        if(gnuplotPipe){
            pclose(gnuplotPipe);
            gnuplotPipe = nullptr;
        }
    }

public:
    MetaheuristicSolver(Scheduling &sched, int jobs) : evaluator(sched), num_jobs(jobs), gnuplotPipe(nullptr), live_plot_enabled(false){
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    void set_live_plot(bool enable){
        live_plot_enabled = enable;
    }

    std::vector<int> iterative_improvement(const std::string &instance_name, int max_iter = 1000){
        // 修改標題，加入測資名稱
        init_gnuplot("II Trajectory - " + instance_name);
        std::vector<int> current = generate_initial_solution();
        int current_makespan = evaluator.run_scheduling(current);

        for(int i = 0; i < max_iter; ++i){
            std::vector<int> neighbor = get_random_neighbor(current);
            int neighbor_makespan = evaluator.run_scheduling(neighbor);

            if(neighbor_makespan < current_makespan){
                current = neighbor;
                current_makespan = neighbor_makespan;
            }

            best_makespan_history.push_back(current_makespan);
            // 【修正】：記錄實際被接受的 current_makespan
            makespan_history.push_back(current_makespan);

            if(i % 10 == 0) update_plot();
        }

        update_plot();
        save_plot("II", instance_name, current_makespan); // 存檔
        close_gnuplot();
        return current;
    }

    std::vector<int> simulated_annealing(const std::string &instance_name, double initial_temp = 100.0, double cooling_rate = 0.95, int max_iter = 1000){
        // 修改標題，加入測資名稱
        init_gnuplot("SA Trajectory - " + instance_name);
        std::vector<int> current = generate_initial_solution();
        std::vector<int> best = current;
        int current_makespan = evaluator.run_scheduling(current);
        int best_makespan = current_makespan;

        double temp = initial_temp;
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

        for(int iter = 0; iter < max_iter; ++iter){
            std::vector<int> neighbor = get_random_neighbor(current);
            int neighbor_makespan = evaluator.run_scheduling(neighbor);

            int delta = neighbor_makespan - current_makespan;

            if(delta < 0 || prob_dist(rng) < std::exp(-delta / temp)){
                current = neighbor;
                current_makespan = neighbor_makespan;

                if(current_makespan < best_makespan){
                    best = current;
                    best_makespan = current_makespan;
                }
            }

            best_makespan_history.push_back(best_makespan);
            // 【修正】：記錄實際被接受的 current_makespan
            makespan_history.push_back(current_makespan);

            temp *= cooling_rate;

            if(iter % 10 == 0) update_plot();
        }

        update_plot();
        save_plot("SA", instance_name, best_makespan); // 存檔
        close_gnuplot();
        return best;
    }

    std::vector<int> tabu_search(const std::string &instance_name, int max_iter = 500, int tabu_tenure = 10){
        // 修改標題，加入測資名稱
        init_gnuplot("TS Trajectory - " + instance_name);
        std::vector<int> current = generate_initial_solution();
        std::vector<int> best = current;
        int current_makespan = evaluator.run_scheduling(current);
        int best_makespan = current_makespan;

        std::vector<std::vector<int>> tabu_list(num_jobs, std::vector<int>(num_jobs, 0));

        for(int iter = 0; iter < max_iter; ++iter){
            std::vector<int> best_neighbor;
            int best_neighbor_makespan = 1e9;
            int best_swap_i = -1, best_swap_j = -1;

            for(int i = 0; i < num_jobs - 1; ++i){
                for(int j = i + 1; j < num_jobs; ++j){
                    std::vector<int> neighbor = current;
                    std::swap(neighbor[i], neighbor[j]);
                    int makespan = evaluator.run_scheduling(neighbor);

                    int job1 = current[i];
                    int job2 = current[j];
                    int min_job = std::min(job1, job2);
                    int max_job = std::max(job1, job2);

                    bool is_tabu = tabu_list[min_job][max_job] > iter;
                    bool aspiration = makespan < best_makespan;

                    if(!is_tabu || aspiration){
                        if(makespan < best_neighbor_makespan){
                            best_neighbor_makespan = makespan;
                            best_neighbor = neighbor;
                            best_swap_i = min_job;
                            best_swap_j = max_job;
                        }
                    }
                }
            }

            if(best_swap_i != -1){
                current = best_neighbor;
                current_makespan = best_neighbor_makespan;
                tabu_list[best_swap_i][best_swap_j] = iter + tabu_tenure;

                if(current_makespan < best_makespan){
                    best = current;
                    best_makespan = current_makespan;
                }
            }
            else{
                break;
            }

            best_makespan_history.push_back(best_makespan);
            makespan_history.push_back(current_makespan); // TS 這邊原本就是對的

            if(iter % 10 == 0) update_plot();
        }

        update_plot();
        save_plot("TS", instance_name, best_makespan); // 存檔
        close_gnuplot();
        return best;
    }

    std::vector<int> get_history() const{
        return best_makespan_history;
    }
};

int main(){
    // 假設你的測試資料都放在與執行檔同目錄的 "test_cases" 資料夾下
    std::string test_case_dir = "./Test_case";

    if(!fs::exists(test_case_dir) || !fs::is_directory(test_case_dir)){
        std::cerr << "找不到資料夾: " << test_case_dir << "\n";
        return 1;
    }

    // 巡覽資料夾內的所有檔案
    for(const auto &entry : fs::directory_iterator(test_case_dir)){
        if(entry.path().extension() == ".txt"){
            std::string filepath = entry.path().string();
            std::cout << "========================================\n";
            std::cout << "正在處理測資: " << filepath << "\n";

            TestCase tc;
            if(!read_taillard_file(filepath, tc)){
                continue; // 讀取失敗則跳過換下一個
            }

            // 初始化排程器與寫入資料
            Scheduling scheduler;
            scheduler.job_mtx = tc.job_mtx;

            // 初始化求解器
            MetaheuristicSolver solver(scheduler, tc.num_jobs);
            solver.set_live_plot(true); // 開啟即時繪圖

            // 執行 II
            std::cout << "Running Iterative Improvement...\n";
            std::vector<int> ii_result = solver.iterative_improvement(tc.instance_name, 200);
            std::cout << "[II] Makespan: " << scheduler.run_scheduling(ii_result) << "\n\n";

            // 執行 SA
            std::cout << "Running Simulated Annealing...\n";
            std::vector<int> sa_result = solver.simulated_annealing(tc.instance_name, 100.0, 0.95, 200);
            std::cout << "[SA] Makespan: " << scheduler.run_scheduling(sa_result) << "\n\n";

            // 執行 TS
            std::cout << "Running Tabu Search...\n";
            std::vector<int> ts_result = solver.tabu_search(tc.instance_name, 200, 10);
            std::cout << "[TS] Makespan: " << scheduler.run_scheduling(ts_result) << "\n\n";
        }
    }

    std::cout << "所有測資執行完畢！\n";
    return 0;
}