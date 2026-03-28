#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <iomanip>
#include <string>
#include <omp.h> // 引入 OpenMP 標頭檔
#include <unordered_map>
#include <cstdint>

namespace fs = std::filesystem;

struct TestCase{
    int num_jobs;
    int num_machines;
    std::string instance_name;
    std::vector<std::vector<int>> job_mtx;
};

bool read_taillard_file(const std::string &filepath, TestCase &tc){
    std::ifstream file(filepath);
    if(!file.is_open()){
        std::cerr << "Error: 無法開啟檔案 " << filepath << "\n";
        return false;
    }

    file >> tc.num_jobs >> tc.num_machines >> tc.instance_name;
    tc.job_mtx.assign(tc.num_machines, std::vector<int>(tc.num_jobs));

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
    std::vector<std::vector<int>> job_mtx;

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

    void close_gnuplot(){
        if(gnuplotPipe){
            pclose(gnuplotPipe);
            gnuplotPipe = nullptr;
        }
    }

    void save_plot(const std::string &save_dir, const std::string &algo_prefix, const std::string &instance_name, int best_makespan){
        if(best_makespan_history.empty()) return;

        std::string filename = save_dir + "/" + algo_prefix + "_" + instance_name + "_" + std::to_string(best_makespan) + ".jpg";

        FILE *pipe = popen("gnuplot", "w");
        if(pipe){
            fprintf(pipe, "set terminal jpeg size 800,600\n");
            fprintf(pipe, "set output '%s'\n", filename.c_str());
            fprintf(pipe, "set title '%s Trajectory - %s'\n", algo_prefix.c_str(), instance_name.c_str());
            fprintf(pipe, "set xlabel 'Iteration'\n");
            fprintf(pipe, "set ylabel 'Makespan'\n");

            fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', '-' with lines title 'Current Makespan' lc rgb 'red'\n");

            for(size_t i = 0; i < best_makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, best_makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            for(size_t i = 0; i < makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            fflush(pipe);
            pclose(pipe);
        }
    }

public:
    MetaheuristicSolver(Scheduling &sched, int jobs) : evaluator(sched), num_jobs(jobs), gnuplotPipe(nullptr), live_plot_enabled(false){
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    void set_live_plot(bool enable){ live_plot_enabled = enable; }

    std::vector<int> iterative_improvement(const std::string &instance_name, const std::string &save_dir, int max_iter = 1000){
        init_gnuplot("II Trajectory - " + instance_name);
        std::cout << "Run II" << std::endl;
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
            makespan_history.push_back(current_makespan);

            if(i % 10 == 0) update_plot();
        }
        std::cout << "II Result: " << current_makespan << std::endl;;
        update_plot();
        save_plot(save_dir, "II", instance_name, current_makespan);
        close_gnuplot();
        return current;
    }

    std::vector<int> simulated_annealing(const std::string &instance_name, const std::string &save_dir, double initial_temp = 100.0, double cooling_rate = 0.95, int max_iter = 1000){
        init_gnuplot("SA Trajectory - " + instance_name + "_T" + std::to_string(initial_temp) + "_C" + std::to_string(cooling_rate));
        std::cout << "Run SA" << std::endl;
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
            makespan_history.push_back(current_makespan);

            temp *= cooling_rate;
            if(iter % 10 == 0) update_plot();
        }
        std::cout << "SA Result: " << current_makespan << std::endl;;

        update_plot();
        save_plot(save_dir, "SA", instance_name + "_T" + std::to_string(initial_temp) + "_C" + std::to_string(cooling_rate), best_makespan);
        close_gnuplot();
        return best;
    }

    std::vector<int> tabu_search(const std::string &instance_name, const std::string &save_dir, int max_iter = 500, int tabu_tenure = 10){
        init_gnuplot("TS Trajectory - " + instance_name);
        std::cout << "Run TS" << std::endl;
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
            else{ break; }

            best_makespan_history.push_back(best_makespan);
            makespan_history.push_back(current_makespan);

            if(iter % 10 == 0) update_plot();
        }

        std::cout << "TS Result: " << best_makespan << std::endl;;
        update_plot();
        save_plot(save_dir, "TS", instance_name, best_makespan);
        close_gnuplot();
        return best;
    }

    std::vector<int> tabu_search2(const std::string &instance_name, const std::string &save_dir, int max_iter = 500, int tabu_tenure = 10){
        init_gnuplot("TS2 Trajectory - " + instance_name);
        std::cout << "Run TS2 (Permutation Hash)" << std::endl;
        std::vector<int> current = generate_initial_solution();
        std::vector<int> best = current;
        int current_makespan = evaluator.run_scheduling(current);
        int best_makespan = current_makespan;

        // 使用 Hash Map 來記錄每個排列的解禁代數
        std::unordered_map<uint64_t, int> tabu_list;

        // 定義 Hash 函數 (Polynomial Rolling Hash)
        auto compute_hash = [](const std::vector<int> &arr) -> uint64_t{
            uint64_t hash_val = 0;
            uint64_t p_pow = 1;
            const uint64_t p = 313; // 質數 base
            // modulo 2^64 會透過 uint64_t 的自然溢位自動達成
            for(size_t i = 0; i < arr.size(); ++i){
                hash_val += (arr[i] + 1) * p_pow;
                p_pow *= p;
            }
            return hash_val;
        };

        // 記錄初始解的 Hash
        tabu_list[compute_hash(current)] = 0;

        for(int iter = 0; iter < max_iter; ++iter){
            std::vector<int> best_neighbor;
            int best_neighbor_makespan = 1e9;
            uint64_t best_neighbor_hash = 0;
            bool found_valid_neighbor = false;

            for(int i = 0; i < num_jobs - 1; ++i){
                for(int j = i + 1; j < num_jobs; ++j){
                    std::vector<int> neighbor = current;
                    std::swap(neighbor[i], neighbor[j]);

                    uint64_t current_hash = compute_hash(neighbor);
                    int makespan = evaluator.run_scheduling(neighbor);

                    // 檢查該排列是否在禁忌期內
                    bool is_tabu = (tabu_list.count(current_hash) && tabu_list[current_hash] > iter);
                    bool aspiration = makespan < best_makespan;

                    if(!is_tabu || aspiration){
                        if(makespan < best_neighbor_makespan){
                            best_neighbor_makespan = makespan;
                            best_neighbor = neighbor;
                            best_neighbor_hash = current_hash;
                            found_valid_neighbor = true;
                        }
                    }
                }
            }

            if(found_valid_neighbor){
                current = best_neighbor;
                current_makespan = best_neighbor_makespan;

                // 將步入的該排列 Hash 記錄到禁忌表中，設定解禁代數
                tabu_list[best_neighbor_hash] = iter + tabu_tenure;

                if(current_makespan < best_makespan){
                    best = current;
                    best_makespan = current_makespan;
                }
            }
            else{ break; } // 若所有鄰居皆被禁忌且無法特赦，則提早結束

            best_makespan_history.push_back(best_makespan);
            makespan_history.push_back(current_makespan);

            if(iter % 10 == 0) update_plot();
        }

        std::cout << "TS2 Result: " << best_makespan << std::endl;
        update_plot();
        save_plot(save_dir, "TS2", instance_name, best_makespan);
        close_gnuplot();
        return best;
    }
};

void calculate_and_write_stats(std::ofstream &csv, const std::string &name, const std::vector<int> &data){
    int min_val = *std::min_element(data.begin(), data.end());
    int max_val = *std::max_element(data.begin(), data.end());
    double avg_val = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    csv << min_val << "," << std::fixed << std::setprecision(2) << avg_val << "," << max_val << ",";
}

int main(){
    std::string test_case_dir = "./Test_case";
    if(!fs::exists(test_case_dir) || !fs::is_directory(test_case_dir)){
        std::cerr << "找不到資料夾: " << test_case_dir << "\n";
        return 1;
    }

    std::ofstream csv_file("results.csv");
    csv_file << "Instance,II_Min,II_Avg,II_Max,SA_Min,SA_Avg,SA_Max,TS_Min,TS_Avg,TS_Max\n";

    int NUM_RUNS = 20;

    for(const auto &entry : fs::directory_iterator(test_case_dir)){
        if(entry.path().extension() == ".txt"){
            std::string filepath = entry.path().string();
            std::cout << "========================================\n";
            std::cout << "開始測試: " << filepath << " (共 " << NUM_RUNS << " 次平行執行)\n";

            TestCase tc;
            if(!read_taillard_file(filepath, tc)) continue;

            Scheduling scheduler;
            scheduler.job_mtx = tc.job_mtx;

            std::vector<int> ii_results(NUM_RUNS);
            std::vector<int> sa_results(NUM_RUNS);
            std::vector<int> sa_results2(NUM_RUNS);
            std::vector<int> sa_results3(NUM_RUNS);
            std::vector<int> sa_results4(NUM_RUNS);
            std::vector<int> ts_results(NUM_RUNS);
            std::vector<int> ts_results2(NUM_RUNS);
            std::vector<int> ts_results3(NUM_RUNS);

            for(int run = 1; run <= NUM_RUNS; ++run){
                fs::create_directories("./img/Test_" + std::to_string(run));
            }

#pragma omp parallel for schedule(dynamic)
            for(int run = 1; run <= NUM_RUNS; ++run){
                std::string save_dir = "./img/Test_" + std::to_string(run);

                MetaheuristicSolver solver(scheduler, tc.num_jobs);
                solver.set_live_plot(false);

                // std::vector<int> ii_res = solver.iterative_improvement(tc.instance_name, save_dir, 1000);
                // ii_results[run - 1] = scheduler.run_scheduling(ii_res);

                // std::vector<int> sa_res = solver.simulated_annealing(tc.instance_name, save_dir, 1000.0, 0.95, 1000);
                // sa_results[run - 1] = scheduler.run_scheduling(sa_res);

                // std::vector<int> sa_res2 = solver.simulated_annealing(tc.instance_name, save_dir, 100.0, 0.95, 1000);
                // sa_results2[run - 1] = scheduler.run_scheduling(sa_res2);

                // std::vector<int> sa_res3 = solver.simulated_annealing(tc.instance_name, save_dir, 1000.0, 0.8, 1000);
                // sa_results3[run - 1] = scheduler.run_scheduling(sa_res3);

                // std::vector<int> sa_res4 = solver.simulated_annealing(tc.instance_name, save_dir, 100.0, 0.8, 1000);
                // sa_results4[run - 1] = scheduler.run_scheduling(sa_res4);


                std::vector<int> ts_res = solver.tabu_search(tc.instance_name + "TS_L15", save_dir, 1000, 15);
                ts_results[run - 1] = scheduler.run_scheduling(ts_res);

                std::vector<int> ts_res2 = solver.tabu_search(tc.instance_name + "TS_L10", save_dir, 1000, 10);
                ts_results2[run - 1] = scheduler.run_scheduling(ts_res2);

                std::vector<int> ts_res3 = solver.tabu_search(tc.instance_name + "TS_L5", save_dir, 1000, 5);
                ts_results3[run - 1] = scheduler.run_scheduling(ts_res3);


                // std::vector<int> ts_res2 = solver.tabu_search2(tc.instance_name, save_dir, 1000, 10);
                // ts_results2[run - 1] = scheduler.run_scheduling(ts_res2);


#pragma omp critical
                {
                    std::cout << "  - 第 " << run << " 次執行完成，圖片已儲存至 " << save_dir << "\n";
                }
            }

            csv_file << tc.instance_name << ",";
            // calculate_and_write_stats(csv_file, "II", ii_results);
            // calculate_and_write_stats(csv_file, "SA", sa_results);
            calculate_and_write_stats(csv_file, "TS_L15", ts_results);
            calculate_and_write_stats(csv_file, "TS_L10", ts_results2);
            calculate_and_write_stats(csv_file, "TS_L5", ts_results3);
            // calculate_and_write_stats(csv_file, "SA_T1000_C0.8", sa_results3);
            // calculate_and_write_stats(csv_file, "SA_T100_C0.8", sa_results4);
            csv_file << "\n";

            std::cout << ">> " << tc.instance_name << " 統計資料已寫入 results.csv\n";
        }
    }

    csv_file.close();
    std::cout << "所有測資執行完畢！結果已匯出至 results.csv\n";
    return 0;
}