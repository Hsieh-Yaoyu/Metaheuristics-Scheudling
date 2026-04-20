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
#include <unordered_set>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

std::unordered_map<std::string, int> bks_map = {
    {"tai20_5_1", 1278},
    {"tai20_10_1", 1582},
    {"tai20_20_1", 2297},
    {"tai50_5_1", 2724},
    {"tai50_10_1", 2991},
    {"tai50_20_1", 3850},
    {"tai100_5_1", 5493},
    {"tai100_10_1", 5770},
    {"tai100_20_1", 6202}
};

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
public:

    enum CrossoverType{ OX, LOX, PMX, CX };

    enum MutationType{ Insertion, Swap };

    enum LSSelectionType{ Elite, Top5Random };

    // --- 新增：定義「個體」的資料結構 ---
    struct Individual{
        std::vector<int> chromosome; // 排列方式 (染色體)
        int makespan;                // 目標函數值 (適應度)

        // 方便排序：Makespan 越小越好
        bool operator<(const Individual &other) const{
            return makespan < other.makespan;
        }
    };
    struct MAResult{
        std::vector<int> chromosome;
        int makespan;
        int total_gen;
        double runtime_sec;
    };

    Individual generate_random_individual(int &ffe_counter){
        Individual ind;
        ind.chromosome.resize(num_jobs);
        for(int i = 0; i < num_jobs; ++i){
            ind.chromosome[i] = i;
        }
        std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), rng);
        ind.makespan = evaluate(ind.chromosome, ffe_counter);
        return ind;
    }

private:

    Scheduling &evaluator;
    int num_jobs;
    std::mt19937 rng;
    std::vector<int> makespan_history, best_makespan_history, median_makespan_history;
    // 產生單一隨機個體


    // --- 新增：MA 選擇機制 (Tournament Selection) ---
    Individual tournament_selection(const std::vector<Individual> &population, int k = 3){
        std::uniform_int_distribution<int> dist(0, population.size() - 1);
        Individual best_participant = population[dist(rng)];

        for(int i = 1; i < k; ++i){
            Individual participant = population[dist(rng)];
            if(participant.makespan < best_participant.makespan){
                best_participant = participant;
            }
        }
        return best_participant;
    }

    // --- 新增：FFE評估預算上限 ---
    int evaluate(const std::vector<int> &chromosome, int &ffe_counter) {
        ffe_counter++; // 每呼叫一次，這裡就自動 +1
        return evaluator.run_scheduling(chromosome);
    }

    // --- 新增：NEH 建構式啟發算法 ---
    std::vector<int> generate_neh_solution(int& ffe_count){
        // 步驟 1 & 2：計算每個工作的總處理時間，並由大到小排序
        std::vector<std::pair<int, int>> job_totals(num_jobs);
        for(int i = 0; i < num_jobs; ++i){
            // 利用 evaluator 評估單一工作 i 的時間 (等同於該工作在所有機器的時間總和)
            int total_time = evaluate({ i }, ffe_count);
            job_totals[i] = { total_time, i };
        }

        // 依照總時間降冪排序 (由大到小)
        std::sort(job_totals.begin(), job_totals.end(), [](const auto &a, const auto &b){
            return a.first > b.first;
        });

        // 步驟 3：依序進行最佳插入 (Best Insertion)
        std::vector<int> current_seq;
        current_seq.push_back(job_totals[0].second); // 先放入總時間最長的工作

        for(int i = 1; i < num_jobs; ++i){
            int job_to_insert = job_totals[i].second;
            int best_makespan = 1e9;
            int best_pos = -1;

            // 測試插入目前序列的所有可能位置 (從 0 到 i)
            for(int pos = 0; pos <= i; ++pos){
                std::vector<int> temp_seq = current_seq;
                temp_seq.insert(temp_seq.begin() + pos, job_to_insert);

                // 評估這個「部分排程」的 Makespan
                int current_makespan = evaluate(temp_seq, ffe_count);

                if(current_makespan < best_makespan){
                    best_makespan = current_makespan;
                    best_pos = pos;
                }
            }
            // 確定最佳位置後，正式插入
            current_seq.insert(current_seq.begin() + best_pos, job_to_insert);
        }

        return current_seq;
    }

    // --- 新增：Order Crossover (OX) 交配算子 ---
    std::pair<Individual, Individual> order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)，兩個子代共用這組切點
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經被放入子代中
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. 複製切點範圍內的基因到對應的子代
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. 交叉填補剩餘的位置
        int p2_idx = (cut2 + 1) % num_jobs;
        int off1_idx = (cut2 + 1) % num_jobs;

        int p1_idx = (cut2 + 1) % num_jobs;
        int off2_idx = (cut2 + 1) % num_jobs;

        for(int i = 0; i < num_jobs; ++i){
            // 處理 offspring1: 從 Parent 2 依序填補
            int gene_from_p2 = p2.chromosome[p2_idx];
            if(!in_offspring1[gene_from_p2]){
                offspring1.chromosome[off1_idx] = gene_from_p2;
                off1_idx = (off1_idx + 1) % num_jobs;
            }
            p2_idx = (p2_idx + 1) % num_jobs;

            // 處理 offspring2: 從 Parent 1 依序填補
            int gene_from_p1 = p1.chromosome[p1_idx];
            if(!in_offspring2[gene_from_p1]){
                offspring2.chromosome[off2_idx] = gene_from_p1;
                off2_idx = (off2_idx + 1) % num_jobs;
            }
            p1_idx = (p1_idx + 1) % num_jobs;
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Linear Order Crossover (LOX) 交配算子 ---
    std::pair<Individual, Individual> linear_order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經被放入子代中
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. 複製切點範圍內的基因到對應的子代 (這部分與 OX 相同)
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. 收集另一個父代中「尚未被使用」的基因 (保持由左至右的順序)
        std::vector<int> remaining_for_off1;
        std::vector<int> remaining_for_off2;

        for(int i = 0; i < num_jobs; ++i){
            if(!in_offspring1[p2.chromosome[i]]){
                remaining_for_off1.push_back(p2.chromosome[i]);
            }
            if(!in_offspring2[p1.chromosome[i]]){
                remaining_for_off2.push_back(p1.chromosome[i]);
            }
        }

        // 4. 由左至右 (Linear) 依序填補子代的空位 (跳過 cut1 到 cut2 的區段)
        int idx1 = 0, idx2 = 0;
        for(int i = 0; i < num_jobs; ++i){
            // 如果是在切點範圍內，直接跳過 (因為已經在步驟 2 填好了)
            if(i >= cut1 && i <= cut2) continue;

            offspring1.chromosome[i] = remaining_for_off1[idx1++];
            offspring2.chromosome[i] = remaining_for_off2[idx2++];
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Partially-Mapped Crossover (PMX) 交配算子 ---
    std::pair<Individual, Individual> partially_mapped_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經「存在於切點範圍內」，用以判斷是否衝突
        std::vector<bool> in_off1_middle(num_jobs, false);
        std::vector<bool> in_off2_middle(num_jobs, false);

        // 建立雙向映射表 (Mapping dictionaries)
        std::vector<int> map_p1_to_p2(num_jobs, -1);
        std::vector<int> map_p2_to_p1(num_jobs, -1);

        // 2. 複製切點範圍內的基因，並建立映射關係
        for(int i = cut1; i <= cut2; ++i){
            int gene1 = p1.chromosome[i];
            int gene2 = p2.chromosome[i];

            // 填入子代 1 與 子代 2 的中間段落
            offspring1.chromosome[i] = gene1;
            in_off1_middle[gene1] = true;

            offspring2.chromosome[i] = gene2;
            in_off2_middle[gene2] = true;

            // 紀錄互相對應的映射關係
            map_p1_to_p2[gene1] = gene2;
            map_p2_to_p1[gene2] = gene1;
        }

        // 3. 填補切點以外的剩餘位置
        for(int i = 0; i < num_jobs; ++i){
            // 跳過中間已經填好的部分
            if(i >= cut1 && i <= cut2) continue;

            // --- 處理 Offspring 1 (繼承 Parent 2 的外圍基因) ---
            int candidate1 = p2.chromosome[i];
            // 如果發生衝突 (該基因已經在 Offspring 1 的中間段出現過)
            // 就透過映射表找出替代基因，這組 while 迴圈能完美解決連鎖映射的問題
            while(in_off1_middle[candidate1]){
                candidate1 = map_p1_to_p2[candidate1];
            }
            offspring1.chromosome[i] = candidate1;

            // --- 處理 Offspring 2 (繼承 Parent 1 的外圍基因) ---
            int candidate2 = p1.chromosome[i];
            // 同樣的邏輯，反向查表
            while(in_off2_middle[candidate2]){
                candidate2 = map_p2_to_p1[candidate2];
            }
            offspring2.chromosome[i] = candidate2;
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Cycle Crossover (CX) 交配算子 ---
    std::pair<Individual, Individual> cycle_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 建立索引查表：能以 O(1) 時間找到「某個基因在 Parent 1 中的位置」
        std::vector<int> p1_index_of(num_jobs, 0);
        for(int i = 0; i < num_jobs; ++i){
            p1_index_of[p1.chromosome[i]] = i;
        }

        // 記錄哪些位置已經被 Cycle 走訪過
        std::vector<bool> visited(num_jobs, false);

        // 1. 找出第一個 Cycle (這裡固定從 index 0 開始)
        int current_idx = 0;

        // 當還沒繞回起點形成迴圈時，繼續追蹤
        while(!visited[current_idx]){
            visited[current_idx] = true;

            // 在 Cycle 內：
            // 子代 1 繼承 Parent 1 該位置的基因
            offspring1.chromosome[current_idx] = p1.chromosome[current_idx];
            // 子代 2 繼承 Parent 2 該位置的基因
            offspring2.chromosome[current_idx] = p2.chromosome[current_idx];

            // 尋找下一個 index：
            // 找出 Parent 2 在當前位置的基因，並查詢該基因在 Parent 1 的位置
            int val_in_p2 = p2.chromosome[current_idx];
            current_idx = p1_index_of[val_in_p2];
        }

        // 2. 將非 Cycle 範圍的剩餘位置，用「另一個 Parent」直接填補
        for(int i = 0; i < num_jobs; ++i){
            if(!visited[i]){
                // 不在 Cycle 內：
                // 子代 1 繼承 Parent 2 該位置的基因
                offspring1.chromosome[i] = p2.chromosome[i];
                // 子代 2 繼承 Parent 1 該位置的基因
                offspring2.chromosome[i] = p1.chromosome[i];
            }
        }

        return { offspring1, offspring2 };
    }


    // --- 新增：突變算子 (Swap Mutation) ---
    void mutate(Individual &ind, double mutation_rate, MutationType mutation_type = Insertion){

        if(mutation_type == Insertion){
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            if(prob(rng) < mutation_rate){
                std::uniform_int_distribution<int> dist(0, num_jobs - 1);
                int idx1 = dist(rng);
                int idx2 = dist(rng);
                while(idx1 == idx2) idx2 = dist(rng);

                // 抽出 idx1 的工作，並插入到 idx2 的位置
                int job = ind.chromosome[idx1];
                ind.chromosome.erase(ind.chromosome.begin() + idx1);
                ind.chromosome.insert(ind.chromosome.begin() + idx2, job);
            }
        }
        else if(mutation_type == Swap){
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            if(prob(rng) < mutation_rate){
                std::uniform_int_distribution<int> dist(0, num_jobs - 1);
                int idx1 = dist(rng);
                int idx2 = dist(rng);
                while(idx1 == idx2) idx2 = dist(rng);
                std::swap(ind.chromosome[idx1], ind.chromosome[idx2]);
            }
        }
    }

    // --- 新增：Tabu Search (swap) 作為個體 local search ---
    void sa_insertion_local_search(
    Individual &ind, int &ffe_counter, int max_ffe, int max_ls_iter = 20, double init_temp = 50, double cooling_rate = 0.90){

        if(ind.chromosome.empty() || ffe_counter >= max_ffe) return;

    std::vector<int> current = ind.chromosome;
    int current_makespan = ind.makespan;
    if(current_makespan == 0){
        current_makespan = evaluate(current, ffe_counter);
    }

    std::vector<int> best = current;
    int best_makespan = current_makespan;

    double temperature = init_temp;
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    for(int iter = 0; iter < max_ls_iter && ffe_counter < max_ffe; ++iter){
        bool moved = false;

        for(int idx_from = 0; idx_from < num_jobs && !moved && ffe_counter < max_ffe; ++idx_from){
            for(int idx_to = 0; idx_to < num_jobs && !moved && ffe_counter < max_ffe; ++idx_to){
                if(idx_from == idx_to) continue;

                std::vector<int> neighbor = current;

                int job = neighbor[idx_from];
                neighbor.erase(neighbor.begin() + idx_from);
                neighbor.insert(neighbor.begin() + idx_to, job);

                int neighbor_makespan = evaluate(neighbor, ffe_counter);
                int delta = neighbor_makespan - current_makespan;

                bool accept = false;

                if(delta < 0){
                    accept = true;
                }
                else if(temperature > 1e-9){
                    double accept_prob = std::exp(-static_cast<double>(delta) / temperature);
                    if(prob(rng) < accept_prob){
                        accept = true;
                    }
                }

                if(accept){
                    current = neighbor;
                    current_makespan = neighbor_makespan;
                    moved = true;

                    if(current_makespan < best_makespan){
                        best = current;
                        best_makespan = current_makespan;
                    }
                }
            }
        }

        if(!moved) break;

        temperature *= cooling_rate;
    }

    ind.chromosome = best;
    ind.makespan = best_makespan;
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

            // 判斷是否有中位數資料 (MA 專用)
            bool has_median = !median_makespan_history.empty();

            if(has_median){
                fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', "
                    "'-' with lines title 'Current/Gen Best Makespan' lc rgb 'red', "
                    "'-' with lines title 'Population Median' lc rgb 'green'\n");
            }
            else{
                fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', "
                    "'-' with lines title 'Current Makespan' lc rgb 'red'\n");
            }

            // 傳送最佳解資料 (藍線)
            for(size_t i = 0; i < best_makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, best_makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            // 傳送當代/探索解資料 (紅線)
            for(size_t i = 0; i < makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            // 如果有中位數，傳送中位數資料 (綠線)
            if(has_median){
                for(size_t i = 0; i < median_makespan_history.size(); ++i){
                    fprintf(pipe, "%zu %d\n", i, median_makespan_history[i]);
                }
                fprintf(pipe, "e\n");
            }

            fflush(pipe);
            pclose(pipe);
        }
    }

    std::string cross_str(MetaheuristicSolver::CrossoverType type){
        switch(type){
            case OX: return "OX";
            case LOX: return "LOX";
            case PMX: return "PMX";
            case CX: return "CX";
            default: return "Unknown";
        }
    }

public:
    MetaheuristicSolver(Scheduling &sched, int jobs) : evaluator(sched), num_jobs(jobs){
        std::random_device rd;
        rng = std::mt19937(rd());
    }



    // ==========================================
    // 迷因演算法 (Memetic Algorithm) 主程式
    // ==========================================
    MAResult memetic_algorithm(
        int pop_size = 50,
        double crossover_rate = 0.8, double mutation_rate = 0.1,
        CrossoverType crossover_type = OX, bool with_tabu = false,
        bool dynamic_tabu = false, bool do_neh = false, bool use_sa_local_search = false, int ls_gen_num = 20, int ls_iters = 20, double init_temp = 20, double cooling_rate = 0.9,
        LSSelectionType ls_selection_type = Elite,  MutationType mutation_type = Insertion, std::string instance_name = "Instance", std::string save_dir = "./img"){
        
        std::cout << "Run MA with " << cross_str(crossover_type) << (with_tabu ? " (Tabu Enabled)" : "") << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now(); //計時開始

        best_makespan_history.clear();
        makespan_history.clear();
        median_makespan_history.clear();

        // 1. 初始化族群 (Initialization)
        int current_ffe = 0;
        int max_ffe = pop_size * 1000;
        std::vector<Individual> population;
        for(int i = 0; i < pop_size; ++i){
            // 加入 NEH 優質個體
            if(do_neh && i == 0){
                Individual neh_seed;
                neh_seed.chromosome = generate_neh_solution(current_ffe);
                neh_seed.makespan = evaluate(neh_seed.chromosome, current_ffe);
                population.push_back(neh_seed);
                std::cout << "  [Info] Injected NEH Seed with Makespan: " << neh_seed.makespan << std::endl;
            }
            // 族群中加入 Random 個體
            else{
                population.push_back(generate_random_individual(current_ffe));
            }
        }

        // 尋找初始最佳解
        Individual global_best = *std::min_element(population.begin(), population.end());

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        // Tabu Hash 相關設定 (僅在 with_tabu == true 時發揮作用)
        std::unordered_map<uint64_t, int> tabu_list;
        int gen_tabu_tenure = 10;

        auto compute_hash = [](const std::vector<int> &arr) -> uint64_t{
            uint64_t hash_val = 0;
            uint64_t p_pow = 1;
            const uint64_t p = 313;
            for(size_t i = 0; i < arr.size(); ++i){
                hash_val += (arr[i] + 1) * p_pow;
                p_pow *= p;
            }
            return hash_val;
        };

        // 世代演化迴圈
        int gen = 0; // 保留 gen 僅用於統計與畫圖，不再作為控制變數
        while(current_ffe < max_ffe){  
            gen++;

            std::vector<Individual> next_generation;
            std::unordered_set<uint64_t> current_pop_hashes;

            // 菁英保留策略 (Elitism)
            Individual current_best = *std::min_element(population.begin(), population.end());
            next_generation.push_back(current_best);
            
            if(with_tabu){
                // 1. 計算進度比例 (0.0 到 1.0)
                double progress = static_cast<double>(current_ffe) / max_ffe;
                
                // 2. 線性衰減公式：初始 tenure 為 10，隨著 FFE 增加衰減至 0
                gen_tabu_tenure = 10 - static_cast<int>(progress * 10); 
                
                // 3. 確保不小於 0
                gen_tabu_tenure = std::max(0, gen_tabu_tenure); 

                // 4. 更新 Tabu List
                // 注意：這裡直接使用當前的 tabu_tenure，不需要額外判斷
                uint64_t best_hash = compute_hash(current_best.chromosome);
                current_pop_hashes.insert(best_hash);
                
                // 當 tabu_tenure 為 0 時，寫入 gen + 0，禁忌檢查會自動失效
                tabu_list[best_hash] = gen + gen_tabu_tenure;
            }

            // 產生下一代 (直到數量等於 pop_size)
            while(next_generation.size() < pop_size){
                Individual p1 = tournament_selection(population);
                Individual p2 = tournament_selection(population);
                Individual off1, off2;
                // ====================================================
                // 路徑 A：原本的純粹 MA (不執行重複排除與排序禁忌)
                // ====================================================
                if(!with_tabu){
                    if(prob(rng) < crossover_rate){
                        if(crossover_type == OX){
                            auto offsprings = order_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == LOX){
                            auto offsprings = linear_order_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == PMX){
                            auto offsprings = partially_mapped_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == CX){
                            auto offsprings = cycle_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                    }
                    else{
                        off1 = (prob(rng) < 0.5) ? p1 : p2;
                        off2 = (prob(rng) < 0.5) ? p1 : p2;
                    }

                    // 突變 (Mutation)
                    mutate(off1, mutation_rate, mutation_type);
                    mutate(off2, mutation_rate, mutation_type);

                    off1.makespan = evaluate(off1.chromosome, current_ffe);
                    off2.makespan = evaluate(off2.chromosome, current_ffe);

                    next_generation.push_back(off1);
                    if(next_generation.size() < pop_size){
                        next_generation.push_back(off2);
                    }
                }
                // ====================================================
                // 路徑 B：加上 Tabu 與同代重複排除機制的 MA
                // ====================================================
                else{
                    bool off1_valid = false;
                    bool off2_valid = false;
                    int retries = 0;

                    while(retries < 10 && (!off1_valid || !off2_valid)){
                        Individual temp1 = p1, temp2 = p2;

                        // 交配 (Crossover)
                        if(prob(rng) < crossover_rate){
                            if(crossover_type == OX){
                                auto offsprings = order_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == LOX){
                                auto offsprings = linear_order_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == PMX){
                                auto offsprings = partially_mapped_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == CX){
                                auto offsprings = cycle_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                        }

                        // 突變 (Mutation)
                        mutate(temp1, mutation_rate, mutation_type);
                        mutate(temp2, mutation_rate, mutation_type);

                        uint64_t h1 = compute_hash(temp1.chromosome);
                        uint64_t h2 = compute_hash(temp2.chromosome);

                        if(!off1_valid){
                            bool is_tabu = ((tabu_list.count(h1) && tabu_list[h1] > gen) || current_pop_hashes.count(h1));
                            if(!is_tabu){
                                temp1.makespan = evaluate(temp1.chromosome, current_ffe);
                                off1 = temp1;
                                off1_valid = true;
                                current_pop_hashes.insert(h1);
                                tabu_list[h1] = gen + gen_tabu_tenure;
                            }
                        }

                        if(!off2_valid && (next_generation.size() + (off1_valid ? 1 : 0) < pop_size)){
                            bool is_tabu = ((tabu_list.count(h2) && tabu_list[h2] > gen) || current_pop_hashes.count(h2));
                            if(!is_tabu){
                                temp2.makespan = evaluate(temp2.chromosome, current_ffe);
                                off2 = temp2;
                                off2_valid = true;
                                current_pop_hashes.insert(h2);
                                tabu_list[h2] = gen + gen_tabu_tenure;
                            }
                        }
                        else if(next_generation.size() + (off1_valid ? 1 : 0) >= pop_size){
                            off2_valid = true;
                        }

                        retries++;
                    }

                    if(!off1_valid){
                        off1 = generate_random_individual(current_ffe);
                        uint64_t h1 = compute_hash(off1.chromosome);
                        current_pop_hashes.insert(h1);
                        tabu_list[h1] = gen + gen_tabu_tenure;
                    }
                    next_generation.push_back(off1);

                    if(next_generation.size() < pop_size){
                        if(!off2_valid){
                            off2 = generate_random_individual(current_ffe);
                            uint64_t h2 = compute_hash(off2.chromosome);
                            current_pop_hashes.insert(h2);
                            tabu_list[h2] = gen + gen_tabu_tenure;
                        }
                        next_generation.push_back(off2);
                    }
                } // End of else (路徑 B)
            }

            // 更新族群
            population = next_generation;

            // 找到當代最佳個體
            auto best_it = std::min_element(population.begin(), population.end());

            // Local search(SA), 某幾代從前5名隨機選1個做 local search
            if(gen % ls_gen_num == 0 && use_sa_local_search && current_ffe < max_ffe){
                if(ls_selection_type == Elite){
                    auto best_it = std::min_element(population.begin(), population.end());
                    if(best_it != population.end()){
                        sa_insertion_local_search(*best_it, current_ffe, max_ffe, ls_iters, init_temp, cooling_rate);
                    }
                }
                else if(ls_selection_type == Top5Random){
                    std::sort(population.begin(), population.end());

                int top_k = std::min(5, static_cast<int>(population.size()));
                std::uniform_int_distribution<int> top_dist(0, top_k - 1);
                int selected_idx = top_dist(rng);

                sa_insertion_local_search(population[selected_idx], current_ffe, max_ffe, ls_iters, init_temp, cooling_rate);
                }
            }
            

            // 更新歷史最佳解
            current_best = *std::min_element(population.begin(), population.end());
            if(current_best.makespan < global_best.makespan){
                global_best = current_best;
            }

            // 計算中位數
            std::vector<int> current_makespans(pop_size);
            for(int i = 0; i < pop_size; ++i){
                current_makespans[i] = population[i].makespan;
            }
            std::nth_element(current_makespans.begin(), current_makespans.begin() + pop_size / 2, current_makespans.end());
            int current_median = current_makespans[pop_size / 2];

            makespan_history.push_back(current_best.makespan);
            best_makespan_history.push_back(global_best.makespan);
            median_makespan_history.push_back(current_median);

        }

        std::cout << "MA Result: " << global_best.makespan << std::endl;
        std::string prefix = "MA_" + cross_str(crossover_type) + (with_tabu ? "_Tabu" : "");
        save_plot(save_dir, prefix, instance_name, global_best.makespan);
        auto end_time = std::chrono::high_resolution_clock::now();
        double runtime_sec = std::chrono::duration<double>(end_time - start_time).count();
        return {global_best.chromosome, global_best.makespan, gen, runtime_sec};
    }

    void cross_test(int &current_ffe){
        std::cout << "Testing Order Crossover (OX)..." << std::endl;
        Individual ind1 = generate_random_individual(current_ffe);
        Individual ind2 = generate_random_individual(current_ffe);
        auto offsprings = order_crossover(ind1, ind2);
        std::cout << "Parent 1: ";
        for(int gene : ind1.chromosome) std::cout << gene << " ";
        std::cout << "\nParent 2: ";
        for(int gene : ind2.chromosome) std::cout << gene << " ";
        std::cout << "\nOffspring: ";
        for(int gene : offsprings.first.chromosome) std::cout << gene << " ";
        std::cout << "\nOffspring: ";
        for(int gene : offsprings.second.chromosome) std::cout << gene << " ";
        std::cout << std::endl;
    }

};

void calculate_and_write_stats(std::ofstream &csv, const std::string &name, const std::vector<int> &data, const std::vector<int> &gens, const std::vector<double> &time){
    int min_val = *std::min_element(data.begin(), data.end());
    int max_val = *std::max_element(data.begin(), data.end());
    double avg_val = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum_sq_diff = 0.0;
    for (int val : data) {
        sum_sq_diff += (val - avg_val) * (val - avg_val);
    }
    double std_dev = std::sqrt(sum_sq_diff / (data.size() > 1 ? data.size() - 1 : 1));
    double avg_gen = std::accumulate(gens.begin(), gens.end(), 0.0) / gens.size();
    double avg_runtime = std::accumulate(time.begin(), time.end(), 0.0) / time.size();

    csv << min_val << ","
        << std::fixed << std::setprecision(2) << avg_val << ","
        << max_val << ","
        << std::fixed << std::setprecision(2) << std_dev << ","
        << std::fixed << std::setprecision(2) << avg_gen << ","
        << std::fixed << std::setprecision(2) << avg_runtime << ",";}

int main(){
std::cout << "HI";
#define TABU_TEST false
    // MetaheuristicSolver solver(*(new Scheduling()), 10);
    // solver.cross_test();
    // /*
    std::string test_case_dir = "./Test_case";
    if(!fs::exists(test_case_dir) || !fs::is_directory(test_case_dir)){
        std::cerr << "找不到資料夾: " << test_case_dir << "\n";
        return 1;
    }

    std::ofstream csv_file("results.csv");
    csv_file << "Instance,LOX_iter20_Min,LOX_iter20_Avg,LOX_iter20_Max,LOX_iter20_Std,LOX_iter20_AvgGen,LOX_iter20_AvgTime,LOX_gen50_Min,LOX_gen50_Avg,LOX_gen50_Max,LOX_gen50_Std,LOX_gen50_AvgGen,LOX_gen50_AvgTime,LOX_gen100_Min,LOX_gen100_Avg,LOX_gen100_Max,LOX_gen100_Std,LOX_gen100_AvgGen,LOX_gen100_AvgTime\n";
    //csv_file << "Instance, OX_Min,OX_Avg,OX_Max,OX_Std,LOX_Min,LOX_Avg,LOX_Max,LOX_Std,PMX_Min,PMX_Avg,PMX_Max,PMX_Std,CX_Min,CX_Avg,CX_Max,CX_Std\n";
    //csv_file << "Instance,MA_LOX_Min,MA_LOX_Avg,MA_LOX_dev,MA_OX_neh_Min,MA_OX_neh_Avg,MA_OX_neh_dev\n";
    //csv_file << "Instance,MA_OX_Min,MA_OX_Avg,MA_OX_Max,MA_LOX_Min,MA_LOX_Avg,MA_LOX_Max,MA_PMX_Min,MA_PMX_Avg,MA_PMX_Max,MA_CX_Min,MA_CX_Avg,MA_CX_Max\n";

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

            std::vector<int> lox_results(NUM_RUNS);
            std::vector<int> lox_results2(NUM_RUNS);
            std::vector<int> lox_results3(NUM_RUNS);
            std::vector<int> lox_gen(NUM_RUNS);
            std::vector<int> lox_gen2(NUM_RUNS);
            std::vector<int> lox_gen3(NUM_RUNS);
            std::vector<double> lox_time(NUM_RUNS);
            std::vector<double> lox_time2(NUM_RUNS);
            std::vector<double> lox_time3(NUM_RUNS);



            for(int run = 1; run <= NUM_RUNS; ++run){
                fs::create_directories("./img/Test_" + std::to_string(run));
            }

#pragma omp parallel for schedule(dynamic)
            for(int run = 1; run <= NUM_RUNS; ++run){
                std::string save_dir = "./img/Test_" + std::to_string(run);

                MetaheuristicSolver solver(scheduler, tc.num_jobs);

                auto lox_res = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, false, false, true, false, 20, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "_MA_LOX_gen10", save_dir);
                lox_results[run - 1] = lox_res.makespan;
                lox_gen[run - 1] = lox_res.total_gen;
                lox_time[run - 1] = lox_res.runtime_sec;
                auto lox_res2 = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, false, false, true, true, 50, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "_MA_LOX_gen20", save_dir);
                lox_results2[run - 1] = lox_res2.makespan;
                lox_gen2[run - 1] = lox_res2.total_gen;
                lox_time2[run - 1] = lox_res2.runtime_sec;
                auto lox_res3 = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, false, false, true, true, 100, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "_MA_LOX_Top5", save_dir);
                lox_results3[run - 1] = lox_res3.makespan;
                lox_gen3[run - 1] = lox_res3.total_gen;
                lox_time3[run - 1] = lox_res3.runtime_sec;


#pragma omp critical
                {
                    std::cout << "  - 第 " << run << " 次執行完成，圖片已儲存至 " << save_dir << "\n";
                }
            }
            csv_file << tc.instance_name << ",";
            calculate_and_write_stats(csv_file, "LOX_gen10", lox_results,lox_gen,lox_time);
            calculate_and_write_stats(csv_file, "LOX_gen20", lox_results2,lox_gen2,lox_time2);
            calculate_and_write_stats(csv_file, "LOX_Top5", lox_results3,lox_gen3,lox_time3);
            csv_file << "\n";

            std::cout << ">> " << tc.instance_name << " 統計資料已寫入 results.csv\n";
        }
    }

    csv_file.close();
    std::cout << "所有測資執行完畢！結果已匯出至 results.csv\n";
    // */
    return 0;
}