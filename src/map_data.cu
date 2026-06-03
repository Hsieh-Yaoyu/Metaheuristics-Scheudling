#include "map_data.cuh"
#include <fstream>
#include <iostream>

vector<string> grid_map_cpu;
int rows = 0;
int cols = 0;

vector<Point> start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];
char *d_grid_map_shared = nullptr;

void init_shared_data(const string &filename){
    // --- 1. 從外部檔案讀取地圖 ---
    ifstream file(filename);
    if(!file.is_open()){
        cerr << "[錯誤] 無法讀取地圖檔案:\0 " << filename << "\n請確認專案目錄下有 data 資料夾，且包含該檔案！\0" << endl;
        exit(EXIT_FAILURE);
    }

    grid_map_cpu.clear();
    string line;
    while(getline(file, line)){
        // 移除 Windows 環境可能產生的 \r 字元
        if(!line.empty() && line.back() == '\r'){
            line.pop_back();
        }
        if(!line.empty()){
            grid_map_cpu.push_back(line);
        }
    }
    file.close();

    if(grid_map_cpu.empty()){
        cerr << "[錯誤] 地圖檔案為空！\0" << endl;
        exit(EXIT_FAILURE);
    }

    // --- 2. 動態更新地圖大小 ---
    rows = grid_map_cpu.size();
    cols = grid_map_cpu[0].size();

    // --- 3. 解析起終點 ---
    for(int t = 0; t < TYPE_COUNT; t++){
        start_pos[t].clear(); end_pos[t].clear();
    }

    vector<char> flat_grid(rows * cols);
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
            flat_grid[y * cols + x] = grid_map_cpu[y][x];
            if(grid_map_cpu[y][x] == 'g') start_pos[GAS].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'G') end_pos[GAS].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'e') start_pos[ELEC].push_back(Point(x, y));
            if(grid_map_cpu[y][x] == 'E') end_pos[ELEC].push_back(Point(x, y));
        }
    }

    // --- 4. 複製到 GPU 共用記憶體 ---
    cudaMalloc(&d_grid_map_shared, rows * cols * sizeof(char));
    cudaMemcpy(d_grid_map_shared, flat_grid.data(), rows * cols * sizeof(char), cudaMemcpyHostToDevice);
}

void free_shared_data(){
    if(d_grid_map_shared){
        cudaFree(d_grid_map_shared);
        d_grid_map_shared = nullptr;
    }
}