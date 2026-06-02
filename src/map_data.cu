#include "map_data.cuh"

const vector<string> grid_map_cpu = {
    "g00000000000000000000000",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "00000000000000E000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "G00000000000000000000000",
    "00000000000000000G000000",
    "000000000000000000000000",
    "00000G000000000000000000",
    "110011111100001111110011",
    "110011111100001111110011",
    "000000000000000000000000",
    "001111001111110011110000",
    "001111001111110011110000",
    "000000000000000000000000",
    "00000000000e000000000000"
};

int rows = grid_map_cpu.size();
int cols = grid_map_cpu[0].size();

vector<Point> start_pos[TYPE_COUNT], end_pos[TYPE_COUNT];
char *d_grid_map_shared = nullptr;

void init_shared_data(){
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
    cudaMalloc(&d_grid_map_shared, rows * cols * sizeof(char));
    cudaMemcpy(d_grid_map_shared, flat_grid.data(), rows * cols * sizeof(char), cudaMemcpyHostToDevice);
}

void free_shared_data(){
    if(d_grid_map_shared){
        cudaFree(d_grid_map_shared);
        d_grid_map_shared = nullptr;
    }
}