#pragma once
#include "config.h"

// 宣告全域變數，供其他模組使用
extern const vector<string> grid_map_cpu;
extern int rows;
extern int cols;
extern vector<Point> start_pos[TYPE_COUNT];
extern vector<Point> end_pos[TYPE_COUNT];
extern char *d_grid_map_shared;

// 初始化與釋放記憶體函式
void init_shared_data();
void free_shared_data();