#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// --- 演算法基本參數 ---
const int ANT_COUNT = 100;
const double MAX_PHERO = 100.0;
const int MAX_ITER = 1000;
const int CELL_SIZE = 25;
const int MAX_PATH_LEN = 800;

const int MAX_ENDPOINTS = 10;
const int SAFE_DISTANCE = 3;

const int POP_SIZE = 80;
const int GA_GENERATIONS = 25;

// --- 視覺化顏色設定 ---
const Scalar COLOR_BG = Scalar(255, 255, 255);
const Scalar COLOR_WALL = Scalar(50, 50, 50);
const Scalar COLOR_GRID = Scalar(200, 200, 200);
const Scalar COLOR_GAS = Scalar(0, 255, 255);
const Scalar COLOR_ELEC = Scalar(255, 225, 0);

enum PipeType{ GAS = 0, ELEC = 1 };
const int TYPE_COUNT = 2;

// --- 基因結構 (Chromosome) ---
struct Chromosome{
    double alpha;
    double beta;
    double rho;
    double Q;
    double turn_w;
    double clear_w;
    double diffusion_rad;
    double fitness = -1.0;
};

// --- 螞蟻結構 ---
struct CUDA_Ant{
    int type;
    int pos_x, pos_y;
    int last_dir_x, last_dir_y;
    int target_x, target_y;
    int target_idx;
    int path_x[MAX_PATH_LEN];
    int path_y[MAX_PATH_LEN];
    int path_length;
    bool reached_end;
    bool stuck;
};