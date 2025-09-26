#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <iomanip>

#define ROW 2000
#define COL 100
const float TOTAL = 200000.f;

void matrix_add_consequent() {
    std::vector<float> m1(ROW * COL), m2(ROW * COL), mr(ROW * COL);

    // init
    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < COL; ++j) {
            int idx = i * COL + j;
            m1[idx] = std::sin((i * ROW + j) / TOTAL);
            m2[idx] = std::cos((i * ROW + j) / TOTAL);
        }

    double start = omp_get_wtime();
    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < COL; ++j) {
            int idx = i * COL + j;
            mr[idx] = m1[idx] + m2[idx];
        }
    double end = omp_get_wtime();

    // time check flags
    double checksum = 0.0;
    for (float v : mr) checksum += v;

    std::cout << std::fixed << std::setprecision(6)
              << "Time (sequential): " << (end - start)
              << " s, checksum=" << checksum << "\n";
}

void matrix_add_parallel() {
    std::vector<float> m1(ROW * COL), m2(ROW * COL), mr(ROW * COL);

    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < COL; ++j) {
            int idx = i * COL + j;
            m1[idx] = std::sin((i * ROW + j) / TOTAL);
            m2[idx] = std::cos((i * ROW + j) / TOTAL);
        }

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < COL; ++j) {
            int idx = i * COL + j;
            mr[idx] = m1[idx] + m2[idx];
        }
    double end = omp_get_wtime();

    double checksum = 0.0;
    for (float v : mr) checksum += v;

    std::cout << std::fixed << std::setprecision(6)
              << "Time (parallel):   " << (end - start)
              << " s, checksum=" << checksum << "\n";
}

int main() {

    // matrix_add_consequent();
    // matrix_add_parallel();

    // matrix_add_consequent();
    // matrix_add_parallel();

    matrix_add_consequent();
    matrix_add_parallel();
    return 0;
}
