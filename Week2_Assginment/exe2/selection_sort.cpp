#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <ctime>

// Selection Sort song song hóa với OpenMP
void parallel_selection_sort(std::vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        int min_value = arr[i];

        // Tìm phần tử nhỏ nhất từ i+1 → n-1 (song song)
        #pragma omp parallel
        {
            int local_min_index = min_index;
            int local_min_value = min_value;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min_value) {
                    local_min_value = arr[j];
                    local_min_index = j;
                }
            }

            // Kết hợp kết quả các thread
            #pragma omp critical
            {
                if (local_min_value < min_value) {
                    min_value = local_min_value;
                    min_index = local_min_index;
                }
            }
        }

        // Hoán đổi nếu tìm thấy min mới
        if (min_index != i) {
            std::swap(arr[i], arr[min_index]);
        }
    }
}

int main() {
    const int N = 100000; // N nhỏ thôi để demo
    std::vector<int> arr(N);

    // Random data
    std::mt19937 rng(time(0));
    std::uniform_int_distribution<int> dist(0, 100000);
    for (int i = 0; i < N; i++) arr[i] = dist(rng);

    double start = omp_get_wtime();
    parallel_selection_sort(arr);
    double end = omp_get_wtime();

    std::cout << "Time elapsed: " << (end - start) << " seconds\n";

    // Kiểm tra mảng đã sort đúng chưa (in 10 phần tử đầu)
    for (int i = 0; i < 10; i++) std::cout << arr[i] << " ";
    std::cout << "\n";

    return 0;
}
