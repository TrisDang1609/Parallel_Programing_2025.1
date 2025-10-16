#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <algorithm>

#define ARRAY_SIZE    1000000
#define MAX_VALUE     1000000
#define NUM_OF_FIRST_ARRAYS 10

using namespace std;

void print_arr(vector<int> &arr){
    for (int i = 0; i < ARRAY_SIZE && i < NUM_OF_FIRST_ARRAYS; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void init_arr(vector<int> &arr, vector<int> &arr2, vector<int> &arr3) {
    cout << "start init" << endl;
    int temp;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, MAX_VALUE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = dis(gen);
        arr2[i] = arr[i];
        arr3[i] = arr[i];
    }
}


static inline int co_rank(int k, const vector<int>& L, const vector<int>& R) {
	int n1 = (int)L.size();
	int n2 = (int)R.size();
	int low = max(0, k - n2);
	int high = min(k, n1);
	while (low < high) {
		int i = (low + high) >> 1;
		int j = k - i;
		if (i < n1 && j > 0 && L[i] < R[j - 1]) {
			low = i + 1;
		} else if (i > 0 && j < n2 && L[i - 1] > R[j]) {
			high = i - 1;
		} else {
			return i;
		}
	}
	return low;
}


static inline void merge_range(const vector<int>& L, const vector<int>& R,
							   vector<int>& out, int out_offset,
							   int i_start, int i_end, int j_start, int j_end) {
	int i = i_start, j = j_start, k = out_offset;
	while (i < i_end && j < j_end) {
		if (L[i] <= R[j]) out[k++] = L[i++];
		else out[k++] = R[j++];
	}
	while (i < i_end) out[k++] = L[i++];
	while (j < j_end) out[k++] = R[j++];
}

void merge(vector<int>& arr, int left, int mid, int right) {
	int n1 = mid - left + 1;
	int n2 = right - mid;
	vector<int> L(n1), R(n2);
	for (int i = 0; i < n1; i++)
		L[i] = arr[left + i];
	for (int j = 0; j < n2; j++)
		R[j] = arr[mid + 1 + j];
	int i = 0, j = 0, k = left;
	while (i < n1 && j < n2) {
		if (L[i] <= R[j]) {
			arr[k] = L[i];
			i++;
		} else {
			arr[k] = R[j];
			j++;
		}
		k++;
	}
	while (i < n1) {
		arr[k] = L[i];
		i++;
		k++;
	}
	while (j < n2) {
		arr[k] = R[j];
		j++;
		k++;
	}
}

void mergeSort(vector<int>& arr, int left, int right) {
	if (left < right) {
		int mid = left + (right - left) / 2;
		mergeSort(arr, left, mid);
		mergeSort(arr, mid + 1, right);
		merge(arr, left, mid, right);
	}
}

// OpenMP version
void merge_omp(vector<int>& arr, int left, int mid, int right) {
	int n1 = mid - left + 1;
	int n2 = right - mid;
	vector<int> L(n1), R(n2);
    
	if (n1 > 2048) {
	#pragma omp parallel proc_bind(spread)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n1; ++i) L[i] = arr[left + i];
    }
    } else {
        for (int i = 0; i < n1; ++i) L[i] = arr[left + i];
    }

    if (n2 > 2048) {
		#pragma omp parallel proc_bind(spread)
        {
            #pragma omp for schedule(static)
            for (int j = 0; j < n2; ++j) R[j] = arr[mid + 1 + j];
        }
    } else {
        for (int j = 0; j < n2; ++j) R[j] = arr[mid + 1 + j];
    }

	
	const int total = n1 + n2;
	const int threshold = 1 << 16;
	if (total <= threshold) {
		
		int i = 0, j = 0, k = left;
		while (i < n1 && j < n2) {
			if (L[i] <= R[j]) arr[k++] = L[i++];
			else arr[k++] = R[j++];
		}
		while (i < n1) arr[k++] = L[i++];
		while (j < n2) arr[k++] = R[j++];
	} else {
		// em sử dụng co-rank - mỗi một luồng sẽ merge một phần riêng biệt, cơ mà không hiệu quả lắm ...
		#pragma omp parallel proc_bind(spread)
		{
			int T = omp_get_num_threads();
			int tid = omp_get_thread_num();
			int k_start = (int)((1LL * tid * total) / T);
			int k_end   = (int)((1LL * (tid + 1) * total) / T);
			int i_start = co_rank(k_start, L, R);
			int j_start = k_start - i_start;
			int i_end   = co_rank(k_end, L, R);
			int j_end   = k_end - i_end;
			merge_range(L, R, arr, left + k_start, i_start, i_end, j_start, j_end);
		}
	}
}

void mergeSort_omp(vector<int>& arr, int left, int right) {
	if (left < right) {
		int mid = left + (right - left) / 2;
		mergeSort_omp(arr, left, mid);
		mergeSort_omp(arr, mid + 1, right);
		merge_omp(arr, left, mid, right);
	}
}

// OpenMP tasks version:
static void mergeSort_omp_task_impl(vector<int>& arr, int left, int right, int cutoff) {
	if (right - left + 1 <= cutoff) {
		// small slice: back to sequential
		mergeSort(arr, left, right);
		return;
	}
	int mid = left + (right - left) / 2;
	#pragma omp task shared(arr) if (right - left + 1 > cutoff)
	mergeSort_omp_task_impl(arr, left, mid, cutoff);
	#pragma omp task shared(arr) if (right - left + 1 > cutoff)
	mergeSort_omp_task_impl(arr, mid + 1, right, cutoff);
	#pragma omp taskwait
	merge(arr, left, mid, right);
}

void mergeSort_omp_tasks(vector<int>& arr, int left, int right, int cutoff = 1 << 14) {
	#pragma omp parallel proc_bind(spread)
	{
		#pragma omp single nowait
		{
			mergeSort_omp_task_impl(arr, left, right, cutoff);
		}
	}
}

int main() {
	
	omp_set_dynamic(0);           
	omp_set_num_threads(24);      
    
	vector<int> arr(ARRAY_SIZE), arr2(ARRAY_SIZE), arr3(ARRAY_SIZE);
	init_arr(arr, arr2, arr3);

	
    // Benchmark standard variant
    print_arr(arr);

	auto start = chrono::high_resolution_clock::now();
	mergeSort(arr, 0, ARRAY_SIZE - 1);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double, std::milli> duration = end - start;

	print_arr(arr);

	cout << "Thoi gian chay MergeSort: " << duration.count() << " ms" << endl;



    // Benchmark OpenMP variant on a fresh array
    print_arr(arr2);

    auto start2 = chrono::high_resolution_clock::now();
	mergeSort_omp(arr2, 0, ARRAY_SIZE - 1);
	auto end2 = chrono::high_resolution_clock::now();
	chrono::duration<double, std::milli> duration2 = end2 - start2;

    print_arr(arr2);
	cout << "Thoi gian chay MergeSort_omp: " << duration2.count() << " ms" << endl;



	// Benchmark OpenMP tasks variant (parallel recursion)
    print_arr(arr3);

	auto start3 = chrono::high_resolution_clock::now();
	mergeSort_omp_tasks(arr3, 0, ARRAY_SIZE - 1); // default cutoff ~16K, adjust as needed
	auto end3 = chrono::high_resolution_clock::now();
	chrono::duration<double, std::milli> duration3 = end3 - start3;

    print_arr(arr3);
	cout << "Thoi gian chay MergeSort_omp_tasks: " << duration3.count() << " ms" << endl;

	return 0;
}
