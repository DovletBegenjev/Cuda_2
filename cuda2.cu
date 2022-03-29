#include <iostream>
#include <omp.h>
#include <ctime>
#include <fstream>
#include <cstdlib>
#include <cuda.h>
#include <math.h>

using namespace std;

const int arr_size = 1000;
#define block_size 1024

// Евклидова норма матрицы
__global__ void EuclidNorm(float *arr1, float *S)
{
    // Статическое выделение разделяемой памяти внутри кэш-памяти L1 мультипроцессора  
    __shared__ float cache[block_size];
    
    // Глобальный индекс потока
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // Номер потока внутри блока
    int cacheId = threadIdx.x;
	
	float temp  = 0;
    while(i < arr_size)
    {
        for (int j = 0; j < arr_size; ++j)
		{
			temp += arr1[i*arr_size + j] * arr1[i*arr_size + j];
		}
        
        i += gridDim.x * blockDim.x;    
    }
	
	// Копируем частичные результаты в разделяемую память. 
    // Для каждого блока потоков записывается в отдельный массив cache
    cache[cacheId] = temp;
    // барьерная синхронизация для каждого блока потоков (для каждого мультипроцессора)
    __syncthreads();
    
    // Операция редукции
    int k = blockDim.x/2;
    while(k != 0)
    {
        if(cacheId < k)
            cache[cacheId] += cache[cacheId + k];
        __syncthreads(); // нужна синхронизация при каждом обращении к разделяемой памяти
        k /= 2;
    }
    
    if(cacheId == 0)
        S[blockIdx.x] = cache[0];  // копируем результаты из каждого блока потоков в массив scal (__syncthreads() здесь необязательно)
}

// C = ||A|| * A + B - A
__global__ void sum(float *arr1, float *arr2, float *arr3, float sqrtS)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < arr_size)
    {
        for (int j = 0; j < arr_size; ++j)
		{
			arr3[i*arr_size + j] = sqrtS * arr1[i*arr_size + j] + arr2[i*arr_size + j] - arr1[i*arr_size + j];
		}
        
        i += gridDim.x * blockDim.x;    
    }
}

int main(int argc, char** argv)
{
	float *arr1, *arr2, *arr3, *S;
	float *d_arr1, *d_arr2, *d_arr3, *d_S;
	
	// Выделение памяти на CPU
	arr1 = (float*)malloc(arr_size * arr_size * sizeof(float));
	arr2 = (float*)malloc(arr_size * arr_size * sizeof(float));
	arr3 = (float*)malloc(arr_size * arr_size * sizeof(float));
	S = (float*)malloc(((arr_size + block_size - 1) / block_size) * sizeof(float));
	
    // Выделение памяти на GPU
    cudaMalloc((void**)&d_arr1, arr_size * arr_size * sizeof(float));
    cudaMalloc((void**)&d_arr2, arr_size * arr_size * sizeof(float));
    cudaMalloc((void**)&d_arr3, arr_size * arr_size * sizeof(float)); 
	cudaMalloc((void**)&d_S, ((arr_size + block_size - 1) / block_size) * sizeof(float));
	
	// Обнуление матрицы C на CPU
    memset(arr3, 0, arr_size * arr_size * sizeof(float));
	memset(S, 0, ((arr_size + block_size - 1) / block_size) * sizeof(float));
	
    // Обнуление матрицы C на GPU
    cudaMemset(d_arr3, 0, arr_size * arr_size * sizeof(float));
	cudaMemset(d_S, 0, ((arr_size + block_size - 1) / block_size) * sizeof(float));

	// Считаем матрицу из файла
	ifstream in("numbers1.txt");
	for (long long i = 0; i < arr_size; i++)
		for (long long j = 0; j < arr_size; j++)
			in >> arr1[i*arr_size + j];
	in.close();

	in.open("numbers2.txt");
	for (long long i = 0; i < arr_size; i++)
		for (long long j = 0; j < arr_size; j++)
			in >> arr2[i*arr_size + j];
	in.close();

	double time = omp_get_wtime();
	
	// Копирование данных из оперативной памяти в видеопамять
	cudaMemcpy(d_arr1, arr1, arr_size * arr_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, arr_size * arr_size * sizeof(float), cudaMemcpyHostToDevice);
	
	// Задаем размерность блока (количество потоков в блоке) и сетки (количество блоков)
    dim3 block(block_size); 
    dim3 grid((arr_size + block_size - 1) / block_size);

	// вызов функции ядра (выполняется на каждом потоке)
    EuclidNorm<<<grid, block>>>(d_arr1, d_S);
	
	// Копирование данных из видеопамяти в оперативную память
    cudaMemcpy(S, d_S, ((arr_size + block_size - 1) / block_size) * sizeof(float), cudaMemcpyDeviceToHost);
	
	float sqrtS = 0;
	for (int i = 0; i < ((arr_size + block_size - 1) / block_size); ++i)
	{
		sqrtS += S[i];
	}
	sqrtS = sqrt(sqrtS);
	cout << "S = " << sqrtS << endl;
	
	//cudaDeviceSynchronize();
		
	sum<<<grid, block>>>(d_arr1, d_arr2, d_arr3, sqrtS);
	
	// Копирование данных из видеопамяти в оперативную память
    cudaMemcpy(arr3, d_arr3, arr_size * arr_size * sizeof(float), cudaMemcpyDeviceToHost);

	double time2 = omp_get_wtime() - time;
	cout << "time = " << time2 << endl;

	ofstream out("out.txt");
	for (int i = 0; i < arr_size; i++)
	{
		for (int j = 0; j < arr_size; j++)
		{
			out << arr3[i*arr_size + j] << " ";
		}
		out << endl;
	}
		
	// освобождение памяти на CPU
    free(arr1);
    free(arr2);
    free(arr3);
	free(S);
	
    // освобождение памяти на GPU
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_arr3);
	cudaFree(d_S);
        
    return 0;
}
