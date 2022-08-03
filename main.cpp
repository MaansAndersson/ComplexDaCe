#include "/home/mansande/turbo-engine-func/.dacecache/single_map_complex64/include/single_map_complex64.h"
#include "/home/mansande/turbo-engine-func/.dacecache/single_map_complex128/include/single_map_complex128.h"
#include "/home/mansande/turbo-engine-func/.dacecache/single_map_Mcomplex64/include/single_map_Mcomplex64.h"
#include "/home/mansande/turbo-engine-func/.dacecache/single_map_Mcomplex128/include/single_map_Mcomplex128.h"
#include "/home/mansande/turbo-engine-func/.dacecache/single_map_float32/include/single_map_float32.h"
#include "/home/mansande/turbo-engine-func/.dacecache/single_map_float64/include/single_map_float64.h"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>


double sum(double A[], int N){
double a = 0; 
for (int i = 0; i<N; i++){
	a += A[i];
}
return a;
};

double mean(double A[], int N){
double a = sum(A,N); 
return a/=N;
}

double inner(double A[], double B[], int N){
double a = 0; 
for (int i = 0; i<N; i++){
	a += A[i]*B[i];
}
return a; 
}

double stdA(double A[], int N){
double meanA = mean(A,N);
return std::sqrt(inner(A,A,N)/N-meanA*meanA);
}

int main(int argc, char **argv) {

using namespace std::chrono;
int N = atoi(argv[1]); // WELL 

//StockahmFFTHandle_t handle;

single_map_complex64Handle_t handle_single_map_complex64; 
single_map_complex128Handle_t handle_single_map_complex128; 
single_map_float32Handle_t handle_single_map_float32;
single_map_float64Handle_t handle_single_map_float64;
single_map_Mcomplex64Handle_t handle_single_map_Mcomplex64;
single_map_Mcomplex128Handle_t handle_single_map_Mcomplex128;

dace::complex64 * __restrict__ xcomplex64 = (dace::complex64*) calloc(N, sizeof(dace::complex64)); 
dace::complex128 * __restrict__ xcomplex128 = (dace::complex128*) calloc(N, sizeof(dace::complex128));
dace::float32 * __restrict__ xfloat32 = (dace::float32*) calloc(N, sizeof(dace::float32)); 
dace::float64 * __restrict__ xfloat64 = (dace::float64*) calloc(N, sizeof(dace::float64)); 
dace::float32 * __restrict__ xMcomplex64 = (dace::float32*) calloc(2*N, sizeof(dace::float32)); 
dace::float64 * __restrict__ xMcomplex128 = (dace::float64*) calloc(2*N, sizeof(dace::float64));

// auto tstart, tstop;


int runs = 1000; 
double t_map_complex64[runs];
double t_map_complex128[runs];
double t_map_float32[runs];
double t_map_float64[runs];
double t_map_Mcomplex64[runs];
double t_map_Mcomplex128[runs]; 

//std::cout<<runs<<std::endl;
//std::cout<<N<<std::endl;
//handle = __dace_init_StockahmFFT(N, k, r); 

handle_single_map_complex64 = __dace_init_single_map_complex64(N);
handle_single_map_complex128 = __dace_init_single_map_complex128(N);
handle_single_map_float32 = __dace_init_single_map_float32(N);
handle_single_map_float64 = __dace_init_single_map_float64(N);
handle_single_map_Mcomplex64 =  __dace_init_single_map_Mcomplex64(N);
handle_single_map_Mcomplex128 = __dace_init_single_map_Mcomplex128(N);

__program_single_map_complex64(handle_single_map_complex64, xcomplex64, N);
__program_single_map_complex128(handle_single_map_complex128, xcomplex128, N);
__program_single_map_float32(handle_single_map_float32, xfloat32, N);
__program_single_map_float64(handle_single_map_float64, xfloat64, N);
__program_single_map_Mcomplex64(handle_single_map_Mcomplex64, xMcomplex64, N);
__program_single_map_Mcomplex128(handle_single_map_Mcomplex128, xMcomplex128, N);

for(int i = 0; i<runs; i++){

	dace::complex64 * __restrict__ xcomplex64 = (dace::complex64*) calloc(N, sizeof(dace::complex64)); 
	dace::complex128 * __restrict__ xcomplex128 = (dace::complex128*) calloc(N, sizeof(dace::complex128));
	dace::float32 * __restrict__ xfloat32 = (dace::float32*) calloc(N, sizeof(dace::float32)); 
	dace::float64 * __restrict__ xfloat64 = (dace::float64*) calloc(N, sizeof(dace::float64)); 
	dace::float32 * __restrict__ xMcomplex64 = (dace::float32*) calloc(2*N, sizeof(dace::float32)); 
	dace::float64 * __restrict__ xMcomplex128 = (dace::float64*) calloc(2*N, sizeof(dace::float64));
	

//#__program_StockahmFFT(handle, x, N, k, r);

	auto tstart = high_resolution_clock::now();
	__program_single_map_complex64(handle_single_map_complex64, xcomplex64, N);
	auto tstop = high_resolution_clock::now();
	t_map_complex64[i]=duration_cast<nanoseconds>(tstop-tstart).count();

	tstart = high_resolution_clock::now();
	__program_single_map_complex128(handle_single_map_complex128, xcomplex128, N);
	tstop = high_resolution_clock::now();
	t_map_complex128[i]=duration_cast<nanoseconds>(tstop-tstart).count();

	tstart = high_resolution_clock::now();
	__program_single_map_float32(handle_single_map_float32, xfloat32, N);
	tstop = high_resolution_clock::now();
	t_map_float32[i]=duration_cast<nanoseconds>(tstop-tstart).count();

	tstart = high_resolution_clock::now();
	__program_single_map_float64(handle_single_map_float64, xfloat64, N);
	tstop = high_resolution_clock::now();
	t_map_float64[i]=duration_cast<nanoseconds>(tstop-tstart).count();

	tstart = high_resolution_clock::now();
	__program_single_map_Mcomplex64(handle_single_map_Mcomplex64, xMcomplex64, N);
	tstop = high_resolution_clock::now();
	t_map_Mcomplex64[i]=duration_cast<nanoseconds>(tstop-tstart).count();

	tstart = high_resolution_clock::now();
	__program_single_map_Mcomplex128(handle_single_map_Mcomplex128, xMcomplex128, N);
	tstop = high_resolution_clock::now();
	t_map_Mcomplex128[i]=duration_cast<nanoseconds>(tstop-tstart).count();


};
//auto stop = high_resolution_clock::now();
//__dace_exit_StockahmFFT(handle);
__dace_exit_single_map_complex64(handle_single_map_complex64);
__dace_exit_single_map_complex128(handle_single_map_complex128);
__dace_exit_single_map_float32(handle_single_map_float32);
__dace_exit_single_map_float64(handle_single_map_float64);


free(xcomplex64);
free(xcomplex128);
free(xfloat32);
free(xfloat64);



//auto duration = duration_cast<microseconds>(stop - start);
//std::cout<<duration.count() <<std::endl;

/*
std::cout<<mean(t_map_complex64,runs) <<std::endl;
std::cout<<stdA(t_map_complex64,runs) <<std::endl;

std::cout<<mean(t_map_complex128,runs) <<std::endl;
std::cout<<stdA(t_map_complex128,runs) <<std::endl;

std::cout<<mean(t_map_float32,runs) <<std::endl;
std::cout<<stdA(t_map_float32,runs) <<std::endl;

std::cout<<mean(t_map_float64,runs) <<std::endl;
std::cout<<stdA(t_map_float64,runs) <<std::endl;


std::cout<<"map_complex64";
	std::cout<<",";
	std::cout<<"map_complex128";
	std::cout<<",";
	std::cout<<"map_float32";
	std::cout<<",";
	std::cout<<"map_float64";
	std::cout<<",";
	std::cout<<"map_Mcomplex64";
	std::cout<<",";
	std::cout<<"map_Mcomplex128"	<<std::endl;

*/
for(int i = 0; i<runs; i++){
	std::cout<<t_map_complex64[i];
	std::cout<<",";
	std::cout<<t_map_complex128[i];
	std::cout<<",";
	std::cout<<t_map_float32[i];
	std::cout<<",";
	std::cout<<t_map_float64[i];
	std::cout<<",";
	std::cout<<t_map_Mcomplex64[i];
	std::cout<<",";
	std::cout<<t_map_Mcomplex128[i]	<<std::endl;


}

}
