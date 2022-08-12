#from ugly_print_code import *
import numpy as np
import dace as dc
import os


def create_all_main_files(list_of_functions,list_of_native_type,lsit_of_dimensions,N):

    for i, function in enumerate(list_of_functions):
        path = os.getcwd()+'/main_'+function+'.cpp'
        function_file = open(path,'w')

        allocation_size = lsit_of_dimensions[i]
        type = list_of_native_type[i]
        main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. """

        main2="""dace::"""+type+""" * __restrict__ x = (dace::"""+type+"""*) calloc("""+allocation_size+""", sizeof(dace::"""+type+"""));""" 
        
        main2="""dace::"""+type+""" * __restrict__ x = (dace::"""+type+"""*) calloc("""+allocation_size+""", sizeof(dace::"""+type+"""));"""

        main3="""double * __restrict__ xi = (double*) calloc(N, sizeof(double));
                 double * __restrict__ xr = (double*) calloc(N, sizeof(double));"""
        
        main4="""float * __restrict__ xi = (float*) calloc(N, sizeof(float));
                 float * __restrict__ xr = (float*) calloc(N, sizeof(float));"""


        main5="""tstart = std::chrono::high_resolution_clock::now();"""
        
        main6="""__program_"""+function+"""(handle, x, N);"""
        main6="""__program_"""+function+"""(handle, xr, xi, N);"""
        
        main7="""tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(x);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
        function_file.write(main)

def create_main_fileDFTc_complex64(function,type,N,M):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')

    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    long long M = """+str(M)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    dace::complex64 * __restrict__ x = (dace::complex64*) calloc(M * N, sizeof(dace::complex64));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(M, N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        dace::complex64 * __restrict__ x = (dace::complex64*) calloc(M * N, sizeof(dace::complex64));
        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, x, M, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(x);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)

def create_main_fileDFTc_complex128(function,type,N,M):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')

    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    long long M = """+str(M)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    dace::complex128 * __restrict__ x = (dace::complex128*) calloc(M * N, sizeof(dace::complex128));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(M, N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        dace::complex128 * __restrict__ x = (dace::complex128*) calloc(M * N, sizeof(dace::complex128));
        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, x, M, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(x);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)
    
def create_main_fileDFTr2r_complex128(function,type,N,M):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')
    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    long long M = """+str(M)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    double * __restrict__ xi = (double*) calloc(N, sizeof(double));
    double * __restrict__ xr = (double*) calloc(N, sizeof(double));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(M, N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        double * __restrict__ xi = (double*) calloc(M * N, sizeof(double));
        double * __restrict__ xr = (double*) calloc(M * N, sizeof(double));
        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, xi, xr, M, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(xi);
    free(xr);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)

    
def create_main_fileDFTr2r_complex64(function,type,N,M):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')
    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    long long M = """+str(M)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    float * __restrict__ xi = (float*) calloc(M * N, sizeof(float));
    float * __restrict__ xr = (float*) calloc(M * N, sizeof(float));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(M, N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        float * __restrict__ xi = (float*) calloc(M * N, sizeof(float));
        float * __restrict__ xr = (float*) calloc(M * N, sizeof(float));
        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, xi, xr, M, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(xi);
    free(xr);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)

def create_main_fileDFTr2rN2_complex64(function,type,N,M):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')
    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    float * __restrict__ x = (float*) calloc((2 * N), sizeof(float));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        float * __restrict__ x = (float*) calloc((2 * N), sizeof(float));

        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, x, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(x);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)


def create_main_fileDFTr2rN2_complex128(function,type,N):
    path = os.getcwd()+'/main_'+function+'.cpp'
    function_file = open(path,'w')
    main="""
//This code was generated by batched_dft.py
#include <cstdlib>
#include ".dacecache/"""+function+"""/include/"""+function+""".h"
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    
    long long N = """+str(N)+""";
    int runs = atoi(argv[1]); //Well lol; 
    double time[runs];
    auto tstart =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    auto tstop =  std::chrono::high_resolution_clock::now(); // Remove ::now()
    double * __restrict__ x = (double*) calloc((2 * N), sizeof(double));
    
    """+function+"""Handle_t handle;
        handle = __dace_init_"""+function+"""(N);

        for(int i = 0; i < runs; i++){
        //change to not only allocate new but re-write random. 

        double * __restrict__ x = (double*) calloc((2 * N), sizeof(double));

        tstart = std::chrono::high_resolution_clock::now();
        
        __program_"""+function+"""(handle, x, N);
        
        tstop = std::chrono::high_resolution_clock::now(); 

        time[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(tstop-tstart).count();
    };
    
    
    __dace_exit_"""+function+"""(handle);

    free(x);

    for(int i = 0; i<runs; i++){
        std::cout << time[i] << std::endl;
    }


    return 0;
}
"""
    function_file.write(main)


def create_compilation_line(list_of_functions):
    path = os.getcwd()+'/compilation_line.sh'
    compilation_line = open(path,'w')
    
    compiler = """g++"""
    firstlines = """
#!/bin/bash
include=/home/mans/.local/lib/python3.8/site-packages/dace/runtime/include
src=.
"""
    compilation_line.write(firstlines)
    
    for function in list_of_functions: 
        newline = """
"""+compiler+""" -o """+function+""" -fopenmp -O3 main_"""+function+""".cpp -I$include \
 $src/.dacecache/"""+function+"""/src/cpu/"""+function+""".cpp $src/.dacecache/"""+function+"""/include/"""+function+""".h"""

        compilation_line.write(newline)
    
