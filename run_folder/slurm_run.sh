#!/bin/bash
#SBATCH --account=deepsea
#SBATCH --job-name="GPU DaCe - Complex Analysis"
#SBATCH -N 1 -p dp-esb
#SBATCH --output=ComplexDace-out.%j
#SBATCH --error=ComplexDace-err.%j
#SBATCH --time=00:30:00


#module load GCC/10.3.0
#module load ParaStationMPI/5.4.9-1
#module load CMake/3.18.0 Score-P/6.0 
#module load PAPI/6.0.0
#module load Scalasca/2.6 


module load GCCcore/.11.2.0
module load Python/3.9.6
module load CMake/3.23.1

module load CUDA

#export SCOREP_FILTERING_FILE=scfilt 

#export SCOREP_PROFILING_ENABLE_CORE_FILES=1
#export SCOREP_TOTAL_MEMORY="4GB"
#export SCOREP_MEM="1GB"
#export SCOREP_ENABLE_PROFILING="1"
#export SCOREP_PROFILING=true
#export SCOREP_ENABLE_TRACING="1"
#export SCOREP_TRACING=true
#export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--thread=omp"
#export PAPI_METRICS="PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_REF_CYC,PAPI_SP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP"
#SCOREP_VERBOSE=true

application=/work/deepsea/andersson1/High-level-SDFG/ComplexDaCe/src/batched_dft.py

A=RUN_$OMP_NUM_THREADS
mkdir $A
cd $A
echo $A
echo $application
python3 $application 
cd ..








