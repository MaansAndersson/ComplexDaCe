import dace as dc
import time
from time import perf_counter
import numpy as np
import os 
from ugly_print_code import *
from sdfg_optimization import *

from dace.transformation.dataflow import (MapReduceFusion, MapFusion,
                                      MapWCRFusion)


from dace.transformation.auto import auto_optimize as aopt

from dace.sdfg.infer_types import set_default_schedule_and_storage_types
    

def make_DFT(r, dtype):
    dftr = np.zeros(shape = (r,r), dtype = dtype)
    dfti = np.zeros(shape = (r,r), dtype = dtype)
    for i in range(r):
        for j in range(r):
            w = np.cos(i*j*2*np.pi/r) - 1j * np.sin(i*j*2*np.pi/r)
            #if abs(np.real(w)) < 1e-15:
                #w = 0 + np.imag(w)
            #if abs(np.imag(w)) < 1e-15:
                #w = 1j*0 + np.real(w)       
            dftr[i,j] = np.real(w)
            dfti[i,j] = np.imag(w)
    dftr = (dftr + dftr.transpose())/2
    dfti = (dfti + dfti.transpose())/2
    return  dftr + 1j*dfti, dftr, dfti

def batched_dftc(
             sdfg : dc.sdfg,  
             dtype : None,
             N : str,   
             M : str,
             is_start_state : bool,              
             src : str, 
             dst : str,
             tasklet_type : str) -> dc.sdfg.state: 
    '''Creates the SDFG for (DFT_m ⊗ I_n )x_mn decided by the input size of x and the operator size DFT_m'''
    '''Or rather for DFT_m @ [x1, y1, x2, y2, ..., ..., x_n, y_n] ''' 
   
    # Create dst node 
    operator_name='batched_dftc'
    state = sdfg.add_state(operator_name+str(int(10000*time.perf_counter_ns())), is_start_state=is_start_state)

    # Create access nodes for reading and writing

    if src.startswith('t'):
        src_node = state.add_access(src)
    elif src.startswith('x'):
        src_node = state.add_read(src)

    if dst.startswith('t'):
        dst_node = state.add_access(dst)
    elif dst.startswith('x'):
        dst_node = state.add_write(dst)

    sdfg.add_transient(operator_name+'t', shape=[N, N, M], dtype=dtype)
    tmp_node = state.add_access(operator_name+'t')

    ranges_dic = {'i' : '0:N', 'j' : '0:N', 'k' : '0:M'} 
    input_dic  = {'inp' : dc.Memlet(data=src_node.data, subset=('j:j,k:k'))} 
    outputs    = {'out' : dc.Memlet(data=tmp_node.data, subset=('i,j,k'))}

    # if tasklet_type == 'Python':
    #     code='''out=(inp[0,0]*(DFTR[j,i]+(1j)*DFTI[j,i]))'''
    #     language = dc.Language.Python
    # if tasklet_type == 'CPP':
    #     language = dc.Language.CPP
    #     if dtype == np.complex128:
    #         code='''out=(inp[0]*(DFTR[j+N*i]+dace::complex128(0,1)*DFTI[j+N*i]));'''
    #     elif dtype == np.complex64:
    #         code='''out=(inp[0]*(DFTR[j+N*i]+dace::complex64(0,1)*DFTI[j+N*i]));'''

    #If DACE_compiler_default_data_types == C
    if tasklet_type == 'Python':
        code='''out=(inp[0,0]*(DFTR[j,i]+(1j)*DFTI[j,i]))'''
        language = dc.Language.Python
    if tasklet_type == 'CPP':
        language = dc.Language.CPP
        if dtype == np.complex128:
            code='''out=(inp[0]*(std::complex<double>(DFTR[j+N*i],DFTI[j+N*i])));'''
        elif dtype == np.complex64:
            code='''out=(inp[0]*(std::complex<float>(DFTR[j+N*i],DFTI[j+N*i])));'''

    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name        = operator_name, 
    map_ranges  = ranges_dic,
    inputs      = input_dic,
    code        = code,
    language    = language, 
    outputs     = outputs,
    schedule    = dc.dtypes.ScheduleType.Default,
    unroll_map  = False,
    propagate   = True,

    )

    state.add_nedge(src_node,
                   map_entry,
                   dc.Memlet(data=src_node.data, subset='0:N,0:M'))
    
    state.add_nedge(map_exit,
                   tmp_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:N,0:M'))

    red_node = state.add_reduce(wcr='lambda a,b: a+b',
                                axes=[1], 
                                identity=0,
                                schedule=dc.dtypes.ScheduleType.Default)

    state.add_nedge(tmp_node,
                   red_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:N,0:M'))

    state.add_nedge(red_node,
                    dst_node,
                    dc.Memlet(data=dst_node.data, subset='0:N,0:M'))

    return state

def batched_dftr2r(
             sdfg : dc.sdfg,  
             dtype : None,
             N : int,
             M : str,   
             is_start_state : bool,              
             src : str, 
             src_complex : str,
             dst : str, 
             dst_complex : str, 
             ) -> dc.sdfg.state: 
    '''Creates the SDFG for (DFT_m ⊗ I_n )x_mn decided by the input size of x and the operator size DFT_m'''
    '''Or rather for DFT_m @ [x1, y1, x2, y2, ..., ..., x_n, y_n] ''' 
   
    # Create dst node 
    operator_name='batched_dft_r2r'
    state = sdfg.add_state(operator_name+str(int(10000*time.perf_counter_ns())), is_start_state=is_start_state)

    # Create access nodes for reading and writing

    if src.startswith('t'):
        src_node = state.add_access(src)
        src_node_complex = state.add_access(src_complex)

    elif src.startswith('x'):
        src_node = state.add_read(src)
        src_node_complex = state.add_read(src_complex)

    if dst.startswith('t'):
        dst_node = state.add_access(dst)
        dst_node_complex = state.add_access(dst_complex)

    elif dst.startswith('y'):
        dst_node = state.add_write(dst)
        dst_node_complex = state.add_read(dst_complex)

    sdfg.add_transient(operator_name+'t', shape=[N, M, N], dtype=dtype)
    tmp_node = state.add_access(operator_name+'t')

    sdfg.add_transient(operator_name+'tc', shape=[N, M, N], dtype=dtype)
    tmp_node_complex = state.add_access(operator_name+'tc')

    ranges_dic = {'i' : '0:N', 'k' : '0:M', 'j' : '0:N'} 

    input_dic  = {'inp' : dc.Memlet(data=src_node.data, subset=('j:j,k:k')),
                  'inp2' : dc.Memlet(data=src_node_complex.data, subset=('j:j,k:k'))} 
    outputs    = {'out' : dc.Memlet(data=tmp_node.data, subset=('i,k,j')),
                  'out2' : dc.Memlet(data=tmp_node_complex.data, subset=('i,k,j'))}

    if dtype == np.float32:
        code='''
float A = DFTR[j+N*i];
float B = DFTI[j+N*i];
float x__ = inp[0];
float y__ = inp2[0];
out = (x__*A - y__*B);
out2=(x__*B + y__*A);
'''
    elif dtype == np.float64:
        code='''
double A = DFTR[j+N*i];
double B = DFTI[j+N*i];
double x__ = inp[0];
double y__ = inp2[0];
out = (x__*A - y__*B);
out2=(x__*B + y__*A);
'''
    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name        =operator_name, 
    map_ranges  =ranges_dic,
    inputs      =input_dic,
    code        =code,
    language    =dc.Language.CPP, 
    outputs     =outputs
    )

    # MAP 
    
    state.add_nedge(src_node,
                   map_entry,
                   dc.Memlet(data=src_node.data, subset='0:N,0:M'))
    
    state.add_nedge(map_exit,
                   tmp_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:M,0:N'))

    state.add_nedge(src_node_complex,
                   map_entry,
                   dc.Memlet(data=src_node_complex.data, subset='0:N,0:M'))
    
    state.add_nedge(map_exit,
                   tmp_node_complex,
                   dc.Memlet(data=tmp_node_complex.data, subset='0:N,0:M,0:N'))

    # REDUCTION

    red_node = state.add_reduce('lambda a,b: a+b', axes=[2], identity=0)
    
    state.add_nedge(tmp_node,
                   red_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:M,0:N'))

    state.add_nedge(red_node,
                    dst_node,
                    dc.Memlet(data=dst_node.data, subset='0:N,0:M'))

    red_node_complex = state.add_reduce('lambda a,b: a+b', axes=[2], identity=0)

    state.add_nedge(tmp_node_complex,
                   red_node_complex,
                   dc.Memlet(data=tmp_node_complex.data, subset='0:N,0:M,0:N'))

    state.add_nedge(red_node_complex,
                    dst_node_complex,
                    dc.Memlet(data=dst_node_complex.data, subset='0:N,0:M'))

    return state

def batched_DFT_AoS(
             sdfg : dc.sdfg,  
             dtype : None,
             N : str,   
             M : str,
             is_start_state : bool,              
             src : str, 
             dst : str,
             ) -> dc.sdfg.state: 
    '''Creates the SDFG for (DFT_m ⊗ I_n )x_mn decided by the input size of x and the operator size DFT_m'''
    '''Or rather for DFT_m @ [x1, y1, x2, y2, ..., ..., x_n, y_n] ''' 
   
    # Create dst node 
    operator_name='batched_DFT_AoS'
    state = sdfg.add_state(operator_name+str(int(10000*time.perf_counter_ns())), is_start_state=is_start_state)

    # Create access nodes for reading and writing

    if src.startswith('t'):
        src_node = state.add_access(src)
    elif src.startswith('x'):
        src_node = state.add_read(src)

    if dst.startswith('t'):
        dst_node = state.add_access(dst)
    elif dst.startswith('y'):
        dst_node = state.add_write(dst)

    sdfg.add_transient(operator_name+'t', shape=[N,M,N], dtype=dtype)
    tmp_node = state.add_access(operator_name+'t')

    ranges_dic = {'j' : '0:N', 'k' : '0:M:2', 'i' : '0:N'} 
    #ranges_dic = {'j' : '0:N', 'i' : '0:N', 'k' : '0:M:2'} 
    input_dic  = {'inp' : dc.Memlet(data=src_node.data, subset=('j:j,k:k+2'))} 
    outputs    = {'out' : dc.Memlet(data=tmp_node.data, subset=('i,k:k+2,j'))}

    #code='''out=inp[0,k%M]*DFTR[j,i]+(-1)**(k+1)*inp[0,(1+k)%M]*DFTI[j,i]'''

#     if dtype == np.float32:
#         code='''
# float A = DFTR[j+N*i];
# float B = DFTI[j+N*i];
# float x__ = inp[0];
# float y__ = inp[1];
# out[0] = (x__*A - y__*B);
# out[1]=(x__*B + y__*A);
# '''
#     elif dtype == np.float64:
#         code='''
# double A = DFTR[j+N*i];
# double B = DFTI[j+N*i];
# double x__ = inp[0];
# double y__ = inp[1];
# out[0] = (x__*A - y__*B);
# out[1]=(x__*B + y__*A);
# '''
    # code='''out=inp[0,k%2]*DFTR[j,i]+(-1)**(k+1)*inp[0,(1+k)%2]*DFTI[j,i]'''

    # code='''out=inp[0+N*k%2]*DFTR[j+N*i]+std::pow((-1),(k+1))*inp[0+N*(1+k)%2]*DFTI[j+N*i];'''
    if dtype == np.float32:
        code='''
A = DFTR[j,i];
B = DFTI[j,i];
x__ = inp[0];
y__ = inp[1];
out[0] = (x__*A - y__*B);
out[1]=(x__*B + y__*A);
'''
    elif dtype == np.float64:
        code='''
A = DFTR[j,i];
B = DFTI[j,i];
x__ = inp[0,0];
y__ = inp[0,1];
out[0] = (x__*A - y__*B);
out[1]=(x__*B + y__*A);
'''

    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name        =operator_name, 
    map_ranges  =ranges_dic,
    inputs      =input_dic,
    code        =code,
    language    =dc.Language.Python, 
    outputs     =outputs
    )
    
    state.add_nedge(src_node,
                   map_entry,
                   dc.Memlet(data=src_node.data, subset='0:N,0:M'))
    
    state.add_nedge(map_exit,
                   tmp_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:M,0:N'))

    red_node = state.add_reduce('lambda a,b: a+b', axes=[2], identity=0)
    
    state.add_nedge(tmp_node,
                   red_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:M,0:N'))

    state.add_nedge(red_node,
                    dst_node,
                    dc.Memlet(data=dst_node.data, subset='0:N,0:M'))

    return state



def batched_dft_r2r_SoA(
             sdfg : dc.sdfg,  
             dtype : None,
             N : int,   
             M : str,
             is_start_state : bool,              
             src : str, 
             dst : str,
             ) -> dc.sdfg.state: 
    '''Creates the SDFG for (DFT_m ⊗ I_n )x_mn decided by the input size of x and the operator size DFT_m'''
    '''Or rather for DFT_m @ [x1, y1, x2, y2, ..., ..., x_n, y_n] ''' 
   
    # Create dst node 
    operator_name='batched_dft_r2r_SoA'
    state = sdfg.add_state(operator_name+str(int(10000*time.perf_counter_ns())), is_start_state=is_start_state)

    # Create access nodes for reading and writing

    if src.startswith('t'):
        src_node = state.add_access(src)
    elif src.startswith('x'):
        src_node = state.add_read(src)

    if dst.startswith('t'):
        dst_node = state.add_access(dst)
    elif dst.startswith('x'):
        dst_node = state.add_write(dst)

    sdfg.add_transient(operator_name+'t', shape=[N, N, M], dtype=dtype)
    tmp_node = state.add_access(operator_name+'t')

    ranges_dic = {'j' : '0:N', 'i' : '0:N', 'k' : '0:M:2'} 
    input_dic  = {'inp' : dc.Memlet(data=src_node.data, subset=('j:j,k:k+2'))} 
    outputs    = {'out' : dc.Memlet(data=tmp_node.data, subset=('i,j,k:k+2'))}

    #code='''out=inp[0,k%M]*DFTR[j,i]+(-1)**(k+1)*inp[0,(1+k)%M]*DFTI[j,i]'''

    if dtype == np.float32:
        code='''
float A = DFTR[j+N*i];
float B = DFTI[j+N*i];
float x__ = inp[0];
float y__ = inp[1];
out[0] = (x__*A - y__*B);
out[1]=(x__*B + y__*A);
'''
    elif dtype == np.float64:
        code='''
double A = DFTR[j+N*i];
double B = DFTI[j+N*i];
double x__ = inp[0];
double y__ = inp[1];
out[0] = (x__*A - y__*B);
out[1]=(x__*B + y__*A);
'''
    # code='''out=inp[0,k%2]*DFTR[j,i]+(-1)**(k+1)*inp[0,(1+k)%2]*DFTI[j,i]'''

    #code='''out=inp[0+N*k%2]*DFTR[j+N*i]+std::pow((-1),(k+1))*inp[0+N*(1+k)%2]*DFTI[j+N*i];'''
    

    tasklet, map_entry, map_exit = state.add_mapped_tasklet(
    name        =operator_name, 
    map_ranges  =ranges_dic,
    inputs      =input_dic,
    code        =code,
    language    =dc.Language.CPP, 
    outputs     =outputs
    )
    
    state.add_nedge(src_node,
                   map_entry,
                   dc.Memlet(data=src_node.data, subset='0:N,0:M'))
    
    state.add_nedge(map_exit,
                   tmp_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:N,0:M'))

    red_node = state.add_reduce('lambda a,b: a+b', axes=[1], identity=0)
    
    state.add_nedge(tmp_node,
                   red_node,
                   dc.Memlet(data=tmp_node.data, subset='0:N,0:N,0:M'))

    state.add_nedge(red_node,
                    dst_node,
                    dc.Memlet(data=dst_node.data, subset='0:N,0:M'))

    return state



def test_batched_DFTc(backend : str, Nr : int, Mr : int, dtype_input : str, aoptBool : bool, tasklet_type : str):

    if dtype_input == 'complex128':
        dtype = np.complex128
        dfttype = np.float64
    elif dtype_input == 'complex64':
        dtype = np.complex64
        dfttype = np.float32
        
    sdfg_name = 'batched_DFTc'+dtype_input+'_'+tasklet_type+'_'+str(Nr)
    sdfg = dc.SDFG(sdfg_name)
    N = dc.symbol('N', dtype=dc.int32)
    M = dc.symbol('M', dtype=dc.int32)

    sdfg.add_array('x', [N, M], dtype=dtype)
    #sdfg.add_array('y', [N, M], dtype=dtype)
    #sdfg.add_transient('t', shape=[N, M], dtype=dtype)

    DFT, DFTr, DFTi = make_DFT(Nr, dfttype)
    sdfg.add_constant('DFTR', DFTr) 
    sdfg.add_constant('DFTI', DFTi)

    bstate = batched_dftc(sdfg, dtype, N, M, True, 'x', 'x', tasklet_type)

    sdfg.fill_scope_connectors()
    #sdfg.apply_strict_transformations()

    #sdfg.simplify()    

    if backend == 'GPU':
        if aoptBool:
            aopt.auto_optimize(sdfg, dc.DeviceType.GPU)
        sdfg.apply_gpu_transformations()
    else: 
        if aoptBool:
            aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    #sdfg.validate()
    #sdfg.is_valid() 

    #aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    #sdfg.apply_transformations([MapFusion, MapWCRFusion])
    #sdfg.apply_transformations(MapReduceFusion)
    
    M = Mr
    N = Nr
    #optimize_for_cpu(sdfg=sdfg, m=M, n=M, k=N)
    F = sdfg.compile()
    T1 = []
    x = (np.random.rand(N,M) + 1j * np.random.rand(N,M)).astype(dtype)
    x1 = x.copy()


    
    t0 = time.perf_counter_ns()    
    F(N = N, M = M, x = x)
    t1 = time.perf_counter_ns()


    
    if dtype_input == 'complex128':
        #assert np.all((1e-10>abs(x-y)))
        #create_main_fileDFTc_complex128(sdfg_name, dtype_input, N, M)
        pass
    elif dtype_input == 'complex64':
        #assert np.all((1e-4>abs(x-y)))
        #create_main_fileDFTc_complex64(sdfg_name, dtype_input, N, M)
        pass


    ##assert (1e-10>max(abs(x-y)))
    return sdfg_name

def test_batched_DFTr2r(backend : str, Nr : int, Mr : int, dtype_input : str, aoptBool : bool):
    if dtype_input == 'complex128':
        dtype = np.float64
        dfttype = np.float64
        dtype_str = 'float64'
    elif dtype_input == 'complex64':
        dtype = np.float32
        dfttype = np.float32
        dtype_str = 'float32'

    aopt_str = ''
    if aoptBool: 
        aopt_str = 'aopt'

    sdfg_name = 'batched_DFTr2r'+'_'+dtype_input+'_'+aopt_str+'_'+str(Nr)
    sdfg = dc.SDFG(sdfg_name)
    N = dc.symbol('N', dtype=dc.int32)
    M = dc.symbol('M', dtype=dc.int32)

    sdfg.add_array('xr', [N, M], dtype=dtype)
    sdfg.add_array('xi', [N, M], dtype=dtype)

    sdfg.add_array('yr', [N, M], dtype=dtype)
    sdfg.add_array('yi', [N, M], dtype=dtype)
    N = Nr

    DFT, DFTr, DFTi = make_DFT(N, dtype)

    sdfg.add_constant('DFTR', DFTr) 
    sdfg.add_constant('DFTI', DFTi)
    
    bstate = batched_dftr2r(sdfg, dtype, N, M, True, 'xr', 'xi', 'yr', 'yi')
    
    sdfg.fill_scope_connectors()
    #sdfg.apply_strict_transformations()

    #sdfg.simplify()    
    sdfg.validate()
    sdfg.is_valid()

    if aoptBool and backend == 'CPU':
        aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    
    if backend == 'GPU':
        if aoptBool:
            aopt.auto_optimize(sdfg, dc.DeviceType.GPU)
        sdfg.apply_gpu_transformations()
    
    optimize_for_cpu(sdfg=sdfg, m=N, n=N, k=M)

    F = sdfg.compile()
    T1 = []
    #for i in range(1):
    M = Mr
    xr = np.random.rand(N,M).astype(dtype) 
    xi = np.random.rand(N,M).astype(dtype)
    
    x1 = xr.copy() + 1j*xi.copy() 
    y = DFT@x1

    t0 = time.perf_counter_ns()    
    F(N = N, M = M, xr = xr, xi = xi, yr = xr, yi = xi)
    t1 = time.perf_counter_ns()
    T1.append((t1-t0))
    x = xr.copy() + 1j*xi.copy() 
    print('DFT-r2r, Mean time:', np.median(T1))

    print(x-y)

    if dtype_input == 'complex128':
        assert np.all(1e-10>(abs(x-y)))
        #create_main_fileDFTr2r_complex128(sdfg_name, dtype_input, N, M)
        pass
    elif dtype_input == 'complex64': 
        #assert np.all(1e-4>(abs(x-y)))
        #create_main_fileDFTr2r_complex64(sdfg_name, dtype_input, N, M)
        pass

    return sdfg_name

def test_batched_DFT_AoS(backend : str, Nr : int, Mr : int, dtype_input : str, aoptBool : bool):
    if dtype_input == 'complex128':
        dtype = np.float64
        dfttype = np.float64
        dtype_str = 'float64'
    elif dtype_input == 'complex64':
        dtype = np.float32
        dfttype = np.float32
        dtype_str = 'float32'
    
    aopt_str = ''
    if aoptBool: 
        aopt_str = 'aopt'

    sdfg_name = 'batched_DFTr2r_AoS'+'_'+dtype_input+'_'+aopt_str+'_'+str(Nr)
    sdfg = dc.SDFG(sdfg_name)
    N = dc.symbol('N', dtype=dc.int32)
    M = dc.symbol('M', dtype=dc.int32)
    
    sdfg.add_array('x', [N, M], dtype=dtype)
    sdfg.add_array('y', [N, M], dtype=dtype)
    #sdfg.add_transient('t', [N, M], dtype=dtype)

    N = Nr

    DFT, DFTr, DFTi = make_DFT(N, dtype)

    sdfg.add_constant('DFTR', DFTr) 
    sdfg.add_constant('DFTI', DFTi)
    
    bstate = batched_DFT_AoS(sdfg, dtype, N, M, True, 'x', 'y')

    sdfg.fill_scope_connectors()
    sdfg.apply_strict_transformations()

    sdfg.simplify()    
    sdfg.validate()
    sdfg.is_valid()

    if aoptBool and backend == 'CPU':
        aopt.auto_optimize(sdfg, dc.DeviceType.CPU)
    
    if backend == 'GPU':
        if aoptBool:
            aopt.auto_optimize(sdfg, dc.DeviceType.GPU)
        sdfg.apply_gpu_transformations()


    sdfg.apply_transformations([MapFusion, MapWCRFusion])
    #sdfg.apply_transformations(MapReduceFusion)
    #sdfg.optimize()
    #F = sdfg.compile()
    T1 = []
    M = Mr
    x = np.random.rand(N,M).astype(dtype) 
    #x = np.ones((N,M)).astype(dtype)
    x1 = x[:,0::2].copy() + 1j*x[:,1::2].copy()
    #x1=x1.reshape([N,int(M/2)])

    #optimize_for_cpu(sdfg=sdfg, m=N, n=N, k=M)
    F2 = sdfg.compile()
    # print(x1)
    # print(x)
    t0 = time.perf_counter_ns()    
    #F(N = N, M = M, x = x)
    t1 = time.perf_counter_ns()
    T1.append((t1-t0))

    print('DFT-r2r-AoS, Mean time:', np.median(T1))
    
    y = DFT@x1

    xxx = np.ones((N,M)).astype(dtype)
    t0 = time.perf_counter_ns()    
    F2(N = N, M = M, x = x, y = xxx)
    t1 = time.perf_counter_ns()

    xx = xxx[:,0::2].copy() + 1j*xxx[:,1::2].copy()
    print('Diff: ',xx-y)

    

    # print('prin xx: ',x)
    # print('prin xx: ',xx)
    #print(y)
    if dtype_input == 'complex128':
        pass
        #assert np.all(1e-10>(abs(xx-y)))
        # create_main_fileDFTr2rN2_complex128(sdfg_name, dtype_str, N, M)
    #elif dtype_input == 'complex64': 
        #assert np.all(1e-4>(abs(xx-y)))
        # create_main_fileDFTr2rN2_complex64(sdfg_name, dtype_str, N, M)


    return sdfg_name

def create_wrapper_function(list_of_functions):
    for func in list_of_functions:
        pass
    pass 

def init_wrapper_function(list_of_functions):
    for func in list_of_functions:
        pass 
    pass

def release_wrapper_function(list_of_functions):
    for func in list_of_functions:
        pass 
    pass

def main():
    dc.Config.set('profiling', value=True)
    dc.Config.set('treps', value=100)
    dc.Config.set('profiling_status',value=False)
    list_of_functions = []
    M = 64
    for N in [64]: #[32, 64, 128, 256, 512]:

        print(N)
        #print('DFTc')
        # list_of_functions.append(test_batched_DFTc      (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex128', aoptBool=False, tasklet_type='Python'))
        #list_of_functions.append(test_batched_DFTc      (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex128', aoptBool=False, tasklet_type='CPP'))
        # #list_of_functions.append(test_batched_DFTc      (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex128', aoptBool=True, tasklet_type='CPP'))
        print('DFT_r2r')
        #list_of_functions.append(test_batched_DFTr2r    (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex128', aoptBool=False))
        # list_of_functions.append(test_batched_DFTr2r    (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex128', aoptBool=True))
        print('DFT_r2r_AoS')
        list_of_functions.append(test_batched_DFT_AoS(backend = 'CPU', Nr=N, Mr=M*2, dtype_input='complex128', aoptBool=False))    
        # list_of_functions.append(test_batched_DFT_AoS(backend = 'CPU', Nr=N, Mr=M*2, dtype_input='complex128', aoptBool=True))
        # #print('DFT_r2r_SoA')
        # print('-----')
        # print('DFTc')
        # # list_of_functions.append(test_batched_DFTc      (backend = 'CPU', Nr=N, dtype_input='complex64', aoptBool=False, tasklet_type='Python'))
        # list_of_functions.append(test_batched_DFTc      (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex64', aoptBool=False, tasklet_type='CPP'))
        # print('DFT_r2r')
        # list_of_functions.append(test_batched_DFTr2r    (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex64' , aoptBool=False))
        # list_of_functions.append(test_batched_DFTr2r    (backend = 'CPU', Nr=N, Mr=M, dtype_input='complex64' , aoptBool=True))
        # print('DFT_r2r_AoS')
        # list_of_functions.append(test_batched_DFT_AoS(backend = 'CPU', Nr=N, Mr=M*2, dtype_input='complex64' , aoptBool=False))
        # list_of_functions.append(test_batched_DFT_AoS(backend = 'CPU', Nr=N, Mr=M*2, dtype_input='complex64' , aoptBool=True))


    #sdfg_name4, time4 = test_batched_DFT_r2r_optimized(Nr)
    #sdfg_name5, time5 = test_batched_DFT_r2r_N2_optimized(Nr=N)
    
    create_compilation_line(list_of_functions)
    #create_compilation_line_CUDA(list_of_functions)
    #create_compilation_line_HIP(list_of_functions) 
    print(list_of_functions)
    #init_wrapper_function()
    #create_wrapper_function()
    #release_wrapper_function()

    print(os.getcwd())

if __name__ == '__main__':
    main()
