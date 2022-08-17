#import click
import dace as dc
import numpy as np
from typing import List, Tuple

# For optimizations
from dace.transformation.dataflow import (DoubleBuffering, MapCollapse, MapExpansion, MapReduceFusion, StripMining,
                                          InLocalStorage, AccumulateTransient, Vectorization)

from dace.transformation import helpers as xfutil


#####################################################################
# Data-centric optimization helpers


def find_map_by_param(sdfg: dc.SDFG, pname: str) -> dc.nodes.MapEntry:
    """ Finds the first map entry node by the given parameter name. """
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dc.nodes.MapEntry) and pname in n.params)


def find_map_and_state_by_param(sdfg: dc.SDFG, pname: str) -> Tuple[dc.nodes.MapEntry, dc.SDFGState]:
    """ Finds the first map entry node by the given parameter name. """
    return next(
        (n, p) for n, p in sdfg.all_nodes_recursive() if isinstance(n, dc.nodes.MapEntry) and pname in n.params)


def find_mapexit_by_param(sdfg: dc.SDFG, pname: str) -> dc.nodes.MapExit:
    """ Finds the first map exit node by the given parameter name. """
    entry, state = find_map_and_state_by_param(sdfg, pname)
    return state.exit_node(entry)


#####################################################################
# Matrix multiplication data-centric optimization schemes

def optimize_for_cpu(sdfg: dc.SDFG, m: int, n: int, k: int):
    """ Optimize the matrix multiplication example for multi-core CPUs. """
    # Ensure integers are 32-bit by default
    dc.Config.set('compiler', 'default_data_types', value='C')

    # Fuse the map and reduce nodes
    sdfg.apply_transformations(MapReduceFusion)
    #sdfg.apply_transformations(MapReduceFusion)


    # # Find multiplication map
    entry = find_map_by_param(sdfg, 'k')

    # # Create a tiling strategy
    divides_evenly = (m % 32 == 0) and (n % 32 == 0) and (k % 256 == 0)
    xfutil.tile(sdfg, entry, divides_evenly, False, k=256, i=32, j=32)
    xfutil.tile(sdfg, entry, divides_evenly, divides_evenly, j=16, i=4)

    print(entry)

    # # Reorder internal map to "k,i,j"
    xfutil.permute_map(entry, [2, 0, 1])
    

    # # Add local storage for B in j tile: we apply InLocalStorage with a
    # # parameter "array" named B, between the two maps of j and i
    regtile_j = find_map_by_param(sdfg, 'tile1_j')
    regtile_i = find_map_by_param(sdfg, 'tile1_i')


    InLocalStorage.apply_to(sdfg, dict(array='x'), node_a=regtile_j, node_b=regtile_i)
    

    if divides_evenly:
        print('Måns bäst!')
        # Add local storage for C
        exit_inner = find_mapexit_by_param(sdfg, 'k')
        exit_rti = find_mapexit_by_param(sdfg, 'tile1_i')
        AccumulateTransient.apply_to(sdfg, dict(array='y', identity=0), map_exit=exit_inner, outer_map_exit=exit_rti)
        
        # Vectorize microkernel map
        postamble = n % 4 != 0
        entry_inner, inner_state = find_map_and_state_by_param(sdfg, 'k')
        Vectorization.apply_to(inner_state.parent,
                               dict(vector_len=64, preamble=False, postamble=postamble),
                               map_entry=entry_inner)

    # # Mark outer tile map as sequential to remove atomics
    find_map_by_param(sdfg, 'tile_k').map.schedule = dc.ScheduleType.Sequential

    # # Collapse maps for more parallelism
    # find_map_by_param(sdfg, 'o0').map.collapse = 2
    tile_i = find_map_by_param(sdfg, 'tile_i')
    tile_j = find_map_by_param(sdfg, 'tile_j')
    MapCollapse.apply_to(sdfg, outer_map_entry=tile_i, inner_map_entry=tile_j)
    tile_ij = find_map_by_param(sdfg, 'tile_i')  # Find newly created map
    tile_ij.map.schedule = dc.ScheduleType.CPU_Multicore
    tile_ij.map.collapse = 2


def optimize_for_gpu(sdfg: dc.SDFG, m: int, n: int, k: int):
    """ Optimize the matrix multiplication example for GPUs. """
    # Ensure integers are 32-bit by default
    dc.Config.set('compiler', 'default_data_types', value='C')

    # Fuse the map and reduce nodes
    sdfg.apply_transformations(MapReduceFusion)

    # Apply GPU transformation
    sdfg.apply_gpu_transformations()

    # Find multiplication map
    entry = find_map_by_param(sdfg, 'k')

    # Create a tiling strategy
    divides_evenly = (m % 64 == 0) and (n % 64 == 0) and (k % 8 == 0)
    xfutil.tile(sdfg, entry, divides_evenly, True, i=64, j=64, k=8)
    xfutil.tile(sdfg, entry, divides_evenly, True, i=8, j=4)

    # Create kernel schedule by collapsing and reordering maps
    gtile_i = find_map_by_param(sdfg, 'tile_i')
    gtile_j = find_map_by_param(sdfg, 'tile_j')
    btile_i = find_map_by_param(sdfg, 'tile1_i')
    btile_j = find_map_by_param(sdfg, 'tile1_j')
    MapCollapse.apply_to(sdfg, outer_map_entry=gtile_i, inner_map_entry=gtile_j, permissive=True)
    MapCollapse.apply_to(sdfg, outer_map_entry=btile_i, inner_map_entry=btile_j, permissive=True)
    btile = find_map_by_param(sdfg, 'tile1_i')
    btile.map.schedule = dc.ScheduleType.GPU_ThreadBlock

    # Add local storage (shared memory) for A and B on GPU
    ktile = find_map_by_param(sdfg, 'tile_k')
    smem_a = InLocalStorage.apply_to(sdfg, dict(array='A'), node_a=ktile, node_b=btile)
    smem_b = InLocalStorage.apply_to(sdfg, dict(array='B'), node_a=ktile, node_b=btile)
    sdfg.arrays[smem_a.data].storage = dc.StorageType.GPU_Shared
    sdfg.arrays[smem_b.data].storage = dc.StorageType.GPU_Shared

    # Add local storage (registers) for A and B
    ttile = find_map_by_param(sdfg, 'k')
    warptile, ttile = xfutil.extract_map_dims(sdfg, ttile, [2])
    InLocalStorage.apply_to(sdfg, dict(array='trans_gpu_A'), node_a=warptile, node_b=ttile)
    InLocalStorage.apply_to(sdfg, dict(array='trans_gpu_B'), node_a=warptile, node_b=ttile)

    # Add local storage (registers) for C
    state = next(s for s in sdfg.nodes() if warptile in s.nodes())
    warptile_exit = state.exit_node(warptile)
    btile_exit = state.exit_node(btile)
    AccumulateTransient.apply_to(sdfg, map_exit=warptile_exit, outer_map_exit=btile_exit)
    # Set C tile to zero on allocation
    c_access = next(n for n in state.data_nodes() if n.data == 'trans_gpu_C')
    c_access.setzero = True

    # Unroll microkernel maps
    ttile.map.unroll = True

    # Apply double-buffering on shared memory
    DoubleBuffering.apply_to(sdfg, map_entry=ktile, transient=smem_a)
