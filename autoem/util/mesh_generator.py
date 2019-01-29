from taskqueue import TaskQueue, MockTaskQueue
import igneous.task_creation as tc
from cloudvolume.lib import Vec
from queue import Queue
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import sys
import argparse
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = MPI.Get_processor_name()
class mpiTaskQueue():
    def __init__(self, queue_name='', queue_server=''):
        self._queue = []
        pass

    def insert(self, task):
        self._queue.append(task)

    def run(self, ind, use_tqdm=False):
        if use_tqdm:
            for i in tqdm(ind):
                self._queue[i].execute()
        else:
            for i in ind:
                self._queue[i].execute()

    def clean(self, ind):
        del self._queue
        self._queue = []
        pass
    def wait(self, progress=None):
        return self

    def kill_threads(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

def main():
    ''' run with 
    mpiexex -n 16 mesh_generator $LABEL_PATH
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument( 'labels', default=None, help="path to precomputed labels")
    parser.add_argument( '--verbose', default=False, help="wether to use progressbar")
    parser.add_argument( '--dim_size', default='64,64,64', help="mesh chunksize")
    args = parser.parse_args()

    if rank == 0:
        in_path = 'file://'+args.labels
        mip = 0
        dim_size = tuple(int(d) for d in args.dim_size.split(','))
        print(dim_size)

        print("Making meshes...")
        mtq = mpiTaskQueue()
        tc.create_meshing_tasks(mtq, in_path, mip, shape=Vec(*dim_size))
        L = len(mtq._queue)
        #print('total', rank,size, L)
        all_range = np.arange(L)
        sub_ranges = np.array_split(all_range, size)
        #print(sub_ranges)
    else:
        sub_ranges = None
        mtq = None

    sub_ranges = comm.bcast(sub_ranges, root=0)
    mtq = comm.bcast(mtq, root=0)
    mtq.run(sub_ranges[rank], args.verbose)
    comm.barrier()
    if rank == 0:
        mtq.clean(all_range)
        print("Cleaned", len(mtq._queue))
        print("Updating metadata...")
        tc.create_mesh_manifest_tasks(mtq, in_path)
        print(len(mtq._queue))
        all_range = np.arange(L)
        sub_ranges = np.array_split(all_range, size)
    else:
        sub_ranges = None
        mtq = None
    
    sub_ranges = comm.bcast(sub_ranges, root=0)
    mtq = comm.bcast(mtq, root=0)
    mtq.run(sub_ranges[rank], args.verbose)
    #mtq.run(sub_ranges[rank])
    comm.barrier()
    if rank == 0:
        command = r'gunzip {}/mesh/*.gz'.format(args.labels)
        print(command)
        subprocess.call(command, shell=True)
        print("Done!")
    
if __name__ == '__main__':
    main()