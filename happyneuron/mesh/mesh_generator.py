#!/usr/bin/env python3
"""Create a mesh based on a CloudVolume segmentation.

Classes
-------
MPITaskQueue
    Parallelize tasks over MPI.

Functions
---------
parse_args
    Collect command line arguments.
generate_mesh
    Generate a mesh of label data.
"""

import argparse
import logging
import os
from queue import Queue
import re
import sys
import subprocess

from cloudvolume.lib import Vec
import igneous.task_creation as tc
from mpi4py import MPI
import numpy as np
from taskqueue import TaskQueue, MockTaskQueue


# Get MPI metadata.
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
HOST = MPI.Get_processor_name()


# Set up logging to STDOUT with rank information.
LOGGER = logging.getLogger('mesh_generator_v3.py')
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)s Rank %(rank)s : %(message)s')
syslog.setFormatter(formatter)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(syslog)
LOGGER = logging.LoggerAdapter(LOGGER, {'rank': str(RANK)})


class MPITaskQueue():
    """Parallelize tasks over MPI.

    Parameters
    ----------
    queue_name : str
        Unused.
    queue_server : str
        Unused.
    """
    def __init__(self, queue_name='', queue_server=''):
        self._queue = []
        pass

    def insert(self, task):
        """Add a task to the queue.

        Parameters
        ----------
        task
            A TaskQueue task to run in the future.
        """
        self._queue.append(task)

    def run(self, ind):
        """Run a subset of tasks in the queue sequentially.

        Parameters
        ----------
        ind : list of int
            Indices of tasks to run.
        """
        for i in ind:
            self._queue[i].execute()

    def clean(self, ind):
        """Clear the queue.

        Parameters
        ----------
        ind : list of int
            Indices of tasks to remove. Unused.
        """
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


def parse_args(args=None):
    """Collect command line arguments.

    Parameters
    ----------
    args : str, optional
        String containing arguments to parse. If ``None``, parse from sys.argv.

    Returns
    -------
    args : argparse.Namespace
        Command line arguments stored in an object.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--labels', type=str,
                   help="path to precomputed labels")
    p.add_argument('--dim_size', type=int, nargs='*', default=[64, 64, 64],
                   help="mesh chunk size")
    p.add_argument('--quiet', action='store_true',
                   help="pass to disable logging")

    return p.parse_args()


def generate_mesh(labels, dim_size=(64, 64, 64), quiet=False):
    """Generate a mesh of label data.

    Create a mesh from a CloudVolume segmentation layer and save the result in
    the correct place for visualization. Mesh generation is parallelized over
    MPI ranks.

    Parameters
    ----------
    labels : str
        Path to a CloudVolume layer of segmentation labels.
    dim_size : tuple of int
        The size of the subvolume assigned to each task.
    quiet : bool
        If True, suppress stdout logging output.

    Notes
    -----
    To take advantage of MPI parallelism, this script must be run as
    ``mpiexec -n <n_ranks> python mesh_generator_v3.py <...args>``.
    """
    args = parse_args()

    # Disable logging if requested.
    if args.quiet:
        LOGGER.logger.removeHandler(syslog)
        noop = logging.NullHandler()
        LOGGER.logger.addHandler(noop)

    LOGGER.info('Starting on host {}'.format(HOST))

    # Set up the meshing task queue on rank 0.
    if RANK == 0:
        # Load in the layer data.
        LOGGER.info('Loading CloudVolume layer at {}'.format(args.labels))
        if os.path.isdir(args.labels) and not re.search(r'^file://', args.labels):
            in_path = 'file://' + args.labels
        mip = 0
        LOGGER.info('Meshing volume with dimensions {}'.format(dim_size))

        # Create a queue of meshing tasks over subvolumes of the layer.
        LOGGER.info("Setting up meshing task queue.")
        mtq = MPITaskQueue()
        tasks = tc.create_meshing_tasks(layer_path=in_path,
                                        mip=mip,
                                        shape=Vec(*dim_size),
                                        mesh_dir='mesh')

        for t in tasks:
            mtq.insert(t)

        # Compute the tasks for each rank to complete.
        LOGGER.info('{} tasks created.'.format(len(mtq._queue)))
        L = len(mtq._queue)
        all_range = np.arange(L)
        sub_ranges = np.array_split(all_range, SIZE)
    else:
        sub_ranges = None
        mtq = None

    # Synchronize and broadast the task queue and assigned tasks to all ranks.
    sub_ranges = COMM.bcast(sub_ranges, root=0)
    mtq = COMM.bcast(mtq, root=0)

    # Run the tasks assigned to this rank, then wait for all to finish.
    LOGGER.info('Running task queue 1.')
    mtq.run(sub_ranges[RANK])
    LOGGER.info('Finished task queue 1.')
    COMM.barrier()

    # Set up the metadata update queue
    if RANK == 0:
        LOGGER.info('Cleaning {} tasks.'.format(len(mtq._queue)))
        mtq.clean(all_range)

        # Create a queue of `igneous` metadata update tasks.
        LOGGER.info('Setting up manifest update task queue.')
        mtq2 = MPITaskQueue()
        tasks = tc.create_mesh_manifest_tasks(in_path, magnitude=3)

        for t in tasks:
            mtq2.insert(t)

        # Compute the tasks for each rank to complete.
        LOGGER.info('Created {} mesh manifest tasks.'.format(len(mtq2._queue)))
        L2 = len(mtq2._queue)
        all_range = np.arange(L2)
        sub_ranges = np.array_split(all_range, SIZE)
    else:
        sub_ranges = None
        mtq2 = None

    # Synchronize and broadcast the metadata update task queue and assignments.
    sub_ranges = COMM.bcast(sub_ranges, root=0)
    mtq2 = COMM.bcast(mtq2, root=0)

    # Run the metadata update queue.
    LOGGER.info('Running task queue 2.')
    mtq2.run(sub_ranges[RANK])
    LOGGER.info('Finished task queue 2.')
    COMM.barrier()
    LOGGER.info('Done')


def main():
    args = parse_args()
    generate_mesh(args.labels, dim_size=args.dim_size, quiet=args.quiet)


if __name__ == '__main__':
    main()
