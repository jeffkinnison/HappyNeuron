#!/bin/bash
#COBALT -t 60
#COBALT -n 2
#COBALT -q debug-flat-quad
#COBALT -A connectomics_aesp

MYNAME=distributed_inference

module load datascience/tensorflow-1.13
module load datascience/mpi4py
module rm darshan
module load cray-hdf5-parallel/1.10.2.0
export HDF5_USE_FILE_LOCKING=FALSE

#export PYTHONPATH=$PYTHONPATH:/projects/connectomics_aesp/hanyuli/envs/klab/lib/python3.6/site-packages
export PYTHONPATH=$PYTHONPATH:/projects/datascience/keceli/pip_ffn
NRANK_PER_NODE=32                                # Number of MPI ranks per node (Max 256)
NTHREAD_PER_CORE=4                              # Number of threads per core (Max 4)
NNODE=$COBALT_JOBSIZE                           # Number of nodes   

#MYSCRIPT=/projects/connectomics_aesp/hanyuli/programs/google_ffn/ffn/run_inference.py
MYSCRIPT=/projects/connectomics_aesp/hanyuli/programs/google_ffn/ffn/run_distributed_inference.py
NRANK=$((NRANK_PER_NODE*NNODE))                 # Number of MPI ranks
NTHREAD=$((NTHREAD_PER_CORE*64/NRANK_PER_NODE)) # Number of threads per MPI rank

START=`date +"%s"`

aprun -n $NRANK -N $NRANK_PER_NODE -e MKL_VERBOSE=0 -e MKLDNN_VERBOSE=0 -e KMP_BLOCKTIME=0 -e KMP_AFFINITY=granularity=fine,compact,1,0 -e OMP_NUM_THREADS=$NTHREAD -cc depth -d $NTHREAD -j $NTHREAD_PER_CORE python $MYSCRIPT \
    --inference_request="$(cat config.pbtxt)" \
    --bounding_box 'start { x:11000 y:12000 z:500 } size { x:1024 y:1024 z:256 }' \
    --subvolume_size 256,256,64 \
    --overlap 32,32,16 \
    --use_cpu True

NOW=`date +"%s"`
echo $(((NOW - START)/60)) minutes
  
