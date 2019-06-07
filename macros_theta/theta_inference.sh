#!/bin/sh
#COBALT -A connectomics_aesp
#COBALT -q debug-flat-quad
#COBALT -t 00:59:00

MYNAME=debug_inference

#module load datascience/tensorflow-1.12 datascience/horovod-0.15.2
module load datascience/tensorflow-1.13 datascience/horovod-0.16.1
module rm darshan
module load cray-hdf5-parallel/1.10.2.0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/lus/theta-fs0/projects/datascience/keceli/pip_ffn

NRANK_PER_NODE=1                                # Number of MPI ranks per node (Max 256)
NTHREAD_PER_CORE=1                              # Number of threads per core (Max 4)
NNODE=$COBALT_JOBSIZE                           # Number of nodes   

MYSCRIPT=/home/keceli/ffn/keceli_ffn/run_inference.py
MYSCRIPT=/home/keceli/ffn/google_ffn/run_inference.py
NNODE=1
NRANK=$((NRANK_PER_NODE*NNODE))                 # Number of MPI ranks
NTHREAD=$((NTHREAD_PER_CORE*64/NRANK_PER_NODE)) # Number of threads per MPI rank
TRAINDIR=train_${MYNAME}_b${BATCHSIZE}_n${NNODE}_p${NRANK}_t${NTHREAD}_r${LRATE}_o${OPTIMIZER}_s${SHARDING_RULE}${SCALING_RULE}_${COBALT_JOBID}

aprun -n $NRANK -N $NRANK_PER_NODE -e MKL_VERBOSE=0 -e MKLDNN_VERBOSE=0 -e KMP_BLOCKTIME=0 -e KMP_AFFINITY=granularity=fine,compact,1,0 -e OMP_NUM_THREADS=$NTHREAD -cc depth -d $NTHREAD -j $NTHREAD_PER_CORE python $MYSCRIPT \
    --inference_request="$(cat config.txt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'

