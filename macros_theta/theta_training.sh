#!/bin/sh
#COBALT -A connectomics_aesp
###COBALT -A datascience
###COBALT -q default
###COBALT -q debug-cache-quad
###COBALT -q debug-flat-quad
###COBALT -t 00:59:00


MYNAME=ffn

#module load datascience/tensorflow-1.12 #datascience/horovod-0.15.2
module load datascience/tensorflow-1.13 #datascience/horovod-0.16.1 automatically loaded with tf
module rm darshan
module load cray-hdf5-parallel/1.10.2.0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/lus/theta-fs0/projects/datascience/keceli/pip_ffn

TFRECORDFILE=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/tf_record_file
GROUNDTRUTH=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/groundtruth.h5 
GRAYSCALE=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/grayscale_maps.h5 

TRAINER=/gpfs/mira-home/keceli/ffn/keceli_ffn/train_hvd.py

NRANK_PER_NODE=1                                # Number of MPI ranks per node (Max 256)
NTHREAD_PER_CORE=1                              # Number of threads per core (Max 4)
NNODE=$COBALT_JOBSIZE                           # Number of nodes   
NINTER=1
MKL_DYNAMIC=false
OMP_NESTED=false
KMP_BLOCKTIME=0
OPTIMIZER=adam
LRATE=0.001
SHARDING_RULE=0 # 0 for no sharding, 1 for sharding
SCALING_RULE=0 # 0 for no scaling, 1 for linear scaling, 2 for sqrt scaling wrt number of ranks, 4 for sqrt scaling wrt number of ranks*batchsize
XLA=0
BATCHSIZE=64
NTHREAD=$((NTHREAD_PER_CORE*64/NRANK_PER_NODE)) # Number of threads per MPI rank
NRANK=$((NRANK_PER_NODE*NNODE))                 # Number of MPI ranks
MKLTHREAD=$NTHREAD

TRAINDIR=train_${MYNAME}_b${BATCHSIZE}_n${NNODE}_p${NRANK}_t${NTHREAD}_i${NINTER}_m${MKLTHREAD}_d${MKL_DYNAMIC}_n${OMP_NESTED}_kk${KMP_BLOCKTIME}__r${LRATE}_o${OPTIMIZER}_s${SHARDING_RULE}${SCALING_RULE}_${COBALT_JOBID}_${XALT_COBALT_QUEUE}
mkdir -p $TRAINDIR  

aprun -n $NRANK -N $NRANK_PER_NODE -e MKLDNN_VERBOSE=0 -e MKL_VERBOSE=0 -e MKL_DYNAMIC=$MKL_DYNAMIC -e OMP_NESTED=$OMP_NESTED  -e KMP_BLOCKTIME=${KMP_BLOCKTIME} -e KMP_AFFINITY=granularity=fine,compact,1,0 -e OMP_NUM_THREADS=$NTHREAD -e MKL_NUM_THREADS=$MKLTHREAD -cc depth -d $NTHREAD -j $NTHREAD_PER_CORE python $TRAINER \
    --train_coords $TFRECORDFILE \
    --data_volumes valdation1:${GRAYSCALE}:raw \
    --label_volumes valdation1:${GROUNDTRUTH}:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33 \
    --batch_size $BATCHSIZE \
    --optimizer $OPTIMIZER \
    --max_steps 400000000 \
    --summary_rate_secs 400 \
    --scaling_rule $SCALING_RULE \
    --sharding_rule $SHARDING_RULE \
    --num_intra_threads $NTHREAD \
    --num_inter_threads $NINTER \
    --train_dir $TRAINDIR &> log_${TRAINDIR}.txt &

