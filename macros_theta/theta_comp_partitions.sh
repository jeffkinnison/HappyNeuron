/bin/bash
#COBALT -t 59
#COBALT -A datascience
#COBALT -q debug-flat-quad

MYNAME=test

module load datascience/tensorflow-1.13
module load datascience/mpi4py
module rm darshan
module load cray-hdf5-parallel/1.10.2.0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PYTHONPATH:/lus/theta-fs0/projects/datascience/keceli/pip_ffn


TFRECORDFILE=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/tf_record_file
GROUNDTRUTH=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/groundtruth.h5
GRAYSCALE=/lus/theta-fs0/projects/datascience/keceli/run/f3n/training/grayscale_maps.h5
TRAINER=/gpfs/mira-home/keceli/ffn/keceli_ffn/compute_partitions_mpi.py
NRANK_PER_NODE=1                                # Number of MPI ranks per node (Max 256)
NTHREAD_PER_CORE=1                              # Number of threads per core (Max 4)
NTHREAD=$((NTHREAD_PER_CORE*64/NRANK_PER_NODE)) # Number of threads per MPI rank
NNODE=$COBALT_JOBSIZE                           # Number of nodes   
NRANK=$((NRANK_PER_NODE*NNODE))                 # Number of MPI ranks
LOM=24
MINSIZE=10000
TRAINDIR=partition_${MYNAME}_n${NNODE}_p${NRANK}_t${NTHREAD}_l${LOM}_m${MINSIZE}_${COBALT_JOBID}
mkdir -p $TRAINDIR

aprun -n $NRANK -N $NRANK_PER_NODE -e KMP_BLOCKTIME=0 -e KMP_AFFINITY=“granularity=fine,compact,1,0” -e OMP_NUM_THREADS=$NTHREAD -cc depth -d $NTHREAD -j $NTHREAD_PER_CORE python $TRAINER \
--input_volume ${GROUNDTRUTH}:stack \
--output_volume ${TRAINDIR}/af.h5:af \
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
--lom_radius ${LOM},${LOM},${LOM}  \
--min_size ${MINSIZE}
