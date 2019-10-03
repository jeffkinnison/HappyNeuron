#COBALT -t 59
#COBALT -A datascience
#COBALT -q debug-flat-quad

source /lus/theta-fs0/projects/connectomics_aesp/software/neuro_env_36/bin/activate
module rm darshan
export HDF5_USE_FILE_LOCKING=FALSE


FIJI=/lus/theta-fs0/projects/connectomics_aesp/software/Fiji.app/ImageJ-linux64
HPN=/lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron
PRE_TILES=$HPN/trakem2/preprocess_tiles.py
MONTAGE=$HPN/trakem2/mpi_montage.py
PRE_STACK=$HPN/trakem2/preprocess_stack.py
ALIGN=$HPN/trakem2/align.py
EXPORT=$HPN/trakem2/mpi_export.py

NRANK_PER_NODE=1                             # Number of MPI ranks per node (Max 256)
NTHREAD_PER_CORE=4                              # Number of threads per core (Max 4)
NTHREAD=$((NTHREAD_PER_CORE*64/NRANK_PER_NODE)) # Number of threads per MPI rank
NNODE=$COBALT_JOBSIZE                           # Number of nodes
NRANK=$((NRANK_PER_NODE*NNODE))                 # Number of MPI ranks

SUBMIT_THETA="aprun -n $NRANK -N $NRANK_PER_NODE -e KMP_BLOCKTIME=0 -e KMP_AFFINITY=granularity=fine,compact,1,0 -e OMP_NUM_THREADS=$NTHREAD -cc depth -d $NTHREAD -j $NTHREAD_PER_CORE"

######USER INPUT
RAW_INPUT=./trakem2_HL00732
PROCESS_FOLDER=./trakem2_HL00732_process_1
MIN=1024
MAX=2048
RANGE="0, 100" #ignored for now


######EXECUTION


python $PRE_TILES $RAW_INPUT $PROCESS_FOLDER/align_raw.txt #raw_input #output
sleep 1

$SUBMIT_THETA python $MONTAGE $PROCESS_FOLDER/align_raw.txt $PROCESS_FOLDER --min $MIN --max $MAX --fiji $FIJI
sleep 1

python $PRE_STACK $PROCESS_FOLDER/output $PROCESS_FOLDER/align_new.txt #input #output 
sleep 1
	
aprun -n 1 python $ALIGN $PROCESS_FOLDER/align_new.txt $PROCESS_FOLDER/align1  --fiji $FIJI
sleep 1

$SUBMIT_THETA python $EXPORT $PROCESS_FOLDER/align_new.txt $PROCESS_FOLDER/align1  --range $RANGE --fiji $FIJI
sleep 1

wait

