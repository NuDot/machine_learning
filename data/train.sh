#!/bin/bash -l

#$$ -P snoplus

#$$ -pe omp 4

#$$ -l gpus=0.25
# Request at least compute capability 3.5
#$$ -l gpu_c=3.5

#$$ -l h_rt=${TIME}

# Give the job a name
#$$ -N ${NAME}

#$$ -j y

# load modules
module load python/3.6.2
module load cuda/9.1
module load cudnn/7.1
module load tensorflow/r1.8

source /project/snoplus/snoing/install/root-5.34.36/bin/thisroot.sh

# Run the test script
python ${PROCESSOR} --signallist ${SGL} --bglist ${BGL} --signal ${SG} --bg ${BG} --outdir ${OUTDIR} --time_index ${TIME_PARA} --qe_index ${QE_PARA}
