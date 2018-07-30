#!/bin/bash -l

#$$ -P snoplus

#$$ -pe omp 2

#$$ -l gpus=0.5
# Request at least compute capability 3.5
#$$ -l gpu_c=3.5

#$$ -l h_rt=${TIME}

# Give the job a name
#$$ -N ${NAME}

#$$ -j y

# load modules
module load python
module load cuda/8.0 
module load cudnn/6.0 
module load tensorflow/r1.4

source /project/snoplus/snoing/install/root-5.34.36/bin/thisroot.sh
export PYTHONPATH=$$PYTHONPATH:/projectnb/snoplus/machine_learning/keras/lib/python2.7/site-packages/

# Run the test script
python ${PROCESSOR} --signallist ${SGL} --bglist ${BGL} --signal ${SG} --bg ${BG} --outdir ${OUTDIR} --time_index ${TIME_PARA} --qe_index ${QE_PARA}
