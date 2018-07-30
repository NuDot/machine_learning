#!/bin/bash -l

# Request 4 cores. This will set NSLOTS=4
#$ -pe omp 2
# Request 1 GPU
#$ -l gpus=0.5
# Request at least compute capability 3.5
#$ -l gpu_c=3.5

#$ -l h_rt=12:00:00

# Give the job a name
#$ -N HP

# load modules
module load python
module load cuda/8.0 
module load cudnn/6.0 
module load tensorflow/r1.4

source /project/snoplus/snoing/install/root-5.34.36/bin/thisroot.sh
export PYTHONPATH=$PYTHONPATH:/projectnb/snoplus/machine_learning/keras/lib/python2.7/site-packages/

# Run the test script
python /projectnb/snoplus/machine_learning/prototype/hyperparameter.py --signallist /projectnb/snoplus/machine_learning/data/networktrain_v2/C10dVrndVtx_3p0mSphere.dat --bglist /projectnb/snoplus/machine_learning/data/networktrain_v2/Xe136dVrndVtx_3p0mSphere.dat --signal C10 --bg Xe136 --outdir /projectnb/snoplus/sphere_data/C10_training_new/C10Xe136time6_qe6 --time_index 7 --qe_index 10
