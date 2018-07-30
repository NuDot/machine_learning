#!/bin/bash -l

source /project/snoplus/snoing/install/root-5.34.36/bin/thisroot.sh

module load python
export PYTHONPATH=$$PYTHONPATH:/projectnb/snoplus/lib/python2.7/site-packages/

#$$ -P snoplus

#$$ -l mem_per_core=8G

#$$ -l h_rt=${TIME}

#$$ -j y


python /projectnb/snoplus/machine_learning/prototype/processing_sparse_t.py --input ${INPUT} --outputdir ${OUTPUT} --start ${START} --end ${END} --elow ${ELOW} --ehi ${EHI}
