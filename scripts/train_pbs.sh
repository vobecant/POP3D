NGPUS=${1}
QUEUE=${2:-"qgpu"} # qgpu_eurohpc
CFG=${3}
OUT=${4}
NAME=${5}
HOURS=${6:-8}
EXTRA=${7}

#PROJECT_NAME=OPEN-26-3
PROJECT_NAME=DD-23-68

NOW=$(date +"%Y%m%d_%H%M%S")
NAME="${NAME}__${NOW}"

PORT=$(shuf -i30000-60000 -n1)

mkdir -p "./jobs"
JOB_FILE="./jobs/${NAME}.job"

echo "#!/bin/bash

#PBS -N ${NAME}
#PBS -A ${PROJECT_NAME}
#PBS -q ${QUEUE}
#PBS -l select=${NGPUS}
#PBS -l walltime=${HOURS}:00:00
#PBS -j oe
#PBS -k o

cd $PROJECT/vobecant/projects/TPVFormer-OpenSet

. /scratch/project/open-26-3/vobecant/miniconda3/etc/profile.d/conda.sh
conda activate OpenOccupancy

pwd
export MASTER_PORT=${PORT}
echo MASTER_PORT
echo ${MASTER_PORT}
echo 'MASTER_PORT should be printed above'

echo ${NAME}
echo ${JOB_FILE}

CUDA_LAUNCH_BLOCKING=1 bash launcher.sh ${CFG} ${OUT} ${EXTRA}" >${JOB_FILE}

echo "run job ${JOB_FILE}"
qsub ${JOB_FILE}
echo ""
