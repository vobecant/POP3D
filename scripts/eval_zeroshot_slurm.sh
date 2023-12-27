POP3D_DIR=${1}
CKPT=${2:-"./pretrained/pop3d_weights.pth"}
NUM_GPUS=${3:-8}
HOURS=${4:-1}
CFG=${5:-"config/pop3d_maskclip_12ep.py"}
EXTRA="--text-embeddings-path ./pretrained/zeroshot_weights.pth"

ACCOUNT="..."
PARTITION="..."

NOW=$(date +"%Y%m%d_%H%M%S")
NAME="zeroshot_eval__${NOW}"
OUT_FILE=${NAME}.err

mkdir -p "./jobs"
JOB_FILE="./jobs/${NAME}.job"

echo "#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name ${NAME}
#SBATCH --account ${ACCOUNT}
#SBATCH --output=${OUT_FILE}
#SBATCH --gpus ${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${HOURS}:00:00

cd ${POP3D_DIR}

conda activate pop3d

echo ${NAME}
echo ${OUT_FILE}
echo ${JOB_FILE}

srun python3 eval.py --py-config ${CFG} --resume-from ${CKPT} --maskclip --no-wandb ${EXTRA}" >${JOB_FILE}

echo "run job ${JOB_FILE}"
echo "write to ${OUT_FILE}"
sbatch ${JOB_FILE}
echo ""
