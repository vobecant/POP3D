POP3D_DIR=${1}
NUM_GPUS=${2:-8}
HOURS=${3:-72}
CFG=${4:-"config/pop3d_maskclip_12ep.py"}
OUT=${5:-"out/pop3d_maskclip_12ep"}
EXPNAME=${6:-'pop3d_maskclip_12ep'}
EXTRA="--agnostic --maskclip --no-class-weights --lr 1e-4"

PARTITION="..."
ACCOUNT="..."
USERNAME="..."

mkdir -p ./jobs
mkdir -p ./out


NOW=$(date +"%Y%m%d_%H%M%S")
EXPNAME=${EXPNAME}_${NOW}
JOB_FILE=./jobs/${EXPNAME}.job

echo "#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name ${EXPNAME}
#SBATCH --account ${ACCOUNT}
#SBATCH --output=${OUT_FILE}
#SBATCH --gpus ${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${HOURS}:00:00
#SBATCH --signal=SIGUSR1@90

cd ${POP3D_DIR}
conda activate pop3d

# run script from above
srun python train.py --py-config ${CFG} --work-dir ${OUT} ${EXTRA}" >${JOB_FILE}
echo "run job ${JOB_FILE}"
echo "write to ${OUT_FILE}"
echo "output directory: ${OUT}"
sbatch ${JOB_FILE}
echo ""
sleep 1

squeue -u ${USERNAME}
