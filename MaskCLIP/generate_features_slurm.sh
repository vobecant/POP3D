NUM_GPUS=${1}
HOURS_TOTAL=${2}
HOURS_PER_GPU=`echo "${HOURS_TOTAL}/${NUM_GPUS}+1" | bc` # computes how long should the job run on each GPU

ACCOUNT=${3:-""}
MASKCLIP_DIR=${4:-""}
FIRST=${5-0} # 0
LAST=${6-240942}  # 36048
STEP=`echo "(${LAST}-${FIRST})/${NUM_GPUS}+1" | bc`

CFG_PATH=configs/maskclip_plus/anno_free/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k__nuscenes_trainvaltest.py
CKPT_PATH=ckpts/maskclip_plus_vit16_deeplabv2_r101-d8_512x512_8k_coco-stuff164k.pth
PROJ_DIR=../data/nuscenes/features/projections/data/nuscenes
OUT_DIR=../data/nuscenes/maskclip_features_projections
PARTITION=qgpu

mkdir -p ./jobs

for START in $(seq ${FIRST} ${STEP} ${LAST}); do
  END=$((${START} + ${STEP}))

  EXPNAME=maskclip_features_${START}_${END}
  JOB_FILE=./jobs/${EXPNAME}.job

  echo "#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --output=${EXPNAME}.err
#SBATCH --mem=15GB
#SBATCH --job-name ${EXPNAME}
#SBATCH --account ${ACCOUNT}
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --partition=${PARTITION}
#SBATCH --time=${HOURS_PER_GPU}:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=4

conda activate maskclip

cd ${MASKCLIP_DIR}

# run script
srun python tools/extract_features.py ${CFG_PATH} --save-dir ${OUT_DIR} --checkpoint ${CKPT_PATH} --projections-dir ${PROJ_DIR} --complete --start ${START} --end ${END}" >${JOB_FILE}
  echo "run job ${JOB_FILE}"
  echo "write to ${EXPNAME}.err"
  sbatch ${JOB_FILE}
  echo ""
  sleep 1
done

squeue -u vobecant
