#!/bin/sh
CONFIG=$1
WORKDIR=$2

shift 2

echo "$@"
echo "${CONFIG}"
echo "${WORKDIR}"
echo python train.py --py-config ${CONFIG} --work-dir ${WORKDIR} "$@"
python train.py --py-config $CONFIG --work-dir $WORKDIR "$@"
