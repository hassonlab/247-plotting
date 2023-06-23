#!/bin/bash
#SBATCH --time=1:10:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -o 'logs/%A.log'

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate /home/kw1166/.conda/envs/247-main-new
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda
    source activate /home/hgazula/.conda/envs/247-main
else
    module load anacondapy
    source activate srm
fi

export TRANSFORMERS_OFFLINE=1

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`
