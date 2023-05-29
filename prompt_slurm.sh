#!/bin/bash

#SBATCH -o log/rte/dgst_%j_%a.log
#SBATCH --array=0,
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2

domain='qqp'
eval_mode='base'
ft_mode='st'
model_type_str='roberta'

# ./prompt_script.sh $domain $eval_mode ft $SLURM_ARRAY_TASK_ID

echo "========================================"

./prompt_script.sh $domain $eval_mode st $SLURM_ARRAY_TASK_ID $model_type_str
# ./prompt_iterate_script.sh $domain $eval_mode st $SLURM_ARRAY_TASK_ID $model_type_str
# ./prompt_iterate_script.sh $domain $eval_mode $ft_mode $SLURM_ARRAY_TASK_ID $model_type_str
