#!/bin/bash

#SBATCH -o log/qqp/prompt_slurm_%j_%a.log
#SBATCH --array=0-7
#SBATCH --exclusive
#SBATCH --gres=gpu:2

set -e

domain='rte'

train_size='12'

eval_mode='base' # `base` for glue, `adv` for AdversarialGLUE

ft_mode='st'

model_type_str='deberta'

eval_model='sc' # sc, proto

# exp_id=$SLURM_ARRAY_TASK_ID
exp_id=0

RED='tput setaf 1'
NC='tput sgr0'

tput setaf 1; printf "\n------------------- Eval Pretrained entailment model -------------------\n"
tput sgr0

python eval_prompt.py --domain $domain --eval-split $eval_mode --model-config pretrain \
    --exp-id $exp_id --model-type $eval_model --model-type-str $model_type_str

for i in {1..5}
do
    tput setaf 1; printf "\n------------------- Round $i -------------------\n"
    tput sgr0
    python proc_data.py $domain train $train_size $exp_id train

    tput setaf 1; printf "\n--------- Prompt_1 ---------\n"
    tput sgr0

    python prompt_cst.py \
        --domain $domain \
        --train-size $train_size \
        --ft-mode $ft_mode \
        --exp-id $exp_id \
        --eval-mode $eval_mode \
        --train-mode $train_mode \
        --model-type-str $model_type_str
    
    tput setaf 1; printf "\n--------- Evaluating on GLUE ---------\n"
    tput sgr0
    python eval_prompt.py --domain $domain --eval-split base --model-config pretrain \
        --exp-id $exp_id --model-type $eval_model --model-type-str $model_type_str
    
    tput setaf 1; printf "\n--------- Evaluating on AdversarialGLUE ---------\n"
    tput sgr0
    python eval_prompt.py --domain $domain --eval-split adv --model-config pretrain \
        --exp-id $exp_id --model-type $eval_model --model-type-str $model_type_str
done