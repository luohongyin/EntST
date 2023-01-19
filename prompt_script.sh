#!/bin/bash

#SBATCH -o log/qqp/prompt_slurm_%j_%a.log
#SBATCH --array=0-9
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high

set -e

domain=$1
# domain='qqp'

train_size='12'

eval_mode_input=$2
# eval_mode='base'

ft_mode=$3
# ft_mode='st'

mix_rate='0.8'
train_mode='plabel' # plabel, proto
eval_model='sc' # sc, proto

exp_id=$4
base_exp_log_name=base-setred-$5\-$ft_mode
exp_log_name=simple-setred-$5\-$ft_mode
model_type_str=$5
# exp_id=$SLURM_ARRAY_TASK_ID

RED='tput setaf 1'
NC='tput sgr0'

tput setaf 1; printf "\n------------------- Eval Pretrain -------------------\n"
tput sgr0

# python eval_prompt.py $domain mt $eval_mode 2 50 pretrain $exp_id $eval_model pretrain

for eval_mode in 'base' 'adv'
do
    touch log/$domain/exp_$eval_mode\_$exp_id\_$base_exp_log_name.json
    touch log/$domain/exp_$eval_mode\_$exp_id\_$exp_log_name.json

    rm log/$domain/exp_$eval_mode\_$exp_id\_$base_exp_log_name.json
    rm log/$domain/exp_$eval_mode\_$exp_id\_$exp_log_name.json
done

touch log/$domain/cal_w_[$exp_id].json
rm log/$domain/cal_w_[$exp_id].json

cp log/qnli/prob_template.json log/$domain/prob_$model_type_str\_ft.json
cp log/qnli/prob_template.json log/$domain/prob_$model_type_str\_st.json
cp log/qnli/w_template.json log/$domain/w_stat.json

for i in {1..10}
do
    tput setaf 1; printf "\n------------------- Round $i -------------------\n"
    tput sgr0
    # python proc_data.py $domain train $train_size 0 train
    python proc_data.py $domain train $train_size $exp_id train
    # python proc_data.py $domain train $train_size $exp_id dev

    tput setaf 1; printf "\n--------- Prompt_1 ---------\n"
    tput sgr0
    python prompt_cst.py $domain prompt_1 $train_size ft $exp_id $eval_mode $train_mode $model_type_str
    # python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model base
    python eval_prompt.py $domain mt base 2 50 self_train $exp_id $eval_model $base_exp_log_name $model_type_str
    python eval_prompt.py $domain mt adv 2 50 self_train $exp_id $eval_model $base_exp_log_name $model_type_str

    # python eval_uncertainty.py $domain deberta $eval_mode 200 $exp_id baseline

    python prompt_cst.py $domain prompt_1 $train_size $ft_mode $exp_id $eval_mode $train_mode $model_type_str
    # python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model $exp_log_name
    # python mix_model.py $domain $mix_rate $model_type_str $exp_id
    python eval_prompt.py $domain mt base 2 50 self_train $exp_id $eval_model $exp_log_name $model_type_str
    python eval_prompt.py $domain mt adv 2 50 self_train $exp_id $eval_model $exp_log_name $model_type_str

    # python eval_uncertainty.py $domain deberta $eval_mode 200 $exp_id uncertain

    # for j in {1..10}
    # do
    #     python prompt_cst.py $domain prompt_1 $train_size boost $exp_id $eval_mode $train_mode
    #     python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model $exp_log_name
    #     python eval_uncertainty.py $domain deberta $eval_mode 200 $exp_id uncertain
    # done

    if [[ "$ft_mode" == "ft" ]]; then
        continue
    fi

    # tput setaf 1; printf "\n--------- Prompt_2 ---------\n"
    # tput sgr0
    # python prompt_cst.py $domain prompt_2 $train_size $ft_mode $exp_id $eval_mode hopfield
    # python prompt_cst.py $domain prompt_2 $train_size $ft_mode $exp_id $eval_mode $train_mode
    # python mix_model.py $domain $mix_rate
    # python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model

    # tput setaf 1; printf "\n--------- Prompt_joint ---------\n"
    # tput sgr0
    # python prompt_cst.py $domain prompt_joint $train_size $ft_mode $exp_id $eval_mode hopfield
    # python prompt_cst.py $domain prompt_joint $train_size $ft_mode $exp_id $eval_mode $train_mode
    # python mix_model.py $domain $mix_rate
    # python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model
done

# cp log/$domain/prob_$model_type_str\_st.json log/$domain/prob_$model_type_str\_$exp_log_name.json

# tput setaf 1; echo '--------- Adv Evaluation ---------'
# tput sgr0
# python eval_adv_offline.py $domain large train

# tput setaf 1; echo '--------- Base Evaluation ---------'
# tput sgr0
# python eval_base_offline.py $domain large train
