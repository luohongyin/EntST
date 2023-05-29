import os
import sys
import json

import random
import argparse

import torch
import torch.nn.functional as F
from prompt_emb_layer import PromptEmbedding, PromptDecoder
from eval_task_adp import load_train_data, save_data

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from proc_data import (
    build_prompt_input, build_ft_data, meta_entailment_prompt
)

from eval_prompt import (
    prompt_seq_cls_relabel,
    prompt_cse_relabel,
    cls_evaluate,
    load_eval_data,
    get_unconf_nodes,
    plabel_neighbor_agreement
)
from train_glue import (
    adapt_glue_func,
)
from transformers.trainer_utils import set_seed

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(
    prog='python prompt_cst.py',
    description='Entailment self-training for NLU',
    epilog='Submit issues on Github for addtional help.'
)

# Task parameters
parser.add_argument('--domain', type=str)
parser.add_argument('--train-model', type=str)
parser.add_argument('--train-size', type=int)
parser.add_argument('--ft-mode', type=str)
parser.add_argument('--exp-id', type=str)

# Model parameters
parser.add_argument("--eval-mode", type=str, default="base")
parser.add_argument("--model-type-str", type=str, default="deberta")

args = parser.parse_args()

model_type_str = args.model_type_str

train_mode = 'prompt_1'
model_tag = 'large'
data_relabel_split = 'syn_data_relabel'


def select_confident_data(
        sent1_list, sent2_list, label_list,
        pseudo_label_list, pseudo_label_scores, num_bot,
        mode = 'sort'
    ):

    data = list(zip(
        sent1_list, sent2_list, label_list,
        pseudo_label_list, pseudo_label_scores, list(range(len(sent1_list)))
    ))
    data_size = len(data)

    if mode == 'sort':
        data = sorted(data, key = lambda x: x[-2])
        data_top = sorted(data[num_bot:], key = lambda x: x[-1])
    elif mode == 'zero':
        data_top = [x for x in data if x[-2] != 0]
    else:
        print(f'\nMode {mode} not supported\n')
        sys.exit()

    idx_list = [x[-1] for x in data_top]

    sent1_list_top = [x[0] for x in data_top]
    sent2_list_top = [x[1] for x in data_top]
    label_list_top = [x[2] for x in data_top]
    pseudo_label_list_top = [x[3] for x in data_top]
    data_top = {
        'sent1_list': sent1_list_top,
        'sent2_list': sent2_list_top,
        'label_list': label_list_top,
        'pseudo_label_list': pseudo_label_list_top
    }
    return data_top, idx_list


def pseudo_label_learning(exp_id):

    # model_type_str = 'roberta'
    model_size_str = 'large'

    if model_type_str == 'roberta':
        tokenizer_path = 'roberta-large'
    elif model_type_str == 'deberta':
        tokenizer_path = 'microsoft/deberta-large'
    else:
        print(f'\nBackbone model {model_type_str} not supported\n')

    model_path = f'luohy/ESP-{model_type_str}-large'

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_size_str}-tok.pt'
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    num_prompt_type = 1

    sent1_list, sent2_list, label_list, _ = load_train_data(
        args.domain, exp_id=args.exp_id
    )
    if sent2_list is None:
        sent2_list = sent1_list
    
    sent1_dev, sent2_dev, label_dev, _ = load_train_data(
        args.domain, exp_id=0
    )
    if sent2_dev is None:
        sent2_dev = sent1_dev

    if 'ft' in args.ft_mode:
        relabel_func = prompt_seq_cls_relabel
        num_prompt_type = 1
        plabel_iter = 1
        turn_on_dropout = False
    else:
        relabel_func = prompt_seq_cls_relabel
        num_prompt_type = 1
        plabel_iter = 7
        turn_on_dropout = True

    score_board_all = 0
    pseudo_label_tensor = 0

    pseudo_label_all = []
    hidden_states_all = []

    num_case = len(sent1_list)

    pseudo_label_list_eval, _, _ = relabel_func(
        args.domain, sent1_list, sent2_list,
        tok = tok, model = model,
        mnli=False,
        model_type_str = model_type_str, model_size_str = model_size_str,
        num_prompt_type = num_prompt_type, prompt_sep = False,
        dropout = False
    )
    pseudo_label_list_eval = torch.LongTensor(pseudo_label_list_eval)

    for x in range(plabel_iter):
        pseudo_label_list, score_board, hidden_states = relabel_func(
            args.domain, sent1_list, sent2_list,
            tok = tok, model = model,
            mnli = False,
            model_type_str = model_type_str, model_size_str = model_size_str,
            num_prompt_type = num_prompt_type, prompt_sep = False,
            dropout = turn_on_dropout
        )
        
        score_board_all += score_board
        pseudo_label_all += pseudo_label_list
        hidden_states_all.append(hidden_states)

    if 'st' in args.ft_mode:
        hidden_states_all = torch.cat(hidden_states_all, dim = 0)
        
        # Uncertainty estimation for SimPLE.
        conf_node, unconf_node, Jsq, r = get_unconf_nodes(
            hidden_states_all, pseudo_label_all,
            k = 9, p = float(sum(pseudo_label_all)) / len(pseudo_label_all)
        )
        
        # For Naive voting without uncertainty estimation,
        # Comment out the `get_unconf_nodes` function call above and use the following:
        #
        # conf_node = [1 for i in range(hidden_states_all.size(0))]
        
        plabel_sum = 0
        conf_node_sum = 0
        for i in range(plabel_iter):
            st_idx = i * num_case
            ed_idx = (i + 1) * num_case
            plabel_batch = pseudo_label_all[st_idx: ed_idx]
            conf_batch = conf_node[st_idx: ed_idx]

            plabel_sum += torch.Tensor(
                [x * y for x, y in zip(plabel_batch, conf_batch)]
            )
            conf_node_sum += torch.Tensor(conf_batch)
        
        conf_vec = conf_node_sum / plabel_iter
        plabel_soft = plabel_sum / conf_node_sum
        even_mask = torch.logical_and(
            plabel_soft != 0.5, conf_node_sum != 0
        )
        pseudo_label_list = (plabel_soft > 0.5).long()#.tolist()

        pseudo_label_list = torch.where(
            even_mask, pseudo_label_list, pseudo_label_list_eval
        ).tolist()

    pseudo_label_acc = (
        torch.Tensor(label_list) == torch.Tensor(pseudo_label_list)
    ).sum().float().item() / len(label_list)

    if args.domain == 'sst2' or args.domain == 'cola' or args.domain == 'qqp':
        pseudo_label_acc = 1 - pseudo_label_acc
    print(f'Pseudo labeling Acc. = {pseudo_label_acc}')

    prompt_list, rvs_map = build_prompt_input(
        args.domain,
        sent1_list,
        sent2_list,
        mlm=False, sep=False
    )

    label_final = pseudo_label_list

    args.train_size = len(sent1_list)
    new_data = build_ft_data(
        args.domain, rvs_map, num_prompt_type, label_final, label_list,
        prompt_list, 'st', # args.ft_mode,
        args.train_mode, args.train_size,
    )
    
    adapt_glue_func(
        args.domain, model_tag, data_relabel_split, 'cls',
        data = new_data, no_train = True, verbose = False,
        from_mnli = True, num_epochs = 4, prompt_mode = args.train_mode,
        exp_id = args.exp_id, model_type_str = model_type_str,
        eval_mode = args.eval_mode, train_mode = args.train_mode,
        model_config_pt = model_path, robust_loss_func = 'gm', c=5e-1
    )


if __name__ == '__main__':

    pseudo_label_learning(args.exp_id)
