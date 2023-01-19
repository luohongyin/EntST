import os
import sys
import json

import random

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
from advst_pipeline import meta_genetic_labeling
from train_glue import (
    adapt_glue_func,
    gnc_glue_func,
    protost_glue_func,
    adpst_glue_func
)
from gen_hypo_eval import (
    get_gen_prompt,
    verify_gp,
    syn_st_func
)
from transformers.trainer_utils import set_seed

# set_seed(42)
torch.cuda.empty_cache()

domain = sys.argv[1]
train_mode = sys.argv[2] # prompt_x
train_size = int(sys.argv[3])

try:
    ft_mode = sys.argv[4]
except:
    ft_mode = 'st'

exp_id = sys.argv[5]
eval_mode = sys.argv[6]

mode = sys.argv[7] # plabel, hopfield
model_type_str = sys.argv[8]

model_tag = 'large' # base, large
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
        print(f'Mode {mode} not supported')
        abort()

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

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_size_str}-tok.pt'
    )

    if model_type_str == 'roberta':
        model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_pr.pt'
    elif model_type_str == 'deberta':
        # model_path = f'model_ft_file/cls_{domain}_large_syn_data_relabel_0.pt'
        model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
    
    if 'boost' in ft_mode:
        model_path = f'model_ft_file/cls_{domain}_{model_type_str}_syn_data_relabel_{exp_id}.pt'


    # model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
    # model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_pr.pt'
    # model_path = 'model_ft_file/mnli_model_sc_5e-06_binary_pb.pt'
    # model_path = 'model_ft_file/mnli_model_sc_5e-06_single_p0.pt'
    # model_path = 'model_ft_file/mnli_model_sc_1e-05_binary_meta_ep499_maml-10-200.pt'

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    num_prompt_type = 1

    sent1_list, sent2_list, label_list, _ = load_train_data(
        domain, exp_id=exp_id
    )
    if sent2_list is None:
        sent2_list = sent1_list
    
    sent1_dev, sent2_dev, label_dev, _ = load_train_data(
        domain, exp_id=0
    )
    if sent2_dev is None:
        sent2_dev = sent1_dev

    if 'ft' in ft_mode:
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
        domain, sent1_list, sent2_list,
        tok=tok, model=model,
        # model_path=f'model_ft_file/cls_{domain}_large_{data_relabel_split}_{exp_id}.pt',
        mnli=False,
        model_type_str=model_type_str, model_size_str=model_size_str,
        num_prompt_type=num_prompt_type, prompt_sep=False,
        dropout = False
    )
    pseudo_label_list_eval = torch.LongTensor(pseudo_label_list_eval)

    for x in range(plabel_iter):
        pseudo_label_list, score_board, hidden_states = relabel_func(
            domain, sent1_list, sent2_list,
            tok=tok, model=model,
            # model_path=f'model_ft_file/cls_{domain}_large_{data_relabel_split}_{exp_id}.pt',
            mnli=False,
            model_type_str=model_type_str, model_size_str=model_size_str,
            num_prompt_type=num_prompt_type, prompt_sep=False,
            dropout = turn_on_dropout
        )
        
        score_board_all += score_board
        pseudo_label_all += pseudo_label_list
        hidden_states_all.append(hidden_states)
    
    raw_label_list = label_list

    if 'st' in ft_mode:
        hidden_states_all = torch.cat(hidden_states_all, dim = 0)
        conf_node, unconf_node, Jsq, r = get_unconf_nodes(
            hidden_states_all, pseudo_label_all,
            tok = tok, model = model,
            k = 9, p = float(sum(pseudo_label_all)) / len(pseudo_label_all)
        )
        # conf_node = [1 for i in range(hidden_states_all.size(0))]

        torch.save(
            hidden_states_all, f'log/qnli/hs_all_{model_type_str}.pt'
        )
        json.dump(
            [pseudo_label_all, conf_node],
            open(f'log/qnli/pl_conf_{model_type_str}.json', 'w')
        )
        # abort()
        
        plabel_sum = 0
        conf_node_sum = 0
        for i in range(plabel_iter):
            st_idx = i * num_case
            ed_idx = (i + 1) * num_case
            plabel_batch = pseudo_label_all[st_idx: ed_idx]
            conf_batch = conf_node[st_idx: ed_idx]
            # conf_batch = [1 for x in conf_batch]

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

        # data_select, _ = select_confident_data(
        #     sent1_list, sent2_list, label_list,
        #     pseudo_label_list, conf_vec, # * even_mask,
        #     len(sent1_list) // 8, mode = 'zero'
        # )

        # sent1_list = data_select['sent1_list']
        # sent2_list = data_select['sent2_list']
        # label_list = data_select['label_list']
        # pseudo_label_list = data_select['pseudo_label_list']
        # print(len(sent1_list))

    # score_board = F.softmax(score_board, 1)
    pseudo_label_scores, _ = score_board_all.max(1)
    # pseudo_label_list = pseudo_label_list.tolist()
    pseudo_label_scores = pseudo_label_scores.tolist()

    score_prob = F.softmax(score_board, dim = 1)
    label_tensor = F.one_hot(torch.LongTensor(raw_label_list), 2).float()

    pseudo_label_prob, _ = score_prob.max(1)
    label_prob = (score_prob * label_tensor).sum(1)

    pseudo_label_acc = (
        torch.Tensor(label_list) == torch.Tensor(pseudo_label_list)
    ).sum().float().item() / len(label_list)

    if domain == 'sst2' or domain == 'cola' or domain == 'qqp':
        pseudo_label_acc = 1 - pseudo_label_acc
    print(f'Pseudo labeling Acc. = {pseudo_label_acc}')

    prob_stat = json.load(open(f'log/{domain}/prob_{model_type_str}_{ft_mode}.json'))
    # prob_stat['pl_prob'].append(pseudo_label_prob.tolist())
    # prob_stat['hl_prob'].append(label_prob.tolist())
    prob_stat['pl_acc'].append(pseudo_label_acc)
    # prob_stat['ll'].append(label_list)
    # prob_stat['pll'].append(pseudo_label_list)
    json.dump(prob_stat, open(f'log/{domain}/prob_{model_type_str}_{ft_mode}.json', 'w'))

    prompt_list, rvs_map = build_prompt_input(
        domain,
        sent1_list,
        sent2_list,
        mlm=False, sep=False
    )

    label_final = pseudo_label_list

    train_size = len(sent1_list)
    new_data = build_ft_data(
        domain, rvs_map, num_prompt_type, label_final, label_list,
        prompt_list, 'st', # ft_mode,
        train_mode, train_size,
    )

    if 'ft' in ft_mode:
        adapt_glue_func(
            domain, model_tag, data_relabel_split, 'cls',
            data = new_data, no_train = True, verbose = False,
            from_mnli = True, num_epochs = 6, prompt_mode = train_mode,
            exp_id = exp_id, model_type_str = model_type_str,
            eval_mode = eval_mode, train_mode = train_mode,
            model_config_pt = model_path, robust_loss_func = 'gm', c=5e-1
        )
    else:
        adapt_glue_func(
            domain, model_tag, data_relabel_split, 'cls',
            data = new_data, no_train = True, verbose = False,
            from_mnli = True, num_epochs = 4, prompt_mode = train_mode,
            exp_id = exp_id, model_type_str = model_type_str,
            eval_mode = eval_mode, train_mode = train_mode,
            model_config_pt = model_path, robust_loss_func = 'gm', c=5e-1
        )


def proto_label_learning(exp_id):

    model_type_str = 'deberta'
    model_size_str = 'large'

    num_gen = 100

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_size_str}-tok.pt'
    )

    model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
    # model_path = 'model_ft_file/mnli_model_sc_5e-06_single_p0.pt'
    # model_path = 'model_ft_file/mnli_model_sc_1e-05_binary_meta_ep499_maml-10-200.pt'

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    # proto_emb = torch.load(f'model_ft_file/{domain}_proto_emb.pt').cuda()

    num_prompt_type = 1

    sent1_list, sent2_list, label_list, dformat = load_train_data(
        domain, exp_id=exp_id
    )
    if sent2_list is None:
        sent2_list = sent1_list

    sent1_dev, sent2_dev, label_dev, _ = load_eval_data(
        domain, 'base', dev_split_id=exp_id
    )
    if sent2_dev is None:
        sent2_dev = sent1_dev

    num_gen_list_raw = [num_gen] * len(sent1_list)

    pseudo_label_list, score_board = prompt_seq_cls_relabel(
        domain, sent1_list, sent2_list,
        tok=tok, model=model,
        mnli=False,
        model_type_str=model_type_str, model_size_str=model_size_str,
        num_prompt_type=num_prompt_type, prompt_sep=False
    )

    num_gen_list = (100 - score_board.max(1).values * 100).long().tolist()
    pseudo_label_scores, _ = score_board.max(1)
    pseudo_label_scores = pseudo_label_scores.tolist()

    gp_all, gs_all = get_gen_prompt(
        domain, num_gen_list_raw, sent1_list, sent2_list, label_list,
        batch_size = 32, pretrained = True
    )

    gs_all, pred_label_list = verify_gp(
        domain, model_type_str, model_path,
        gs_all=gp_all, from_file=False, batch_size=32
    )

    syn_st_func(domain, gs_all, pred_label_list, dformat)

    '''data_top, data_bot, idx_list = select_confident_data(
        sent1_list, sent2_list, label_list,
        pseudo_label_list, pseudo_label_scores, len(sent1_list) // 8
    )
    sent1_list = data_top['sent1_list'] # + data_bot['sent1_list']
    sent2_list = data_top['sent2_list'] # + data_bot['sent2_list']

    # print(idx_list)
    # abort()
    label_list = [label_list[x] for x in idx_list]'''

    gp_all, gs_all = get_gen_prompt(
        domain, num_gen_list_raw, sent1_list, sent2_list, label_list,
        batch_size = 32, pretrained = False
    )

    prompt_list, pseudo_label_list, real_idx_list, p = plabel_neighbor_agreement(
        domain, tok, model.cuda(), gp_all, label_list, train_mode
    )

    real_plabels = [pseudo_label_list[x] for x in real_idx_list]
    pseudo_label_acc = (
        torch.Tensor(label_list) == torch.Tensor(real_plabels)
    ).sum().float().item() / len(label_list)

    if domain == 'sst2' or domain == 'cola' or domain == 'qqp':
        pseudo_label_acc = 1 - pseudo_label_acc
    print(f'Pseudo labeling Acc. = {pseudo_label_acc}')

    new_data = {
        'sent1_list': prompt_list,
        'sent2_list': None,
        'label_list': pseudo_label_list,
    }

    adapt_glue_func(
        domain, model_tag, data_relabel_split, 'cls',
        data = new_data, no_train = True, verbose = False,
        from_mnli = True, num_epochs = 6, prompt_mode = train_mode,
        exp_id = exp_id, model_type_str = model_type_str,
        eval_mode = eval_mode, train_mode = train_mode,
    )


def meta_seq_relabel(
        sent1_list, sent2_list, pseudo_label_list,
        tok = None, me_model = None, batch_size = 32,
        pseudo_label_scores = None
    ):

    def diag_mask(num_case):
        eye = torch.eye(num_case * 2)
        diag = -torch.diag(torch.ones(num_case), diagonal=num_case)
        value = (eye + diag + diag.T).cuda()
        mask = (value == 0).float()
        return value, mask

    pseudo_label_scores = (pseudo_label_scores - 0.5) * 2

    # model = model.cuda()
    me_model.eval()
    num_case = len(sent1_list)

    meta_input_list, meta_label_list, pair_idx_list = meta_entailment_prompt(
        domain, sent1_list, sent2_list,
        pseudo_label_list, mlm = False, skip_self = False,
        # sep = (domain != 'sst2')
    )

    pseudo_label_binary = pseudo_label_list # + [
    #     1 - x for x in pseudo_label_list
    # ]

    for i, pl in enumerate(pseudo_label_binary):
        if pl == 0:
            pseudo_label_binary[i] = -1

    pseudo_label_binary = torch.Tensor(
        pseudo_label_binary
    ).cuda().float().unsqueeze(1)

    pseudo_meta_probs = []
    pseudo_meta_probs_inv = []
    for i in range(0, len(meta_input_list), batch_size):
        meta_input_batch = meta_input_list[i: i + batch_size]

        true_logits, false_logits = cls_evaluate(
            tok, me_model, meta_input_batch, 0,
            model_type = 'sc', mnli = False, temperature=10,
            softmax = True
        )
        true_logits = (true_logits - 0.5) * 10

        pseudo_meta_probs.append(true_logits)
        pseudo_meta_probs_inv.append(false_logits)

    pseudo_meta_probs = torch.cat(pseudo_meta_probs, dim = 0)
    pseudo_meta_probs = pseudo_meta_probs.view(2 * num_case, -1)
    # value, mask = diag_mask(num_case)

    pseudo_meta_probs = (
        pseudo_meta_probs[:num_case, :num_case] + \
        pseudo_meta_probs[num_case:, num_case:] + \
        ( - pseudo_meta_probs[:num_case, num_case:]) + \
        ( - pseudo_meta_probs[num_case:, :num_case])
    ) / 1
    pseudo_meta_probs = pseudo_meta_probs + pseudo_meta_probs.T
    # print(pseudo_meta_probs)
    # abort()
    mask = torch.eye(num_case).cuda()
    # mask *= pseudo_label_scores.unsqueeze(1) * pseudo_label_scores.unsqueeze(0)

    # pseudo_meta_probs -= pseudo_meta_probs.mean(1, keepdim=True)
    pseudo_meta_probs = pseudo_meta_probs * (1 - mask) + mask * 1 # - 0.5
    # pseudo_meta_probs -= pseudo_meta_probs.mean(1, keepdim=True)

    # pseudo_meta_probs *= pseudo_label_scores.unsqueeze(1) * pseudo_label_scores.unsqueeze(0)

    # print(pseudo_label_scores)
    # abort()
    hopfield_step = 0
    while hopfield_step < 40:
        pseudo_label_binary *= pseudo_label_scores.unsqueeze(1)
        hopfield_logits = torch.mm(pseudo_meta_probs, pseudo_label_binary)

        new_labels = torch.where(
            hopfield_logits > 0,
            torch.ones_like(pseudo_label_binary),
            -torch.ones_like(pseudo_label_binary)
        )
        # print(hopfield_logits)

        if torch.all(new_labels == pseudo_label_binary):
            print(f'Converaged at {hopfield_step} steps')
            break

        pseudo_label_binary = new_labels
        hopfield_step += 1
    # abort()

    # new_labels = (new_labels > 0).long()
    new_labels = new_labels.squeeze(1).tolist()[:num_case]
    return new_labels, torch.abs(hopfield_logits - hopfield_logits.mean()).tolist()


def hopfield_self_training(exp_id):

    def bin_label_trans(label_list):
        return [int(x > 0) for x in label_list]

    model_type_str = 'deberta'
    model_size_str = 'large'

    num_epoch = 1

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_size_str}-tok.pt'
    )
    pt_model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
    # pt_model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_meta_ps1.pt'
    # me_model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
    me_model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_meta_pmso.pt'
    st_model_path = f'model_ft_file/cls_{domain}_large_syn_data_relabel_{exp_id}.pt'

    num_prompt_type = 1

    if mode == 'plabel':
        sent1_list, sent2_list, label_list, dformat = load_train_data(
            domain, exp_id=exp_id
        )
    elif mode == 'hopfield':
        sent1_list, sent2_list, label_list, dformat = load_eval_data(
            domain, 'base', dev_split_id=exp_id
        )
    else:
        print(f'Mode {mode} not supported')
        abort()

    if sent2_list is None:
        sent2_list = sent1_list

    for e in range(num_epoch):
        if e == 0:
            model_path = pt_model_path
        else:
            model_path = st_model_path
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        pseudo_label_list, score_board = prompt_seq_cls_relabel(
            domain, sent1_list, sent2_list,
            tok=tok, model=model, model_path=model_path,
            model_type_str=model_type_str, model_size_str=model_size_str,
            num_prompt_type=num_prompt_type, prompt_sep=False
        )
        score_board = F.softmax(score_board.cuda(), 1)
        pseudo_label_scores, _ = score_board.max(1)
        pseudo_label_scores_list = pseudo_label_scores.tolist()

        # '''
        data_top, data_bot, idx_list = select_confident_data(
            sent1_list, sent2_list, label_list,
            pseudo_label_list, pseudo_label_scores_list, len(sent1_list) // 8
        )

        meta_input_list, meta_label_list, pair_idx_list = meta_entailment_prompt(
            domain, sent1_list, sent2_list, pseudo_label_list,
            mlm = False, mode = 'sample', sample_pool=set(idx_list),
            # sep = (domain != 'sst2')
        )
        meta_data = list(zip(meta_input_list, meta_label_list))
        random.shuffle(meta_data)
        meta1_list = [x[0] for x in meta_data]
        meta_label_list = [x[1] for x in meta_data]

        meta_ent_data = {
            'sent1_list': meta1_list,
            'sent2_list': None,
            'label_list': meta_label_list,
            'dformat': dformat
        }

        adapt_glue_func(
            domain, model_tag, data_relabel_split, 'cls',
            data = meta_ent_data, no_train = True, verbose = False,
            from_mnli = False, num_epochs = 2, prompt_mode = train_mode,
            exp_id = exp_id, model_type_str = model_type_str,
            eval_mode = eval_mode, model_config_pt = me_model_path
        )
        # '''

        me_model = AutoModelForSequenceClassification.from_pretrained(
            # st_model_path
            me_model_path
        ).cuda()

        pseudo_label_acc = (
            torch.Tensor(label_list) == torch.Tensor(pseudo_label_list)
        ).sum().float().item() / len(label_list)

        if domain == 'sst2' or domain == 'cola' or domain == 'qqp':
            pseudo_label_acc = 1 - pseudo_label_acc
        print(f'Pseudo labeling Acc. = {pseudo_label_acc}')

        hopfield_label_list, hopfield_scores = meta_seq_relabel(
            sent1_list, sent2_list, pseudo_label_list,
            tok = tok, me_model = me_model, batch_size = 32,
            pseudo_label_scores = pseudo_label_scores
        )

        pseudo_label_list = bin_label_trans(pseudo_label_list)
        hopfield_label_list = bin_label_trans(hopfield_label_list)

        # print(bin_label_trans(pseudo_label_list))
        # print(bin_label_trans(hopfield_label_list))
        # abort()

        pseudo_label_acc = (
            torch.Tensor(label_list) == torch.Tensor(hopfield_label_list)
        ).sum().float().item() / len(label_list)

        if domain == 'sst2' or domain == 'cola' or domain == 'qqp':
            pseudo_label_acc = 1 - pseudo_label_acc
        print(f'Pseudo labeling Acc. = {pseudo_label_acc}')

        new_dataset = {
            'sent1_list': sent1_list,
            'sent2_list': sent2_list,
            'label_list': hopfield_label_list,
            'dformat': dformat
        }

        if mode == 'hopfield':
            save_data(new_dataset, domain, 'dev', split_id = exp_id)
            return

        prompt_list, rvs_map = build_prompt_input(
            domain, sent1_list, sent2_list, mlm=False, sep=False
        )

        train_size = len(sent1_list)
        new_data = build_ft_data(
            rvs_map, num_prompt_type, hopfield_label_list, label_list,
            prompt_list, ft_mode, train_mode, train_size,
        )

        adapt_glue_func(
            domain, model_tag, data_relabel_split, 'cls',
            data = new_data, no_train = True, verbose = False,
            from_mnli = True, num_epochs = 6, prompt_mode = train_mode,
            exp_id = exp_id, model_type_str = model_type_str,
            eval_mode = eval_mode
        )


if __name__ == '__main__':

    if mode == 'plabel':
        if ft_mode == 'ft':
            pseudo_label_learning(exp_id)
        else:
            pseudo_label_learning(exp_id)
            # hopfield_self_training(exp_id)
    elif mode == 'hopfield':
        hopfield_self_training(exp_id)
    elif mode == 'proto':
        proto_label_learning(exp_id)
    else:
        print(f'mode {mode} not defined.')
