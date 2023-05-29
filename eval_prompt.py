import sys
import json

import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification
)

from proc_data import coordinate, build_prompt_input
from prompt_emb_layer import (
    PromptEmbedding,
    PromptDecoder,
)
from eval_task_adp import (
    load_train_data, load_adv_eval, load_base_eval
)

torch.cuda.empty_cache()

def load_eval_data(dataset_name, eval_split, dev_split_id=None):
    _, _, _, dformat = load_train_data(dataset_name)

    if dev_split_id is not None:
        sent1_list, sent2_list, label_list, _ = load_base_eval(
            dataset_name, split='dev', dformat=dformat,
            dev_split_id = dev_split_id
        )

    elif eval_split == 'adv':
        sent1_list, sent2_list, label_list = load_adv_eval(
            dataset_name, dformat
        )

    elif eval_split == 'base':
        sent1_list, sent2_list, label_list, _ = load_base_eval(
            dataset_name, split='dev', dformat=dformat
        )

    return sent1_list, sent2_list, label_list, dformat


def get_base_logits(tok, model, t_idx, f_idx, ok_idx,
                    num_prompt, prompt_str = None, mlm = True):
    if not mlm:
        return 0, 0, 0

    if prompt_str is None:
        input_txt = [
            'It is [MASK] that true.',
            'It is [MASK] that frue.',
            'It is [MASK] that ok.'
        ]
        offset = 0
    else:
        input_txt = [
            f'{prompt_str} It is [MASK] that .',
            f'{prompt_str} It is [MASK] that .',
            f'{prompt_str} It is [MASK] that .'
        ]
        offset = num_prompt
    input_enc = tok(
        input_txt, return_tensors = 'pt',
        padding = 'longest', truncation = True
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(input_ids, attn_mask)

    t_base = result.logits[0][offset + 3][t_idx].item()
    f_base = result.logits[1][offset + 3][f_idx].item()
    ok_base = result.logits[2][offset + 3][ok_idx].item()

    return t_base, f_base, ok_base


def get_proto_emb(
        tok, model, sent1_list, sent2_list, label_list, batch_size=32
    ):

    hidden_state_list = []
    for j in range(0, len(sent1_list), batch_size):
        sent1_batch = sent1_list[j: j + batch_size]

        if sent2_list is None:
            sent2_batch = None
        else:
            sent2_batch = sent2_list[j: j + batch_size]
        label_batch = label_list[j: j + batch_size]

        input_enc = tok(
            text = sent1_batch,
            text_pair = sent2_batch,
            max_length = 512,
            padding = 'longest',
            return_tensors = 'pt',
            truncation = True,
            return_attention_mask = True,
            verbose = False
        )
        label_tensor = torch.Tensor(label_batch).long().cuda()

        input_ids = input_enc['input_ids'].cuda()
        attn_mask = input_enc['attention_mask'].cuda()

        with torch.no_grad():
            result = model(
                input_ids = input_ids,
                attention_mask = attn_mask,
                output_hidden_states = True
            )
            output_emb = result.hidden_states[-1][:, 0, :]
            # if hasattr(model.module, 'pooler'):
            #     output_emb = model.module.pooler(result.hidden_states[-1])
            # else:
            #     output_emb = model.module.bert.pooler(result.hidden_states[-1])

        hidden_state_list.append(output_emb)

    hidden_states = torch.cat(hidden_state_list, dim=0)
    label_tensor = torch.Tensor(label_list).cuda().unsqueeze(1)

    proto_emb_0 = (hidden_states * (1 - label_tensor)).sum(0, keepdim=True) / (1 - label_tensor).sum(0)
    proto_emb_1 = (hidden_states * label_tensor).sum(0, keepdim=True) / label_tensor.sum(0)

    proto_emb = torch.cat(
        [proto_emb_0, proto_emb_1], dim = 0
    )

    return proto_emb.contiguous().detach(), hidden_states.detach()


def get_unconf_nodes(
        hidden_states, pseudo_label_list, # pseudo_label_scores,
        k = 20, p = 0.5
    ):

    data_size = len(pseudo_label_list)

    pseudo_label_list = torch.Tensor(pseudo_label_list).cuda()

    pseudo_label_scores = pseudo_label_list * p +\
         (1 - pseudo_label_list) * (1 - p)

    label_mismatch_mat = (
        pseudo_label_list.unsqueeze(1) != pseudo_label_list.unsqueeze(0)
    ).float()

    emb_dist_mat = torch.cdist(hidden_states, hidden_states)
    mm_dist = emb_dist_mat * label_mismatch_mat + (
        1 - label_mismatch_mat) * 10000
    r = (mm_dist.min() / 32).item() / 4

    rvs_dist_mat = 1. / (1 + emb_dist_mat)
    neighbor_k_mat, _ = torch.topk(rvs_dist_mat, k, dim = 1)
    ngb_k_thresh = neighbor_k_mat[:, -1:]
    topk_mask = (rvs_dist_mat > ngb_k_thresh).float()

    pscore_usq = pseudo_label_scores.unsqueeze(1)
    # pscore_usq = 0.5
    topk_dist = rvs_dist_mat * topk_mask

    mu = (1 - pscore_usq) * topk_dist.sum(1, keepdim=True)
    sigma = pscore_usq * (1 - pscore_usq) * (topk_dist * topk_dist).sum(1, keepdim=True)

    J = (topk_dist * label_mismatch_mat).sum(1, keepdim=True)
    # print(J.squeeze())
    # sys.exit()
    confidence = (J - mu) / torch.sqrt(sigma)

    conf_sq = confidence.squeeze(1)
    thres_topk, _ = torch.topk(conf_sq, data_size // 5)

    # conf_node = (confidence < -1).long().squeeze(1)
    conf_node = (confidence < thres_topk[-1]).long().squeeze(1)
    unconf_node = 1 - conf_node

    return conf_node.tolist(), unconf_node.tolist(), J.squeeze(1), r


def plabel_neighbor_agreement(
        domain, tok, model, gp_all, label_list, train_mode
    ):

    def list_acc(domain, l1, l2, k, top = True):
        if k == 0:
            k = len(l1)
        elif top:
            k = len(l1) - k
            l1 = l1[:k]
            l2 = l2[:k]
        else:
            l1 = l1[-k:]
            l2 = l2[-k:]

        # print(l1)
        # print(l2)
        acc = sum([int(x == y) for x, y in zip(l1, l2)]) / float(k)
        if domain not in {'qnli', 'rte'}:
            acc = 1 - acc
        return acc

    num_case = len(gp_all)
    nb_agree = [[i, 0] for i in range(num_case)]

    prompt_list = []
    prompt_con_list = []
    pred_list = []
    real_idx_list = []

    real_idx = 0
    model.eval()

    for i, gp_case_list in enumerate(gp_all):
        gp_anno_list, gp_gen_list = gp_case_list
        data_size = len(gp_gen_list) // 2

        gp_ent_list = gp_gen_list[:data_size]
        gp_con_list = gp_gen_list[data_size:]

        num_gen = data_size - 1
        prompt_batch = [x[0] for x in gp_ent_list]
        prompt_con_batch = [x[0] for x in gp_con_list]

        true_logits, false_logits, hidden_states = cls_evaluate(
            tok, model,
            prompt_batch, 0,
            output_hidden_states = True
        )
        pred_labels = (false_logits > true_logits).long().tolist()
        pred_logits_all = torch.maximum(true_logits, false_logits)

        dist_vec = -torch.cdist(
            hidden_states[:1], hidden_states
        ).squeeze(0)

        tk_v, tk_idx = dist_vec.topk(
            k = max(data_size // 5 * 4, 1), dim=0
        )
        tk_idx_list = sorted(tk_idx.tolist())
        # tk_idx_list = [x for x in range(len(prompt_batch))]
        # print(tk_idx)
        # sys.exit()

        prompt_batch = [prompt_batch[i] for i in tk_idx_list]
        prompt_con_batch = [prompt_con_batch[i] for i in tk_idx_list]
        pred_labels = [pred_labels[i] for i in tk_idx_list]

        # mj_label = int(sum(pred_labels) / len(pred_labels) > 0.5)
        # pred_labels = [mj_label for x in pred_labels]

        pred_logits = pred_logits_all[0]
        real_idx_list.append(real_idx)

        # pred_list.append(pred_labels[0].item())
        if train_mode == 'prompt_1':
            prompt_list += prompt_batch
            pred_list += pred_labels
            real_idx += len(prompt_batch)

        if train_mode == 'prompt_2':
            prompt_list += prompt_con_batch
            pred_list += pred_labels
            real_idx += len(prompt_batch)

        if train_mode == 'prompt_joint':
            prompt_list += prompt_batch + prompt_con_batch
            pred_list += pred_labels + pred_labels
            real_idx += len(prompt_batch) * 2

    p = float(sum(pred_list)) / len(pred_list)
    return prompt_list, pred_list, real_idx_list, p

    '''if num_gen == 0:
            continue

        # print(true_logits.size())
        # print(hidden_states.size())
        # sys.exit()
        dist_list = 1 / (1 + torch.cdist(
            hidden_states[1:], hidden_states[:1]
        )).squeeze(1)
        # dist_list /= dist_list.max()
        threh, tidx = torch.topk(dist_list, k=8, dim=0)
        neighbor_mask = (dist_list > threh[-1]).float()

        # print(pred_labels)
        # sys.exit()

        label_diff = (pred_labels[1:] != pred_labels[0]).float()
        # label_diff = (label_diff / (1 + dist_list) / pred_logits).sum() / num_gen
        # label_diff = 1 / pred_logits
        pred_logits_gap = (
            pred_logits_all[1:] * neighbor_mask
        ).max() / pred_logits
        # pred_logits_gap = 1 / pred_logits
        label_diff = (
            label_diff * neighbor_mask * dist_list * pred_logits_gap
        ).sum() # / label_diff.sum()

        nb_agree[i][1] = label_diff

    nb_sorted = sorted(nb_agree, key = lambda x: x[1])
    pred_sorted = [pred_list[x[0]] for x in nb_sorted]
    label_sorted = [label_list[x[0]] for x in nb_sorted]

    k = num_case // 8
    # print(nb_sorted)
    all_acc = list_acc(domain, pred_list, label_list, 0, top=True)
    top_acc = list_acc(domain, pred_sorted, label_sorted, k, top=True)
    bot_acc = list_acc(domain, pred_sorted, label_sorted, k, top=False)

    print(f'all_acc = {all_acc}, top_acc = {top_acc}, bot_acc = {bot_acc}')
    sys.exit()

    nb_filter = sorted(nb_sorted[:-k], key = lambda x: x[0])
    data_idx = [x[0] for x in nb_filter]
    pred_filter = [pred_list[x[0]] for x in nb_filter]

    return data_idx, pred_filter
    # '''


def mlm_evaluate(
        tok, model, input_list, rvs_flag,
        t_idx = None, f_idx = None, ok_idx = None,
        t_base = None, f_base = None, ok_base = None,
        model_type = 'mlm', num_prompt = None, mnli = False,
        proto_emb = None
    ):

    input_enc = tok(
        input_list,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(
            input_ids = input_ids,
            attention_mask = attn_mask
        )

    _, pred = result.logits.max(2)

    if num_prompt is None:
        offset = 0
    else:
        offset = num_prompt

    # print(input_ids[0][offset + 3])
    # sys.exit()

    true_logits = result.logits[:, offset + 3, t_idx] - t_base
    false_logits = result.logits[:, offset + 3, f_idx] - f_base
    ok_logits = result.logits[:, offset + 3, ok_idx] - ok_base

    if rvs_flag == 0:
        return true_logits, false_logits #, ok_logits
    elif rvs_flag == 1:
        return false_logits, true_logits #, ok_logits
    else:
        print('Rvs_flag not supported')
        sys.exit()


def cls_evaluate(
        tok, model, input_list, rvs_flag,
        t_idx = None, f_idx = None, ok_idx = None,
        t_base = None, f_base = None, ok_base = None,
        model_type='sc', num_prompt = None, mnli = False,
        temperature = 1, softmax = True, proto_emb = None,
        output_hidden_states = False, phi = 1
    ):

    input_enc = tok(
        input_list,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(
            input_ids = input_ids,
            attention_mask = attn_mask,
            output_hidden_states = output_hidden_states
        )

    logits = result.logits
    if not mnli:
        logits[:, 1] -= 10000

    if softmax:
        logits = F.softmax(logits * temperature, dim = -1)
    else:
        logits = result.logits

    true_logits = logits[:, 0] * phi
    false_logits = logits[:, -1]

    if mnli:
        ok_logits = logits[:, 1]
        if rvs_flag == 0:
            return true_logits, ok_logits, false_logits
        else:
            return false_logits, ok_logits, true_logits

    if output_hidden_states:
        # if hasattr(model, 'pooler'):
        #     output_emb = model.pooler(result.hidden_states[-1])
        # else:
        #     output_emb = model.bert.pooler(result.hidden_states[-1])
        output_emb = result.hidden_states[-6][:, 0, :]

        if rvs_flag == 0:
            return true_logits, false_logits, output_emb
        else:
            return false_logits, true_logits, output_emb

    if rvs_flag == 0:
        return true_logits, false_logits
    else:
        return false_logits, true_logits


def proto_evaluate(
        tok, model, input_list, rvs_flag,
        t_idx = None, f_idx = None, ok_idx = None,
        t_base = None, f_base = None, ok_base = None,
        model_type='sc', num_prompt = None, mnli = False,
        temperature = 1, softmax = True, proto_emb = None
    ):

    input_enc = tok(
        input_list,
        max_length = 512,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    with torch.no_grad():
        result = model(
            input_ids = input_ids,
            attention_mask = attn_mask,
            output_hidden_states = True
        )
        if hasattr(model, 'module'):
            if hasattr(model.module, 'pooler'):
                output_emb = model.module.pooler(result.hidden_states[-1])
            else:
                output_emb = model.module.bert.pooler(result.hidden_states[-1])
        else:
            if hasattr(model, 'pooler'):
                output_emb = model.pooler(result.hidden_states[-1])
            else:
                output_emb = model.bert.pooler(result.hidden_states[-1])


    # logits = torch.mm(output_emb, proto_emb.T)
    output_emb = (output_emb - proto_emb.mean()) / proto_emb.std()
    logits = -torch.cdist(output_emb, proto_emb)
    if not mnli:
        logits[:, 1] -= 10000
    if softmax:
        logits = F.softmax(result.logits * temperature, dim = -1)

    true_logits = logits[:, 0]
    false_logits = logits[:, -1]

    if rvs_flag == 0:
        return true_logits, false_logits
    else:
        return false_logits, true_logits


def gen_prompt_tok(num_prompt):
    prompt_tokens = [f'<prompt_token_{i}>' for i in range(num_prompt)]
    return prompt_tokens


def add_prompt_layer(model, dataset_name, num_prompt, model_type_str):
    if model_type_str == 'bert':
        model.bert.embeddings.word_embeddings = PromptEmbedding(
            model.bert.embeddings.word_embeddings, num_prompt,
            # f'model_ft_file/{dataset_name}_prompt_emb.pt'
            prompt_path = f'model_ft_file/mnli_prompt_emb_binary.pt'
        )

        model.cls.predictions.decoder = PromptDecoder(
            model.cls.predictions.decoder,
            num_prompt,
            model.bert.embeddings.word_embeddings.prompt_emb
        )
    elif model_type_str == 'deberta':
        pass
    else:
        print(f'Model {model_type_str} not supported')
        sys.exit()
    return model


def eval_prompt_seq_cls(
        dataset_name, eval_mode, model_tag, data_split,
        data = None, mlm = False, model_type = 'sc'
    ):

    tok = AutoTokenizer.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
        # f'model_file/{model_type_str}-large-tok.pt'
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f'model_ft_file/cls_{dataset_name}_{model_tag}_{data_split}.pt'
    )

    model.cuda()
    model.eval()

    model = nn.DataParallel(model)

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_mode
    )

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    num_crr = 0

    batch_size = 32

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        label_batch = torch.Tensor(label_list[i: i + batch_size]).long()

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli':
            score_board = torch.zeros(cur_bs, 3)
        else:
            score_board = torch.zeros(cur_bs, 2)

        for j in range(num_prompt_type):
            false_scores, true_scores, ok_scores = cls_evaluate(
                tok, model, prompt_input_group[j], rvs_map[j], model_type
            )

            # '''
            score_board[:, 0] += false_scores.cpu() # - f_base
            if dataset_name == 'mnli':
                score_board[:, 2] += true_scores.cpu() # - t_base
                score_board[:, 1] += ok_scores.cpu()
            else:
                score_board[:, 1] += true_scores.cpu()
            # '''

            # print(score_board)
            # sys.exit()
            _, pred = score_board.max(1)

        _, pred = score_board.max(1)
        num_crr += (pred == label_batch).float().sum()

        '''
        print('')
        print(pred)
        print(label_batch)
        sys.exit()
        # '''

    acc = num_crr / num_case

    print('------------------------')
    print(f'Accuracy = {acc}')
    print('------------------------\n')


def const_distance(f_logits_1, t_logits_1, f_logits_2, t_logits_2):
    log_prob_1 = F.log_softmax(torch.cat(
        [f_logits_1.unsqueeze(1), t_logits_1.unsqueeze(1)], dim = 1
    ), dim = 1)
    log_prob_2 = F.log_softmax(torch.cat(
        [t_logits_2.unsqueeze(1), f_logits_2.unsqueeze(1)], dim = 1
    ), dim = 1)
    dist = log_prob_1 - log_prob_2
    entropy_1 = log_prob_1[:, 0] - log_prob_1[:, 1]
    entropy_2 = log_prob_2[:, 0] - log_prob_2[:, 1]
    entropy = (entropy_1 * entropy_1 + entropy_2 * entropy_2).mean()
    return (dist * dist).mean() * 1 + entropy * 0.05


def eval_prompt_const(
        dataset_name, eval_mode, model_tag, data_split,
        data = None, mlm = False, model_type = 'sc'
    ):

    tok = AutoTokenizer.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
        # f'model_file/{model_type_str}-large-tok.pt'
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f'model_ft_file/cls_{dataset_name}_{model_tag}_{data_split}.pt'
    )

    model.cuda()
    model.train()

    model = nn.DataParallel(model)

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_mode
    )

    if data is not None:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    total_loss = 0

    batch_size = 32

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        false_scores_1, true_scores_1 = cls_evaluate(
            tok, model, prompt_input_group[0], rvs_map[0], model_type
        )

        false_scores_2, true_scores_2 = cls_evaluate(
            tok, model, prompt_input_group[1], rvs_map[1], model_type
        )

        loss = const_distance(
            false_scores_1, true_scores_1, false_scores_2, true_scores_2
        )

        total_loss += loss * cur_bs

    print(f'Total_loss = {total_loss.item()}\n')
    return total_loss / num_case


def calibrate_weight(dataset_name, tok, model):
    if dataset_name == 'sst2':
        null_p = 'the movie is good is entailed by comment..'
    if dataset_name == 'qnli':
        null_p = 'the answer to question? is entailed by sentence..'
    if dataset_name == 'qqp':
        null_p = 'the answer to question? is entailed by another question?.'
    if dataset_name == 'rte':
        null_p = 'the hypothesis is entailed by the premise..'
    if dataset_name == 'cola':
        null_p = 'the sentence. is fluent.'

    input_list = [null_p]
    true_logits, false_logits = cls_evaluate(
        tok, model, input_list, 0,
        model_type='sc', mnli = False,
        temperature = 1, softmax = False
    )

    weight = false_logits / true_logits
    return weight


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def prompt_seq_cls_relabel(
        dataset_name, sent1_list, sent2_list, model_type = 'sc',
        tok = None, model = None, model_path = None, mnli = False,
        model_type_str='deberta', model_size_str='large',
        mlm=False, batch_size=16, num_prompt_type=1, prompt_sep=False,
        proto_emb = None, dropout = False
    ):

    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            # f'model_file/deberta-large-tok.pt'
            f'model_file/{model_type_str}-{model_size_str}-tok.pt'
        )

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

    model.cuda()
    model.eval()

    if dropout:
        model.train()

    model = nn.DataParallel(model)

    num_case = len(sent1_list)
    pred_list = []
    score_board_list = []
    hidden_states_list = []

    if proto_emb is None:
        eval_func = cls_evaluate
    else:
        eval_func = proto_evaluate

    # cal_w = calibrate_weight(dataset_name, tok, model).cpu()

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        if sent2_list is None:
            sent2_batch = sent1_batch
        else:
            sent2_batch = sent2_list[i: i + batch_size]

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm, sep=prompt_sep
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli' or mnli:
            score_board = torch.zeros(cur_bs, 3)
        else:
            score_board = torch.zeros(cur_bs, 2)

        for j in range(0, num_prompt_type):

            if dataset_name == 'mnli' or mnli:
                false_scores, ok_scores, true_scores = eval_func(
                    tok, model,
                    prompt_input_group[j],
                    int(j==1 and rvs_map[1] != rvs_map[0]),
                    model_type, mnli = True, softmax = True,
                    proto_emb = proto_emb
                )
            else:
                false_scores, true_scores, hidden_states = eval_func(
                    tok, model,
                    prompt_input_group[j],
                    int(j==1 and rvs_map[1] != rvs_map[0]),
                    model_type, mnli = False, softmax = False,
                    proto_emb = proto_emb, output_hidden_states = True
                )

            score_board[:, 0] += false_scores.cpu() # * cal_w # - f_base
            if dataset_name == 'mnli' or mnli:
                score_board[:, 2] += true_scores.cpu() # - t_base
                score_board[:, 1] += ok_scores.cpu()
            else:
                score_board[:, 1] += true_scores.cpu()

            _, pred = score_board.max(1)

            hidden_states_list.append(hidden_states)

        _, pred = score_board.max(1)
        pred_list += pred.tolist()
        score_board_list.append(score_board)

    score_board_tensor = torch.cat(score_board_list, dim = 0)
    hidden_states_tensor = torch.cat(hidden_states_list, dim = 0)

    return pred_list, score_board_tensor, hidden_states_tensor


def prompt_cse_relabel(
        dataset_name, sent1_list, sent2_list, model_type = 'sc',
        tok = None, model = None, model_path = None, mnli = False,
        model_type_str='deberta', model_size_str='large',
        mlm=False, batch_size=16, num_prompt_type=1, prompt_sep=False,
        proto_emb = None
    ):

    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            # f'model_file/deberta-large-tok.pt'
            f'model_file/{model_type_str}-{model_size_str}-tok.pt'
        )

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

    model.cuda()
    model.eval()

    model = nn.DataParallel(model)

    num_case = len(sent1_list)
    pred_list = []
    score_board_list = []

    if proto_emb is None:
        eval_func = cls_evaluate
    else:
        eval_func = proto_evaluate

    # cal_w = calibrate_weight(dataset_name, tok, model).cpu()

    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        if sent2_list is None:
            sent2_batch = sent1_batch
        else:
            sent2_batch = sent2_list[i: i + batch_size]

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch, mlm, sep=prompt_sep
        )

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli' or mnli:
            score_board = torch.zeros(cur_bs, 3)
        else:
            score_board = torch.zeros(cur_bs, 2)

        for j in range(0, num_prompt_type):

            if dataset_name == 'mnli' or mnli:
                false_scores, ok_scores, true_scores = eval_func(
                    tok, model,
                    prompt_input_group[j],
                    int(j==1 and rvs_map[1] != rvs_map[0]),
                    model_type, mnli = True, softmax=True,
                    proto_emb = proto_emb
                )
            else:
                false_scores, true_scores = eval_func(
                    tok, model,
                    prompt_input_group[j], 0,
                    # int(j==1 and rvs_map[1] != rvs_map[0]),
                    model_type, mnli = False, softmax=False,
                    proto_emb = proto_emb
                )

            iter_flag = 1. - j * 2
            false_scores *= iter_flag
            true_scores *= iter_flag

            score_board[:, 0] += false_scores.cpu() # * cal_w # - f_base
            if dataset_name == 'mnli' or mnli:
                score_board[:, 2] += true_scores.cpu() # - t_base
                score_board[:, 1] += ok_scores.cpu()
            else:
                score_board[:, 1] += true_scores.cpu()

            # _, pred = score_board.max(1)
            # print(score_board)
            # print(pred)
            # sys.exit()

        # _, pred = score_board.max(1)
        pred = (score_board[:, 0] > 0).int()
        if rvs_map[0]:
            pred = 1 - pred

        pred_list += pred.tolist()
        score_board_list.append(score_board)

    return pred_list, torch.cat(score_board_list, dim=0)


def evaluate_func(
        dataset_name,
        model_mode,
        eval_split,
        model_type,
        num_prompt_type,
        num_prompt,
        tok,
        model,
        prompt_str = None,
        dev_split_id = None,
        return_loss = False,
        exp_id = None,
        model_type_str = 'none'
    ):

    if model_type == 'mlm':
        eval_func = mlm_evaluate
    elif model_type == 'sc':
        eval_func = cls_evaluate
    elif model_type == 'proto':
        eval_func = proto_evaluate
    else:
        print(f'model_type {model_type} not supported')
        sys.exit()
    loss_fn = nn.CrossEntropyLoss()

    t_idx = tok.convert_tokens_to_ids('true')
    f_idx = tok.convert_tokens_to_ids('false')
    ok_idx = tok.convert_tokens_to_ids('ok')

    # model.cuda()
    model.eval()

    sent1_list, sent2_list, label_list, dformat = load_eval_data(
        dataset_name, eval_split, dev_split_id=dev_split_id
    )

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    num_crr_0 = 0
    num_crr_1 = 0
    num_crr = 0
    loss_0 = 0
    loss_1 = 0
    loss = 0

    batch_size = 32

    t_base, f_base, ok_base = get_base_logits(
        tok, model, t_idx, f_idx, ok_idx,
        num_prompt, prompt_str = prompt_str,
        mlm = (model_type == 'mlm')
    )

    # cal_w = calibrate_weight(dataset_name, tok, model)
    try:
        w0, w1 = json.load(open(
            f'log/{dataset_name}/cal_w_{exp_id}.json'
        ))
    except:
        # print('no cal_w ckpt')
        w0, w1 = [1., 1.]
    cal_w = w0 / w1

    # print(dataset_name)
    # print(exp_id)

    try:
        # proto_emb = torch.load(
        #     f'model_ft_file/proto_emb_{dataset_name}_syn_data_relabel_{exp_id}.pt'
        # ).cuda()
        proto_emb = torch.load(f'model_ft_file/{dataset_name}_proto_emb.pt').cuda()
    except:
        # sys.exit()
         #print('No self-trained checkpoint, using pretrained')
        proto_emb = None
        # proto_emb = torch.load(f'model_ft_file/{dataset_name}_proto_emb.pt')

    conf_list = []
    crr_list = []
    for i in range(0, num_case, batch_size):
        sent1_batch = sent1_list[i: i + batch_size]
        sent2_batch = sent2_list[i: i + batch_size]
        label_batch = torch.LongTensor(
            label_list[i: i + batch_size]
        ).cuda()

        cur_bs = len(sent1_batch)

        prompt_input_list, rvs_map = build_prompt_input(
            dataset_name, sent1_batch, sent2_batch,
            mlm = (model_type == 'mlm'), sep = False
        )

        if model_mode == 'pt':
            prompt_input_list  = [
                f'{prompt_str} {x}' for x in prompt_input_list
            ]

        prompt_input_group = [
            prompt_input_list[:cur_bs], prompt_input_list[cur_bs:]
        ]

        if dataset_name == 'mnli':
            score_board_0 = torch.zeros(cur_bs, 3).cuda()
            score_board_1 = torch.zeros(cur_bs, 3).cuda()
            score_board = torch.zeros(cur_bs, 3).cuda()
        else:
            score_board_0 = torch.zeros(cur_bs, 2).cuda()
            score_board_1 = torch.zeros(cur_bs, 2).cuda()
            score_board = torch.zeros(cur_bs, 2).cuda()

        for j in range(0, num_prompt_type):

            if dataset_name == 'mnli':
                false_scores, ok_scores, true_scores = eval_func(
                    tok, model, prompt_input_group[j], rvs_map[j],
                    t_idx = t_idx, f_idx = f_idx, ok_idx = ok_idx,
                    t_base = t_base, f_base = f_base, ok_base = ok_base,
                    num_prompt = num_prompt, model_type = model_type,
                    mnli = True, proto_emb = proto_emb
                )
            else:
                false_scores, true_scores = eval_func(
                    tok, model, prompt_input_group[j], rvs_map[j],
                    t_idx = t_idx, f_idx = f_idx, ok_idx = ok_idx,
                    t_base = t_base, f_base = f_base, ok_base = ok_base,
                    num_prompt = num_prompt, model_type = model_type,
                    mnli = False, softmax = True, proto_emb = proto_emb,
                    # phi = cal_w
                )

            # if rvs_map[j] == 0:
            #     false_scores *= cal_w
            # else:
            #     true_scores *= cal_w

            if j == 0:
                score_board_0[:, 0] += false_scores # - f_base
                if dataset_name == 'mnli':
                    score_board_0[:, 2] += true_scores # - t_base
                    score_board_0[:, 1] += ok_scores
                else:
                    score_board_0[:, 1] += true_scores
            if j == 1:
                score_board_1[:, 0] += false_scores # - f_base
                if dataset_name == 'mnli':
                    score_board_1[:, 2] += true_scores # - t_base
                    score_board_1[:, 1] += ok_scores
                else:
                    score_board_1[:, 1] += true_scores

            score_board[:, 0] += false_scores # - f_base
            if dataset_name == 'mnli':
                score_board[:, 2] += true_scores # - t_base
                # score_board[:, 1] += ok_scores
            else:
                score_board[:, 1] += true_scores

            _, pred = score_board.max(1)

        conf, pred_0 = score_board_0.max(1)
        crr = (pred_0 == label_batch).float()
        num_crr_0 += crr.sum().item()
        loss_0 += loss_fn(score_board_0, label_batch).item() * cur_bs

        _, pred_1 = score_board_1.max(1)
        num_crr_1 += (pred_1 == label_batch).float().sum().item()
        loss_1 += loss_fn(score_board_1, label_batch).item() * cur_bs

        _, pred = score_board.max(1)
        num_crr += (pred == label_batch).float().sum().item()
        loss += loss_fn(score_board, label_batch).item() * cur_bs

        conf_list += conf.tolist()
        crr_list += crr.tolist()

    acc_0 = num_crr_0 / num_case
    acc_1 = num_crr_1 / num_case
    acc = num_crr / num_case

    loss_0 /= num_case
    loss_1 /= num_case
    loss /= num_case

    conf_data = {
        'conf_list': conf_list,
        'crr_list': crr_list
    }

    json.dump(conf_data, open(
        f'log/confidence/{dataset_name}_ent_{model_type_str}_{eval_split}.json', 'w'
    ))

    if not return_loss:
        return acc_0, acc_1, acc
    else:
        return acc_0, acc_1, acc, loss_0, loss_1, loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='python eval_prompt.py',
        description='Evaluation of self-trained classifiers with different prompts',
        epilog='Submit issues on Github for addtional help.'
    )

    # Task parameters
    parser.add_argument('--domain', type=str)
    parser.add_argument('--model-mode', type=str, default='mt')
    parser.add_argument('--eval-split', type=str)
    parser.add_argument('--num-prompt-type', type=int, default=2)
    parser.add_argument('--num-prompt', type=int, default=50)

    # Model parameters
    parser.add_argument("--model-config", type=str, default="self_train")
    parser.add_argument('--exp-id', type=str)
    parser.add_argument('--model-type', type=str, help='If the model is a classifier or MLM.', default='sc')
    parser.add_argument("--model-type-str", type=str, default="deberta")

    args = parser.parse_args()

    dataset_name = args.domain
    model_mode = args.model_mode
    eval_split = args.eval_split
    num_prompt_type = args.num_prompt_type
    num_prompt = args.num_prompt
    model_config = args.model_config
    exp_id = args.exp_id
    model_type = args.model_type
    model_type_str = args.model_type_str

    if model_config == 'pretrain':
        model_path = f'luohy/ESP-{model_type_str}-large'
    elif model_config == 'self_train':
        model_path = f'model_ft_file/cls_{dataset_name}_{model_type_str}_syn_data_relabel_{exp_id}.pt'
    elif model_config == 'adp_train':
        model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
        # model_path = 'model_ft_file/mnli_model_sc_3e-06_binary_pr.pt'
    else:
        print(f'\n{model_config} not supported.\n')
        sys.exit()

    # model_type_str = 'roberta'
    if model_type_str == 'roberta':
        tokenizer_path = 'roberta-large'
    
    elif model_type_str == 'deberta':
        tokenizer_path = 'microsoft/deberta-large'
    
    else:
        print(f'\nBackbone model {model_type_str} not supported.\n')

    tok = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    if model_type == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(
            model_path
        )
        eval_func = mlm_evaluate
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        eval_func = cls_evaluate

    if model_mode == 'mt':
        prompt_str = None
        num_prompt = None

    elif model_mode == 'pt':
        model = AutoModelForMaskedLM.from_pretrained(
            f'model_file/{model_type_str}-large-mlm.pt'
        )

        prompt_tok_list = gen_prompt_tok(num_prompt)
        tok.add_tokens(prompt_tok_list)
        prompt_str = ' '.join(prompt_tok_list)

        model = add_prompt_layer(
            model, dataset_name, num_prompt, model_type_str
        )

    else:
        print(f'\nModel mode {model_mode} not supported.\n')
        sys.exit()

    model.eval()
    model = model.cuda()
    model = nn.DataParallel(model)

    acc_0, acc_1, acc = evaluate_func(
        dataset_name, model_mode, eval_split, model_type,
        num_prompt_type, num_prompt, tok, model,
        prompt_str = prompt_str, exp_id = exp_id,
        model_type_str = model_type_str
    )

    print(f'\nAcc with suppotion 0 = {acc_0}, Acc with supposition 1 = {acc_1}, Acc_joint = {acc}\n')