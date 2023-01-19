from distutils.log import log
import sys
import json
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DebertaTokenizerFast,
    DebertaForSequenceClassification,
    utils
)
from eval_task_adp import proc_input, proc_output

from mix_model import mix
from eval_prompt import (
    evaluate_func, get_proto_emb, get_unconf_nodes, cls_evaluate
)

utils.logging.set_verbosity_warning()

domain_dict = {
    'sst2': 'SST-2',
    'qqp': 'QQP',
    'mnli': 'MNLI',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'cola': 'CoLA',
    'mrpc': 'MRPC'
}

def shuffle_data(sent1_list, sent2_list, label_list, dtype_list):
    if sent2_list is None:
        data = list(zip(sent1_list, label_list, dtype_list))
    else:
        data = list(zip(sent1_list, sent2_list, label_list, dtype_list))

    random.shuffle(data)
    sent1_list = [x[0] for x in data]
    label_list = [x[-2] for x in data]
    dtype_list = [x[-1] for x in data]

    if sent2_list is not None:
        sent2_list = [x[1] for x in data]
    return sent1_list, sent2_list, label_list, dtype_list


def load_annotate_data(domain):
    all_data = json.load(open(
        f'data/glue_data/{domain_dict[domain]}/train_proc.json'
    ))

    sent1_list = all_data['sent1_list']
    sent2_list = all_data['sent2_list']
    label_list = all_data['label_list']
    dformat = all_data['dformat']

    return sent1_list, sent2_list, label_list, dformat


def load_train_data(domain, data_split, no_train=False):
    all_data = json.load(open(
        f'data/glue_data/{domain_dict[domain]}/{data_split}_proc.json'
    ))

    sent1_list = all_data['sent1_list']
    sent2_list = all_data['sent2_list']
    label_list = all_data['label_list']
    dformat = all_data['dformat']

    dtype_list = [0] * len(sent1_list)

    if 'syn_' in data_split and 'next' not in data_split and not no_train:

        print(f'\nNum of synthetic cases = {len(sent1_list)}\n')

        sent1_anno, sent2_anno, label_anno, _ = load_annotate_data(
            domain
        )

        dtype_list += [1] * len(sent1_anno)

        sent1_list += sent1_anno
        if sent2_list is not None:
            sent2_list += sent2_anno
        label_list += label_anno

        '''try:
            rl_data = json.load(open(f'log/{domain}_rl_data.json'))

            sent1_rl = rl_data['sent1_list']
            sent2_rl = rl_data['sent2_list']
            label_rl = rl_data['label_list']

            print(f'\nNum of RL synthetic cases = {len(sent1_rl)}\n')

            sent1_list += sent1_rl
            if sent2_list is not None:
                sent2_list += sent2_rl
            label_list += label_rl

            dtype_list += [1] * len(sent1_rl)
        except:
            pass'''

    sent1_list, sent2_list, label_list, dtype_list = shuffle_data(
        sent1_list, sent2_list, label_list, dtype_list
    )

    return sent1_list, sent2_list, label_list, dtype_list, dformat


def adapt(
        model,
        tok,
        sent1_list,
        sent2_list,
        label_list,
        dtype_list,
        dformat,
        num_epochs = 1,
        batch_size = 32,
        num_labels = 2,
        dataset_name = 'sst2',
        model_tag = 'large',
        data_split = 'train',
        task = 'cls',
        save = True,
        parallel = True,
        verbose = True,
        from_mnli = True,
        prompt_mode = None,
        eval = False,
        exp_id = None,
        eval_mode = 'base',
        early_stop = False,
    ):

    def dev_condition(acc, acc_val, loss, loss_val):
        if acc > acc_val:
            return True
        elif acc == acc_val and loss < loss_val - 0.01:
            return True
        else:
            return False

    t_total = int(len(sent1_list) / batch_size * num_epochs)
    if exp_id is not None:
        data_split = f'{data_split}_{exp_id}'

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-6, eps=1e-6, # weight_decay=1e-7
    )
    # model_base = copy.deepcopy(model)
    # model_base.train()

    if eval:
        model.eval()
    else:
        model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    model = model.cuda()
    # model_base = model_base.cuda()

    if parallel:
        model = nn.DataParallel(model)

    step_id = 0
    acc_val = 0
    loss_val = 100

    for epoch in range(num_epochs):
        model.train()

        for j in range(0, len(sent1_list), batch_size):
            sent1_batch = sent1_list[j: j + batch_size]
            
            if sent2_list is None:
                sent2_batch = None
            else:
                sent2_batch = sent2_list[j: j + batch_size]
            label_batch = label_list[j: j + batch_size]
            
            if dtype_list is None:
                dtype_batch = None
            else:
                dtype_batch = dtype_list[j: j + batch_size]
            
            data_size = len(sent1_batch) // 2

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
            
            dtype_tensor = None
            if dtype_batch:
                dtype_tensor = torch.Tensor(dtype_batch).cuda()

            input_ids = input_enc['input_ids'].cuda()
            attn_mask = input_enc['attention_mask'].cuda()

            result = model(
                input_ids = input_ids,
                attention_mask = attn_mask,
            )
            
            '''
            with torch.no_grad():
                result_base = model_base(
                    input_ids = input_ids,
                    attention_mask = attn_mask,
                )
            # '''

            if from_mnli:
                proc_logits = proc_output(dataset_name, result.logits)
                # proc_logits_base = proc_output(dataset_name, result_base.logits)
            else:
                proc_logits = result.logits
                # proc_logits_base = result_base.logits
            
            loss = loss_fct(proc_logits, label_tensor).mean()
            # reg = (loss - 1e-7) ** 2
            # loss = (loss + 1e-2 * reg).mean()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            model.zero_grad()

            if step_id % 100 == 0 and verbose:
                if dtype_tensor is not None:
                    print(dtype_tensor)
                print(f'Trained {step_id} / {t_total} steps, loss = {loss}\n')
            step_id += 1
        
        # print(model.module.device)
        # abort()
        if early_stop:
            acc_0, acc_1, acc, loss_0, loss_1, loss = evaluate_func(
                dataset_name, 'mt', eval_mode, 'sc', 2, 50, tok, model.module,
                dev_split_id = exp_id, return_loss = True
            )
            
            acc_best = acc_0
            loss_best = loss_0
            # acc_best = min([acc_0, acc_1, acc])
            # loss_best = min([loss_0, loss_1, loss])
            dev_satis = dev_condition(acc_best, acc_val, loss_best, loss_val)
            
        if save and early_stop and (dev_satis or epoch < 6):
            # print(
            #     f'Current epoch: {epoch}, Acc. = {acc_best}, Loss = {loss_best}'
            # )
            # if dev_satis:
            acc_val = acc_best
            loss_val = loss_best
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(
                f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
            )
            # print(f'---- Epoch {epoch} Checkpoint saved ----')
            
        sent1_list, sent2_list, label_list, _ = shuffle_data(
            sent1_list, sent2_list, label_list, label_list
        )
        
        # model.module = mix(model_base, model.module, 0.5)

        if verbose:
            print('Epoch {} finished'.format(epoch))
            print('-' * 89)

    if save and not early_stop:
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(
            f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
        )
        # print(f'---- Epoch {epoch} Checkpoint saved ----')

    return model.module if hasattr(model, 'module') else model


def adapt_glue_func(
        dataset_name, model_tag, data_split, task,
        data = None, no_train = False, verbose = True,
        from_mnli = True, num_epochs = 1, prompt_mode = None,
        eval = False, exp_id = None, model_type_str = 'deberta',
        eval_mode = 'base', model_config_pt = None, train_mode = None,
        robust_loss_func = 'gm', c = 1e-2
    ):

    if dataset_name == 'sst2':
        batch_size = 32
    elif dataset_name == 'mnli':
        batch_size = 12
    elif dataset_name == 'rte':
        batch_size = 16
    elif dataset_name == 'qqp':
        batch_size = 16
    elif dataset_name == 'qnli':
        batch_size = 8
    else:
        batch_size = 32

    try:
        sent1_list, sent2_list, label_list, dtype_list, dformat = load_train_data(
            dataset_name, data_split, no_train=no_train
        )
        
        sent1_list, sent2_list = proc_input(
            dataset_name, sent1_list, sent2_list
        )
    except:
        dformat = {'label_dict': {0: 0, 1: 1}}

    if data is not None:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']
        label_list = data['label_list']

    num_labels = len(dformat['label_dict'])
    
    if verbose:
        print(f'\nBatch_size = {batch_size}')
        print(f'\nUsing model deberta-{model_tag}-sc-{num_labels}.pt\n')

    tok = AutoTokenizer.from_pretrained(
        # 'model_file/bert-large-tok.pt'
        f'model_file/{model_type_str}-{model_tag}-tok.pt'
    )

    if task == 'cls':
        if from_mnli:
            # model_config = f'model_ft_file/cls_mnli_{model_tag}_train.pt'
            # model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
            model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_pr.pt'
            # model_config = 'model_ft_file/mnli_model_sc_5e-06_binary_pb.pt'
            # model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_pk24.pt'
            # model_config = 'model_ft_file/mnli_model_sc_1e-05_binary_meta_ep99_maml-10-200.pt'
        else:
            model_config = 'model_file/deberta-large-sc-2.pt'
    else:
        model_config = f'model_ft_file/dis_{dataset_name}_{model_tag}_syn_data_relabel.pt'

    if model_config_pt is not None:
        model_config = model_config_pt
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config
    )

    if dataset_name == 'cola':
        dtype_list = label_list

    model = adapt(
        model, tok,
        sent1_list, sent2_list, label_list, dtype_list,
        dformat, num_epochs, batch_size, num_labels,
        dataset_name, model_tag, data_split, task,
        save = True, parallel = True, verbose = verbose,
        from_mnli = from_mnli, prompt_mode = prompt_mode,
        eval = eval, exp_id = exp_id, early_stop = False,
        eval_mode = eval_mode
    )


def gnc(
        model,
        tok,
        sent1_list,
        sent2_list,
        label_list,
        dtype_list,
        dformat,
        num_epochs = 1,
        batch_size = 32,
        num_labels = 2,
        dataset_name = 'sst2',
        model_tag = 'large',
        data_split = 'train',
        task = 'cls',
        save = True,
        parallel = True,
        verbose = True,
        from_mnli = True,
        prompt_mode = None,
        eval = False,
        exp_id = None,
        eval_mode = 'base',
        early_stop = False,
        w_list = None,
        phi = 1
    ):

    def dev_condition(acc, acc_val, loss, loss_val):
        if acc > acc_val:
            return True
        elif acc == acc_val and loss < loss_val - 0.01:
            return True
        else:
            return False
    
    # phi_tensor = torch.Tensor([[phi, 1]]).cuda()
    phi_tensor = phi

    t_total = int(len(sent1_list) / batch_size * num_epochs)
    if exp_id is not None:
        data_split = f'{data_split}_{exp_id}'

    optimizer = torch.optim.AdamW(
        model.parameters(),
        # [model.classifier.weight],
        lr=4e-6, eps=1e-6,
        # weight_decay=1e-5
    )
    # model_base = copy.deepcopy(model)
    # model_base.train()

    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    # model = model.cuda()
    # model_base = model_base.cuda()

    if parallel:
        model = nn.DataParallel(model)

    step_id = 0
    acc_val = 0
    loss_val = 100

    for epoch in range(num_epochs):
        model.train()

        for j in range(0, len(sent1_list), batch_size):
            sent1_batch = sent1_list[j: j + batch_size]
            
            if sent2_list is None:
                sent2_batch = None
            else:
                sent2_batch = sent2_list[j: j + batch_size]
            label_batch = label_list[j: j + batch_size]
            
            if dtype_list is None:
                dtype_batch = None
            else:
                dtype_batch = dtype_list[j: j + batch_size]
            
            w_batch = w_list[j: j + batch_size]
            
            data_size = len(sent1_batch) // 2

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
            
            dtype_tensor = None
            if dtype_batch:
                dtype_tensor = torch.Tensor(dtype_batch).cuda()

            input_ids = input_enc['input_ids'].cuda()
            attn_mask = input_enc['attention_mask'].cuda()

            result = model(
                input_ids = input_ids,
                attention_mask = attn_mask,
            )
            
            '''
            with torch.no_grad():
                result_base = model_base(
                    input_ids = input_ids,
                    attention_mask = attn_mask,
                )
            # '''

            if from_mnli:
                proc_logits = proc_output(dataset_name, result.logits)
                # proc_logits_base = proc_output(dataset_name, result_base.logits)
            else:
                proc_logits = result.logits
                # proc_logits_base = result_base.logits
            
            # proc_logits *= phi_tensor
            loss = (
                loss_fct(proc_logits, label_tensor) * w_batch
            ) #.mean()
            
            # reg = (loss - 1e-7) ** 2
            # loss = (loss + 1e-4 * reg) * w_batch #.mean()

            loss.mean().backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            model.zero_grad()

            if step_id % 100 == 0 and verbose:
                if dtype_tensor is not None:
                    print(dtype_tensor)
                print(f'Trained {step_id} / {t_total} steps, loss = {loss}\n')
            step_id += 1
            
        # sent1_list, sent2_list, label_list, _ = shuffle_data(
        #     sent1_list, sent2_list, label_list, label_list
        # )

        if verbose:
            print('Epoch {} finished'.format(epoch))
            print('-' * 89)
    
    loss_list = []
    new_label_list = []
    
    model.eval()
    for j in range(0, len(sent1_list), batch_size):
        sent1_batch = sent1_list[j: j + batch_size]
        
        if sent2_list is None:
            sent2_batch = None
        else:
            sent2_batch = sent2_list[j: j + batch_size]
        label_batch = label_list[j: j + batch_size]
        
        if dtype_list is None:
            dtype_batch = None
        else:
            dtype_batch = dtype_list[j: j + batch_size]
        
        data_size = len(sent1_batch) // 2

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
        
        dtype_tensor = None
        if dtype_batch:
            dtype_tensor = torch.Tensor(dtype_batch).cuda()

        input_ids = input_enc['input_ids'].cuda()
        attn_mask = input_enc['attention_mask'].cuda()

        with torch.no_grad():
            result = model(
                input_ids = input_ids,
                attention_mask = attn_mask,
            )

        if from_mnli:
            proc_logits = proc_output(dataset_name, result.logits)
            # proc_logits_base = proc_output(dataset_name, result_base.logits)
        else:
            proc_logits = result.logits
            # proc_logits_base = result_base.logits
        
        # proc_logits = F.log_softmax(proc_logits, dim = 1)
        # proc_logits[:, 0] *= phi
        
        loss = loss_fct(proc_logits * phi_tensor, label_tensor)#.mean()
        # loss = -(
        #     proc_logits * F.one_hot(label_tensor) * phi_tensor
        # ).sum(1) # * w_batch
        loss_list.append(loss)

        _, pred = proc_logits.max(1)
        new_label_list.append(pred)
    
    loss_list = torch.cat(loss_list, dim = 0)
    new_label_list = torch.cat(new_label_list, dim = 0).long().tolist()

    if save and not early_stop:
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(
            f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
        )
        # print(f'---- Epoch {epoch} Checkpoint saved ----')

    return model.module if hasattr(model, 'module') else model, loss_list, new_label_list


def gnc_glue_func(
        dataset_name, model_tag, data_split, task,
        data = None, no_train = False, verbose = True,
        from_mnli = True, num_epochs = 1, prompt_mode = None,
        eval = False, exp_id = None, model_type_str = 'deberta',
        eval_mode = 'base', model_config_pt = None, train_mode = None,
        robust_loss_func = 'gm', c = 1e-2
    ):

    if dataset_name == 'sst2':
        batch_size = 32
    elif dataset_name == 'mnli':
        batch_size = 12
    elif dataset_name == 'rte':
        batch_size = 16
    elif dataset_name == 'qqp':
        batch_size = 16
    elif dataset_name == 'qnli':
        batch_size = 8
    else:
        batch_size = 32

    try:
        sent1_list, sent2_list, label_list, dtype_list, dformat = load_train_data(
            dataset_name, data_split, no_train=no_train
        )
        
        sent1_list, sent2_list = proc_input(
            dataset_name, sent1_list, sent2_list
        )
    except:
        dformat = {'label_dict': {0: 0, 1: 1}}

    if data is not None:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']
        label_list = data['label_list']

    num_labels = len(dformat['label_dict'])
    
    if verbose:
        print(f'\nBatch_size = {batch_size}')
        print(f'\nUsing model deberta-{model_tag}-sc-{num_labels}.pt\n')

    tok = AutoTokenizer.from_pretrained(
        # 'model_file/bert-large-tok.pt'
        f'model_file/{model_type_str}-{model_tag}-tok.pt'
    )

    if task == 'cls':
        if from_mnli:
            # model_config = f'model_ft_file/cls_mnli_{model_tag}_train.pt'
            model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
            # model_config = 'model_ft_file/mnli_model_sc_5e-06_binary_pb.pt'
            # model_config = 'model_ft_file/mnli_model_sc_5e-06_single_p0.pt'
            # model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_pk24.pt'
            # model_config = 'model_ft_file/mnli_model_sc_1e-05_binary_meta_ep99_maml-10-200.pt'
        else:
            model_config = 'model_file/deberta-large-sc-2.pt'
    else:
        model_config = f'model_ft_file/dis_{dataset_name}_{model_tag}_syn_data_relabel.pt'

    if model_config_pt is not None:
        model_config = model_config_pt
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config
    ).cuda()

    if dataset_name == 'cola':
        dtype_list = label_list
    
    w_list = torch.ones(len(sent1_list)).cuda()
    eta = 1e-2
    mu_step = 2.8
    prev_mu_rsq = -100

    if robust_loss_func == 'gm':
        mu = 10000
    else:
        mu = 1e-5
    
    num_case = len(sent1_list)
    data_size = num_case // 2

    num_episode = 0
    max_episode = num_epochs

    '''_, _, edge_cut, c = get_unconf_nodes(
        sent1_list, label_list, # pseudo_label_scores,
        tok = tok, model = model,
        tok_path = None, model_path = None,
        k = len(sent1_list) // 4,
        p = sum(label_list) / float(len(label_list))
    )'''

    # c = 1e-4
    c_sq = c * c
    c_sum = 0

    cl = 0
    # cl = 1e-7
    
    phi = 1
    phi_tensor = torch.Tensor([[1, 1]]).cuda()

    while True:
        w_list = torch.clamp(w_list, min = 0., max = 1.)
        # print(w_list)

        _, loss_list, _ = gnc(
            model,
            tok,
            sent1_list, sent2_list, label_list, dtype_list,
            dformat, 1, batch_size, num_labels,
            dataset_name, model_tag, data_split, task,
            save = (mu / mu_step < 1 or num_episode == max_episode),
            parallel = True,
            verbose = verbose,
            from_mnli = from_mnli, prompt_mode = prompt_mode,
            eval = eval, exp_id = exp_id, early_stop = False,
            eval_mode = eval_mode, w_list = w_list, phi = phi_tensor
        )

        c = loss_list.mean() - 2.4 * loss_list.std() / math.sqrt(len(sent1_list))
        c *= .1

        cl = loss_list.mean() - 2.4 * loss_list.std() / math.sqrt(len(sent1_list))
        cl *= .01

        loss_sq = (loss_list - cl) ** 2

        c_sq = c * c
        
        if robust_loss_func == 'gm':
            if mu == 10000.:
                mu = 2 * loss_sq.max().item() / c_sq
            else:
                mu /= mu_step
            
            if mu < 1 or num_episode == max_episode:
                break
            
            w_list_new = (mu * c_sq / (loss_sq + mu * c_sq))
            w_list_new *= w_list_new

            w_list = w_list_new

            if num_episode == 0:
                w_stat = json.load(open('log/qnli/w_stat.json'))
                w_stat.append(w_list.tolist())
                json.dump(w_stat, open(
                    'log/qnli/w_stat.json', 'w'
                ))
        
        elif robust_loss_func == 'tls':
            low_bar = mu / (1 + mu) * c_sq
            high_bar = (1 + mu) / mu * c_sq

            w_zeros = torch.zeros_like(w_list)
            w_ones = torch.ones_like(w_list)
            w_linear = c / loss_list * math.sqrt(
                mu * (mu + 1)
            ) - mu

            zero_mask = (loss_sq > high_bar).float()
            one_mask = (loss_sq < low_bar).float()
            linear_mask = 1 - (zero_mask + one_mask)

            if linear_mask.sum() == 0 or num_episode == max_episode:
                break

            w_list = zero_mask * w_zeros + one_mask * w_ones + \
                        w_linear * linear_mask
            
            if mu == 1e-5:
                mu = c_sq / (2 * loss_sq.max().item() - c_sq)
            else:
                mu *= mu_step
            
            new_mu_rsq = (w_list * loss_sq).sum()
        
        else:
            print('Robust function not supported')
            abort()

        num_episode += 1
    
        '''
        label_tensor = torch.Tensor(label_list).cuda()
        zero_tensor = 1 - label_tensor
        w0 = (zero_tensor * w_list).sum() / zero_tensor.sum()
        w1 = (label_tensor * w_list).sum() / label_tensor.sum()

        phi_tensor = torch.Tensor([[w0.item(), w1.item()]]).cuda()

        phi_tensor = torch.clamp(
            phi_tensor,
            min = phi_tensor.max() * 0.97,
            max = phi_tensor.max()
        )
        phi = (phi_tensor[0][0] / phi_tensor[0][1]).item()

        true_logits, false_logits = cls_evaluate(
            tok, model, sent1_list, 0,
            model_type='sc', num_prompt = None, mnli = False,
            temperature = 1, softmax = False, proto_emb = None,
            output_hidden_states = False, phi = 1
        )
        label_list = (
            false_logits > true_logits * phi
        ).long().tolist()
        # '''

    print(f'NE = {num_episode}')
    print(w_list)
    # print(phi_tensor.tolist()[0])
    # abort()

    # json.dump(phi_tensor.tolist()[0], open(
    #     f'log/{dataset_name}/cal_w_{exp_id}.json', 'w'
    # ))


def protost(
        model,
        tok,
        sent1_list,
        sent2_list,
        label_list,
        dtype_list,
        dformat,
        num_epochs = 1,
        batch_size = 32,
        num_labels = 2,
        dataset_name = 'sst2',
        model_tag = 'large',
        data_split = 'train',
        task = 'cls',
        save = True,
        parallel = True,
        verbose = True,
        from_mnli = True,
        prompt_mode = None,
        eval = False,
        exp_id = None,
        eval_mode = 'base',
        early_stop = False,
    ):

    def dev_condition(acc, acc_val, loss, loss_val):
        if acc > acc_val:
            return True
        elif acc == acc_val and loss < loss_val - 0.01:
            return True
        else:
            return False

    t_total = int(len(sent1_list) / batch_size * num_epochs)
    if exp_id is not None:
        data_split = f'{data_split}_{exp_id}'

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-6, eps=1e-6, # weight_decay=1e-7
    )

    if eval:
        model.eval()
    else:
        model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    model = model.cuda()

    if parallel and not hasattr(model, 'module'):
        model = nn.DataParallel(model)

    step_id = 0
    acc_val = 0
    loss_val = 100

    for epoch in range(num_epochs):
        model.train()

        proto_emb, _ = get_proto_emb(
            tok, model,
            sent1_list, sent2_list, label_list, batch_size
        )

        for j in range(0, len(sent1_list), batch_size):
            sent1_batch = sent1_list[j: j + batch_size]
            
            if sent2_list is None:
                sent2_batch = None
            else:
                sent2_batch = sent2_list[j: j + batch_size]
            label_batch = label_list[j: j + batch_size]
            
            if dtype_list is None:
                dtype_batch = None
            else:
                dtype_batch = dtype_list[j: j + batch_size]
            
            data_size = len(sent1_batch) // 2

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
            
            dtype_tensor = None
            if dtype_batch:
                dtype_tensor = torch.Tensor(dtype_batch).cuda()

            input_ids = input_enc['input_ids'].cuda()
            attn_mask = input_enc['attention_mask'].cuda()

            result = model(
                input_ids = input_ids,
                attention_mask = attn_mask,
                output_hidden_states = True
            )

            if hasattr(model.module, 'pooler'):
                hidden_states = model.module.pooler(
                    result.hidden_states[-1]
                )
            else:
                hidden_states = model.module.bert.pooler(
                    result.hidden_states[-1]
                )
            
            proc_logits = -torch.cdist(hidden_states, proto_emb)
            # print(proc_logits.size())
            # abort()
            # pc_dist = torch.cdist(
            #     hidden_states, hidden_states
            # ).mean() * 1e-2

            pc_dist = torch.norm(
                hidden_states[:data_size] - hidden_states[data_size:],
                dim = 1
            ).mean() * 5e-2
            
            loss = loss_fct(proc_logits, label_tensor).mean() # - pc_dist
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            model.zero_grad()

            if step_id % 100 == 0 and verbose:
                if dtype_tensor is not None:
                    print(dtype_tensor)
                print(f'Trained {step_id} / {t_total} steps, loss = {loss}\n')
            step_id += 1
        
        if early_stop:
            acc_0, acc_1, acc, loss_0, loss_1, loss = evaluate_func(
                dataset_name, 'mt', eval_mode, 'sc', 2, 50, tok, model.module,
                dev_split_id = exp_id, return_loss = True
            )
            
            acc_best = acc_0
            loss_best = loss_0
            dev_satis = dev_condition(acc_best, acc_val, loss_best, loss_val)
            
        if save and early_stop and (dev_satis or epoch < 6):
            acc_val = acc_best
            loss_val = loss_best
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(
                f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
            )
            
        sent1_list, sent2_list, label_list, _ = shuffle_data(
            sent1_list, sent2_list, label_list, label_list
        )

        if verbose:
            print('Epoch {} finished'.format(epoch))
            print('-' * 89)

    if save and not early_stop:
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(
            f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
        )
    
    proto_emb, _ = get_proto_emb(
        tok, model,
        sent1_list, sent2_list, label_list, batch_size
    )

    torch.save(
        proto_emb.cpu(),
        f'model_ft_file/proto_emb_{dataset_name}_{data_split}.pt'
    )

    return model.module if hasattr(model, 'module') else model


def protost_glue_func(
        dataset_name, model_tag, data_split, task,
        data = None, no_train = False, verbose = True,
        from_mnli = True, num_epochs = 1, prompt_mode = None,
        eval = False, exp_id = None, model_type_str = 'deberta',
        eval_mode = 'base', model_config_pt = None
    ):

    if dataset_name == 'sst2':
        batch_size = 32
    elif dataset_name == 'mnli':
        batch_size = 12
    elif dataset_name == 'rte':
        batch_size = 16
    elif dataset_name == 'qqp':
        batch_size = 16
    elif dataset_name == 'qnli':
        batch_size = 8
    else:
        batch_size = 32

    try:
        sent1_list, sent2_list, label_list, dtype_list, dformat = load_train_data(
            dataset_name, data_split, no_train=no_train
        )
        
        sent1_list, sent2_list = proc_input(
            dataset_name, sent1_list, sent2_list
        )
    except:
        dformat = {'label_dict': {0: 0, 1: 1}}

    if data is not None:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']
        label_list = data['label_list']

    num_labels = len(dformat['label_dict'])
    
    if verbose:
        print(f'\nBatch_size = {batch_size}')
        print(f'\nUsing model deberta-{model_tag}-sc-{num_labels}.pt\n')

    tok = AutoTokenizer.from_pretrained(
        # 'model_file/bert-large-tok.pt'
        f'model_file/{model_type_str}-{model_tag}-tok.pt'
    )

    if task == 'cls':
        # model_config = f'model_ft_file/cls_mnli_{model_tag}_train.pt'
        # model_config = 'model_file/deberta-large-sc-3.pt'
        if from_mnli:
            # model_config = f'model_ft_file/cls_mnli_{model_tag}_train.pt'
            # model_config = 'model_ft_file/mnli_model_sc_3e-06_binary_p1.pt'
            # model_config = 'model_ft_file/mnli_model_sc_5e-06_single_p0.pt'
            model_config = 'model_ft_file/mnli_model_sc_1e-05_binary_meta_ep499_maml-10-200.pt'
        else:
            model_config = 'model_file/deberta-large-sc-2.pt'
    else:
        model_config = f'model_ft_file/dis_{dataset_name}_{model_tag}_syn_data_relabel.pt'

    if model_config_pt is not None:
        model_config = model_config_pt
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config
    )

    if dataset_name == 'cola':
        dtype_list = label_list

    model = protost(
        model, tok,
        sent1_list, sent2_list, label_list, dtype_list,
        dformat, num_epochs, batch_size, num_labels,
        dataset_name, model_tag, data_split, task,
        save = True, parallel = True, verbose = verbose,
        from_mnli = from_mnli, prompt_mode = prompt_mode,
        eval = eval, exp_id = exp_id, early_stop = False,
        eval_mode = eval_mode
    )


def train(
        model,
        tok,
        sent1_list,
        sent2_list,
        label_list,
        dtype_list,
        dformat,
        num_epochs = 1,
        batch_size = 32,
        num_labels = 2,
        dataset_name = 'sst2',
        model_tag = 'large',
        data_split = 'train',
        task = 'cls',
        save = True,
        parallel = True
    ):

    t_total = int(len(sent1_list) / batch_size * num_epochs)
    warmup_steps = int(t_total / 20)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, eps=1e-6
    )

    # lr_sche = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    # )

    model.train()

    if parallel:
        model = nn.DataParallel(model)

    step_id = 0

    for epoch in range(num_epochs):

        for j in range(0, len(sent1_list), batch_size):
            sent1_batch = [
                f'sentence 1: {x}' for x in sent1_list[j: j + batch_size]
            ]
            if sent2_list is None:
                sent2_batch = None
            else:
                sent2_batch = [
                    f'sentence 2: {x}' for x in sent2_list[j: j + batch_size]
                ]

            label_batch = label_list[j: j + batch_size]
            if dtype_list is None:
                dtype_batch = None
            else:
                dtype_batch = dtype_list[j: j + batch_size]
            
            if task == 'dis':
                sent2_batch = None
            '''
            if task == 'dis':
                # sent2_batch = None
                label_txt_batch = [
                    dformat['label_explain'][str(x)] for x in label_batch
                ]
                if sent2_batch is None:
                    sent2_batch = label_txt_batch
                else:
                    sent2_batch = [
                        f'label: {x}. context: {y}' for x, y in zip(label_txt_batch, sent2_batch)
                    ]
            # '''

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
            dtype_tensor = None
            if dtype_batch:
                dtype_tensor = torch.Tensor(dtype_batch).cuda()

            input_ids = input_enc['input_ids'].cuda()
            attn_mask = input_enc['attention_mask'].cuda()

            if task == 'cls':
                result = model(
                    input_ids = input_ids,
                    attention_mask = attn_mask,
                    labels = label_tensor
                )

            elif task == 'dis':
                result = model(
                    input_ids = input_ids,
                    attention_mask = attn_mask,
                    labels = dtype_tensor
                )

            else:
                print(f'Task {task} not supported')
                abort()

            loss = result.loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # lr_sche.step()
            model.zero_grad()

            if step_id % 100 == 0:
                if dtype_tensor is not None:
                    print(dtype_tensor)
                print(f'Trained {step_id} / {t_total} steps, loss = {loss}\n')
            step_id += 1

        print('Epoch {} finished'.format(epoch))
        print('-' * 89)

    if save:
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(
            f'model_ft_file/{task}_{dataset_name}_{model_tag}_{data_split}.pt'
        )
        print('Checkpoint saved')

    return model.module if hasattr(model, 'module') else model


def train_glue_func(dataset_name, model_tag, data_split, task):
    num_epochs = 1

    if dataset_name == 'sst2':
        batch_size = 32
    elif dataset_name == 'mnli':
        batch_size = 12
    elif dataset_name == 'rte':
        batch_size = 16
    elif dataset_name == 'qqp':
        batch_size = 16
    elif dataset_name == 'qnli':
        batch_size = 8
    else:
        batch_size = 32

    print(f'\nBatch_size = {batch_size}')

    sent1_list, sent2_list, label_list, dtype_list, dformat = load_train_data(
        dataset_name, data_split
    )

    num_labels = len(dformat['label_dict'])
    print(f'\nUsing model deberta-{model_tag}-sc-{num_labels}.pt\n')

    tok = DebertaTokenizerFast.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
    )

    if task == 'cls':
        model_config = f'model_file/deberta-{model_tag}-sc-{num_labels}.pt'
        # model_config = f'model_ft_file/cls_{dataset_name}_{model_tag}_syn_data_relabel.pt'
    else:
        # model_config = f'model_ft_file/dis_{dataset_name}_{model_tag}_syn_data_relabel.pt'
        model_config = f'model_file/deberta-{model_tag}-sc-2.pt'

    model = DebertaForSequenceClassification.from_pretrained(
        model_config
    ).cuda()

    model = train(
        model, tok,
        sent1_list, sent2_list, label_list, dtype_list,
        dformat, num_epochs, batch_size, num_labels,
        dataset_name, model_tag, data_split, task,
        save = True, parallel = True
    )


if __name__ == '__main__':
    num_epochs = 1

    dataset_name = sys.argv[1]
    model_tag = sys.argv[2]
    data_split = sys.argv[3]
    task = sys.argv[4]

    # train_glue_func(dataset_name, model_tag, data_split, task)
    adapt_glue_func(dataset_name, model_tag, data_split, task)
