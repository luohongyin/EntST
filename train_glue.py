from distutils.log import log
import sys
import time
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

from eval_prompt import (
    evaluate_func, get_proto_emb, get_unconf_nodes, cls_evaluate
)
from prompt_emb_layer import (
    TuringAdaptorSCModel,
    encode_inputs
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


def load_train_data(domain, data_split, exp_id=None, no_train=False):
    if exp_id is not None:
        all_data = json.load(open(
            f'data/glue_data/{domain_dict[domain]}/{data_split}_proc_{exp_id}.json'
        ))
    else:
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
        model_type_str = None,
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
            else:
                proc_logits = result.logits
            
            loss = loss_fct(proc_logits, label_tensor).mean()
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
            f'model_ft_file/{task}_{dataset_name}_{model_type_str}_{data_split}.pt'
        )

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
    
    if from_mnli:
        model_path = f'luohy/ESP-{model_type_str}-large'
        if model_type_str == 'roberta':
            tokenizer_path = 'roberta-large'
        else:
            tokenizer_path = 'microsoft/deberta-large'
    else:
        if model_type_str == 'roberta':
            tokenizer_path = 'roberta-large'
        else:
            tokenizer_path = 'microsoft/deberta-large'
        model_path = tokenizer_path

    if model_config_pt is not None:
        model_path = model_config_pt
    
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
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
        eval_mode = eval_mode, model_type_str = model_type_str
    )


def train(
        model,
        tok,
        model_type_str,
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
                sys.exit()

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
            f'model_ft_file/{task}_{dataset_name}_{model_tag}_{model_type_str}.pt'
        )
        print('Checkpoint saved')

    return model.module if hasattr(model, 'module') else model


def train_glue_func(
        dataset_name,
        model_type_str,
        data_split,
        task,
        exp_id = None
    ):
    num_epochs = 6

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
        dataset_name, data_split, exp_id = exp_id
    )

    model_tag = 'large'

    num_labels = len(dformat['label_dict'])
    print(f'\nUsing model {model_type_str}-{model_tag}-sc-{num_labels}.pt\n')

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_tag}-tok.pt'
    )

    if task == 'cls':
        model_config = f'model_file/{model_type_str}-{model_tag}-sc-{num_labels}.pt'
        # model_config = f'model_ft_file/cls_{dataset_name}_{model_tag}_syn_data_relabel.pt'
    else:
        # model_config = f'model_ft_file/dis_{dataset_name}_{model_tag}_syn_data_relabel.pt'
        model_config = f'model_file/{model_type_str}-{model_tag}-sc-2.pt'

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config
    ).cuda()

    model = train(
        model, tok, model_type_str,
        sent1_list, sent2_list, label_list, dtype_list,
        dformat, num_epochs, batch_size, num_labels,
        dataset_name, model_tag, data_split, task,
        save = True, parallel = True
    )





if __name__ == '__main__':
    num_epochs = 1

    dataset_name = sys.argv[1]
    model_type_str = sys.argv[2]
    data_split = sys.argv[3]
    task = sys.argv[4]
    exp_id = sys.argv[5]

    train_glue_func(
        dataset_name, model_type_str, data_split, task, exp_id
    )
    # adapt_glue_func(dataset_name, model_tag, data_split, task)
