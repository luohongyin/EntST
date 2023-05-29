import sys
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    utils,
    DebertaTokenizerFast,
    DebertaForSequenceClassification
)
from proc_data import coordinate

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

def load_train_data(domain, exp_id=None):
    if exp_id is not None:
        all_data = json.load(open(
            f'data/glue_data/{domain_dict[domain]}/train_proc_{exp_id}.json'
        ))
    else:
        all_data = json.load(open(
            f'data/glue_data/{domain_dict[domain]}/train_proc.json'
        ))

    sent1_list = all_data['sent1_list']
    sent2_list = all_data['sent2_list']
    label_list = all_data['label_list']
    dformat = all_data['dformat']
    
    return sent1_list, sent2_list, label_list, dformat

def load_adv_eval(domain, dformat):
    dev_data = json.load(open('data/adv_dev/dev.json'))[domain]
    sent1_title = dformat['sentence1_eval']
    sent2_title = dformat['sentence2_eval']
    sent1_list = [x[sent1_title] for x in dev_data]

    if sent2_title:
        sent2_list = [x[sent2_title] for x in dev_data]
    else:
        sent2_list = None

    label_list = [x['label'] for x in dev_data]
    return sent1_list, sent2_list, label_list


def save_data(data, domain, split, split_id=None):
    dataset_name = domain_dict[domain]
    if split_id is None:
        json.dump(data, open(
            f'data/glue_data/{dataset_name}/{split}_proc.json', 'w'
        ))
    else:
        json.dump(data, open(
            f'data/glue_data/{dataset_name}/{split}_proc_{split_id}.json', 'w'
        ))


def load_base_eval(domain, split='dev', dformat=None, dev_split_id=None):
    if dev_split_id is None:
        all_data = json.load(open(
            f'data/glue_data/{domain_dict[domain]}/{split}_proc.json'
        ))
    else:
        all_data = json.load(open(
            f'data/glue_data/{domain_dict[domain]}/{split}_proc_{dev_split_id}.json'
        ))

    if all_data is None:
        return [], [], [], None

    sent1_list = all_data['sent1_list']
    sent2_list = all_data['sent2_list']
    label_list = all_data['label_list']
    dformat = all_data['dformat']
    
    return sent1_list, sent2_list, label_list, dformat


def proc_input(dataset_name, sent1_list, sent2_list):
    dataset_coll = set([
        'sst2', 'qqp', 'mnli', 'qnli', 'rte'
    ])

    if dataset_name == 'sst2':
        sent1_list = [
            f'sentence 1: {x}' for x in sent1_list
        ]
        sent2_list = ['sentence 2: it is a bad movie'] * len(sent1_list)
    
    elif dataset_name == 'qqp':
        sent1_list = [
            f'sentence 1: {x}' for x in sent1_list
        ]
        sent2_list = [
            f'sentence 2: {x}' for x in sent2_list
        ]
    
    elif dataset_name == 'qnli':
        sent1_list = [
            f'sentence 1: {x}' for x in sent1_list
        ]
        sent2_list = [
            f'sentence 2: {x}' for x in sent2_list
        ]
    
    elif dataset_name == 'cola':
        sent1_list = [
            f'sentence 1: {x}' for x in sent1_list
        ]
        sent2_list = [
            'sentence 2: The grammar is incorrect.' for x in sent1_list
        ]
    
    elif dataset_name in dataset_coll:
        sent1_list = [
            f'sentence 1: {x}' for x in sent1_list
        ]
        sent2_list = [
            f'sentence 2: {x}' for x in sent2_list
        ]
    
    else:
        print(f'Dataset {dataset_name} not supported')
        sys.exit()

    return sent1_list, sent2_list


def proc_output(dataset_name, output_logits, no_neu = True):

    ent_logits = output_logits[:, :1]
    neu_logits = output_logits[:, 1: 2]
    con_logits = output_logits[:, 2:]

    flip_flag = {
        'sst2': 0, 'qqp': 0, 'qnli': 0, 'rte': 0, 'cola': 0
    }

    if dataset_name not in flip_flag:
        print('Dataset name not in flip flag')
        abort()

    if dataset_name == 'mnli':
        return output_logits

    if flip_flag[dataset_name]:
        proc_logits = torch.cat(
            [con_logits, ent_logits, neu_logits], dim = 1
        )

    else:
        proc_logits = torch.cat(
            [ent_logits, con_logits, neu_logits], dim = 1
        )
    
    if no_neu:
        proc_logits = proc_logits[:, :2]
    
    return proc_logits#.contiguous()


def evaluate(
        model, tok, sent1_list, sent2_list, label_list, dataset_name,
        eval_mode, num_epochs=1, batch_size=32, save=True, parallel=True,
        return_loss = False, const = False, from_mnli = True
    ):

    # if return_loss:
    #     model.train()
    # else:
    model.eval()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    if parallel:
        model = nn.DataParallel(model)

    num_case = len(sent1_list)
    num_correct = 0

    logits_list = []
    loss_list = []
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
            )
        
        if from_mnli:
            proc_logits = proc_output(dataset_name, result.logits)
        else:
            proc_logits = result.logits
            
        _, pred_id = proc_logits.max(dim=1)
        crr = (pred_id == label_tensor).float().sum()

        num_correct += crr
        loss_batch = loss_fct(proc_logits, label_tensor)
        # print(loss_batch)
        # abort()
        loss_list.append(loss_batch)

        # print(loss)
        # abort()
    
    # print(proc_logits)
    # print(pred_id)
    acc = num_correct / num_case
    # print(f'\nAccuracy = {acc}\n')

    loss = torch.cat(loss_list, dim=0).mean().item()
    # print(f'Loss = {loss}\n')
    # abort()

    if not save:
        return loss if return_loss else [loss, acc]

    try:
        result_list = json.load(open(
            f'log/{dataset_name}_{eval_mode}_results.json'
        ))
    except:
        result_list = []
    
    result_list.append(acc.item())
    
    json.dump(result_list, open(
        f'log/{dataset_name}_{eval_mode}_results.json', 'w'
    ))
    return loss if return_loss else acc


def eval_adp_func(
        dataset_name, eval_mode, model_tag, data_split,
        data=None, save=True, return_loss=False,
        const=False, from_mnli=True
    ):

    num_epochs = 2
    batch_size = 32

    _, _, _, dformat = load_train_data(dataset_name)
    
    if data is None:
        if eval_mode == 'adv':
            try:
                sent1_list, sent2_list, label_list = load_adv_eval(dataset_name, dformat)
            except:
                print('No adv dev set')
                return
        
        elif eval_mode == 'base':
            sent1_list, sent2_list, label_list, _ = load_base_eval(
                dataset_name, split='dev', dformat=dformat
            )
        
        elif eval_mode == 'train':
            sent1_list, sent2_list, label_list, _ = load_base_eval(
                dataset_name, split='train', dformat=dformat
            )
        
        elif eval_mode == 'relabel':
            sent1_list, sent2_list, label_list, _ = load_base_eval(
                dataset_name, split='syn_data_relabel', dformat=dformat
            )
        
        elif eval_mode == 'mix':
            sent1_list, sent2_list, label_list, _ = load_base_eval(
                dataset_name, split='train', dformat=dformat
            )
            
            try:
                sent1_syn, sent2_syn, label_syn, _ = load_base_eval(
                    dataset_name, split='syn_eval', dformat=dformat
                )
                sent1_list += sent1_syn
                label_list += label_syn
                
                if sent2_list is not None:
                    sent2_list += sent2_syn
            
            except:
                pass
            
        else:
            print('Mode not supported')
            abort()

        sent1_list, sent2_list = proc_input(dataset_name, sent1_list, sent2_list)
    else:
        sent1_list = data['sent1_list']
        sent2_list = data['sent2_list']
        label_list = data['label_list']
   
    try:
        num_labels = len(dformat['label_dict'])
    except:
        num_labels = 2
    # print(f'\nUsing model deberta-{model_tag}-sc-{num_labels}.pt\n')

    if sent2_list is None:
        sent2_list = ['The grammar is incorrect'] * len(sent1_list)

    if dataset_name == 'sst2':
        print(sent1_list[0])
        print(sent2_list[0])
        abort()

    tok = DebertaTokenizerFast.from_pretrained(
        f'model_file/deberta-{model_tag}-tok.pt'
    )
    
    model = DebertaForSequenceClassification.from_pretrained(
        f'model_ft_file/cls_mnli_{model_tag}_{data_split}.pt'
        # f'model_ft_file/cls_{dataset_name}_{model_tag}_{data_split}.pt'
    ).cuda()

    acc = evaluate(
        model, tok,
        sent1_list, sent2_list, label_list,
        dataset_name, eval_mode, num_epochs, batch_size,
        save = save, parallel = True,
        return_loss = return_loss, const = const, from_mnli = from_mnli
    )

    return acc


if __name__ == '__main__':
    num_epochs = 2
    batch_size = 32

    dataset_name = sys.argv[1]
    eval_mode = sys.argv[2]
    model_tag = sys.argv[3]
    data_split = sys.argv[4]

    acc = eval_adp_func(dataset_name, eval_mode, model_tag, data_split)
    print(acc)