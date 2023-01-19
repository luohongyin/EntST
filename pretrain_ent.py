import sys
import json

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import *

from proc_data import coordinate, build_prompt_input
from prompt_emb_layer import PromptEmbedding, PromptDecoder

from eval_task_adp import (
    load_train_data, load_adv_eval, load_base_eval
)

dataset_name = sys.argv[1]
model_mode = sys.argv[2] # mt or pt
model_type = 'sc' # mlm or sc
data_split = sys.argv[3] # train
num_prompt = int(sys.argv[4]) # 20, 40, etc
cord_mode = sys.argv[5] # single, binary
exp_name = sys.argv[6]

def get_base_logits(tok, model, t_idx, f_idx, ok_idx,
                    num_prompt, prompt_str = None, model_type = None):
    if model_type == 'sc':
        return 0, 0, 0
    
    if prompt_str is None:
        input_txt = ['It is [MASK] that']
        offset = 0
    else:
        input_txt = [f'{prompt_str} It is [MASK] that']
        offset = num_prompt
    input_enc = tok(input_txt, return_tensors = 'pt')

    input_ids = input_enc['input_ids'].cuda()

    with torch.no_grad():
        result = model(input_ids)

    t_base = result.logits[0][offset + 3][t_idx].item()
    f_base = result.logits[0][offset + 3][f_idx].item()
    ok_base = result.logits[0][offset + 3][ok_idx].item()
    
    return t_base, f_base, ok_base


def mlm_train_loss(tok, model, loss_fn, input_list, label_list):
    
    input_enc = tok(
        input_list,
        max_length = 384,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()

    result = model(
        input_ids = input_ids,
        attention_mask = attn_mask
    )

    if model_mode == 'pt':
        offset = num_prompt
    else:
        offset = 0

    logits = result.logits[:, offset + 3, :]
    loss = loss_fn(logits, torch.LongTensor(label_list).cuda())

    return loss


def sc_train_loss(tok, model, loss_fn, input_list, label_list):
    
    input_enc = tok(
        input_list,
        max_length = 384,
        padding = 'longest',
        return_tensors = 'pt',
        truncation = True,
        return_attention_mask = True,
        verbose = False
    )

    input_ids = input_enc['input_ids'].cuda()
    attn_mask = input_enc['attention_mask'].cuda()
    label_tensor = label_list.cuda()

    result = model(
        input_ids = input_ids,
        attention_mask = attn_mask,
        labels = label_tensor
    )

    loss = result.loss.mean()

    return loss


def gen_prompt_tok(prompt_len):
    prompt_tokens = [f'<prompt_token_{i}>' for i in range(prompt_len)]
    return prompt_tokens


def add_prompt_layer(model, dataset_name, num_prompt, model_type_str):
    if model_type_str == 'bert':
        model.bert.embeddings.word_embeddings = PromptEmbedding(
            model.bert.embeddings.word_embeddings, num_prompt
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
        abort()
    return model


def get_batch_rvs_label(label_batch, rvs_map, t_idx, f_idx, ok_idx):
    new_labels = []
    for tag in rvs_map:
        if tag == 0:
            label_mapping = [t_idx, ok_idx, f_idx]
        else:
            label_mapping = [f_idx, ok_idx, t_idx]
        new_labels += [label_mapping[x] for x in label_batch]
    return new_labels


if __name__ == '__main__':

    model_type_str = 'bert'
    model_size_str = 'large'
    log_step = 100

    tok = AutoTokenizer.from_pretrained(
        f'model_file/{model_type_str}-{model_size_str}-tok.pt'
    )
    t_idx = tok.convert_tokens_to_ids('true')
    f_idx = tok.convert_tokens_to_ids('false')
    ok_idx = tok.convert_tokens_to_ids('ok')

    print(f't_idx: {t_idx}, f_idx: {f_idx}, ok_idx: {ok_idx}')

    if model_type == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(
            f'model_file/{model_type_str}-large-mlm.pt'
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            f'model_file/{model_type_str}-{model_size_str}-sc-3.pt'
        )

    loss_fn = nn.CrossEntropyLoss()
    if model_type_str == 'bert':
        lr = 5e-6
    else:
        lr = 3e-6
    num_epoch = 2
    batch_size = 16
    
    if model_mode == 'mt':
        prompt_str = None
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = lr,
            weight_decay = 1e-5,
            eps = 1e-6
        )

    elif model_mode == 'pt':
        prompt_tok_list = gen_prompt_tok(num_prompt)
        tok.add_tokens(prompt_tok_list)
        prompt_str = ' '.join(prompt_tok_list)

        model = add_prompt_layer(
            model, dataset_name, num_prompt, model_type_str
        )

        optimizer = Adafactor(
            [model.bert.embeddings.word_embeddings.prompt_emb],
            lr = 10,
            weight_decay = 1e-5,
            scale_parameter = False,
            relative_step = False
        )
    else:
        print(f'Model mode {model_mode} not supported.')
        abort()

    model.cuda()
    model = nn.DataParallel(model)

    t_base, f_base, ok_base = get_base_logits(
        tok, model, t_idx, f_idx, ok_idx,
        num_prompt, prompt_str = prompt_str, model_type = model_type
    )

    print(t_base, f_base, ok_base)
    # abort()

    model.train()
    
    sent1_list, sent2_list, label_list, dformat = load_train_data(
        dataset_name
    )

    sent1_dev_list, sent2_dev_list, label_dev_list, _ = load_train_data(
        dataset_name, exp_id = 0
    )

    if sent2_list is None:
        sent2_list = sent1_list

    num_case = len(sent1_list)
    step_id = 0
    best_val_loss = 10000
    val_count = 0

    if cord_mode == 'binary':
        batch_size = batch_size // 2

    for e in range(num_epoch):
        for i in range(0, num_case, batch_size):

            sent1_batch = sent1_list[i: i + batch_size]
            sent2_batch = sent2_list[i: i + batch_size]
            label_batch = torch.Tensor(label_list[i: i + batch_size]).long()


            cur_bs = len(sent1_batch)

            prompt_input_list, rvs_map = build_prompt_input(
                dataset_name, sent1_batch, sent2_batch,
                mlm = (model_type == 'mlm')
            )

            if cord_mode == 'single':
                prompt_input_list = prompt_input_list[:cur_bs]
                rvs_map = rvs_map[:1]

            if model_type == 'mlm':
                batch_label_list = get_batch_rvs_label(
                    label_batch, rvs_map, t_idx, f_idx, ok_idx
                )
            elif cord_mode == 'binary':
                batch_label_list = torch.cat(
                    [label_batch, 2 - label_batch], dim = 0
                )
            else:
                batch_label_list = label_batch

            if model_mode == 'pt':
                prompt_input_list = [
                    f'{prompt_str} {x}' for x in prompt_input_list
                ]

            if model_type == 'mlm':
                loss = mlm_train_loss(
                    tok, model, loss_fn, prompt_input_list, batch_label_list
                )
            else:
                loss = sc_train_loss(
                    tok, model, loss_fn, prompt_input_list, batch_label_list
                )

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if step_id % log_step == 0:
                '''prompt_dev_list, _ = build_prompt_input(
                    dataset_name, sent1_dev_list, sent2_dev_list,
                    mlm = (model_type == 'mlm')
                )
                dev_label_list = get_batch_rvs_label(
                    label_dev_list, rvs_map, t_idx, f_idx, ok_idx
                )
                t_base, f_base, ok_base = get_base_logits(
                    tok, model, t_idx, f_idx, ok_idx,
                    num_prompt, prompt_str = prompt_str, model_type = model_type
                )
                with torch.no_grad():
                    val_loss = mlm_train_loss(
                        tok, model, loss_fn, prompt_dev_list, dev_label_list
                    )
                if val_loss < best_val_loss:
                    model.module.save_pretrained(
                        f'model_ft_file/{dataset_name}_model_{model_type}_{lr}_{cord_mode}_{exp_name}-val.pt'
                    )
                    best_val_loss = val_loss
                    val_count = 0
                else:
                    val_count += 1
                    if val_count > 5:
                        print(step_id, val_count)
                        print('Early stop.')
                        sys.exit()
                
                print(t_base, f_base, ok_base)'''
                print(f'Step_id = {step_id}, loss = {loss.item()}\n')
            
            step_id += 1
        
        if model_mode == 'pt':
            torch.save(
                model.module.bert.embeddings.word_embeddings.prompt_emb.data,
                f'model_ft_file/{dataset_name}_prompt_emb_{cord_mode}.pt'
            )
        else:
            model.module.save_pretrained(
                f'model_ft_file/{dataset_name}_model_{model_type}_{lr}_{cord_mode}_{exp_name}.pt'
            )
            print(f'Epoch {e} finished')
            print('Checkpoint saved')
            print('-' * 89)
            print(' ')