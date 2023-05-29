import transformers
import datasets
from datasets import load_dataset
import random
import pandas as pd
from IPython.display import display, HTML
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import gc
from tqdm import tqdm
import torch.nn as nn
import argparse
import os
import numpy as np

"""Utils Functions"""
def get_unconf_nodes(
        hidden_states, pseudo_label_list, # pseudo_label_scores,
        k = 20, p_list = 0.5, divisor = 5
    ):
    '''
    Graph-based method for predicting uncertain pseudo labels.
    
    Inputs -
        hidden_states: Pytorch float tensor, (N, d) sized -
            hidden states of input data
        pseudo_label_list: List of (N) interger - 
            list of predicted pseudo labels
        k: number of neighbors to be considered, usually 5 to 20.
        p_list: prior label distribution, (N) probabilities
    
    Outputs -
        conf_node: a binary list of node confidents [0, 1, 0, ...]
            1 stands for confident, 0 stands for uncertain
            same index as hidden_states and pseudo_label_list
    '''
    with torch.no_grad():
        data_size = len(pseudo_label_list)
        pseudo_label_scores = torch.Tensor([
            p_list[x] for x in pseudo_label_list
        ]).to(device)
        pseudo_label_list = torch.Tensor(pseudo_label_list).to(device)
        # pseudo_label_scores = torch.Tensor(pseudo_label_scores).cuda()

        # pseudo_label_scores = pseudo_label_list * p +\
        #      (1 - pseudo_label_list) * (1 - p)

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
        # abort()
        confidence = (J - mu) / torch.sqrt(sigma)

        conf_sq = confidence.squeeze(1)
        thres_topk, _ = torch.topk(conf_sq, data_size//divisor)

        # conf_node = (confidence < -1).float()
        conf_node = (confidence < thres_topk[-1]).long().squeeze(1)
        unconf_node = 1 - conf_node
        pseudo_label_scores=[]
        pseudo_label_list=[]
    torch.cuda.empty_cache()
    return conf_node.tolist()

"""get dataset for a given dataset name"""
def get_dataset(dataset_name, new_cache_dir="./cache2", num=2400, dataset_type="train"):
    if dataset_name == "amazon_reviews_multi":
        dataset= load_dataset("amazon_reviews_multi","en", cache_dir=new_cache_dir)
    elif dataset_name == "copa":
        dataset= load_dataset("super_glue", 'copa', cache_dir=new_cache_dir)
    else:
        dataset= load_dataset(dataset_name, cache_dir=new_cache_dir)
    df=pd.DataFrame(dataset[dataset_type])
    df=df.sample(num)
    df_train=pd.DataFrame()
    if dataset_name == 'ag_news':
        df_train['inference_pair_1']="This is a world news is entailed by "+df["text"]
        df_train['inference_pair_2']="This is a sports news is entailed by "+df["text"]
        df_train['inference_pair_3']="This is a business news is entailed by "+df["text"]
        df_train['inference_pair_4']="This is a science news is entailed by "+df["text"]
        df_train['label']=df['label']
    elif dataset_name == 'emotion':
        df_train['inference_pair_0']="i feel unhappy is entailed by "+df["text"]
        df_train['inference_pair_1']="i feel happy is entailed by "+df["text"]
        df_train['inference_pair_2']="i feel love is entailed by "+df["text"]
        df_train['inference_pair_3']="i feel angry is entailed by "+df["text"]
        df_train['inference_pair_4']="i feel afraid is entailed by "+df["text"]
        df_train['inference_pair_5']="i feel shocked is entailed by "+df["text"]
        df_train['label']=df['label']
    elif dataset_name == 'copa':
        df_train['inference_pair_1']=df['choice1']+" is entailed by "+df['premise']
        df_train['inference_pair_2']=df['choice2']+" is entailed by "+df['premise']
        df_train['label']=df['label']
    elif dataset_name == 'amazon_reviews_multi':
        df_train['inference_pair_1']="This is terrible is entailed by "+df["review_title"]+" "+df["review_body"]
        df_train['inference_pair_2']="This is bad is entailed by "+df["review_title"]+" "+df["review_body"]
        df_train['inference_pair_3']="This is alright is entailed by "+df["review_title"]+" "+df["review_body"]
        df_train['inference_pair_4']="This is good is entailed by "+df["review_title"]+" "+df["review_body"]
        df_train['inference_pair_5']="This is awesome is entailed by "+df["review_title"]+" "+df["review_body"]
        df_train['label']=df['stars'] 
    else:
        raise NotImplementedError
    return df_train

def get_input_and_attention(sentences):
    sentences=list(sentences)
    encoded_dict = tokenizer(
              sentences,           
              padding="max_length", 
              max_length=512,
              truncation=True,
              return_tensors="pt",
    )

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def get_dataloader(dataset_name, batch_size = 4):
    if dataset_name == "ag_news":
        sentences_train_1 = df_train.inference_pair_1.values
        sentences_train_2 = df_train.inference_pair_2.values
        sentences_train_3 = df_train.inference_pair_3.values
        sentences_train_4 = df_train.inference_pair_4.values
        labels_train = df_train.label.values
        labels_train= torch.tensor(labels_train)
        input_ids_train_1, attention_masks_train_1 = get_input_and_attention(sentences_train_1)
        input_ids_train_1=input_ids_train_1.clip(0, 28995)
        input_ids_train_2, attention_masks_train_2 = get_input_and_attention(sentences_train_2)
        input_ids_train_2=input_ids_train_2.clip(0, 28995)
        input_ids_train_3, attention_masks_train_3 = get_input_and_attention(sentences_train_3)
        input_ids_train_3=input_ids_train_3.clip(0, 28995)
        input_ids_train_4, attention_masks_train_4 = get_input_and_attention(sentences_train_4)
        input_ids_train_4=input_ids_train_4.clip(0, 28995)
        dataset_train = TensorDataset(input_ids_train_1, attention_masks_train_1,input_ids_train_2, attention_masks_train_2,input_ids_train_3, attention_masks_train_3,input_ids_train_4, attention_masks_train_4,labels_train)
        input_ids = torch.cat((input_ids_train_1, input_ids_train_2, input_ids_train_3, input_ids_train_4), 0)
        input_masks = torch.cat((attention_masks_train_1, attention_masks_train_2, attention_masks_train_3, attention_masks_train_4),0)
    elif dataset_name == 'copa':
        pass
    elif dataset_name == 'emotion':
        pass
    elif dataset_name == 'amazon_reviews_multi':
        pass
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(
                dataset_train, 
                batch_size = batch_size 
            )
    return train_dataloader, input_ids, input_masks

def get_base_value(dataset_name, PATH, model_checkpoint, new_cache_dir='./cache2', device='cuda', device_ids=[0, 1, 2, 3]):
    with torch.no_grad(): 
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        if device_ids != None:
            model = nn.DataParallel(model, device_ids = device_ids)
        model.eval() 
        if dataset_name == 'ag_news':
            inference_pair_1=["This is a world news is entailed by ."]
            inference_pair_2=["This is a sports news is entailed by ."]
            inference_pair_3=["This is a business news is entailed by ."]
            inference_pair_4=["This is a science news is entailed by ."]
            input_1, attention_1 = get_input_and_attention(inference_pair_1)
            input_1=input_1.clip(0, 28995).reshape(1, -1)
            input_2, attention_2 = get_input_and_attention(inference_pair_2)
            input_2=input_2.clip(0, 28995).reshape(1, -1)
            input_3, attention_3 = get_input_and_attention(inference_pair_3)
            input_3=input_3.clip(0, 28995).reshape(1, -1)
            input_4, attention_4 = get_input_and_attention(inference_pair_4)
            input_4=input_4.clip(0, 28995).reshape(1, -1)
            output_1 = model(input_1.to(device), 
                                     attention_mask=attention_1.to(device))
            output_2 = model(input_2.to(device), 
                                     attention_mask=attention_2.to(device))
            output_3 = model(input_3.to(device), 
                                     attention_mask=attention_3.to(device))
            output_4 = model(input_4.to(device), 
                                     attention_mask=attention_4.to(device))
            torch.cuda.empty_cache()
            return output_1.logits, output_2.logits, output_3.logits, output_4.logits
        elif dataset_name == 'copa':
            pass
        elif dataset_name == 'emotion':
            pass
        elif dataset_name == 'amazon_reviews_multi':
            pass
        else:
            raise NotImplementedError

def train(dataloader, PATH, num_epochs=5, device_ids=[0,1,2,3], device='cuda', new_cache_dir='./cache2'):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
    model.load_state_dict(torch.load(PATH))
    model = nn.DataParallel(model, device_ids = device_ids)
    model.train()
    # custom training parameters
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-6)
    for i in range(num_epochs):
        for batch in tqdm(dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device).long()
            output = model(b_input_ids, 
                                         attention_mask=b_input_mask, labels = b_labels)
            output.loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
    return model
    