import transformers
import datasets
from datasets import load_dataset
import datasets
import random
import pandas as pd
from IPython.display import display, HTML
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import gc
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn as nn
import argparse
import os
import numpy as np
from .utils import *
parser = argparse.ArgumentParser(
    prog='copa_finetune.py',
    description='Finetune DeBERTa/RoBERTa on COPA',
)

parser.add_argument('--index', type=int, required=True,
                     help='index of pth file')
parser.add_argument('--algo', type=str, required=True,
                    choices=['deberta', 'roberta'], help='model name')

args = parser.parse_args()

pth_index = args.index
model_name = args.algo

new_cache_dir="./cache"
model_checkpoint = {
    'deberta': 'microsoft/deberta-large',
    'roberta': 'roberta-large',
}[model_name]
PATH = {
    'deberta': './model_files/mnli_model_sc_3e-06_binary_p1.pt/pytorch_model.bin',
    'roberta': './model_files/mnli_model_sc_5e-06_binary_pr.pt/pytorch_model.bin',
}[model_name]
cls_layer = {
    'deberta': -6,
    'roberta': -2,
}[model_name]

df_train = get_dataset('copa')
labels=torch.tensor(df_train.label.values).reshape(1, -1).int()
labels=labels[0]
labels=2-labels*2
labels_count=torch.bincount(labels).float()
labels_sum=torch.sum(labels_count)
labels_count/=labels_sum
#becuase copa is special, we use distribution of 0.5, 0, 0.5. other multiclassification tasks don't follow this setting
labels_count[0]=0.5
labels_count[2]=0.5


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) 
sentences_train_1 = df_train.inference_pair_1.values
sentences_train_2 = df_train.inference_pair_2.values
labels_train = df_train.label.values


input_ids_train_1, attention_masks_train_1 = get_input_and_attention(sentences_train_1)
input_ids_train_2, attention_masks_train_2 = get_input_and_attention(sentences_train_2)
input_ids_train_1=input_ids_train_1.clip(0, 28995)
input_ids_train_2=input_ids_train_2.clip(0, 28995)
labels_train= torch.tensor(labels_train)
dataset_train = TensorDataset(input_ids_train_1, input_ids_train_2, attention_masks_train_1, attention_masks_train_2, labels_train)
batch_size = 32

train_dataloader = DataLoader(
            dataset_train, 
            batch_size = batch_size 
        )

torch.cuda.empty_cache()
device=torch.device('cuda')
input_ids = torch.cat((input_ids_train_1, input_ids_train_2), 0)
input_masks = torch.cat((attention_masks_train_1, attention_masks_train_2),0)
total_steps = len(train_dataloader)
progress_bar = tqdm(range(total_steps))
num=0
num0=0

def get_eval_label():
    with torch.no_grad():
        new_training_sample_1=[]
        new_training_sample_2=[]
        cls_embeddings_1=[]
        cls_embeddings_2=[]
        new_training_sample_list_1=[]
        new_training_sample_list_2=[]
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        device_ids=[0,1, 2, 3]
        model = nn.DataParallel(model, device_ids = device_ids)
        model.eval() 
        for batch in train_dataloader:   
            b_input_ids_1 = batch[0].to(device)
            b_input_ids_2 = batch[1].to(device)
            b_input_mask_1 = batch[2].to(device)
            b_input_mask_2 = batch[3].to(device)
            b_labels = batch[4].to(device)
            output_1 = model(b_input_ids_1, 
                                     attention_mask=b_input_mask_1, output_hidden_states=True)
            output_2 = model(b_input_ids_2, 
                                     attention_mask=b_input_mask_2, output_hidden_states=True)
            ans=torch.tensor(output_1.logits[:,2]<output_2.logits[:,2])*2
            new_training_sample_1.append((2-ans).tolist()) 
            new_training_sample_2.append(ans.tolist())
            cls_embeddings_1.append(output_1.hidden_states[cls_layer][:, 0, :].to('cpu'))
            cls_embeddings_2.append(output_2.hidden_states[cls_layer][:, 0, :].to('cpu'))
            del output_1, output_2
            del ans
            torch.cuda.empty_cache()
        for x in new_training_sample_1:
            for item in x:
                new_training_sample_list_1.append(item)
        for x in new_training_sample_2:
            for item in x:
                new_training_sample_list_2.append(item)
        new_training_sample_choice_1=torch.tensor(new_training_sample_list_1).reshape(1, -1)
        new_training_sample_choice_2=torch.tensor(new_training_sample_list_2).reshape(1, -1)
        sample_choice_full = torch.cat((new_training_sample_choice_1, new_training_sample_choice_2), 1).int().tolist()
        sample_choice_full = sample_choice_full[0]
        cls_embeddings_1 = torch.cat(cls_embeddings_1, dim=0)
        cls_embeddings_2 = torch.cat(cls_embeddings_2, dim=0)
        cls_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=0)
    return sample_choice_full, cls_embeddings

eval_labels, eval_cls_embeddings = get_eval_label()
eval_cls_embeddings = eval_cls_embeddings.to(device)
eval_labels = torch.tensor(eval_labels)
num_0 = torch.sum((eval_labels == 0)).float()
num_2 = torch.sum((eval_labels == 2)).float()
labels_count = torch.tensor([num_0, 0, num_2])
labels_sum=torch.sum(labels_count)
labels_count/=labels_sum

input_ids_list = []
attention_list = []
eval_labels_list = []
confidence_unconf_once = get_unconf_nodes(eval_cls_embeddings, eval_labels.long(), k=10, p_list=labels_count)
num_case = len(confidence_unconf_once)
confidence_unconf_once = torch.tensor(confidence_unconf_once)
for i in range(num_case):
    if confidence_unconf_once[i]==1:
        input_ids_list.append(input_ids[i, :].reshape(1, -1))
        attention_list.append(input_masks[i, :].reshape(1, -1))
        eval_labels_list.append(eval_labels[i])
input_ids_unconf_tensor = torch.cat(input_ids_list, dim=0)
attention_unconf_tensor = torch.cat(attention_list, dim=0)
ans3 = torch.tensor(eval_labels_list)


def get_label_and_confidence():
    with torch.no_grad():
        new_training_sample_1=[]
        new_training_sample_2=[]
        new_training_sample_list_1=[]
        new_training_sample_list_2=[]
        cls_embeddings_list_1 = []
        cls_embeddings_list_2 = []
        cls_embeddings_1 = []
        cls_embeddings_2 = []
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        device_ids=[0,1, 2, 3]
        model = nn.DataParallel(model, device_ids = device_ids)
        model.train() #was: model.eval()
        for batch in train_dataloader:
            b_input_ids_1 = batch[0].to(device)
            b_input_ids_2 = batch[1].to(device)
            b_input_mask_1 = batch[2].to(device)
            b_input_mask_2 = batch[3].to(device)
            b_labels = batch[4].to(device)
            output_1 = model(b_input_ids_1, 
                                     attention_mask=b_input_mask_1, output_hidden_states=True)
            output_2 = model(b_input_ids_2, 
                                     attention_mask=b_input_mask_2, output_hidden_states=True)
            ans=torch.tensor(output_1.logits[:,2]<output_2.logits[:,2])*2
            new_training_sample_1.append((2-ans).tolist()) 
            new_training_sample_2.append(ans.tolist())
            cls_embeddings_1.append(output_1.hidden_states[cls_layer][:, 0, :])
            cls_embeddings_2.append(output_2.hidden_states[cls_layer][:, 0, :])
            del output_1, output_2
            del b_input_ids_1, b_input_ids_2
            del b_input_mask_1, b_input_mask_2
            torch.cuda.empty_cache()
        for x in new_training_sample_1:
            for item in x:
                new_training_sample_list_1.append(item)
        for x in new_training_sample_2:
            for item in x:
                new_training_sample_list_2.append(item)
        cls_embeddings_list_1=torch.cat(tuple(cls_embeddings_1), 0)
        cls_embeddings_list_2=torch.cat(tuple(cls_embeddings_2), 0)
        new_training_sample_choice_1=torch.tensor(new_training_sample_list_1).reshape(1, -1)
        new_training_sample_choice_2=torch.tensor(new_training_sample_list_2).reshape(1, -1)
        sample_choice_full = torch.cat((new_training_sample_choice_1, new_training_sample_choice_2), 1).int().tolist()
        sample_choice_full = sample_choice_full[0]
        sample_choice_full = torch.tensor(sample_choice_full)
        hidden_states = torch.cat((cls_embeddings_list_1, cls_embeddings_list_2), 0)
    return hidden_states, sample_choice_full

def get_ans(numiter=5):
    with torch.no_grad():
        tot_confidence_list=[]
        tot_label_list=[]
        num=0
        label_unconf_once=[]
        confidence_unconf_once=[]
        num_iter = numiter
        for i in range(num_iter):
            num+=1
            hidden_states, sample_choice = get_label_and_confidence()
            tot_confidence_list.append(hidden_states)
            tot_label_list.append(sample_choice)
        
        tot_confidence_list = torch.cat(tot_confidence_list,dim=0)
        tot_label_list=torch.cat(tot_label_list, dim=0)
        
        #unconf+dropout
        tot_confidence_list=torch.tensor(tot_confidence_list)
        tot_label_list=torch.tensor(tot_label_list)
        tot_confidence_list = get_unconf_nodes(tot_confidence_list, tot_label_list, k=10, p_list=labels_count)
        tot_confidence_list = torch.tensor(tot_confidence_list)    
        tot_confidence_list *= 2
        tot_confidence_list -=1 
        tot_sum = tot_confidence_list*(tot_label_list+1)
        tot_mask = tot_sum >=0
        tot_sum = tot_sum*tot_mask 
        tot_sum = tot_sum.tolist()
        ans = []
        for i in range(num_iter):
            ans.append(torch.tensor(tot_sum[(i*num_case):(i+1)*num_case]).reshape(1, -1))
        ans = torch.cat(ans, dim=0)
        count_0 = torch.sum((ans == 1), dim=0).reshape(1, -1)
        count_2 = torch.sum((ans == 3), dim=0).reshape(1, -1)
        mask_0 = ((count_0)>count_2).float()
        mask_2 = ((count_2)>count_0).float()
        ans = mask_0*1 + mask_2*3
        ans_mask = 1-(ans>0).float()
        ans -= 1 
        ans *= (1-ans_mask)
        ans_2 = ans_mask * eval_labels
        ans += ans_2
        ans = ans[0]
        
        #only dropout
        ans_3 = []
        tot_label_list = tot_label_list.tolist()
        for i in range(num_iter):
            ans_3.append(torch.tensor(tot_label_list[(i*num_case):(i+1)*num_case]).reshape(1, -1))
        ans_3 = torch.cat(ans_3, dim=0)
        print(ans_3)
        count_0 = torch.sum((ans_3 == 0), dim=0).reshape(1, -1)
        count_2 = torch.sum((ans_3 == 2), dim=0).reshape(1, -1)
        mask_0 = ((count_0)>count_2).float()
        mask_2 = ((count_2)>count_0).float()
        ans_3 =  mask_0*0 + mask_2*2
        ans_3 = ans_3[0]
    return ans, ans_3

ans, ans2= get_ans()

input_ids_all = torch.cat((input_ids_train_1, input_ids_train_2), dim=0)
attention_all = torch.cat((attention_masks_train_1, attention_masks_train_2), dim=0)


#unconf+dropout
dataset_pseudo_train = TensorDataset(input_ids_all, attention_all, ans)
batch_size=4
train_pseudo_dataloader = DataLoader(
            dataset_pseudo_train, 
            batch_size = batch_size 
        )
torch.cuda.empty_cache()
model = train(train_pseudo_dataloader, PATH)
torch.save(model.module.state_dict(), f'./copa-{model_name}/finetuned_pseudo_{pth_index}.pth')

#confidence
dataset_confidence_train = TensorDataset(input_ids_all, attention_all, eval_labels)
batch_size=4
train_confidence_dataloader = DataLoader(
            dataset_confidence_train, 
            batch_size = batch_size 
        )

torch.cuda.empty_cache()
model = train(train_confidence_dataloader, PATH)
torch.save(model.module.state_dict(), f'./copa-{model_name}/finetuned_confidence_{pth_index}.pth')

#dropout
dataset_dropout_train = TensorDataset(input_ids_all, attention_all, ans2)
batch_size=4
train_dropout_dataloader = DataLoader(
            dataset_dropout_train, 
            batch_size = batch_size 
        )

torch.cuda.empty_cache()
model = train(train_dropout_dataloader, PATH)
torch.save(model.module.state_dict(), f'./copa-{model_name}/finetuned_dropout_{pth_index}.pth')            

#unconf
dataset_unconf_train = TensorDataset(input_ids_unconf_tensor, attention_unconf_tensor, ans3)
batch_size=4
train_unconf_dataloader = DataLoader(
            dataset_unconf_train, 
            batch_size = batch_size 
        )
torch.cuda.empty_cache()
model = train(train_unconf_dataloader, PATH)
torch.save(model.module.state_dict(), f'./copa-{model_name}/finetuned_unconf_{pth_index}.pth')  
