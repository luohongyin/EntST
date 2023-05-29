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
    prog='emotion_classification_finetune.py',
    description='finetune emotion classification model',
)

parser.add_argument('--index', type=int, required=True,
                    help='index of pth file')
parser.add_argument('--algo', type=str, required=True,
                    choices=['deberta', 'roberta'], help='model name')

args = parser.parse_args()

pth_index = args.index
model_name = args.algo

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
new_cache_dir="./cache2"
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
val_k = {
    'deberta': 10,
    'roberta': 10,
}[model_name]
divisor = {
    'deberta': 8,
    'roberta': 5,
}[model_name]

#random sample 2400 data point
df_train = get_dataset('emotion')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) 
sentences_train_0 = df_train.inference_pair_0.values
sentences_train_1 = df_train.inference_pair_1.values
sentences_train_2 = df_train.inference_pair_2.values
sentences_train_3 = df_train.inference_pair_3.values
sentences_train_4 = df_train.inference_pair_4.values
sentences_train_5 = df_train.inference_pair_5.values
labels_train = df_train.label.values

input_ids_train_0, attention_masks_train_0 = get_input_and_attention(sentences_train_0)
input_ids_train_0=input_ids_train_0.clip(0, 28995)
input_ids_train_1, attention_masks_train_1 = get_input_and_attention(sentences_train_1)
input_ids_train_1=input_ids_train_1.clip(0, 28995)
input_ids_train_2, attention_masks_train_2 = get_input_and_attention(sentences_train_2)
input_ids_train_2=input_ids_train_2.clip(0, 28995)
input_ids_train_3, attention_masks_train_3 = get_input_and_attention(sentences_train_3)
input_ids_train_3=input_ids_train_3.clip(0, 28995)
input_ids_train_4, attention_masks_train_4 = get_input_and_attention(sentences_train_4)
input_ids_train_4=input_ids_train_4.clip(0, 28995)
input_ids_train_5, attention_masks_train_5 = get_input_and_attention(sentences_train_5)
input_ids_train_5=input_ids_train_5.clip(0, 28995)
labels_train= torch.tensor(labels_train)
dataset_train = TensorDataset(input_ids_train_0, attention_masks_train_0, input_ids_train_1, attention_masks_train_1,input_ids_train_2, attention_masks_train_2,input_ids_train_3, attention_masks_train_3,input_ids_train_4, attention_masks_train_4,input_ids_train_5, attention_masks_train_5,labels_train)
batch_size = 4

train_dataloader = DataLoader(
            dataset_train, 
            batch_size = batch_size 
        )

torch.cuda.empty_cache()
device=torch.device('cuda')
input_ids = torch.cat((input_ids_train_0, input_ids_train_1, input_ids_train_2, input_ids_train_3, input_ids_train_4, input_ids_train_5), 0)
input_masks = torch.cat((attention_masks_train_0, attention_masks_train_1, attention_masks_train_2, attention_masks_train_3, attention_masks_train_4, attention_masks_train_5),0)
total_steps = len(train_dataloader)
progress_bar = tqdm(range(total_steps))
 
def get_base_value():
    with torch.no_grad(): 
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        device_ids=[0, 1, 2, 3]
        model = nn.DataParallel(model, device_ids = device_ids)
        model.eval() 
        inference_pair_0=["i feel unhappy is entailed by ."]
        inference_pair_1=["i feel happy is entailed by ."]
        inference_pair_2=["i feel love is entailed by ."]
        inference_pair_3=["i feel angry is entailed by ."]
        inference_pair_4=["i feel afraid is entailed by ."]
        inference_pair_5=["i feel shocked is entailed by ."]
        input_0, attention_0 = get_input_and_attention(inference_pair_0)
        input_0=input_0.clip(0, 28995).reshape(1, -1)
        input_1, attention_1 = get_input_and_attention(inference_pair_1)
        input_1=input_1.clip(0, 28995).reshape(1, -1)
        input_2, attention_2 = get_input_and_attention(inference_pair_2)
        input_2=input_2.clip(0, 28995).reshape(1, -1)
        input_3, attention_3 = get_input_and_attention(inference_pair_3)
        input_3=input_3.clip(0, 28995).reshape(1, -1)
        input_4, attention_4 = get_input_and_attention(inference_pair_4)
        input_4=input_4.clip(0, 28995).reshape(1, -1)
        input_5, attention_5 = get_input_and_attention(inference_pair_5)
        input_5=input_5.clip(0, 28995).reshape(1, -1)
        output_0 = model(input_0.to(device), 
                                 attention_mask=attention_0.to(device))
        output_1 = model(input_1.to(device), 
                                 attention_mask=attention_1.to(device))
        output_2 = model(input_2.to(device), 
                                 attention_mask=attention_2.to(device))
        output_3 = model(input_3.to(device), 
                                 attention_mask=attention_3.to(device))
        output_4 = model(input_4.to(device), 
                                 attention_mask=attention_4.to(device))
        output_5 = model(input_5.to(device), 
                                 attention_mask=attention_5.to(device))
        torch.cuda.empty_cache()
    return output_0.logits, output_1.logits, output_2.logits, output_3.logits, output_4.logits, output_5.logits

base_0, base_1, base_2, base_3, base_4, base_5=get_base_value()
base = torch.cat((base_0[0].reshape(1, -1), base_1[0].reshape(1, -1), base_2[0].reshape(1, -1), base_3[0].reshape(1, -1), base_4[0].reshape(1, -1), base_5[0].reshape(1, -1)), dim=0)

confidence_sample=[]

def get_eval_label():
    with torch.no_grad():
        input_ids_list=[]
        attention_list=[]
        label_list = []
        choice_list = []
        cls_embeddings_list = []
        cls_embeddings = []
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        device_ids=[0, 1, 2, 3]
        model = nn.DataParallel(model, device_ids = device_ids)
        model.eval() 
        for batch in tqdm(train_dataloader):
            b_labels = batch[12].to(device)
            b_input_ids_0 = batch[0].to(device)
            b_input_mask_0 = batch[1].to(device)
            b_input_ids_1 = batch[2].to(device)
            b_input_mask_1 = batch[3].to(device)
            b_input_ids_2 = batch[4].to(device)
            b_input_mask_2 = batch[5].to(device)
            b_input_ids_3 = batch[6].to(device)
            b_input_mask_3 = batch[7].to(device)
            b_input_ids_4 = batch[8].to(device)
            b_input_mask_4 = batch[9].to(device)
            b_input_ids_5 = batch[10].to(device)
            b_input_mask_5 = batch[11].to(device)
            
            b_input_ids = torch.cat((torch.unsqueeze(b_input_ids_0, 0), torch.unsqueeze(b_input_ids_1, 0), 
                                     torch.unsqueeze(b_input_ids_2, 0), torch.unsqueeze(b_input_ids_3, 0), 
                                     torch.unsqueeze(b_input_ids_4, 0), torch.unsqueeze(b_input_ids_5, 0)), dim=0)
            b_input_ids = b_input_ids.to("cpu")
            b_input_ids = torch.transpose(b_input_ids, 0, 1)
            
            b_input_mask = torch.cat((torch.unsqueeze(b_input_mask_0, 0), torch.unsqueeze(b_input_mask_1, 0), 
                                      torch.unsqueeze(b_input_mask_2, 0), torch.unsqueeze(b_input_mask_3, 0), 
                                      torch.unsqueeze(b_input_mask_4, 0), torch.unsqueeze(b_input_mask_5, 0)), dim=0)
            b_input_mask = b_input_mask.to("cpu")
            b_input_mask = torch.transpose(b_input_mask, 0, 1)
            
            output_0 = model(b_input_ids_0, 
                                 attention_mask=b_input_mask_0, output_hidden_states=True)
            output_1 = model(b_input_ids_1, 
                                 attention_mask=b_input_mask_1, output_hidden_states=True)
            output_2 = model(b_input_ids_2, 
                                 attention_mask=b_input_mask_2, output_hidden_states=True)
            output_3 = model(b_input_ids_3, 
                                 attention_mask=b_input_mask_3, output_hidden_states=True)
            output_4 = model(b_input_ids_4, 
                                 attention_mask=b_input_mask_4, output_hidden_states=True)
            output_5 = model(b_input_ids_5, 
                                 attention_mask=b_input_mask_5, output_hidden_states=True)
            
            output_0.logits[:, 1]-=1000000
            output_1.logits[:, 1]-=1000000
            output_2.logits[:, 1]-=1000000
            output_3.logits[:, 1]-=1000000
            output_4.logits[:, 1]-=1000000
            output_5.logits[:, 1]-=1000000
            
            output_0.logits-=base_0[0]
            output_1.logits-=base_1[0]
            output_2.logits-=base_2[0]
            output_3.logits-=base_3[0]
            output_4.logits-=base_4[0]
            output_5.logits-=base_5[0]
            
            output_0.logits=torch.nn.functional.softmax(output_0.logits, dim=1)
            output_1.logits=torch.nn.functional.softmax(output_1.logits, dim=1)
            output_2.logits=torch.nn.functional.softmax(output_2.logits, dim=1)
            output_3.logits=torch.nn.functional.softmax(output_3.logits, dim=1)
            output_4.logits=torch.nn.functional.softmax(output_4.logits, dim=1)
            output_5.logits=torch.nn.functional.softmax(output_5.logits, dim=1)
            
            #Get the largeset entailment value and label it as entailed
            ans_entail_0=output_0.logits[:, 0]
            ans_entail_1=output_1.logits[:, 0]
            ans_entail_2=output_2.logits[:, 0]
            ans_entail_3=output_3.logits[:, 0]
            ans_entail_4=output_4.logits[:, 0]
            ans_entail_5=output_5.logits[:, 0]
            ans_ent=torch.cat((ans_entail_0, ans_entail_1, ans_entail_2, ans_entail_3, ans_entail_4, ans_entail_5), dim=0).reshape(6, -1).T
            max_entail_label = ans_ent.argmax(1).tolist()
           
                  
            ans_0=output_0.logits[:, 0]<output_0.logits[:, 2]
            ans_1=output_1.logits[:, 0]<output_1.logits[:, 2]
            ans_2=output_2.logits[:, 0]<output_2.logits[:, 2]
            ans_3=output_3.logits[:, 0]<output_3.logits[:, 2]
            ans_4=output_4.logits[:, 0]<output_4.logits[:, 2]
            ans_5=output_5.logits[:, 0]<output_5.logits[:, 2]
            
            ans_label = torch.cat((ans_0, ans_1, ans_2, ans_3, ans_4, ans_5), dim=0).reshape(6, -1).T
            ans_label = ans_label.to("cpu").float()
            ans_label *= 2
            
            b_cls_embeddings = torch.cat((torch.unsqueeze(output_0.hidden_states[cls_layer][:, 0, :], 0),
                                          torch.unsqueeze(output_1.hidden_states[cls_layer][:, 0, :], 0), 
                                          torch.unsqueeze(output_2.hidden_states[cls_layer][:, 0, :], 0),
                                          torch.unsqueeze(output_3.hidden_states[cls_layer][:, 0, :], 0), 
                                          torch.unsqueeze(output_4.hidden_states[cls_layer][:, 0, :], 0),
                                          torch.unsqueeze(output_5.hidden_states[cls_layer][:, 0, :], 0)), dim=0)
            b_cls_embeddings = b_cls_embeddings.to("cpu")
            b_cls_embeddings = torch.transpose(b_cls_embeddings, 0, 1)
#             ans_label_cont = ans_label[:,max_cont_label]
            
            for i in range(len(max_entail_label)):
                input_ids_list.append(b_input_ids[i, max_entail_label[i], :].reshape(1, -1))
                attention_list.append(b_input_mask[i, max_entail_label[i], :].reshape(1, -1))
                label_list.append(ans_label[i, max_entail_label[i]])
                choice_list.append(max_entail_label[i])
                cls_embeddings.append(b_cls_embeddings[i, max_entail_label[i], :].reshape(1, -1))

            del output_0, output_1, output_2, output_3, output_4, output_5, ans_0, ans_1, ans_2, ans_3, ans_4, ans_5
            torch.cuda.empty_cache()
        input_ids_list = torch.cat(input_ids_list, dim=0)
        attention_list = torch.cat(attention_list, dim=0)
        cls_embeddings = torch.cat(cls_embeddings, dim=0)
        print(input_ids_list.shape)
        
    return input_ids_list.to("cpu"), attention_list.to("cpu"), torch.tensor(label_list).to("cpu"), torch.tensor(choice_list).to("cpu"), cls_embeddings


input_ids_tensor, attention_tensor, eval_labels, choice_labels, eval_cls_embeddings=get_eval_label()
num_0 = torch.sum((eval_labels == 0)).float()
num_2 = torch.sum((eval_labels == 2)).float()
labels_count = torch.tensor([num_0, 0, num_2])
labels_sum=torch.sum(labels_count)
labels_count/=labels_sum
print(labels_count)

input_ids_list = []
attention_list = []
eval_labels_list = []
choice_labels_list = []
eval_cls_embeddings = eval_cls_embeddings.to(device)
confidence_unconf_once = get_unconf_nodes(eval_cls_embeddings, eval_labels.long(), k=val_k, p_list=labels_count)

num_case = len(confidence_unconf_once)
confidence_unconf_once = torch.tensor(confidence_unconf_once)
for i in range(num_case):
    if confidence_unconf_once[i]==1:
        input_ids_list.append(input_ids_tensor[i, :].reshape(1, -1))
        attention_list.append(attention_tensor[i, :].reshape(1, -1))
        eval_labels_list.append(eval_labels[i])
        choice_labels_list.append(choice_labels[i])
input_ids_unconf_tensor = torch.cat(input_ids_list, dim=0)
attention_unconf_tensor = torch.cat(attention_list, dim=0)
ans3 = torch.tensor(eval_labels_list)

dataset_unconf_train = TensorDataset(input_ids_unconf_tensor, attention_unconf_tensor, ans3)
batch_size=4
train_unconf_dataloader = DataLoader(
            dataset_unconf_train, 
            batch_size = batch_size 
        )

dataset_train_pseudo = TensorDataset(input_ids_tensor, attention_tensor, eval_labels, choice_labels)
batch_size = 1

train_dataloader_pseudo = DataLoader(
            dataset_train_pseudo, 
            batch_size = batch_size 
        )

def get_label_and_confidence():
    with torch.no_grad():
        new_training_sample=[]
        ans_list=[]
        cls_embeddings_list = []
        cls_embeddings = []
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
        model.load_state_dict(torch.load(PATH))
        model=model.to(device)
        device_ids=[0, 1, 2, 3]
        model = nn.DataParallel(model, device_ids = device_ids)
        model.train() #was: model.eval()
        for batch in tqdm(train_dataloader_pseudo):
            b_choice = batch[3].to(device)
            b_labels = batch[2].to(device)
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            output = model(b_input_ids, 
                                     attention_mask=b_input_mask, output_hidden_states=True)
            output.logits -= base[b_choice, :]
            output.logits[:, 1]-=1000000
            output.logits=torch.nn.functional.softmax(output.logits, dim=1)
            ans=(output.logits[:, 0]<output.logits[:, 2]).to('cpu')
            ans_list.append(2*ans)
            cls_embeddings.append(output.hidden_states[-2][:, 0, :].to('cpu')) #roberta: -2, deberta: -6
            del output, ans
            torch.cuda.empty_cache()
        cls_embeddings=torch.cat(tuple(cls_embeddings), 0)
        ans_list = torch.cat(ans_list, 0).reshape(1, -1)
        cls_embeddings = cls_embeddings.to(device)
    return cls_embeddings, ans_list


def get_ans():
    with torch.no_grad():
        tot_confidence_list=[]
        tot_label_list=[]
        num=0
        label_unconf_once=[]
        confidence_unconf_once=[]
        num_iter = 5
        for i in range(num_iter):
            num+=1
            hidden_states, sample_choice = get_label_and_confidence()
            tot_confidence_list.append(hidden_states)
            tot_label_list.append(sample_choice[0])
        
        tot_confidence_list = torch.cat(tot_confidence_list,dim=0)
        tot_label_list=torch.cat(tot_label_list, dim=0)
        
        #unconf+dropout
        tot_confidence_list=torch.tensor(tot_confidence_list)
        tot_label_list=torch.tensor(tot_label_list)
        tot_confidence_list = get_unconf_nodes(tot_confidence_list, tot_label_list, k=10, p_list=labels_count)
        tot_confidence_list = torch.tensor(tot_confidence_list)    
        tot_confidence_list *= 2
        tot_confidence_list -=1 #-1: unconfident, 1: confident
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
        count_0 = torch.sum((ans_3 == 0), dim=0).reshape(1, -1)
        count_2 = torch.sum((ans_3 == 2), dim=0).reshape(1, -1)
        mask_0 = ((count_0)>count_2).float()
        mask_2 = ((count_2)>count_0).float()
        ans_3 =  mask_0*0 + mask_2*2
        ans_3 += (1-(mask_0+mask_2))*eval_labels
        ans_3 = ans_3[0]
    return ans, ans_3

ans, ans2= get_ans()

input_ids_all = input_ids_tensor
attention_all = attention_tensor

dataset_pseudo_train = TensorDataset(input_ids_all, attention_all, ans)
batch_size=4
train_pseudo_dataloader = DataLoader(
            dataset_pseudo_train, 
            batch_size = batch_size 
        )
model = train(train_pseudo_dataloader, PATH)

torch.save(model.module.state_dict(), f'./emotion-{model_name}/finetuned_pseudo_{pth_index}.pth')

#confidence train
dataset_confidence_train = TensorDataset(input_ids_all, attention_all, eval_labels)
batch_size=4
train_confidence_dataloader = DataLoader(
            dataset_confidence_train, 
            batch_size = batch_size 
        )

model = train(train_confidence_dataloader, PATH)
torch.save(model.module.state_dict(), f'./emotion-{model_name}/finetuned_confidence_{pth_index}.pth')            

#dropout
dataset_dropout_train = TensorDataset(input_ids_all, attention_all, ans2)
batch_size=4
train_dropout_dataloader = DataLoader(
            dataset_dropout_train, 
            batch_size = batch_size 
        )

torch.cuda.empty_cache()
model = train(train_dropout_dataloader, PATH)
torch.save(model.module.state_dict(), f'./emotion-{model_name}/finetuned_dropout_{pth_index}.pth')            

#dropout
torch.cuda.empty_cache()
model = train(train_unconf_dataloader, PATH)
torch.save(model.module.state_dict(), f'./emotion-{model_name}/finetuned_unconf_{pth_index}.pth')  