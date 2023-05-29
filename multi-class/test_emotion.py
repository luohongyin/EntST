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
from .utils import *

parser = argparse.ArgumentParser(
    prog='emotion_classification_finetune_test.py',
    description='finetune emotion classification model',
)

parser.add_argument('--index', type=int, required=True,
                    help='index of pth file')
parser.add_argument('--algo', type=str, required=True,
                    choices=['deberta', 'roberta'], help='model name')
parser.add_argument('--type', type=str, required=True,
                   choices=['pseudo', 'confidence', 'dropout', 'unconf'])
args = parser.parse_args()

pth_index = args.index
model_name = args.algo
pth_type = args.type

new_cache_dir="./cache2"
model_checkpoint = {
    'deberta': 'microsoft/deberta-large',
    'roberta': 'roberta-large',
}[model_name]

df_test = get_dataset('copa', dataset_type="test")
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) 
sentences_train_0 = df_test.inference_pair_0.values
sentences_train_1 = df_test.inference_pair_1.values
sentences_train_2 = df_test.inference_pair_2.values
sentences_train_3 = df_test.inference_pair_3.values
sentences_train_4 = df_test.inference_pair_4.values
sentences_train_5 = df_test.inference_pair_5.values
labels_train = df_test.label.values

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
dataset_train = TensorDataset(input_ids_train_0, attention_masks_train_0,input_ids_train_1, attention_masks_train_1,input_ids_train_2, attention_masks_train_2,input_ids_train_3, attention_masks_train_3,input_ids_train_4, attention_masks_train_4,input_ids_train_5, attention_masks_train_5,labels_train)
batch_size = 32

train_dataloader = DataLoader(
            dataset_train, 
            sampler = RandomSampler(dataset_train), 
            batch_size = batch_size 
        )
# PATH = {
#     'deberta': './model_files/mnli_model_sc_3e-06_binary_p1.pt/pytorch_model.bin',
#     'roberta': './model_files/mnli_model_sc_5e-06_binary_pr.pt/pytorch_model.bin',
# }[model_name]
PATH=f'./emotion-{model_name}/finetuned_{pth_type}_{pth_index}.pth'
torch.cuda.empty_cache()
device=torch.device('cuda')
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
model.load_state_dict(torch.load(PATH))
model=model.to(device)
device_ids=[0, 1, 2, 3]
model = nn.DataParallel(model, device_ids = device_ids)
model.eval()
total_steps = len(train_dataloader)
progress_bar = tqdm(range(total_steps))
accurate=0.0
total=0
with torch.no_grad():
    def get_base_value():
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
        input_i3=input_3.clip(0, 28995).reshape(1, -1)
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

with torch.no_grad():
    for batch in train_dataloader:
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
        output_0 = model(b_input_ids_0, 
                             attention_mask=b_input_mask_0)
        output_1 = model(b_input_ids_1, 
                             attention_mask=b_input_mask_1)
        output_2 = model(b_input_ids_2, 
                             attention_mask=b_input_mask_2)
        output_3 = model(b_input_ids_3, 
                             attention_mask=b_input_mask_3)
        output_4 = model(b_input_ids_4, 
                             attention_mask=b_input_mask_4)
        output_5 = model(b_input_ids_5, 
                             attention_mask=b_input_mask_5)
        output_0.logits-=base_0[0]
        output_1.logits-=base_1[0]
        output_2.logits-=base_2[0]
        output_3.logits-=base_3[0]
        output_4.logits-=base_4[0]
        output_5.logits-=base_5[0]
        output_0.logits[:, 1]-=1000000
        output_1.logits[:, 1]-=1000000
        output_2.logits[:, 1]-=1000000
        output_3.logits[:, 1]-=1000000
        output_4.logits[:, 1]-=1000000
        output_5.logits[:, 1]-=1000000
        output_0.logits=torch.nn.functional.softmax(output_0.logits, dim=1)
        output_1.logits=torch.nn.functional.softmax(output_1.logits, dim=1)
        output_2.logits=torch.nn.functional.softmax(output_2.logits, dim=1)
        output_3.logits=torch.nn.functional.softmax(output_3.logits, dim=1)
        output_4.logits=torch.nn.functional.softmax(output_4.logits, dim=1)
        output_5.logits=torch.nn.functional.softmax(output_5.logits, dim=1)
        ans_0=output_0.logits[:, 0]
        ans_1=output_1.logits[:, 0]
        ans_2=output_2.logits[:, 0]
        ans_3=output_3.logits[:, 0]
        ans_4=output_4.logits[:, 0]
        ans_5=output_5.logits[:, 0]
        ans=torch.cat((ans_0, ans_1, ans_2, ans_3, ans_4, ans_5), dim=0).reshape(6, -1).T
        accurate += (ans.argmax(1) == b_labels).sum() 
        total+=batch_size
        progress_bar.update(1)
        torch.cuda.empty_cache()
           
        
with open("test_result.txt", "a") as f:
    print("path: ", PATH, file=f)
    print("accuracy is: ", accurate/total, file=f) 
    f.close()
print("accuracy is: ", accurate/total) 