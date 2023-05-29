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
import sys
from .utils import *

parser = argparse.ArgumentParser(
    prog='copa_finetune_test.py',
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

model_checkpoint = {
    'deberta': 'microsoft/deberta-large',
    'roberta': 'roberta-large',
}[model_name]

new_cache_dir="./cache"
df_test = get_dataset('copa', dataset_type='test')
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) 
sentences_train_1 = df_test.inference_pair_1.values
sentences_train_2 = df_test.inference_pair_2.values
labels_train = df_test.label.values

input_ids_train_1, attention_masks_train_1 = get_input_and_attention(sentences_train_1)
input_ids_train_2, attention_masks_train_2 = get_input_and_attention(sentences_train_2)
input_ids_train_1=input_ids_train_1.clip(0, 28995)
input_ids_train_2=input_ids_train_2.clip(0, 28995)
labels_train= torch.tensor(labels_train)
dataset_train = TensorDataset(input_ids_train_1, input_ids_train_2, attention_masks_train_1, attention_masks_train_2, labels_train)
batch_size = 16

train_dataloader = DataLoader(
            dataset_train, 
            sampler = RandomSampler(dataset_train), 
            batch_size = batch_size 
        )
PATH=f'./copa-{model_name}/finetuned_{pth_type}_{pth_index}.pth'
torch.cuda.empty_cache()
device=torch.device('cuda')
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,cache_dir=new_cache_dir, num_labels=3)
model=model.to(device)
model.load_state_dict(torch.load(PATH))
model=model.to(device)
model.eval()
total_steps = len(train_dataloader)
progress_bar = tqdm(range(total_steps))
accurate=0.0
total=0
with torch.no_grad():
    for batch in train_dataloader:
        b_input_ids_1 = batch[0].to(device)
        b_input_ids_2 = batch[1].to(device)
        b_input_mask_1 = batch[2].to(device)
        b_input_mask_2 = batch[3].to(device)
        b_labels = batch[4].to(device)
        output_1 = model(b_input_ids_1, 
                             attention_mask=b_input_mask_1)
        output_2 = model(b_input_ids_2, 
                             attention_mask=b_input_mask_2)
        ans=torch.tensor(output_1.logits[:,2]>output_2.logits[:,2])
        accurate += (ans == b_labels).sum() 
        total+=ans.shape[0]
        progress_bar.update(1)
        torch.cuda.empty_cache()

with open("test_result.txt", "a") as f:
    print("path: ", PATH, file=f)
    print("accuracy is: ", accurate/total, file=f) 
    f.close()
print(accurate/total)