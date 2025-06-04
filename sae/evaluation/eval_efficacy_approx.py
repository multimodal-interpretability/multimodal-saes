


"""
Steering Evaluation

"""
from functools import partial
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import json
import random
from baukit import Trace
from joblib import Parallel, delayed
import shutil
import torch.nn.functional as F
import pandas as pd
import json

random.seed(42)
base_imagenet = '/datasets/imagenet_pytorch_new'
labels_path = f'{base_imagenet}/imagenet_class_index.json'
with open(labels_path, 'r') as f:
    imagenet_labels = json.load(f)

imagenet_labels = pd.DataFrame(imagenet_labels).T.rename(columns=({0: 'id', 1: 'label'}))

layer = 8
gpus = ["cuda:1","cuda:2","cuda:3","cuda:5","cuda:6","cuda:0"]
w = open("class_pairs.txt", 'r').read().splitlines()
l=2
steering_vectors = torch.load(f'steering_probe/lambda_{l}_steering.pt')
for i in w:
    cls1 = i.split("@")[0]
    cls2 = i.split("@")[1]
models = []
for i in range(len(gpus)):
    models.append(LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(gpus[i]))



processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
ds = json.load(open("ShareGPT/sharegpt.json"))

torch.set_grad_enabled(False)



def intervene(output,vec,k):
    meow = output[0].squeeze(0).float()
    meow[:1183] += k*vec
    return (meow.unsqueeze(0).half(),output[1])

def compute_loss(inp,prompt,ans,vector,k, ind):
    model_in = processor(inp,prompt,return_tensors="pt").to(models[ind].device)
    intervention = partial(intervene,k=k,vec=vector.clone().to(models[ind].device))
    with Trace(models[ind].language_model.model.layers[layer-1],edit_output=intervention) as ret:
        _ = models[ind](**model_in)
    logits = _.logits[0][-1]
    logits = F.softmax(logits,dim=0)[ans]
    return logits.detach().cpu().numpy().item()
    
    


def compute_activations(inputs,prompt,ans,vector,k, ind):
    summer =0
    for inp in inputs:
        with torch.no_grad():
            summer+=compute_loss(inp,prompt,ans,vector,k, ind)
    
    return summer/len(inputs)

def split_list_into_2d_lists(input_list,num_gpus):
    #thx gpt
    total_elements = len(input_list)
    base_size = total_elements // num_gpus
    remainder = total_elements % num_gpus
    lists_2d = [[] for _ in range(num_gpus)]
    index = 0
    for i in range(num_gpus):
        if remainder > 0:
            sublist_length = base_size + 1
            remainder -= 1
        else:
            sublist_length = base_size
        sublist = input_list[index:index + sublist_length]
        index += sublist_length
        lists_2d[i] = sublist
    return lists_2d


damage_values = []
ks = list(range(20))#[0,4,12,20,32,64]
for k in ks:
    vals = []
    summer = 0
    cnt = 0
    eek = tqdm(range(50))
    for i in eek:
        vector = steering_vectors[i]
        cls1 = w[i].split("@")[0]
        cls2 = w[i].split("@")[1]
        cl_id = int(imagenet_labels[imagenet_labels['label'] == cls1].index[0])
        cl_path = imagenet_labels.iloc[cl_id].id
        cl_dir = f'/imagenet_pytorch_new/val/{cl_path}'
        paths = os.listdir(cl_dir)
        prompt = f'[INST] <image>\n Reply either "{cls1}" or "{cls2}" in lowercase depending on what is shown in the image. Do not output any other text. [/INST]'
        answer = processor(cls2,return_tensors="pt").input_ids[0][-1].item()
        inputs = []
        for j in range(14):
            inputs.append(Image.open(cl_dir+'/'+paths[j]).resize((336,336)))
        indices = split_list_into_2d_lists(inputs,len(gpus))
        lis = Parallel(n_jobs=len(gpus),backend="threading")(delayed(compute_activations)(indices[j],prompt,answer,vector,k,j) for j in range(len(gpus)))
        summer +=np.mean(lis).item()
        vals.append(np.mean(lis).item())
        cnt +=1
        eek.set_description(str(summer/cnt))
    
    damage_values.append(vals)

np.save(f"diffmeans_l{layer}_saeprobe_{l}_steering_efficacy.npy",np.array(damage_values))
