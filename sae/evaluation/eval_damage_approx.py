


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
random.seed(42)
def load(item):
    prompt = '[INST] <image>\nWhat is shown in this image? [/INST] ' +item['conversations'][1]['value']
    try:
        image = Image.open("/data/vision/torralba/selfmanaged/torralba/scratch/schwett/meow/ShareGPT/"+ item['image']).resize((336,336))
    except:
        return 'failed','failed'
    return (prompt, image)
layer = 8
gpus = ["cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:0"]
w = open("class_pairs.txt", 'r').read().splitlines() 
l=0
steering_vectors = torch.load(f'steering_probe/lambda_{l}_steering.pt')
for i in w:
    cls1 = i.split("@")[0]
    cls2 = i.split("@")[1]
models = []
for i in range(len(gpus)):
    models.append(LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(gpus[i]))

for i in range(len(gpus)):
    for param in models[i].parameters():
        param.requires_grad = False

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
ds = json.load(open("/data/vision/torralba/selfmanaged/torralba/scratch/schwett/meow/ShareGPT/sharegpt.json"))

torch.set_grad_enabled(False)

nums = torch.randint(600000, 900000, (100,)).sort().values.tolist()
lookup = {}
for gpu in gpus:
    lookup[gpu] = []
for index in nums:
    e = load(ds[index])
    for gpu in gpus:
        lookup[gpu].append((processor(e[0],e[1], return_tensors="pt").to(gpu),processor(ds[index]['conversations'][1]['value'], return_tensors="pt").to(gpu)))
def intervene(output,vec,k):
    meow = output[0].squeeze(0).float()
    meow[:1183] += k*vec
    return (meow.unsqueeze(0).half(),output[1])

def compute_loss(index,vector,k, gpu,ind):
    intervention = partial(intervene,k=k,vec=vector.clone().to(gpu))
    inp, inp2 = lookup[gpu][index]
    with Trace(models[ind].language_model.model.layers[layer-1],edit_output=intervention) as ret:
        _ = models[ind](**inp)
    logits = _.logits[0][-(len(inp2.input_ids[0])+1):-1]
    correct_answers = inp2.input_ids[0]
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = F.nll_loss(log_probs, correct_answers, reduction='mean')
    _ = models[ind](**inp)
    logits = _.logits[0][-(len(inp2.input_ids[0])+1):-1]
    correct_answers = inp2.input_ids[0]
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss2 = F.nll_loss(log_probs, correct_answers, reduction='mean')
    return nll_loss.detach().cpu().numpy().item()-nll_loss2.detach().cpu().numpy().item()


def compute_activations(indices,vector,k, gpu,ind):
    summer = 0
    for i in indices:
        with torch.no_grad():
            summer+=compute_loss(i,vector,k,gpu,ind)
    
    return summer/len(indices)

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
ks = list(range(20))
for k in ks:
    vals = []
    summer = 0
    cnt = 0
    eek = tqdm(steering_vectors[:50])
    for vector in eek:
        indices = split_list_into_2d_lists(list(range(14)),len(gpus))
        lis = Parallel(n_jobs=len(gpus),backend="threading")(delayed(compute_activations)(indices[j],vector,k,gpus[j],j) for j in range(len(gpus)))
        summer +=np.mean(lis).item()
        vals.append(np.mean(lis).item())
        cnt +=1
        eek.set_description(str(summer/cnt))
    
    damage_values.append(vals)

np.save(f"diffmeans_l{layer}_saeprobe_{l}_steering_damage.npy",np.array(damage_values))