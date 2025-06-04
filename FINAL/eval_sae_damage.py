


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
torch.set_grad_enabled(False)
def load(item):
    prompt = '[INST] <image>\nWhat is shown in this image? [/INST] ' +item['conversations'][1]['value']
    try:
        image = Image.open("/data/vision/torralba/selfmanaged//torralba/scratch/schwett/meow/ShareGPT/"+ item['image']).resize((336,336))
    except:
        return 'failed','failed'
    return (prompt, image)


gpus = ["cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6"]
w = open("class_pairs.txt", 'r').read().splitlines() 
layer = 8
file = f"/data/vision/torralba/selfmanaged/torralba/scratch/schwett/meow/SAE_imagetext_layer{layer}_models_longrun/" +os.listdir(f"/data/vision/torralba/selfmanaged/torralba/scratch/schwett/meow/SAE_imagetext_layer{layer}_models_longrun/")[0]
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(4096,int(8*4096),bias=True) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(8*4096), 4096,bias=True) 
        self.initialize_weights()
    #Biases are initialized to zero, 
    def initialize_weights(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        #fun trick, gaussians in high dimensions are uniform "soap bubbles"
        W_d = torch.randn(self.fc2.weight.size())
        W_d = W_d / W_d.norm(dim=1, keepdim=True) * 0.1  # Adjust 0.1 as needed
        self.fc2.weight.data = W_d
        self.fc1.weight.data = W_d.t()
    def forward(self, x):
        features = self.relu1(self.fc1(x))
        x_hat = self.fc2(features)
        return x_hat,features 
def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from keys in a PyTorch state dictionary.

    Args:
        state_dict (dict): The state dictionary to clean.

    Returns:
        dict: The cleaned state dictionary.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Remove the first 7 characters ("module.")
        else:
            new_state_dict[key] = value
    return new_state_dict
models = []
saes = []
for i in range(len(gpus)):
    models.append(LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(gpus[i]))
    sae = SAE()
    sae = nn.DataParallel(sae, device_ids=[i])
    sae = sae.cuda(device=i)
    sae.load_state_dict(torch.load(file))
    saes.append(sae)
    

for i in range(len(gpus)):
    for param in models[i].parameters():
        param.requires_grad = False

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
ds = json.load(open("/data/vision/torralba/selfmanaged/torralba/scratch/schwett/meow/ShareGPT/sharegpt.json"))



nums = torch.randint(600000, 900000, (500,)).sort().values.tolist()
lookup = {}

def intervene(output,i):
    meow = output[0].squeeze(0).float()
    #print(torch.max(meow.unsqueeze(0)))
    meow = saes[i](meow.unsqueeze(0)*8)[0].squeeze(0)/8
    return (meow.unsqueeze(0).half(),output[1])
def zero_intervene(output):
    meow = output[0].squeeze(0).float()
    meow = meow*0.0
    return (meow.unsqueeze(0).half(),output[1])
def compute_loss(index, gpu,ind):
    e = load(ds[index])
    if e[0] == "failed":
        return -1
    inp = processor(e[0],e[1], return_tensors="pt").to(gpu)
    inp2 = processor(ds[index]['conversations'][1]['value'], return_tensors="pt").to(gpu)
    intervention = partial(intervene,i=ind)

    with Trace(models[ind].language_model.model.layers[layer-1],edit_output=intervention) as ret:
        _ = models[ind](**inp)
    logits = _.logits[0][-(len(inp2.input_ids[0])+1):-1]
    correct_answers = inp2.input_ids[0]
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = F.nll_loss(log_probs, correct_answers, reduction='mean').detach().cpu().numpy().item()
    with Trace(models[ind].language_model.model.layers[layer-1],edit_output=zero_intervene) as ret:
        _ = models[ind](**inp)
    logits = _.logits[0][-(len(inp2.input_ids[0])+1):-1]
    correct_answers = inp2.input_ids[0]
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss3 = F.nll_loss(log_probs, correct_answers, reduction='mean').detach().cpu().numpy().item()
    _ = models[ind](**inp)
    logits = _.logits[0][-(len(inp2.input_ids[0])+1):-1]
    correct_answers = inp2.input_ids[0]
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss2 = F.nll_loss(log_probs, correct_answers, reduction='mean').detach().cpu().numpy().item()
    return 1-(nll_loss-nll_loss2)/(nll_loss3-nll_loss2)


def compute_activations(indices, gpu,ind):
    summer = 0
    cnter = 0
    for i in indices:
        with torch.no_grad():
            loss = compute_loss(i,gpu,ind)
            if loss ==-1:
                continue
            summer+=loss
            cnter +=1
            
    
    return summer/cnter

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

   
indices = split_list_into_2d_lists(nums,len(gpus))
lis = Parallel(n_jobs=len(gpus),backend="threading")(delayed(compute_activations)(indices[j],gpus[j],j) for j in range(len(gpus)))
print(f"Loss Recovered for layer {layer}:" + str(np.mean(lis).item()))