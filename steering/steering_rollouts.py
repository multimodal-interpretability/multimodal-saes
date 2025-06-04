gpu_index = int(input("Input GPU: "))
in_index = int(input("Input Index: "))


def split_range(start, end, parts):
    # Calculate the range of numbers
    numbers = list(range(start, end + 1))
    
    # Calculate the size of each part
    avg_size = len(numbers) / parts
    
    # Create the sublists
    sublists = [numbers[int(i * avg_size):int((i + 1) * avg_size)] for i in range(parts)]
    return sublists

# Split range 0-99 into 7 parts
sublists = split_range(0, 49, 7)
section = sublists[in_index]

sparsity_levels = [0.005,0.0005,0.00005,0]
meow = 3


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
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm
import pandas as pd

from functools import partial
layer=20

torch.set_grad_enabled(False)

training_gpu = "cuda:"+str(gpu_index)
base_imagenet = 'datasets/imagenet_pytorch_new'
labels_path = f'{base_imagenet}/imagenet_class_index.json'
with open(labels_path, 'r') as f:
    imagenet_labels = json.load(f)

imagenet_labels = pd.DataFrame(imagenet_labels).T.rename(columns=({0: 'id', 1: 'label'}))
from random import sample

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(training_gpu)

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

mtx = model.language_model.lm_head.weight.data.float().cpu()

def get_vector(cls1,cl2):
    a = processor(cls1,return_tensors="pt").input_ids[0][-1].item()
    b = processor(cls2,return_tensors="pt").input_ids[0][-1].item()
    return torch.nn.functional.normalize(mtx[b]-mtx[a],dim=0)
    
def intervene(output, vec, k):

    meow = output[0].squeeze(0).float()
    meow[:1183]+=k * vec
    #meow[:1183]=0
    return (meow.unsqueeze(0).half(), output[1])

def forward_ablate(text_input,image, tokens, vec, k):
    intervention = partial(intervene, vec=vec, k=k)
    inp = processor(text_input, image, return_tensors="pt").to(training_gpu)

    with Trace(model.language_model.model.layers[0], retain_input=True, stop=True) as ret:
        out = model(**inp)
    embeddings = ret.input[0]  
    
    
    with Trace(model.language_model.model.layers[layer-1], edit_output=intervention) as ret:
        out = model.language_model(
            inputs_embeds=embeddings.unsqueeze(0),
            use_cache=True)

    logits = out.logits[0][-1]
    past_key_values = out.past_key_values
    ans = torch.argmax(logits)
    new_token_embed = model.language_model.model.embed_tokens(ans)

    outs = [ans.detach().cpu()]
    
    for i in tqdm(range(tokens - 1)):  
        out = model.language_model(
            inputs_embeds = new_token_embed.unsqueeze(0).unsqueeze(0),  # [1, 1, hidden_dim]
            past_key_values=past_key_values,
            use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[0][-1]
        ans = torch.argmax(logits)
        outs.append(ans.detach().cpu())
        new_token_embed = model.language_model.model.embed_tokens(ans)
        

    return outs

def generate_ablate(input_text,image, num_tokens,vec, k):
    ids = forward_ablate(input_text,image, num_tokens,vec, k)
    return processor.decode(np.array(ids), skip_special_tokens=True)

classes = os.listdir("steering_vectors")
loaded = []
cls1 = []
cls2 = []
for cls in classes[:50]:
    cls1.append(cls.split("_to_")[0])
    cls2.append(cls.split("_to_")[1])
    loaded.append(torch.load(f'steering_vectors/{cls}/rollouts_layer{layer}.pt'))#get_vector(cls1[-1],cls2[-1]))
    #loaded.append(torch.nn.functional.normalize(torch.randn(4096),dim=0))



for cls_index in section:
    os.makedirs(f'rollouts/{classes[cls_index]}', exist_ok=True)
    cl_id = int(imagenet_labels[imagenet_labels['label'] == cls1[cls_index]].index[0])
    cl_path = imagenet_labels.iloc[cl_id].id
    cl_dir = f'datasets/imagenet_pytorch_new/val/{cl_path}'
    paths = os.listdir(cl_dir)
    vec = loaded[cls_index].to(training_gpu)
    
    prompts = ['[INST] <image>\nWhat is shown in this image? [/INST] ', 
               f'[INST] <image>\nIs this image related in any way to the concept "{cls1[cls_index]}"? [/INST] ',
               f'[INST] <image>\nIs this image related in any way to the concept "{cls2[cls_index]}"? [/INST] ',
               '[INST] <image>\nDescribe the primary purpose or function of the entity shown in this image. [/INST] ',
               '[INST] <image>\nIf you were to explain this image to a child, what analogy or simplified description would you use? [/INST] ',
               '[INST] <image>\nIn what environment or setting would you most likely encounter this? [/INST] ',
               '[INST] <image>\nWhat are some key distinctive features or characteristics that make this thing unique? [/INST] ',
               '[INST] <image>\nDescribe this image in detail. [/INST] ',
               '[INST] <image>\nHow large is the main subject of this image typically? [/INST] ',
               f'[INST] <image>\nIs this image more related to the concept "{cls1[cls_index]}" or the concept "{cls2[cls_index]}"? [/INST] '
          ]
    for k in np.linspace(0,30,20):
        file = open(f"rollouts/{classes[cls_index]}/layer_{layer}_k_{k}.txt",'w')
        for path in paths[:5]:
            file.write("################ NEW IMAGE ################\n")
            file.write("IMAGE PATH: " +cl_dir+'/'+path+'\n')
            file.write("################ NEW QUESTION ################\n")
            image = Image.open(cl_dir+'/'+path).resize((336,336))
            for prompt in prompts[:-1]:
                file.write(prompt+str(generate_ablate(prompt,image,100,vec,k))+"\n")
                file.write("################ NEW QUESTION ################\n")
            file.write(prompts[-1]+str(generate_ablate(prompts[-1],image,100,vec,k))+"\n")
        file.close()

