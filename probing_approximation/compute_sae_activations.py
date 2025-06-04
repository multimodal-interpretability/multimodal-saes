from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from tqdm import tqdm
import os
import json
from baukit import Trace
from PIL import Image
import pandas as pd
import torch.nn as nn
layer=8

gpu_index = int(input("Select a GPU"))
training_gpu = "cuda:"+str(gpu_index)
torch.set_grad_enabled(False)

classes = open('imagenet_classes.txt','r').read().splitlines() 
ranges = ranges = [
    range(0, 15),    # 0-14
    range(15, 30),   # 15-29
    range(30, 45),   # 30-44
    range(45, 60),   # 45-59
    range(60, 74),   # 60-73
    range(74, 88),   # 74-87
    range(88, 100)   # 88-99
]
selected_range = ranges[int(input("Select a Range (0-6): "))]


base_imagenet = '/data/vision/torralba/datasets/imagenet_pytorch_new'
labels_path = f'{base_imagenet}/imagenet_class_index.json'
with open(labels_path, 'r') as f:
    imagenet_labels = json.load(f)

imagenet_labels = pd.DataFrame(imagenet_labels).T.rename(columns=({0: 'id', 1: 'label'}))

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(training_gpu)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
device = torch.device("cuda:"+str(gpu_index))

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
sae = SAE()
sae = nn.DataParallel(sae, device_ids=[gpu_index])
sae = sae.cuda(device=gpu_index)
dirr = os.listdir(f"/SAE_imagetext_layer{layer}_models_longrun/")[0]
sae.load_state_dict(torch.load(f"/SAE_imagetext_layer{layer}_models_longrun/"+dirr))
print("starting?")

for i in selected_range:
    clas = classes[i]
    e1 = []#5
    print("working on class: " +clas)
    rng = range(0,500)
    cl_id = int(imagenet_labels[imagenet_labels['label'] == clas].index[0])
    cl_path = imagenet_labels.iloc[cl_id].id
    cl_dir = f'/data/vision/torralba/datasets/imagenet_pytorch_new/train/{cl_path}'
    paths = os.listdir(cl_dir)
    
    for ind in tqdm(rng):
        image = Image.open(cl_dir+'/'+paths[ind]).resize((336,336))
        vlm_inputs = processor('<image>',image, return_tensors="pt").to(training_gpu)
        with Trace(model.language_model.model.layers[8],retain_input=True,detach=False) as ret1:
            _ = model(**vlm_inputs)
        e1.append(torch.mean(sae(8*ret1.input[0].float().detach().cpu())[1],dim=0))
       
    tens = torch.stack(e1)
    torch.save(tens,'image_activations/'+clas+'/sae.pt')
