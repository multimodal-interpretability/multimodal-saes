"""
script to train SAEs. Note that it stores all data in vram, as well as loads some questionable models

"""
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
random.seed(42)
from gpuparallel import GPUParallel, Delayed
gpu_index =0
training_gpu = "cuda:"+str(gpu_index)

gpus = ["cuda:3","cuda:4","cuda:5","cuda:6"]
density_scaler = 5.0
expand_factor = 16.0
learning_rate = 0.00005

torch.multiprocessing.set_start_method('spawn', force=True)
models = []
for i in range(len(gpus)):
    models.append(LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(gpus[i]))

for i in range(len(gpus)):
    for param in models[i].parameters():
        param.requires_grad = False

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
ds = json.load(open("ShareGPT/sharegpt.json"))


embedding_mtx=models[0].language_model.model.embed_tokens
torch.set_grad_enabled(True)
# PROMPT [INST] <image>\nWhat is shown in this image? [/INST]
inst_tokens = torch.tensor([[1,733,16289, 28793, 28705]]).to(gpus[0])
prompt_tokens = torch.tensor([[13,3195,349,4894,297,456,3469,28804,733,28748,16289,28793]]).to(gpus[0]) 
kill_embeddings = [embedding_mtx(inst_tokens)[0].to(training_gpu).sum(dim=1),embedding_mtx(prompt_tokens)[0].sum(dim=1).to(training_gpu)]
print("Embeddings Computed")
init = torch.randn(nn.Linear(int(expand_factor*4096), 4096,bias=True).weight.data.size())

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(4096,int(expand_factor*4096),bias=True) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(expand_factor*4096), 4096,bias=True) 
        self.initialize_weights()
    #Biases are initialized to zero, 
    def initialize_weights(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        #fun trick, gaussians in high dimensions are uniform "soap bubbles"
        W_d = init
        W_d = W_d / W_d.norm(dim=1, keepdim=True) * 0.1  # Adjust 0.1 as needed
        self.fc2.weight.data = W_d
        self.fc1.weight.data = W_d.t()
    def forward(self, x):
        features = self.relu1(self.fc1(x))
        x_hat = self.fc2(features)
        return x_hat,features #so I can use anthropic's loss function later

sae1 = SAE()
sae2 = SAE()
sae3 = SAE()

epochs = 1500 #this seems fine, could go higher or lower

sae1 = nn.DataParallel(sae1, device_ids=[0])
sae1 = sae1.cuda(device=0)
sae2 = nn.DataParallel(sae2, device_ids=[1])
sae2 = sae2.cuda(device=1)
sae3 = nn.DataParallel(sae3, device_ids=[2])
sae3 = sae3.cuda(device=2)
saes = [sae1,sae2,sae3]
optimizer1 = optim.Adam(sae1.parameters(),lr=learning_rate,betas=(0.9, 0.999)) #params from blog post
optimizer2 = optim.Adam(sae2.parameters(),lr=learning_rate,betas=(0.9, 0.999))
optimizer3 = optim.Adam(sae3.parameters(),lr=learning_rate,betas=(0.9, 0.999))
optimizers = [optimizer1,optimizer2,optimizer3]
#initialize schedule for lambda, LR
lambda_schedule = list(np.arange(0,density_scaler,density_scaler/(0.05*epochs)))
lambda_schedule.extend(np.ones(epochs)*density_scaler)
lambda_schedule = np.array(lambda_schedule).astype(np.float32)
def lr(epoch):
    if epoch <  epochs* 0.8:
        return 1.0
    else:
        return 1.0 - (epoch - epochs * 0.8) / (epochs * 0.2)
scheduler1 = LambdaLR(optimizer1, lr)
scheduler2 = LambdaLR(optimizer2, lr)
scheduler3 = LambdaLR(optimizer3, lr)
schedulers = [scheduler1,scheduler2,scheduler3]

def anthropicSAELoss(output,target,features,out_weight,lamb):
    #shape of output is (batch_size,4096)
    #shape of features is (batch_size, 4096*expansion)
    #shape of out_weight is (4096,4096*expansion)
    mse = torch.sum((output-target)**2,axis=1)
    density = torch.matmul(torch.abs(features),torch.linalg.vector_norm(out_weight,axis=0).unsqueeze(-1)).squeeze()
    return torch.mean(mse + lamb*density), torch.mean(mse), torch.mean(density)
    
def l0_norm(x, eps=1e-6):
    return torch.sum(torch.where(torch.abs(x) > eps, torch.ones_like(x), torch.zeros_like(x)))

#Setup Done, its time to train!
path = "SAE_imagetext_layer5_models_longrun/"
try:
    os.mkdir(path)
except:
    shutil.rmtree(path)
    os.mkdir(path)

def load(item):
    prompt = '[INST] <image>\nWhat is shown in this image? [/INST] ' +item['conversations'][1]['value']
    try:
        image = Image.open("ShareGPT/"+ item['image'])
    except:
        return 'failed','failed'
    return (prompt, image)


def apply_blacklist_mask(main_tensor, blacklist, mask):
    #claude vectorization magic
    n = main_tensor.size(0)
    k = blacklist.size(0)
    windows = main_tensor.unfold(0, k, 1)
    matches = torch.all(windows == blacklist, dim=1)
    expanded_matches = torch.zeros(n, dtype=torch.bool, device=main_tensor.device)
    for i in range(k):
        expanded_matches[i:n-k+1+i] |= matches
    mask[expanded_matches] = False
    
    return mask

def compute_activations(prompts,images, gpu,ind):
    """
    Compute activations for a list of inputs (Each 10 samples) 
    three main steps (for each input)
    1. compute activations
    2. mask out tokens containing the prompt (and other instruct tokens) (500 ms)
    2. mask out zeros used for padding (fast)

    """
    tim =0
    total = torch.zeros((1,4096)).to(gpu)
    for prompt,image in list(zip(*[prompts,images])):
        vlm_inputs = processor(prompt,image, return_tensors="pt").to(gpu)
        with torch.no_grad():
            with Trace(models[ind].language_model.model.layers[0],retain_input=True,retain_output=False,detach=True,stop=True) as ret:
                _ = models[ind](**vlm_inputs)           
            main_tensor = torch.flatten(ret.input,end_dim=1).sum(dim=1)
            mask = torch.ones(main_tensor.size(0), dtype=torch.bool)
            
            apply_blacklist_mask(main_tensor,kill_embeddings[0].to(gpu),mask)
            apply_blacklist_mask(main_tensor,kill_embeddings[1].to(gpu),mask)
            with Trace(models[ind].language_model.model.layers[5],retain_input=True,retain_output=False,detach=True,stop=True) as ret:
                _ = models[ind](**vlm_inputs)

            total = torch.concat([total,torch.flatten(ret.input,end_dim=1)[mask]],axis=0)
            vlm_inputs = processor(prompt[len('[INST] <image>\nWhat is shown in this image? [/INST] '):], return_tensors="pt").to(gpu)
            with Trace(models[ind].language_model.model.layers[5],retain_input=True,retain_output=False,detach=True,stop=True) as ret:
                _ = models[ind](**vlm_inputs)
            total = torch.concat([total,torch.flatten(ret.input,end_dim=1)],axis=0)
    return total[1:,:].cpu()

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


buffer = torch.zeros((1,4096))
normsum=0
index=0
cnt = 0


loaded = Parallel(n_jobs=32)(delayed(load)(ds[k]) for k in range(index,index+200))

prompts = []
images = []
for e in loaded:
    if e[0] !='failed':
        prompts.append(e[0])
        images.append(e[1])

cnt += len(prompts)
prompts = split_list_into_2d_lists(prompts,len(gpus))
images = split_list_into_2d_lists(images,len(gpus))

print("Computing activations")
lis = Parallel(n_jobs=len(gpus),backend="threading")(delayed(compute_activations)(prompts[j],images[j],gpus[j],j) for j in range(len(gpus)))
for activation in lis:
    cnt += activation.size(0)
    buffer = torch.cat([buffer,activation],axis=0)
    normsum +=torch.linalg.norm(activation,axis=1).sum().cpu().numpy()

i+=200
def aggregate_gradients(saes):
    lis1 = list(saes[0].parameters())
    lis2 = list(saes[1].parameters())
    lis3 = list(saes[2].parameters())
    for ind in range(len(lis1)):
        g1 = lis1[ind].grad.detach().cpu()
        g2 = lis2[ind].grad.detach().cpu()
        g3 = lis3[ind].grad.detach().cpu()
        true_grad = (g1+g2+g3)/3
        lis1[ind].grad *=0.0
        lis1[ind].grad +=true_grad.cuda(device=0)
        lis2[ind].grad *=0.0
        lis2[ind].grad +=true_grad.cuda(device=1)
        lis3[ind].grad *=0.0
        lis3[ind].grad +=true_grad.cuda(device=2)
def backward(sae,optimizer,inputs,gpu_id):

    outputs,features = sae(inputs)
    loss, mse, density = anthropicSAELoss(outputs,inputs,features, sae.module.fc2.weight.data,lambda_schedule[epoch])
    mean = torch.mean(inputs,axis=0) 
    var = 1-(torch.sum(inputs-outputs)**2)/(torch.sum(inputs-mean)**2)
       
    optimizer.zero_grad()
    loss.backward()
    l0 = l0_norm(features.detach().cpu())/features.size(dim=0)
    return loss.detach().cpu(), mse.detach().cpu(), density.detach().cpu(),l0,var.detach().cpu()

def apply_gradients(optimizer):
    optimizer.step()
    
epoch_monitor = tqdm(range(epochs))#only train on 1/6 of the data to save time
for epoch in epoch_monitor:
    epoch_monitor.set_description("Loading Dataset")
    loaded = Parallel(n_jobs=32)(delayed(load)(ds[k]) for k in range(index,index+200))

    prompts = []
    images = []
    for e in loaded:
        if e[0] !='failed':
            prompts.append(e[0])
            images.append(e[1])
    
    
    prompts = split_list_into_2d_lists(prompts,len(gpus))
    images = split_list_into_2d_lists(images,len(gpus))

    print("Computing activations")
    lis = Parallel(n_jobs=len(gpus),backend="threading")(delayed(compute_activations)(prompts[j],images[j],gpus[j],j) for j in range(len(gpus)))
    for activation in lis:
        cnt += activation.size(0)
        buffer = torch.cat([buffer,activation],axis=0)
        normsum +=torch.linalg.norm(activation,axis=1).sum().cpu().numpy()

    i+=200
    batch_size = 4192
    num_training_gpus = 3
    maxint = 0
    epoch_monitor.set_description("Done Loading Dataset, Training")
    buffer = buffer[torch.randperm(buffer.size(0))]
    for batch in range(0,int(len(buffer)/2),batch_size*num_training_gpus):
        maxint = max(maxint,batch+3*batch_size)
        input1 = 64/(normsum/cnt)*buffer[batch:batch+batch_size].to("cuda:0")
        input2 = 64/(normsum/cnt)*buffer[batch+batch_size:batch+2*batch_size].to("cuda:1")
        input3 = 64/(normsum/cnt)*buffer[batch+2*batch_size:batch+3*batch_size].to("cuda:2")
        inputs = [input1,input2,input3]
        
        loss, mse, density, l0,var = list(Parallel(n_jobs=len(gpus),backend="threading")(delayed(backward)(saes[j],optimizers[j],inputs[j],j) for j in range(3)))[0]
        aggregate_gradients(saes)
        grad_norm = torch.norm(torch.cat([torch.flatten(p.grad.detach()) for p in sae1.module.parameters()]), 2)
        torch.nn.utils.clip_grad_norm_(sae1.parameters(), 1)
        grad_norm = torch.norm(torch.cat([torch.flatten(p.grad.detach()) for p in sae2.module.parameters()]), 2)
        torch.nn.utils.clip_grad_norm_(sae2.parameters(), 1)
        grad_norm = torch.norm(torch.cat([torch.flatten(p.grad.detach()) for p in sae3.module.parameters()]), 2)
        torch.nn.utils.clip_grad_norm_(sae3.parameters(), 1)
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

    epoch_monitor.set_description(str({"mse": mse.item(),"density": density.item(), "loss": loss.item(),"var":var.item(),"l0":l0.item(), "grad_norm":grad_norm}))
    buffer = buffer[maxint:]
    for scheduler in schedulers:
        scheduler.step()
print("Training finished!")
print("Final Norm Value: " + str(normsum))
torch.save(sae.state_dict(), path +'sae_final_layer5_16x.pkl')
