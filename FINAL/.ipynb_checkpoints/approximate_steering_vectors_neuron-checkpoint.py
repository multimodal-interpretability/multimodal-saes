layer = 8
lambs = [0.01, 0.00265914795, 0.000707106781, 0.000188030155, 5e-05]
gpu_index = int(input("Which GPU? "))
lamb = lambs[gpu_index]

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os




device = "cuda:"+str(gpu_index)
mtx = torch.load("mtx.pt").detach().cpu().to(device)
w = open("class_pairs.txt", 'r').read().splitlines() 
steering_vectors = []
classes = os.listdir("steering_vectors")
cls1 = []
cls2 = []
for cls in classes[:15]:
    a = cls.split("_to_")[0]
    b = cls.split("_to_")[1]
    cls1.append(a)
    cls2.append(b)
    steering_vectors.append(torch.nn.functional.normalize(torch.load(f"steering_vectors/{cls}/diffmeans_layer8.pt").detach().cpu(),dim=0))
zs = []
coeffs = []
dots = []
l0s = []
vecs = []
for j in range(15):
    target_vec = steering_vectors[j].to(device)
    reciptemp =1.5
    mask = torch.zeros(14336, requires_grad=True,device=device)
    coefs = torch.randn_like(mask, requires_grad=True,device=device)
    optimizer = optim.Adam([coefs,mask], lr=0.001)
    criterion = torch.nn.MSELoss()
    # Train the model
    torch.set_grad_enabled(True)
    pbar = tqdm(range(0, 20000))
    for i in pbar:
        #take in real number mask and compute 0-1 using HardConcrete
        u = torch.clip(torch.rand(14336,device=device, requires_grad=True),0.0001,0.9999)
        s = torch.sigmoid(reciptemp*((u/(1-u)).log() + mask))
        z = torch.clip(s*(1.2)-0.1,0,1)
        # Forward pass
        out = torch.nn.functional.normalize(torch.matmul(mtx,coefs*z),dim=0)
        dot_loss = -torch.dot(out, target_vec) #maximize dot product
        sparsity_loss = torch.sum(torch.sigmoid(mask-np.log(0.1 / 1.1)/reciptemp))
        loss = dot_loss+ lamb*sparsity_loss
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10 ==0:
            z = 1.0*(z>0.5)
            out = torch.nn.functional.normalize(torch.matmul(mtx,coefs*z),dim=0)
            final_dot = torch.dot(out, target_vec).item()
            final_l0 = torch.sum(z).item()
            pbar.set_description("Dot Product: "+str(final_dot)+" L0: "+str(final_l0))
    z = 1.0*(z>0.5)
    coeffs.append(coefs*z)
    out = torch.nn.functional.normalize(torch.matmul(mtx,coefs*z),dim=0)
    vecs.append(out)
    dots.append(torch.dot(out, target_vec).item())
    l0s.append(torch.sum(z).item())
        
torch.save(torch.stack(coeffs,dim=0).detach().cpu(),f"steering_approx/neuron/lambda_{lamb}_reconstruction_coeffs.pt")
torch.save(torch.stack(vecs,dim=0).detach().cpu(),f"steering_approx/neuron/lambda_{lamb}_reconstruction_vectors.pt")
np.save(f"steering_approx/neuron/lambda_{lamb}_reconstruction_dots.npy",np.array(dots))
np.save(f"steering_approx/neuron/lambda_{lamb}_reconstruction_l0s.npy",np.array(l0s))
