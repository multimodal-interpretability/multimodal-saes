
layer = 8
gpu_index = int(input("Which GPU? "))

sparsity_levels = [0.005,0.0005,0.00005,0.00005,0]
meow = int(input("Which Lamb (0-4)? "))

lamb = sparsity_levels[meow]
rngs = [range(0,16),range(16,32),range(32,50)]
rng = rngs[int(input("Which Range (0-2)"))]
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
device = torch.device("cuda:"+str(gpu_index))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



w = open("class_pairs.txt", 'r').read().splitlines() 
steering_vectors = []
classes = os.listdir("steering_vectors")
cls1 = []
cls2 = []
for cls in classes[:50]:
    a = cls.split("_to_")[0]
    b = cls.split("_to_")[1]
    cls1.append(a)
    cls2.append(b)

for index in rng:
    data1 = torch.load(f'image_activations/{cls1[index]}/sae.pt').detach().cpu().numpy()
    data2 = torch.load(f'image_activations/{cls2[index]}/sae.pt').detach().cpu().numpy()
    
    negative_labels = np.zeros((500, 1))
    positive_labels = np.ones((500, 1))
    
    X = np.vstack((data1, data2))
    y = np.vstack((negative_labels, positive_labels))
    
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    class SimpleLinearModel(nn.Module):
        def __init__(self, input_dim):
            super(SimpleLinearModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    
    input_dim = X_train.shape[1]
    linear_model = SimpleLinearModel(input_dim).to(device)
    reciptemp =1.5
    mask = torch.zeros(4096*8, requires_grad=True,device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(list(linear_model.parameters())+[mask], lr=0.001)
    
    num_epochs = 1000
    batch_size = 32
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        permutation = torch.randperm(X_train.size()[0])
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            u = torch.clip(torch.rand(4096*8,device=device, requires_grad=True),0.0001,0.9999)
            s = torch.sigmoid(reciptemp*((u/(1-u)).log() + mask))
            z = torch.clip(s*(1.2)-0.1,0,1)
            # Forward pass
            batch_x = batch_x*z
            outputs = linear_model(batch_x)
            loss = criterion(outputs, batch_y) + lamb*torch.sum(torch.sigmoid(mask-np.log(0.1 / 1.1)/reciptemp))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #TODO: combine the weight matrix
    u = torch.clip(torch.rand(4096*8,device=device, requires_grad=True),0.0001,0.9999)
    s = torch.sigmoid(reciptemp*((u/(1-u)).log() + mask))
    z = 1.0*(torch.clip(s*(1.2)-0.1,0,1)>0.5).unsqueeze(0)
    linear_model.linear.weight.data *= z
    

    with torch.no_grad():
        y_pred = linear_model(X_test)
        y_pred = torch.round(y_pred)
        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())  # Move to CPU for accuracy calculation
        print(f"Test accuracy on : {accuracy:.4f}")
    #TODO: save model
    torch.save(linear_model.state_dict(), 'probes/lambda_'+str(meow)+'/'+classes[index]+'_linear_classifier.pt')
        
