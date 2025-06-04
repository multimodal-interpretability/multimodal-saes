gpu_index = int(input("Select a GPU (0-6): "))
def split_range_to_ranges(start, end, n):
    # Calculate the size of each piece
    range_size = (end - start)
    piece_size = range_size // n
    remainder = range_size % n  # Remainder for uneven splits

    # Create the ranges
    ranges = []
    current_start = start

    for i in range(n):
        current_end = current_start + piece_size
        # Distribute the remainder
        if i < remainder:
            current_end += 1

        ranges.append(range(current_start, current_end))
        current_start = current_end

    return ranges

ranges = result = split_range_to_ranges(0, 1000, 21)
selected_range = ranges[int(input("Select a Range (0-20): "))]
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from tqdm import tqdm
import os
import json
from baukit import Trace
from PIL import Image
import pandas as pd



training_gpu = "cuda:"+str(gpu_index)
torch.set_grad_enabled(False)

classes = open('imagenet_classes.txt','r').read().splitlines() 



base_imagenet = '/data/vision/torralba/datasets/imagenet_pytorch_new'
labels_path = f'{base_imagenet}/imagenet_class_index.json'
with open(labels_path, 'r') as f:
    imagenet_labels = json.load(f)

imagenet_labels = pd.DataFrame(imagenet_labels).T.rename(columns=({0: 'id', 1: 'label'}))

classes = imagenet_labels['label'].values

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True).to(training_gpu)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

print("starting?")

for i in selected_range:
    clas = classes[i]
    os.makedirs(f'image_activations/{clas}', exist_ok = True)
    e1 = []#5
    e2 = []#8
    e3 = []#12
    e4 = []#16
    e5 = []#20
    e6 = []#24
    print("working on class: " +clas)
    rng = range(0,500)
    cl_id = int(imagenet_labels[imagenet_labels['label'] == clas].index[0])
    cl_path = imagenet_labels.iloc[cl_id].id
    cl_dir = f'/data/vision/torralba/datasets/imagenet_pytorch_new/train/{cl_path}'
    paths = os.listdir(cl_dir)
    
    for ind in tqdm(rng):
        image = Image.open(cl_dir+'/'+paths[ind]).resize((336,336))
        vlm_inputs = processor('<image>',image, return_tensors="pt").to(training_gpu)
        with Trace(model.language_model.model.layers[5],retain_input=True,detach=False) as ret1:
            with Trace(model.language_model.model.layers[8],retain_input=True,detach=False) as ret2:
                with Trace(model.language_model.model.layers[12],retain_input=True,detach=False) as ret3:
                    with Trace(model.language_model.model.layers[16],retain_input=True,detach=False) as ret4:
                        with Trace(model.language_model.model.layers[20],retain_input=True,detach=False) as ret5:
                            with Trace(model.language_model.model.layers[24],retain_input=True,detach=False) as ret6:
                                _ = model(**vlm_inputs)
        e1.append(torch.mean(ret1.input[0].float().detach().cpu(),dim=0))
        e2.append(torch.mean(ret2.input[0].float().detach().cpu(),dim=0))
        e3.append(torch.mean(ret3.input[0].float().detach().cpu(),dim=0))
        e4.append(torch.mean(ret4.input[0].float().detach().cpu(),dim=0))
        e5.append(torch.mean(ret5.input[0].float().detach().cpu(),dim=0))
        e6.append(torch.mean(ret6.input[0].float().detach().cpu(),dim=0))
       
    tens = torch.stack(e1)
    torch.save(tens,'image_activations/'+clas+'/layer5.pt')
    tens = torch.stack(e2)
    torch.save(tens,'image_activations/'+clas+'/layer8.pt')
    tens = torch.stack(e3)
    torch.save(tens,'image_activations/'+clas+'/layer12.pt')
    tens = torch.stack(e4)
    torch.save(tens,'image_activations/'+clas+'/layer16.pt')
    tens = torch.stack(e5)
    torch.save(tens,'image_activations/'+clas+'/layer20.pt')
    tens = torch.stack(e6)
    torch.save(tens,'image_activations/'+clas+'/layer24.pt')
