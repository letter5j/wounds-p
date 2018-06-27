import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from model import build_whole_model
import torch
from torchvision import transforms
import PIL.Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


file = '001.jpg'
# transforms.ToTensor()
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = PIL.Image.open(file)

print(type(img))
print(img.size)
inputs = data_transforms(img)
inputs.unsqueeze_(0)
print(type(inputs))
print(inputs.size)
inputs = inputs.to(device)




print('model is loading!')
model = build_whole_model()
PATH = os.path.abspath(os.path.dirname(__file__))
model.load_state_dict(torch.load(os.path.join(PATH, 'model.pth'), map_location='cpu'))
model.to(device)
model_p = model
print('model is loaded!')

outputs = model_p(inputs)
_, preds = torch.max(outputs, 1) 
print(preds)
# return render_template('change_avatar.html')
