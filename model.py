
import torch
import torch.nn as nn
from torch.autograd import Variable

import pretrainedmodels

import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

def build_model(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrained model

    model_name = model_name # could be fbresnet152 or inceptionresnetv2

    if(model_name == 'senet154'):
        model = pretrainedmodels.senet154(pretrained='imagenet')
    elif(model_name == 'se_resnet152'):
        model = pretrainedmodels.se_resnet152(pretrained='imagenet')
    elif(model_name == 'se_resnext101_32x4d'):
        model = pretrainedmodels.se_resnext101_32x4d(pretrained='imagenet')
    elif(model_name == 'resnet152'):
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif(model_name == 'resnet101'):
        model = pretrainedmodels.resnet101(pretrained='imagenet')
    elif(model_name == 'densenet201'):
        model = pretrainedmodels.densenet201(pretrained='imagenet')

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.last_linear.in_features

    class CustomModel(nn.Module):
        def __init__(self, model):
            super(CustomModel, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-1]  )
            self.classifier = nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.Dropout(0.3),  # drop 50% of the neuron
                torch.nn.Linear(128, 7)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    model = CustomModel(model)
    freeze_layer(model.features)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    
    class CustomModel1(nn.Module):
        def __init__(self, model):
            super(CustomModel1, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Sequential(
                *[list(model.classifier.children())[i] for i in [0]]
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    CustomModel = CustomModel1(model)
    num_ftrs = list(CustomModel.classifier.children())[-1].out_features
    CustomModel.to(device)
    return CustomModel, num_ftrs


def build_whole_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model1, num_ftrs1 = build_model('senet154')
    model2, num_ftrs2 = build_model('se_resnet152')
    model3, num_ftrs3 = build_model('se_resnext101_32x4d')
    model4, num_ftrs4 = build_model('resnet152')
    model5, num_ftrs5 = build_model('resnet101')
    # model5, num_ftrs5 = build_model('densenet201')

    total_num_ftrs = num_ftrs1 + num_ftrs2 + num_ftrs3 + num_ftrs4 + num_ftrs5
    class Net(nn.Module):
        def __init__(self, model1, model2, model3, model4, model5):
            super(Net, self).__init__()
            self.net1 = model1 # segment 1 from VGG
            self.net2 = model2 #segment 2 from VGG
            self.net3 = model3 #segment 2 from VGG
            self.net4 = model4
            self.net5 = model5
            
            self.classifier = nn.Sequential(
                torch.nn.Linear(total_num_ftrs, 512),
                torch.nn.Dropout(0.3),
                  # drop 30% of the neuron
                torch.nn.Linear(512, 7),
                
            )

        def forward(self, x):
            x1 = self.net1(x)
            x2 = self.net2(x)             
            x3 = self.net3(x)
            x4 = self.net4(x)             
            x5 = self.net5(x) 
            x6 = torch.cat((x1, x2, x3, x4, x5),1)
            x6 = x6.view(x6.size(0), -1)
            out = self.classifier(x6)
            return out


    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    
    net = Net(model1, model2, model3, model4, model5)
    freeze_layer(net.net1)
    freeze_layer(net.net2)
    freeze_layer(net.net3)
    freeze_layer(net.net4)
    freeze_layer(net.net5)

    net.name = 'model_128_all_512_7'
    net.to(device)
    return net

