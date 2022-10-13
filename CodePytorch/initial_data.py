import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable, grad

# Calculer la n-ieme dérivée d'une fonction
def nth_gradient(f, wrt,n):  # f est la fonction, wrt est la variable par rapport à laquelle on dérive
    for i in range(n):
        grads = grad(f, wrt, create_graph=True)[
            0]  # on dérive f par rapport à wrt
        f = grads
        if grads is None:
            print("Bad Gradient")
            return torch.tensor(0.)
    return grads

#Calcul du résidu de l'équation (en gros la diff entre les 2 membres de l'équation)
def residu(x,t):
    u=net(x,t) #u est la solution de l'équation
    u_xx=nth_gradient(u,x,2) #u_xx est la 2eme dérivée de u par rapport à x
    u_tt=nth_gradient(u,t,2) #u_tt est la 2eme dérivée de u par rapport à t
    residual=u_tt-4*u_xx
    return residual

#Calcul de la perte de l'équation et des bords (b) et initial (i)
def loss(x_r,t_r,
        x_b,t_b,u_b,
        x_i,t_i,u_i):
    loss_residual=torch.mean(residu(x_r,t_r)**2) #perte du résidu

    u_pred_b=net(x_b,t_b) #u_pred_b est la solution de l'équation aux bords
    loss_b=torch.mean((u_pred_b-u_b)**2) #perte des bords

    u_pred_i=net(x_i,t_i) #u_pred_i est la solution de l'équation à l'initial
    loss_i=torch.mean((u_pred_i-u_i)**2) #perte de l'initial

    return loss_residual+loss_b+loss_i
    