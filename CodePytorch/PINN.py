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


class NN(nn.Module):
    # dimension = nombre de neurones en entrée
    def __init__(self, num_layers=3, num_neurons=128, dimension=2):
        super(NN, self).__init__()
        # Here we define the network architecture ()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dimension, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 10))

    def forward(self, x):  # x est l'entrée et là on la fait juste passer dans le réseau
        last_layer = self.model(x)
        return F.log_softmax(last_layer, dim=1)

# Entrainement


def train_step(model, loss_func, opt, x, y):  # x est l'entrée et y le resultat attendu
    opt.zero_grad()
    out = model(x)
    loss = loss_func(out, y)
    loss.backward()  # calcul des gradients et des pertes
    opt.step()  # mets à jour les paramètres du modèle pr l'optimiser
    return loss.item(), out


# epoch = nombre de fois qu'on passe sur tout le dataset
def train(model, loss_fn, opt, train_loader, epochs=10):
    for epoch in range(epochs):
        correct = 0
        bar = tqdm(train_loader)
        for x, y in bar:
            loss, out = train_step(model, loss_fn, opt, x, y)
            bar.set_description(f"Epoch {epoch} loss {loss}")
            out = torch.argmax(out, dim=1)  # on prend la classe prédite
            # on compte le nombre de bonnes réponses
            correct += (out == y).sum().item()
        # on affiche le taux de bonne réponse
        print(f"Epoch {epoch} accuracy {correct/len(train_loader.dataset)}")


# Load data
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)


# Define the network
model = NN(dimension=28*28)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, loss, opt, train_loader, epochs=10)
