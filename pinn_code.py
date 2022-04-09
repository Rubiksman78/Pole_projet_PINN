#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from time import time
import matplotlib.animation as animation
from initial import *
from model import *
from plot import * 
#%%
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

##Constantes à modifier (c:vitesse de l'onde,dimension:2D ou 3D,k : constante pour la gaussienne de l'onde à l'instant initial)
c = 0.5
dimension = 2
w = np.pi*c*np.sqrt(2**2+3**2)

tmin,tmax = 0.,1.0
xmin,xmax = -1.,1.
#Number of points
N_0 = 1000 #Nombre de points pour la condition initiale
N_b = 1000 #Nombre de points pour la condition aux bords
N_r = 10000 #Nombre de points pour le résidu (donc à l'intérieur du domaine)

X_data,u_data,time_x,X_r = set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)

plot_training_points(dimension,time_x)
#%%
bound1 = [tmin] + [xmin for _ in range(dimension)]
bound2 = [tmax] + [xmax for _ in range(dimension)]
lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)

##Pour les paramètres d'entrainement
#lr : taux d'apprentissage
#opt : Optimiseur (+ simple : descente de gradient, Adam : version améliorée + efficace)
#epochs : le nombre de fois où l'on parcours tout le dataset

lr = 1e-3
opt = keras.optimizers.Adam(learning_rate=lr)
epochs = 100
hist = []
t0 = time()

pinn = PINN(3,1,dimension,ub,lb,c)
pinn.compile(opt)

#Boucle d'entrainement
def train():
    for i in range(epochs+1):
        loss = pinn.train_step(X_r,X_data,u_data) #on récupère la loss après chaque train_step
        hist.append(loss.numpy()) 

        if i%500 == 0:
            print(f'It {i}: loss = {loss}') #On print la loss tous les 500 epochs

    print('\nComputation time: {} seconds'.format(time()-t0))

train() 
# %%
model = pinn.model
N = 50
fps = 5
tspace = np.linspace(lb[0], ub[0], N + 1)

plot2d(lb,ub,N,tspace,model,fps)
# %%
