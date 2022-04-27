#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from initial import *
from model import *
from plot import * 
#%%
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

##Constantes à modifier (c:vitesse de l'onde,dimension:2D ou 3D,k : constante pour la gaussienne de l'onde à l'instant initial)
c = 2
dimension = 1
w = np.pi*c*np.sqrt(2**2+3**2)

tmin,tmax = 0.,1.
xmin,xmax = 0.,1.

def train():
    #Number of points
    #N_0 = 500 #Nombre de points pour la condition initiale
    #N_b = 500 #Nombre de points pour la condition aux bords
    #N_r = 10000 #Nombre de points pour le résidu (donc à l'intérieur du domaine)
    N_b = 1000
    N_0 = 1000
    N_r = 100000
    X_data,u_data,time_x,X_r = set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)

    plot_training_points(dimension,time_x)

    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)

    ##Pour les paramètres d'entrainement
    #lr : taux d'apprentissage
    #opt : Optimiseur (+ simple : descente de gradient, Adam : version améliorée + efficace)
    #epochs : le nombre de fois où l'on parcours tout le dataset

    lr = 1e-3
    opt = keras.optimizers.Adam(learning_rate=lr)
    epochs = 1000
    hist = []
    t0 = time()
    pinn = PINN(dimension+1,1,dimension,ub,lb,c)
    pinn.compile(opt)
    #Boucle d'entrainement
    def train():
        for i in range(epochs+1):
            loss = pinn.train_step(X_r,X_data,u_data) #on récupère la loss après chaque train_step
            hist.append(loss.numpy()) 

            if i%100 == 0:
                print(f'It {i}: loss = {loss}') #On print la loss tous les 500 epochs

        print('\nComputation time: {} seconds'.format(time()-t0))

    train() 
    return pinn,lb,ub

def multi_train():
    times = []
    points = np.concatenate(
        (np.arange(0,100,10),np.arange(100,1050,50)),axis=0
        )
    for N_0 in points:
        N_b = N_0
        N_r = 10*N_0
        X_data,u_data,time_x,X_r = set_training_data(
            tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r
            )

        plot_training_points(dimension,time_x)

        bound1 = [tmin] + [xmin for _ in range(dimension)]
        bound2 = [tmax] + [xmax for _ in range(dimension)]
        lb,ub = tf.constant(
            bound1,dtype=DTYPE
            ),\
            tf.constant(
                bound2,dtype=DTYPE
                )

        lr = 1e-3
        opt = keras.optimizers.Adam(learning_rate=lr)
        epochs = 1000
        hist = []
        t0 = time()

        pinn = PINN(dimension+1,1,dimension,ub,lb,c)
        pinn.compile(opt)
  
        def train():
            for i in range(epochs+1):
                loss = pinn.train_step(X_r,X_data,u_data)
                hist.append(loss.numpy()) 
            times.append(time()-t0)

        train() 
        return times
    

# %%
pinn,lb,ub = train()
model = pinn.model
N = 70
fps = 5
tspace = np.linspace(lb[0], ub[0], N + 1)

plot1d(lb,ub,N,tspace,model,fps)
# %%
def plot1d(lb,ub,N,tspace,model,fps):
    ###1D Wave
    x1space = np.linspace(lb[1], ub[1], N + 1)

    T,X1 = np.meshgrid(tspace,x1space)
    Xgrid = tf.stack([T.flatten(),X1.flatten()],axis=-1)

    upred = model(tf.cast(Xgrid,DTYPE))
    U = upred.numpy().reshape(N+1,N+1)
    z_array = np.zeros((N+1,N+1))
    for i in range(N):
        z_array[:,i]= U[i]

    plt.style.use('dark_background')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(T,X1,c=U, marker='X', vmin=0, vmax=1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x1$')
    plt.show()

plot1d(lb,ub,N,tspace,model,fps)      
# %%
