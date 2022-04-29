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

#Boucle d'entrainement
def train(epochs,pinn,X_r,X_data,u_data):
    hist = []
    t0 = time()
    for i in range(epochs+1):
        loss_i,loss_b1,loss_b2,loss_r,lambda_b,lambda_r = \
            pinn.train_step(X_r,X_data,u_data) #on récupère la loss après chaque train_step
        loss = loss_i + loss_b1 + loss_b2 + loss_r
        hist.append(loss.numpy()) 
        if i % 100 == 0:
            print(f'It {i}: residual_loss = {loss_r} \
                | initial_loss = {loss_i} \
                | boundary_loss_x = {loss_b1} \
                | boundary_loss_v = {loss_b2}\
                | lambda_b = {lambda_b} \
                | lambda_r = {lambda_r}') #On print la loss tous les 500 epochs
    print('\nComputation time: {} seconds'.format(time()-t0))
    return hist


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
        lr = 1e-2
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
if __name__ == '__main__':
    config = {
    'c' : 2,
    'dimension' : 1,
    'tmin' : 0.,
    'tmax' : 1.,
    'xmin': 0.,
    'xmax': 1.,
    'N_b' : 300,
    'N_r' : 300,
    'N_0' : 300,
    'learning_rate' : 1e-3,
    'epochs': 5000
}
    c,dimension,tmin,tmax,xmin,xmax,N_b,N_r,N_0,lr,epochs = \
            config['c'],\
            config['dimension'],\
            config['tmin'],\
            config['tmax'],\
            config['xmin'],\
            config['xmax'],\
            config['N_b'],\
            config['N_r'],\
            config['N_0'],\
            config['learning_rate'],\
            config['epochs']
    X_data,u_data,time_x,X_r = \
        set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)

    plot_training_points(dimension,time_x)

    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)

    opt = keras.optimizers.Adam(learning_rate=lr)
    hist = []
    pinn = PINN(dimension+1,1,dimension,ub,lb,c)
    pinn.compile(opt)
    train(epochs,pinn,X_r,X_data,u_data)

#%%
    model = pinn.model
    model.save_weights('pinn.h5')
    N = 70
    fps = 5
    tspace = np.linspace(lb[0], ub[0], N + 1)

    plot1d(lb,ub,N,tspace,model,fps)

# %%
N = 200
fps = 5
tspace = np.linspace(lb[0], ub[0], N + 1)

def plot1dgrid(lb,ub,N,tspace,model):
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

plot1dgrid(lb,ub,N,tspace,model)      
# %%
