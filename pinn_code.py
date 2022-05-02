#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from initial import *
from model import *
from plot import * 
from equation import *
from config import *
#%%
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

#Boucle d'entrainement
def train(epochs,pinn,X_r,X_data,u_data,f_real,N,dimension):
    hist = []
    t0 = time()
    X_test = tf.random.shuffle(X_r)[:1000]
    for i in range(epochs+1):
        loss_i,loss_b1,loss_b2,loss_r,lambda_b,lambda_bv,lambda_r = \
            pinn.train_step(X_r,X_data,u_data,i) #on récupère la loss après chaque train_step
        loss = loss_i + loss_b1 + loss_b2 + loss_r 
        hist.append(loss.numpy()) 
        if (i+1) % 1000 == 0:
            """
            print(f'It {i}: residual_loss = {loss_r}\
                | initial_loss = {loss_i}\
                | boundary_loss_x = {loss_b1}\
                | boundary_loss_v = {loss_b2}\
                | lambda_b = {lambda_b}\
                | lambda_bv = {lambda_bv}\
                | lambda_r = {lambda_r}')
            """
            print(f"It {i}: loss = {loss}") #On print la loss tous les 500 epochs
        if (i+1) % 1000 == 0:
            val_loss = pinn.test_step(X_test,f_real)
            print(f"It {i}: val_loss : {val_loss}")
        if (i+1) % 2000 ==0:
            pinn.model.save_weights('pinn2.h5')
        if (i+1) % 2000 ==0:
            if dimension == 1:
                plot1dgrid(lb,ub,N,pinn.model,i)
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
    config = define_config()
    c,a,dimension,tmin,tmax,xmin,xmax,N_b,N_r,N_0,lr,epochs = \
            config['c'],\
            config['a'],\
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
    X_data,u_data,time_x,X_r = set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r)

    #plot_training_points(dimension,time_x)

    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)
    #plot1dgrid(lb,ub,N,tspace,lambda x: true_u(x,a,c))
    opt = keras.optimizers.Adam(learning_rate=lr)
    hist = []
    pinn = PINN(dimension+1,1,dimension,ub,lb,c)
    pinn.compile(opt)
    #pinn.model.load_weights('pinn1.h5')
    train(epochs,pinn,X_r,X_data,u_data,true_u,N=50,dimension=dimension)

#%%
    model = pinn.model
    N = 70
    fps = 5
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1d(lb,ub,N,tspace,model,fps)
    N = 200
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1dgrid(lb,ub,N,model,0)      
# %%
