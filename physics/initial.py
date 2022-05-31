import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from physics.equation import *

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

def domain_boundaries(tmin,tmax,xmin,xmax):
    #Boundaries of the domain
    lb,ub = tf.constant([tmin,xmin],dtype=DTYPE),tf.constant([tmax,xmax],dtype=DTYPE) #Frontières basse et haute pour toute variable
    return lb,ub 

def set_training_data(tmin,tmax,xmin,xmax,dimension,N_0,N_b,N_r,speed=True):
    lb,ub = domain_boundaries(tmin,tmax,xmin,xmax)

    #Initial conditions
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0] #On fixe t à la valeur lb[0] donc t_0
    x_0 = tf.random.uniform((N_0,dimension), lb[1], ub[1], dtype=DTYPE) #On prend x_0 suivant une loi uniforme sur R2 ou R3
    X_0 = t_0
    for i in range(dimension):
        X_0 = tf.concat([X_0, tf.expand_dims(x_0[:,i],axis=-1)], axis=1) #On prend X_0 = (t_0,x_0) qui sera celui utilisé pour donner directement toutes les variables au réseau de neurones
    u_0 = u0(t_0,x_0) #Condition initiale sur u
    
    #Initial_speed
    v_0 = v0(t_0,x_0,dimension) #Condition initiale sur u

    #Boundary conditions
    t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE) #On prend t suivant une loi uniforme
    x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,dimension), 0.5, dtype=DTYPE) #x_b est situé soit en lb[1] soit en ub[1] (les bords du domaine)
    X_b = t_b
    for i in range(dimension):
        X_b = tf.concat([X_b, tf.expand_dims(x_b[:,i],axis=-1)], axis=1) #Pareil on prend X_b = (t_b,x_b) pour le modèle
    u_b = u_bound(t_b,x_b,dimension)

    #Residual of the equation
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE) 
    x_r = tf.random.uniform((N_r,dimension), lb[1], ub[1], dtype=DTYPE) #On prend t et x uniforme pour le résidu dans le domaine
    X_r = t_r
    for i in range(dimension):
        X_r = tf.concat([X_r, tf.expand_dims(x_r[:,i],axis=-1)], axis=1) #Idem X_r = (t_r,x_r) pour le modèle

    #Training data
    if speed:
        X_data = [X_0,X_b,X_0] #Les points d'entrainement sont les points limites (bords et à l'instant initial)
        u_data = [u_0,u_b,v_0] #Les données d'entrainement visées sont les valeurs de u en ces points
    else:
        X_data = [X_0,X_b]
        u_data = [u_0,u_b] 
    time_x = [t_0,t_b,t_r,x_0,x_b,x_r,u_0,u_b]
    return X_data,u_data,time_x,X_r

#Tracé des points d'entrainement
def plot_training_points(dimension,time_x):
    t_0,t_b,t_r,x_0,x_b,x_r,u_0,u_b = time_x
    if dimension == 1:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.scatter(t_0,x_0[:,0],c=u_0, marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b,x_b[:,0],c=u_b, marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r,x_r[:,0],c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
    if dimension == 2:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(t_0,x_0[:,0],x_0[:,1],c=u_0[:,0], marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b,x_b[:,0],x_b[:,1],c=u_b[:,0], marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r,x_r[:,0],x_r[:,1],c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
        ax.set_zlabel('$x2$')
    if dimension == 3:
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(t_0,x_0[:,0],x_0[:,1],c=u_0[:,0], marker='X', vmin=-1, vmax=1)
        ax.scatter(t_b,x_b[:,0],x_b[:,1],c=u_b[:,0], marker='X', vmin=-1, vmax=1)
        ax.scatter(t_r,x_r[:,0],x_r[:,1],c='r', marker='.', alpha=0.1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x1$')
        ax.set_zlabel('$x2$')

    ax.set_title('Positions of collocation points and boundary data')
    plt.show()
