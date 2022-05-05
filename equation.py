import tensorflow as tf
import numpy as np

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
### u est la fonction que l'on cherche à modéliser u(t,x) avec t réel (temps) et x un vecteur de R2 ou R3
def u0(t,x):
    return t + 1*(tf.sin(np.pi*x) + 0.5*tf.sin(4*np.pi*x))
    #return tf.sin(np.pi*x)

#Speed intial condition
def v0(t,x,dimension):
    n = x.shape[0]
    res = tf.zeros((n,dimension), dtype=DTYPE) #Ici c'est juste v=0 aux bords
    return res

#Boundary condition
def u_bound(t,x,dimension):
    n = x.shape[0]
    res = tf.zeros((n,dimension), dtype=DTYPE) #Ici c'est juste u=0 aux bords
    return res

#Residual of the PDE
def residual(t,x,u_t,u_tt,u_xx,c):
    #return u_t - u_xx + tf.exp(-t)*(tf.sin(np.pi*x)-np.pi**2*np.sin(np.pi*x))
    return u_tt - (c**2)*u_xx #L'équation est d²u/dx²=(1/c²)*d²u/dt² donc on prend le résidu r=d²u/dx²-(1/c²)*d²u/dt² et on veut r -> 0 

#True solution to compare with prediction
def true_u(x,a=0.5,c=2):
    t = x[:,0]
    x = x[:,1]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)