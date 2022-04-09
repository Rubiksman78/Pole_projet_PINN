import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

### u est la fonction que l'on cherche à modéliser u(t,x) avec t réel (temps) et x un vecteur de R2 ou R3
def u0(t,x):
    return t + 3*tf.sin(np.pi*x/2) #Ici c'est un sinus modulé avec une gaussienne mais à modifier

#Initial condition on derivative
def du0_dt(t,x):
    """
    with tf.GradientTape() as tape: #module pour calculer des gradients
        tape.watch(t) #regarder la variable par rapport à laquelle on veut dériver
        u = u0(t,x) #on récupère la condition initiale 
    du_dt = tape.gradient(u,t) #on calcule les gradients de u par rapport à t (i.e. vitesse par exemple)
    """
    return t + 0.5*tf.sin(np.pi*x) #Ici c'est un sinus modulé avec une gaussienne mais à modifier

#Boundary condition
def u_bound(t,x):
    n = x.shape[0]
    res = tf.zeros((n,1), dtype=DTYPE) #Ici c'est juste u=0 aux bords
    return res

#Residual of the PDE
def residual(u_tt,u_xx,c):
    return u_xx - (1/c**2) * u_tt #L'équation est d²u/dx²=(1/c²)*d²u/dt² donc on prend le résidu r=d²u/dx²-(1/c²)*d²u/dt² et on veut r -> 0 

def domain_boundaries(tmin,tmax,xmin,xmax):
    #Boundaries of the domain
    lb,ub = tf.constant([tmin,xmin],dtype=DTYPE),tf.constant([tmax,xmax],dtype=DTYPE) #Frontières basse et haute pour toute variable
    return lb,ub 

def set_training_data(tmin,tmax,xmin,xmax,dimension):
    #Number of points
    N_0 = 1000 #Nombre de points pour la condition initiale
    N_b = 1000 #Nombre de points pour la condition aux bords
    N_r = 10000 #Nombre de points pour le résidu (donc à l'intérieur du domaine)

    lb,ub = domain_boundaries(tmin,tmax,xmin,xmax)

    #Initial conditions
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0] #On fixe t à la valeur lb[0] donc t_0
    x_0 = tf.random.uniform((N_0,dimension), lb[1], ub[1], dtype=DTYPE) #On prend x_0 suivant une loi uniforme sur R2 ou R3
    X_0 = t_0
    for i in range(dimension):
        X_0 = tf.concat([X_0, tf.expand_dims(x_0[:,i],axis=-1)], axis=1) #On prend X_0 = (t_0,x_0) qui sera celui utilisé pour donner directement toutes les variables au réseau de neurones
    u_0 = u0(t_0,x_0) #Condition initiale sur u 
    v_0 = du0_dt(t_0,x_0) #Condition initiale sur du/dt

    #Boundary conditions
    t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE) #On prend t suivant une loi uniforme
    x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,dimension), 0.5, dtype=DTYPE) #x_b est situé soit en lb[1] soit en ub[1] (les bords du domaine)
    X_b = t_b
    for i in range(dimension):
        X_b = tf.concat([X_b, tf.expand_dims(x_b[:,i],axis=-1)], axis=1) #Pareil on prend X_b = (t_b,x_b) pour le modèle
    u_b = u_bound(t_b,x_b)

    #Residual of the equation
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE) 
    x_r = tf.random.uniform((N_r,dimension), lb[1], ub[1], dtype=DTYPE) #On prend t et x uniforme pour le résidu dans le domaine
    X_r = t_r
    for i in range(dimension):
        X_r = tf.concat([X_r, tf.expand_dims(x_r[:,i],axis=-1)], axis=1) #Idem X_r = (t_r,x_r) pour le modèle

    #Training data
    X_data = [X_0,X_0,X_b] #Les points d'entrainement sont les points limites (bords et à l'instant initial)
    u_data = [u_0,v_0,u_b] #Les données d'entrainement visées sont les valeurs de u en ces points

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