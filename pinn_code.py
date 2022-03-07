#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as kl
from keras import models
import matplotlib.pyplot as plt
from time import time
import matplotlib.animation as animation
#%%
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
##Constantes à modifier (c:vitesse de l'onde,dimension:2D ou 3D,k : constante pour la gaussienne de l'onde à l'instant initial)
c = 1
dimension = 3
k = 1

### u est la fonction que l'on cherche à modéliser u(t,x) avec t réel (temps) et x un vecteur de R2 ou R3
#Initial condition
def u0(t,x):
    x = tf.expand_dims(x[:,i],axis=-1)
    z = x- c* t
    return tf.sin(z) * tf.exp(-(k*z)**2) #Ici c'est un sinus modulé avec une gaussienne mais à modifier

#Initial condition on derivative
def du0_dt(t,x):
    with tf.GradientTape() as tape: #module pour calculer des gradients
        tape.watch(t) #regarder la variable par rapport à laquelle on veut dériver
        u = u0(t,x) #on récupère la condition initiale 
    du_dt = tape.gradient(u,t) #on calcule les gradients de u par rapport à t (i.e. vitesse par exemple)
    return du_dt

#Boundary condition
def u_bound(t,x):
    n = x.shape[0]
    res = tf.zeros((n,1), dtype=DTYPE) #Ici c'est juste u=0 aux bords
    return res

#Residual of the PDE
def residual(t,x,u,u_t,u_x,u_tt,u_xx):
    return u_xx - (1/c**2) * u_tt #L'équation est d²u/dx²=(1/c²)*d²u/dt² donc on prend le résidu r=d²u/dx²-(1/c²)*d²u/dt² et on veut r -> 0 

#Number of points
N_0 = 100 #Nombre de points pour la condition initiale
N_b = 100 #Nombre de points pour la condition aux bords
N_r = 5000 #Nombre de points pour le résidu (donc à l'intérieur du domaine)

#Boundaries of the domain
tmin,tmax = 0.,1.0
xmin,xmax = -1.,1.
lb,ub = tf.constant([tmin,xmin],dtype=DTYPE),tf.constant([tmax,xmax],dtype=DTYPE) #Frontières basse et haute pour toute variable

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
X_data = [X_0,X_b,X_0] #Les points d'entrainement sont les points limites (bords et à l'instant initial)
u_data = [u_0,u_b,v_0] #Les données d'entrainement visées sont les valeurs de u en ces points

#Tracé des points d'entrainement
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(t_0,x_0[:,0],x_0[:,1],c=u_0, marker='X', vmin=-1, vmax=1)
ax.scatter(t_b,x_b[:,0],x_b[:,1],c=u_b, marker='X', vmin=-1, vmax=1)
ax.scatter(t_r,x_r[:,0],x_r[:,1],c='r', marker='.', alpha=0.1)
ax.set_xlabel('$t$')
ax.set_ylabel('$x1$')
ax.set_zlabel('$x2$')

ax.set_title('Positions of collocation points and boundary data')
plt.show()

#%%
bound1 = [tmin] + [xmin for _ in range(dimension)]
bound2 = [tmax] + [xmax for _ in range(dimension)]
lb,ub = tf.constant(bound1,dtype=DTYPE),tf.constant(bound2,dtype=DTYPE)

##Modèle allant de num_inputs(3 ou 4 en incluant le temps) -> num_outputs(1 car on veut un réel qui est u(t,x1,x2,(x3)))
def define_net(num_inputs,num_outputs):
    input = models.Input(shape=(num_inputs))
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0) #Normalisation des points en [-1,1]
    x = scaling_layer(input) 
    for neurons in [64,32,32,64]: #Ici 4 couches denses mais on peut en mettre moins
        x = kl.Dense(neurons,activation='relu',kernel_initializer='glorot_normal')(x)
    x = kl.Dense(num_outputs)(x)
    return models.Model(input,x)

##Obtention du résidu de l'équation en calculant les dérivées secondes (ou plutôt laplaciens dans R2 ou R3)
def get_r(model,X_r):#X_r est les points où l'on calcule le résidu
    with tf.GradientTape(persistent=True) as tape: 
        t = X_r[:,0] #t est la première composante
        x = [X_r[:,i] for i in range(1,dimension+1)] #x est le reste des composantes spatiales dans une liste

        tape.watch(t) #on regarde t
        for xi in x:
            tape.watch(xi) #on regarde chaque coordonnée spatiale

        u = model(tf.stack([t]+[tf.expand_dims(xi,axis=-1) for xi in x],axis=1)) #on obtient le résultat du modèle (u prédit) sur (t,x1,x2,(x3))

        u_t = tape.gradient(u,t) #On calcule les gradients de notre u prédit par rapport à t (du/dt)
        gradient_x = [tape.gradient(u,x[i]) for i in range(len(x))] #On calcule les gradients de u par rapport à chaque xi dans une liste (du/dxi)
        u_x = tf.reduce_sum(gradient_x,axis=0) #divergence selon x (somme de ces gradients)

    u_tt = tape.gradient(u_t,t) #On calcule la dérivée seconde par rapport à t (d(du/dt)/dt) = (d²u/dt²)
    double_gradients_x = [tape.gradient(gradient_x[i],x[i]) for i in range(len(x))] #Pareil pour chaque xi d(du/dxi)/dxi = d²u/dxi²
    u_xx = tf.reduce_sum(double_gradients_x,axis=0) #laplacien selon x
    del tape

    return residual(t,x,u,u_t,u_x,u_tt,u_xx) #On renvoie le résidu selon les dérivées qu'on vient de calculert

##La fonction de coût que notre réseau va chercher à minimiser
def compute_loss(model,X_r,X_data,u_data):
    res = get_r(model,X_r) #notre résidu
    phi_r = tf.reduce_mean(tf.square(res)) #MSE (mean squarred error = somme des erreurs²) sur le résidu
    loss = phi_r 
    for i in range(len(X_data)):
        u_pred = model(X_data[i]) #On prédit le résultat sur les points au bord
        loss += tf.reduce_mean(tf.square(u_data[i] - u_pred)) #On ajoute à notre coût Somme((u_bord_prédit-u_bord_réel)²)
    return loss

##train_step représente ce qu'on fait à chaque étape de l'entraînement
@tf.function
def train_step(model,X_r,X_data,u_data,opt):
    with tf.GradientTape() as tape:
        loss = compute_loss(model,X_r,X_data,u_data) #On calcule le coût par rapport aux données
    grads = tape.gradient(loss,model.trainable_variables) #On calcule les gradients par rapport à chaque paramètre du réseau
    opt.apply_gradients(zip(grads,model.trainable_variables)) #On applique la correction à chaque paramètre en fonction du gradient(cf descente de gradient)
    return loss

##Pour les paramètres d'entrainement
#lr : taux d'apprentissage
#opt : Optimiseur (+ simple : descente de gradient, Adam : version améliorée + efficace)
#epochs : le nombre de fois où l'on parcours tout le dataset
lr = 1e-3
opt = keras.optimizers.Adam(learning_rate=lr)
epochs = 5000
hist = []
t0 = time()
model = define_net(4,1)

#Boucle d'entrainement
def train():
    for i in range(epochs+1):
        loss = train_step(model,X_r,X_data,u_data,opt) #on récupère la loss après chaque train_step
        hist.append(loss.numpy()) 

        if i%500 == 0:
            print(f'It {i}: loss = {loss}') #On print la loss tous les 500 epochs

    print('\nComputation time: {} seconds'.format(time()-t0))

train() 
# %%

###Affichage du résultat dans une grille 2D ou 3D
N = 20
fps = 5
tspace = np.linspace(lb[0], ub[0], N + 1)
x1space = np.linspace(lb[1], ub[1], N + 1)
x2space = np.linspace(lb[2], ub[2], N + 1)
x3space = np.linspace(lb[2], ub[2], N + 1) #On échantillonne les composantes entre les bords avec N+1 points

T,X1, X2,X3 = np.meshgrid(tspace,x1space, x2space,x3space) #Découpage d'une grille de points
Xgrid = np.vstack([T.flatten(),X1.flatten(),X2.flatten(),X3.flatten()]).T

# Determine predictions of u(t, x)
upred = model(tf.cast(Xgrid,DTYPE))

U = upred.numpy().reshape(N+1,N+1,N+1,N+1)
z_array = np.zeros((N+1,N+1,N+1,N+1))
for i in range(N+1):
    z_array[:,:,:,i]= U[i]

def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    #plot[0] = ax.plot_surface(X1, X2, zarray[:,:,frame_number], cmap="magma")
    plot[0] = ax.scatter(X1, X2,X3, c=zarray[:,:,:,frame_number])

X1, X2,X3 = np.meshgrid(x1space, x2space,x3space)
fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot(111,projection='3d')
#plot = [ax.plot_surface(X1, X2, z_array[:,:,0], cmap='magma',rstride=1,cstride=1)]
plot = [ax.scatter(X1, X2,X3, c=z_array[:,:,:,0])]
ax.set_xlabel('$x1$')
ax.set_ylabel('$x2$')
ax.set_zlabel('$u_\\theta(x1,x2)$')
ax.set_title('SolutioPoln of Wave equation')

plt.rcParams['animation.ffmpeg_path'] = "C:/SAMUEL/ffmpeg-5.0-full_build/ffmpeg-5.0-full_build/bin/ffmpeg.exe" #Faut installer un truc sombre pour faire l'animation vidéo, pas important
anim = animation.FuncAnimation(fig,update_plot,N+1,fargs=(z_array,plot),interval=1000/fps)
fn = 'plot_surface_animation_funcanimation'
anim.save(fn+'.gif',writer='ffmpeg',fps=fps)
plt.show()
# %%
