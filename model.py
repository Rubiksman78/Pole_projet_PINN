import tensorflow as tf
from tensorflow import keras
import keras.layers as kl
from keras import models
from initial import *

##Modèle allant de num_inputs(3 ou 4 en incluant le temps) -> num_outputs(1 car on veut un réel qui est u(t,x1,x2,(x3)))
def define_net(num_inputs,num_outputs,ub,lb):
    input = models.Input(shape=(num_inputs))
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0) #Normalisation des points en [-1,1]
    x = scaling_layer(input) 
    for neurons in [32,256,32]: #Ici 4 couches denses mais on peut en mettre moins
        x = kl.Dense(neurons,activation='tanh')(x)
    x = kl.Dense(num_outputs)(x)
    return models.Model(input,x)

class PINN(models.Model):

    def __init__(self,num_inputs,num_outputs,dimension,ub,lb,c,**kwargs):
        super().__init__(**kwargs)
        self.model = define_net(num_inputs,num_outputs,ub,lb)
        self.dimension = dimension
        self.c = c

    def compile(self,opt):
        super().compile()
        self.opt = opt

    ##Obtention du résidu de l'équation en calculant les dérivées secondes (ou plutôt laplaciens dans R2 ou R3)
    @tf.function
    def get_r(self,X_r):#X_r est les points où l'on calcule le résidu
        with tf.GradientTape(persistent=True) as tape: 
            t = X_r[:,0] #t est la première composante
            x = [X_r[:,i] for i in range(1,self.dimension+1)] #x est le reste des composantes spatiales dans une liste

            tape.watch(t) #on regarde t
            for xi in x:
                tape.watch(xi) #on regarde chaque coordonnée spatiale

            u = self.model(tf.stack([t]+[xi for xi in x],axis=1)) #on obtient le résultat du modèle (u prédit) sur (t,x1,x2,(x3))

            u_t = tape.gradient(u,t) #On calcule les gradients de notre u prédit par rapport à t (du/dt)
            gradient_x = [tape.gradient(u,x[i]) for i in range(len(x))] #On calcule les gradients de u par rapport à chaque xi dans une liste (du/dxi)
            #u_x = tf.reduce_sum(gradient_x,axis=0) #divergence selon x (somme de ces gradients)

        u_tt = tape.gradient(u_t,t) #On calcule la dérivée seconde par rapport à t (d(du/dt)/dt) = (d²u/dt²)
        double_gradients_x = [tape.gradient(gradient_x[i],x[i]) for i in range(len(x))] #Pareil pour chaque xi d(du/dxi)/dxi = d²u/dxi²
        u_xx = tf.reduce_sum(double_gradients_x,axis=0) #laplacien selon x
        del tape

        return residual(u_tt,u_xx,self.c) #On renvoie le résidu selon les dérivées qu'on vient de calculer

    ##La fonction de coût que notre réseau va chercher à minimiser
    def compute_loss(self,X_r,X_data,u_data):
        res = self.get_r(X_r) #notre résidu
        phi_r = tf.reduce_mean(tf.square(res)) #MSE (mean squarred error = somme des erreurs²) sur le résidu
        loss = phi_r 
        for i in range(len(X_data)):
            u_pred = self.model(X_data[i]) #On prédit le résultat sur les points au bord
            loss += tf.reduce_mean(tf.square(u_data[i] - u_pred)) #On ajoute à notre coût Somme((u_bord_prédit-u_bord_réel)²)
        return loss

    ##train_step représente ce qu'on fait à chaque étape de l'entraînement
    @tf.function
    def train_step(self,X_r,X_data,u_data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X_r,X_data,u_data) #On calcule le coût par rapport aux données
        grads = tape.gradient(loss,self.model.trainable_variables) #On calcule les gradients par rapport à chaque paramètre du réseau
        self.opt.apply_gradients(zip(grads,self.model.trainable_variables)) #On applique la correction à chaque paramètre en fonction du gradient(cf descente de gradient)
        return loss
