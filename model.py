import tensorflow as tf
from tensorflow import keras
import keras.layers as kl
from keras import models
from initial import *
from kernel_upgrade import *

##Modèle allant de num_inputs(3 ou 4 en incluant le temps) -> num_outputs(1 car on veut un réel qui est u(t,x1,x2,(x3)))
def define_net(num_inputs,num_outputs,ub,lb,n_layers = 8, n_neurons = 512):
    input = models.Input(shape=(num_inputs))
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0) #Normalisation des points en [-1,1]
    x = scaling_layer(input) 
    #x = kl.Dense(n_neurons,activation='tanh')(input)
    for _ in range(n_layers): #Ici 4 couches denses mais on peut en mettre moins
        x = kl.Dense(n_neurons,activation='tanh')(x)
        x = kl.BatchNormalization()(x)
        #x = kl.GRU(neurons,return_sequences=True)(x)
    x = kl.Dense(num_outputs,activation='linear')(x)
    return models.Model(input,x)

class PINN(models.Model):
    def __init__(self,
    num_inputs,
    num_outputs,
    dimension,
    ub,
    lb,
    c,
    init_weight = 1,
    bound_weight = 1,
    res_weight = 1,
    bound_speed_weight = 1,
    **kwargs):
        super().__init__(**kwargs)
        self.model = define_net(num_inputs,num_outputs,ub,lb)
        self.dimension = dimension
        self.c = c
        self.init_weight = init_weight
        self.bound_weight = bound_weight
        self.res_weight = res_weight
        self.bound_speed_weight = bound_speed_weight

    def compile(self,opt):
        super().compile()
        self.opt = opt

    ##Obtention du résidu de l'équation en calculant les dérivées secondes (ou plutôt laplaciens dans R2 ou R3)
    def get_r(self,X_r):#X_r est les points où l'on calcule le résidu
        with tf.GradientTape(persistent=True) as tape2:
            X_r_unstacked = tf.unstack(X_r, axis=1)
            tape2.watch(X_r_unstacked)
            with tf.GradientTape(persistent=True) as tape1: 
                x_stacked = tf.stack(X_r_unstacked, axis=1)
                tape1.watch(x_stacked)
                u = self.model(x_stacked) #on obtient le résultat du modèle (u prédit) sur (t,x1,x2,(x3))
            gradient_x = tape1.gradient(u,x_stacked)
            gradient_x_unstacked = tf.unstack(gradient_x,axis=1)
    
        d2f_dx2 = []
        for df_dxi,xi in zip(gradient_x_unstacked, X_r_unstacked):
            d2f_dx2.append(tape2.gradient(df_dxi, xi))
        u_xx = tf.stack(d2f_dx2, axis=1)
        u_tt = u_xx[:,0]
        u_xx = tf.reduce_sum([u_xx[:,i] for i in range(1,self.dimension+1)],axis=0)
        return residual(X_r[:,0],X_r[:,1],gradient_x[:,0],u_tt,u_xx,self.c) #On renvoie le résidu selon les dérivées qu'on vient de calculer

    ##La fonction de coût que notre réseau va chercher à minimiser
    def compute_loss(self,X_r,X_data,u_data):
        res = self.get_r(X_r) #notre résidu
        phi_r = tf.reduce_mean(tf.square(res)) #MSE (mean squarred error = somme des erreurs²) sur le résidu
        loss_i = 0
        loss_b1 = 0
        loss_b2 = 0
        for i in range(len(X_data)):
            if i == 1:
                u_pred = self.model(X_data[i]) #On prédit le résultat sur les points au bord
                loss_b1 += tf.reduce_mean(tf.square(u_data[i]-u_pred))
            elif i == 0:
                u_pred = self.model(X_data[i]) #On prédit le résultat sur les points au bord
                loss_i += tf.reduce_mean(tf.square(u_data[i]-u_pred)) #On ajoute à notre coût Somme((u_bord_prédit-u_bord_réel)²)
            else:
                with tf.GradientTape() as tape: 
                    t = X_data[i][:,0]
                    tape.watch(t)
                    x = [X_data[i][:,j] for j in range(1,self.dimension+1)]
                    u_pred = self.model(tf.stack([t]+[xi for xi in x],axis=1)) #On prédit le résultat sur les points au bord
                    v_pred = tape.gradient(u_pred,t) 
                loss_b2 += tf.reduce_mean(tf.square(u_data[i]-tf.expand_dims(v_pred,axis=-1)))
        return loss_i,loss_b1,loss_b2,phi_r

    ##train_step représente ce qu'on fait à chaque étape de l'entraînement
    def train_step(self,X_r,X_data,u_data,i,use_kernel = False):
        with tf.GradientTape() as tape:
            if use_kernel and (i+1) % 100 ==0:
                Ju = compute_Ju(tf.concat([X_data[0],X_data[1]],axis=0),self.model)
                Jr = compute_Jr(X_r,self.model,self.c,self.dimension)
                Jut = compute_Ju(X_data[2],self.model)
                Kuu = compute_Ki(tf.concat([X_data[0],X_data[1]],axis=0),Ju)
                Krr = compute_Ki(X_r,Jr)
                Kut = compute_Ki(X_data[2],Jut)
                lambda_b = compute_eigen(Kuu,Krr,Kut,name='Kuu')
                lambda_r = compute_eigen(Kuu,Krr,Kut,name='Krr')
                lambda_bv = compute_eigen(Kuu,Krr,Kut,name='Kut')
                self.init_weight = lambda_b
                self.bound_weight = lambda_b
                self.res_weight = lambda_r
                self.bound_speed_weight = lambda_bv

            lambda_b = self.init_weight
            lambda_r = self.res_weight
            lambda_bv = self.bound_speed_weight
            loss_i,loss_b1,loss_b2,loss_r = self.compute_loss(X_r,X_data,u_data) #On calcule le coût par rapport aux données
            loss = self.init_weight * loss_i + \
                self.bound_weight * loss_b1 + \
                self.bound_speed_weight * loss_b2 + \
                self.res_weight * loss_r

        grads = tape.gradient(loss,self.model.trainable_variables) #On calcule les gradients par rapport à chaque paramètre du réseau
        self.opt.apply_gradients(zip(grads,self.model.trainable_variables)) #On applique la correction à chaque paramètre en fonction du gradient(cf descente de gradient)
        return loss_i,loss_b1,loss_b2,loss_r,lambda_b,lambda_bv,lambda_r

    def test_step(self,x,f_real):
        y_real = f_real(x)
        y_pred = self.model(x)
        loss = tf.keras.losses.MeanSquaredError()(y_real,y_pred)
        return loss