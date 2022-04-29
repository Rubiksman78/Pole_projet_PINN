import tensorflow as tf
import numpy as np
from initial import *

def compute_Kuu(x,Ju):
    D = tf.shape(x)[0]
    Kuu = tf.zeros((D,D))
    for x in Ju:
        if x == None:
            continue
        else:
            J = tf.reshape(x,shape=(D,-1))
            K = tf.matmul(J,J,transpose_b=True)
            Kuu = Kuu + K
    return Kuu

def compute_Krr(x,Jr):
    D = tf.shape(x)[0]
    Krr = tf.zeros((D,D))
    for x in Jr:
        if x == None:
            continue
        else:                
            J = tf.reshape(x,shape=(D,-1))
            K = tf.matmul(J,J,transpose_b=True)
            Krr = Krr + K
    return Krr

def compute_K(Ju,Jr):
    a = tf.concat([Ju,Jr],axis=1)
    b = tf.concat([tf.transpose(Ju),tf.transpose(Jr)],axis=0)
    return tf.linalg.matmul(a, b)

@tf.function
def compute_Ju(x,model):
    with tf.GradientTape() as tape:
        u = model(x)
        theta = model.weights
    Ju = tape.jacobian(u,theta)
    return Ju

@tf.function
def compute_Jr(x,model,c,dimension):
    with tf.GradientTape() as tape3:
        tape3.watch(model.trainable_variables)
        theta = model.weights
        with tf.GradientTape(persistent=True) as tape2:
            X_r_unstacked = tf.unstack(x, axis=1)
            tape2.watch(X_r_unstacked)
            with tf.GradientTape(persistent=True) as tape1: 
                x_stacked = tf.stack(X_r_unstacked, axis=1)
                tape1.watch(x_stacked)
                u = model(x_stacked)
                gradient_x = tape1.gradient(u,x_stacked)
                gradient_x_unstacked = tf.unstack(gradient_x,axis=1)
        d2f_dx2 = []
        for df_dxi,xi in zip(gradient_x_unstacked, X_r_unstacked):
            d2f_dx2.append(tape2.gradient(df_dxi, xi))
        u_xx = tf.stack(d2f_dx2, axis=1)
        u_tt = u_xx[:,0]
        u_xx = tf.reduce_sum([u_xx[:,i] for i in range(1,dimension+1)],axis=0)
        res = residual(u_tt,u_xx,c)
    Jr = tape3.jacobian(res,theta)
    return Jr

def compute_eigen(Ku,Kr,iskr = True):
    trace_K = tf.linalg.trace(Ku) + tf.linalg.trace(Kr)
    if iskr:
        return trace_K/tf.linalg.trace(Kr)
    else:
        return trace_K/tf.linalg.trace(Ku)
