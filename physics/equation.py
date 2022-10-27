import tensorflow as tf
import numpy as np

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


def u0(t, x):
    """
    Input: t,x = time and space points for initial condition
    Output: u_0(t,x) = solution on initial condition
    """
    return t + 1*(tf.sin(np.pi*x) + 0.5*tf.sin(4*np.pi*x))
    # return t + tf.sin(np.pi*x) * tf.exp(-x*x/4)


def v0(t, x, dimension):
    """
    Input: t,x = time and space points for speed initial condition
           dimension = space dimension for model
    Output: v_0(t,x) = speed on initial condition
    """
    n = x.shape[0]
    res = tf.zeros((n, dimension), dtype=DTYPE)
    return res


def u_bound(t, x, dimension):
    """
    Input: t,x = time and space points for boundary condition
           dimension = space dimension for model
    Output: u_b(t,x) = solution on boundary condition
    """
    n = x.shape[0]
    res = tf.zeros((n, dimension), dtype=DTYPE)
    return res


def residual(t, x, u_t, u_tt, u_xx, c):
    """
    Input: t,x and derivatives of u
    Ouput : residual of PDE
    """
    return u_tt - (c**2)*u_xx


def true_u(x, a=0.5, c=2):
    """
    Input: x and hyperparameters
    Ouput : true forward solution 
    """
    t = x[:, 0]
    x = x[:, 1]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
        a * np.sin(2 * c * np.pi * x) * np.cos(4 * c * np.pi * t)
