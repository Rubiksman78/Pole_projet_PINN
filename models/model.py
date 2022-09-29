import tensorflow as tf
from tensorflow import keras
import keras.layers as kl
from keras import models
from physics.initial import *
from models.kernel_upgrade import *
import keras


def define_net(num_inputs, num_outputs, ub, lb, n_layers=8, n_neurons=512):
    """
    Neural network architecture for PINN
    Input : num_inputs = dimension of input vector
            num_outputs = dimension of output vector (scalar=1 here)
            ub,lb : upper bound and lower bound of vectorial domain
            n_layers : number of hidden layers
            n_neurons : number of neurons per hidden layer
    Ouput : Tensorflow model f: num_inputs -> num_outputs
    """
    input = keras.Input(shape=(num_inputs))
    # Normalisation des points en [-1,1]
    scaling_layer = tf.keras.layers.Lambda(
        lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
    x = scaling_layer(input)
    for _ in range(n_layers):
        x = kl.Dense(n_neurons, activation='tanh')(x)
        x = kl.BatchNormalization()(x)
    x = kl.Dense(num_outputs, activation='linear')(x)
    return models.Model(input, x)


class PINN(models.Model):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 dimension,
                 ub,
                 lb,
                 c,
                 init_weight=1,
                 bound_weight=1,
                 res_weight=1,
                 bound_speed_weight=1,
                 n_layers=6,
                 n_neurons=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = define_net(num_inputs, num_outputs,
                                ub, lb, n_layers, n_neurons)
        self.dimension = dimension
        self.c = c
        self.init_weight = init_weight
        self.bound_weight = bound_weight
        self.res_weight = res_weight
        self.bound_speed_weight = bound_speed_weight

    def compile(self, opt):
        super().compile()
        self.opt = opt

    def get_r(self, X_r):
        """
        Compute the residual of the PDE
        Input : X_r = residual points sampled to compute the residual
        Ouput : values of the residual over X_r
        """
        # Compute the first order gradient
        with tf.GradientTape(persistent=True) as tape2:
            X_r_unstacked = tf.unstack(X_r, axis=1)
            tape2.watch(X_r_unstacked)
            with tf.GradientTape(persistent=True) as tape1:
                x_stacked = tf.stack(X_r_unstacked, axis=1)
                tape1.watch(x_stacked)
                u = self.model(x_stacked)
            gradient_x = tape1.gradient(u, x_stacked)
            gradient_x_unstacked = tf.unstack(gradient_x, axis=1)

        # Compute second order gradient
        d2f_dx2 = []
        for df_dxi, xi in zip(gradient_x_unstacked, X_r_unstacked):
            d2f_dx2.append(tape2.gradient(df_dxi, xi))
        u_xx = tf.stack(d2f_dx2, axis=1)
        u_tt = u_xx[:, 0]
        u_xx = tf.reduce_sum([u_xx[:, i]
                              for i in range(1, self.dimension+1)], axis=0)

        return residual(X_r[:, 0], X_r[:, 1], gradient_x[:, 0], u_tt, u_xx, self.c)

    def compute_loss(self, X_r, X_data, u_data, use_r=True):
        """
        Compute the residual, boundary and initial losses
        Input : X_r = residual points
                X_data = boundary and initial points
                u_data = target value on X_data
        Output : losses on initial, boundary, boundary speed condition and mean residual 
        """
        if use_r:
            # Residual loss
            res = self.get_r(X_r)
            phi_r = tf.reduce_mean(tf.square(res))
        else:
            phi_r = 0
        loss_i = 0
        loss_b1 = 0
        loss_b2 = 0
        for i in range(len(X_data)):
            if i == 1:
                # Loss boundary condition
                u_pred = self.model(X_data[i])
                loss_b1 += tf.reduce_mean(tf.square(u_data[i]-u_pred))
            elif i == 0:
                # Loss initial condition
                u_pred = self.model(X_data[i])
                loss_i += tf.reduce_mean(tf.square(u_data[i]-u_pred))
            else:
                # Loss initial speed condition
                with tf.GradientTape() as tape:
                    t = X_data[i][:, 0]
                    tape.watch(t)
                    x = [X_data[i][:, j] for j in range(1, self.dimension+1)]
                    u_pred = self.model(tf.stack([t]+[xi for xi in x], axis=1))
                    v_pred = tape.gradient(u_pred, t)
                loss_b2 += tf.reduce_mean(
                    tf.square(u_data[i]-tf.expand_dims(v_pred, axis=-1)))
        return loss_i, loss_b1, loss_b2, phi_r

    def train_step(self, X_r, X_data, u_data, i, use_kernel=True):
        with tf.GradientTape() as tape:
            # Computing lambda coefficients
            if use_kernel and (i+1) % 500 == 0:
                Ju = compute_Ju(
                    tf.concat([X_data[0], X_data[1]], axis=0), self.model)
                Jr = compute_Jr(X_r, self.model, self.c, self.dimension)
                Jut = compute_Ju(X_data[2], self.model)
                Kuu = compute_Ki(tf.concat([X_data[0], X_data[1]], axis=0), Ju)
                Krr = compute_Ki(X_r, Jr)
                Kut = compute_Ki(X_data[2], Jut)
                lambda_b = compute_eigen(Kuu, Krr, Kut, name='Kuu')
                lambda_r = compute_eigen(Kuu, Krr, Kut, name='Krr')
                lambda_bv = compute_eigen(Kuu, Krr, Kut, name='Kut')
                self.init_weight = lambda_b
                self.bound_weight = lambda_b
                self.res_weight = lambda_r
                self.bound_speed_weight = lambda_bv

            lambda_b = self.init_weight
            lambda_r = self.res_weight
            lambda_bv = self.bound_speed_weight
            # Summing losses with their lambda weights
            loss_i, loss_b1, loss_b2, loss_r = self.compute_loss(
                X_r, X_data, u_data)  # On calcule le coût par rapport aux données
            loss = self.init_weight * loss_i + \
                self.bound_weight * loss_b1 + \
                self.bound_speed_weight * loss_b2 + \
                self.res_weight * loss_r
        # Gradient optimization
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_i, loss_b1, loss_b2, loss_r, lambda_b, lambda_bv, lambda_r

    def test_step(self, x, f_real):
        # To validate the model with the real prediction
        y_real = f_real(x)
        y_pred = self.model(x)
        loss = tf.keras.losses.MeanSquaredError()(y_real, y_pred)
        return loss
