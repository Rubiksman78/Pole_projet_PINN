# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from time import time
from physics.initial import *
from models.model import *
from utils.plot import *
from physics.equation import *
from config import *
from tqdm import tqdm
# %%
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Training loop
# Nb de boucle du NN, modele, Valeurs du dataset, CI de X, ..., fonction reelle, N,dim,
# reduit le calcul par petit groupe, barre de progression, ratio de data utilisé
# pour validation/test/entrainement


def train(epochs, pinn, X_r, X_data, u_data, f_real, N, dimension, batch_size, render_bar=True, val_ratio=0.1, begin_from=80000):
    losses = []
    val_losses = []
    t0 = time()

    # Setting validation dataset size
    total_size = tf.shape(X_r)[0].numpy()
    val_size = (val_ratio * total_size).astype(int)
    if batch_size > total_size - val_size:
        print(total_size - val_size)
        raise ValueError("Batch size > train size")

    # Setting train and validation data
    X_data_test = tf.concat(X_data, axis=0)[-val_size:]
    X_test = tf.concat([X_r[-val_size:], X_data_test], axis=0)
    X_r, X_data, u_data = X_r[:-val_size], [x[:-val_size]
                                            for x in X_data], [x[:-val_size] for x in u_data]
    num_steps = (np.ceil(tf.shape(X_r)[0].numpy())/batch_size).astype(int)

    val_loss = pinn.test_step(X_test, f_real)
    val_losses.append(val_loss)

    if render_bar:
        progress_bar = tqdm(range(epochs+1))
    else:
        progress_bar = range(epochs+1)

    # Main training loop over epochs
    for i in progress_bar:
        loss = 0
        # Loop over all batches
        for j in range(num_steps):
            idx_i = j*batch_size
            idx_e = (j+1)*batch_size
            X_rj, X_dataj, u_dataj = X_r[idx_i:idx_e], tf.constant(np.array(X_data)[:, idx_i:idx_e]),\
                tf.constant(np.array(u_data)[:, idx_i:idx_e])
            loss_i, loss_b1, loss_b2, loss_r, lambda_b, lambda_bv, lambda_r = \
                pinn.train_step(X_rj, X_dataj, u_dataj,
                                i)  # Calling train step on batch
            loss_j = loss_i + loss_b1 + loss_b2 + loss_r
            loss += loss_j
        loss = loss / num_steps
        losses.append(loss)
        # Rendering training metrics
        if render_bar:
            progress_bar.set_description(f"Epoch {i}: Loss= {loss}")
            if (i+1) % 50 == 0:
                val_loss = pinn.test_step(X_test, f_real)
                val_losses.append(val_loss)
                #print(f"Epoch {i}: val_loss : {val_loss}")

        if (i+1) % 10000 == 0:
            print(f'It {i}: residual_loss = {loss_r}\
                    | initial_loss = {loss_i}\
                    | boundary_loss_x = {loss_b1}\
                    | boundary_loss_v = {loss_b2}\
                    | lambda_b = {lambda_b}\
                    | lambda_bv = {lambda_bv}\
                    | lambda_r = {lambda_r}')

        if (i+1) % 1000 == 0:
            # Change name file for another train
            pinn.model.save_weights('results/pinn200.h5')
        if (i+1) % 2000 == 0:
            if dimension == 1:
                plot1dgrid(lb, ub, N, pinn.model, i+begin_from)
        if (i+1) % 50 == 0:
            plot_curve(i, losses, val_losses, 'plot2.png')
    print('\nComputation time: {} seconds'.format(time()-t0))
    return losses, val_losses

# Multiple trainings for different number of points


def multi_train():
    times = []
    points = np.concatenate(
        (np.arange(0, 100, 10), np.arange(100, 1050, 50)), axis=0
    )
    for N_0 in points:
        config = define_config()
        c, a, dimension, tmin, tmax, xmin, xmax, N_b, N_r, N_0, lr, epochs = \
            config['c'],\
            config['a'],\
            config['dimension'],\
            config['tmin'],\
            config['tmax'],\
            config['xmin'],\
            config['xmax'],\
            N_0,\
            N_0,\
            N_0,\
            config['learning_rate'],\
            config['epochs']
        X_data, u_data, time_x, X_r = set_training_data(
            tmin, tmax, xmin, xmax, dimension, N_0, N_b, N_r)
        bound1 = [tmin] + [xmin for _ in range(dimension)]
        bound2 = [tmax] + [xmax for _ in range(dimension)]
        lb, ub = tf.constant(bound1, dtype=DTYPE), tf.constant(
            bound2, dtype=DTYPE)
        opt = keras.optimizers.Adam(learning_rate=lr)
        pinn = PINN(dimension+1, 1, dimension, ub, lb, c)
        pinn.compile(opt)
        t_0 = time()
        train(epochs, pinn, X_r, X_data, u_data, true_u,
              N=100, dimension=dimension, batch_size=32)
        times.append(time()-t_0)
    return times


# %%
if __name__ == '__main__':
    config = define_config()
    c, a, dimension, tmin, tmax, xmin, xmax, N_b, N_r, N_0, lr, epochs = \
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
    X_data, u_data, time_x, X_r = set_training_data(
        tmin, tmax, xmin, xmax, dimension, N_0, N_b, N_r)

    plot_training_points(dimension, time_x)

    bound1 = [tmin] + [xmin for _ in range(dimension)]
    bound2 = [tmax] + [xmax for _ in range(dimension)]
    lb, ub = tf.constant(bound1, dtype=DTYPE), tf.constant(bound2, dtype=DTYPE)
    plot1dgrid_real(lb, ub, 200, lambda x: true_u(x, a, c), 999999)
    opt = keras.optimizers.Adam(learning_rate=lr)
    hist = []
    pinn = PINN(dimension+1, 1, dimension, ub, lb, c)
    pinn.compile(opt)
    pinn.model.load_weights('results/pinn100.h5')
    batch_size_max = int(0.9*N_b)  # 30% of train dataset
    train(epochs, pinn, X_r, X_data, u_data, true_u, N=100,
          dimension=dimension, batch_size=batch_size_max)

    # Test
    model = pinn.model
    N = 70
    fps = 5
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1d(lb, ub, N, tspace, model, fps)
    N = 100
    tspace = np.linspace(lb[0], ub[0], N + 1)
    plot1dgrid(lb, ub, N, model, 0)
