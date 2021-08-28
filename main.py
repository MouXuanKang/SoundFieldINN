"""
@Title:
    Solving Helmholtz equation with PDDO-PINNs.
    And This is model function
@author:
    Xu Liang
    Department of Underwater Acoustic
    Harbin Engineering University
    2013053118@hrbeu.edu.cn

Created on 2021
"""
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import pickle
import pathlib
import time
import matplotlib.pyplot as plt
from sciann import Variable, Functional, Data, SciModel, PDE, Parameter
import tensorflow.keras.callbacks as callbacks


def data_tmp(num=10000, random=True):
    # load .pickle
    with open('Data/G00.pickle', 'rb') as f:
        Family_Node_G00 = pickle.load(f)
    f.close()
    with open('Data/G01.pickle', 'rb') as f:
        Family_Node_G01 = pickle.load(f)
    f.close()
    with open('Data/G10.pickle', 'rb') as f:
        Family_Node_G10 = pickle.load(f)
    f.close()
    with open('Data/G02.pickle', 'rb') as f:
        Family_Node_G02 = pickle.load(f)
    f.close()
    with open('Data/G20.pickle', 'rb') as f:
        Family_Node_G20 = pickle.load(f)
    f.close()
    with open('Data/FamilyRZ.pickle', 'rb') as f:
        Family_Node_R = pickle.load(f)
        Family_Node_Z = pickle.load(f)
    f.close()
    with open('Data/RePImP.pickle', 'rb') as f:
        Family_Node_Rep = pickle.load(f)
        Family_Node_Imp = pickle.load(f)
    f.close()
    # load .mat
    data = loadmat('Data/cylinder_pre_c0_w0.mat')

    Re_P_star = data['ReP_star'][:-1, :].T
    Im_P_star = data['ImP_star'][:-1, :].T
    Rho_star = data['rho_star'][1:-1, :]
    c_star = data['c_star'][1:-1, :]
    omega = data['omega_star']
    # FamilyList = data['Family_list']
    FamilySize = data['Family_size'][0]
    # c0 = data['c0']
    R = data['R']  # R x 1
    Z = data['Z']  # Z x 1
    # shape
    nr = R.shape[1]
    nz = Z.shape[1]
    horizont = (FamilySize[0] + FamilySize[1] + 1) * (FamilySize[2] + FamilySize[3] + 1)

    # PI = 3.1415926
    k_star = omega / c_star
    #
    if random:
        idx = np.random.choice((nr - 1) * (nz - 1), num, replace=False)
    else:
        idx = np.arange(0, (nr - 1) * (nz - 1))
    G00_train = []
    G02_train = []
    G20_train = []
    R_train = []
    Z_train = []
    Rep_train = []
    Imp_train = []
    for i, val in enumerate(idx):
        idx_r = val % nr
        idx_z = val // nr
        G00_train.append(Family_Node_G00[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        G02_train.append(Family_Node_G02[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        G20_train.append(Family_Node_G20[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        R_train.append(Family_Node_R[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        Z_train.append(Family_Node_Z[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        Rep_train.append(Family_Node_Rep[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())
        Imp_train.append(Family_Node_Imp[idx_z, idx_r, :, :].flatten().astype(np.float64).tolist())

    k_train = k_star.flatten()[idx, None]
    rho_train = Rho_star.flatten()[idx, None]
    Rep_target = Re_P_star.flatten()[idx, None]
    Imp_target = Im_P_star.flatten()[idx, None]
    return k_train, rho_train, Rep_target, Imp_target, G00_train, G02_train, G20_train, \
           R_train, Z_train, Rep_train, Imp_train, horizont


if __name__ == "__main__":
    # prepare data
    k_train, rho_train, Re_p_target, Im_p_target, G00_train, G02_train, G20_train, r_train, z_train, \
    Re_p_train, Im_p_train, horizont = data_tmp()

    # flag
    # IsTrain = False
    IsTrain = True
    k0_train = 150.0 * 3.1415926 / 1500.0
    # Variables and Fields
    r = Variable("r", units=horizont, dtype='float64')
    z = Variable("z", units=horizont, dtype='float64')
    k = Variable("k", units=horizont, dtype='float64')
    G00 = Variable("G00", units=horizont, dtype='float64')
    G02 = Variable("G02", units=horizont, dtype='float64')
    G20 = Variable("G20", units=horizont, dtype='float64')
    layers = 4 * [49]
    p_real = Functional("p_real", [r, z, k], layers, 'tanh')
    p_imag = Functional("p_imag", [r, z, k], layers, 'tanh')

    # Define constrains
    d1 = Data(p_real)
    d2 = Data(p_imag)
    d3 = Data(k)
    d4 = Data(G00)
    d5 = Data(G02)
    d6 = Data(G20)
    # d4 = Data(k0)
    c1 = PDE(k ** 2 * tf.reduce_sum(tf.matmul(G00, p_real), axis=1) +
             tf.reduce_sum(tf.matmul(G20, p_real), axis=1) +
             tf.reduce_sum(tf.matmul(G02, p_real)), axis=1)
    c2 = PDE(k ** 2 * tf.reduce_sum(tf.matmul(G00, p_imag), axis=1) +
             tf.reduce_sum(tf.matmul(G20, p_imag), axis=1) +
             tf.reduce_sum(tf.matmul(G02, p_imag)), axis=1)

    # data rename
    data_d1 = Re_p_train
    data_d2 = Im_p_train
    data_d3 = k_train
    data_d4 = G00_train
    data_d5 = G20_train
    data_d6 = G02_train
    # data_d4 = k0_train
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    # constraints rename
    input_ = [r, z, k, G00, G02, G20]
    input_data = [r_train, z_train, k_train, G00_train, G02_train, G20_train]
    cons_ = [d1, d2, d3, d4, d5, d6, c1, c2]
    cons_data = [data_d1, data_d2, data_d3, data_d4, data_d5, data_d6, data_c1, data_c2]

    model = SciModel(input_, cons_)

    # callbacks
    current_file_path = pathlib.Path(__file__).parents[0]
    checkpoint_filepath = str(current_file_path.joinpath('callbacks/pddo.ckpt'))
    save_path = str(current_file_path.joinpath('callbacks/PDDO/Helmholtz2D.hdf5'))
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True
    )
    if IsTrain:
        t = time.time()
        history = model.train(
            input_data,
            cons_data,
            epochs=1e3,
            batch_size=49,
            adaptive_weights={"method": "NTK", "freq": 100},
            shuffle=True,
            learning_rate=1e-4,
            reduce_lr_after=10,
            stop_loss_value=1e-8,
            callbacks=[model_checkpoint_callback]
        )
        t = time.time() - t
        model.save_weights(save_path)
        fig1 = plt.figure(1)
        plt.semilogy(history.history['loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('figures/fig1_loss_history.png')
    else:
        # load the best performing model
        model.load_weights(save_path)
