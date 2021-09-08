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
from sciann import Variable, Functional, Data, SciModel, PDE, Field, dot
import tensorflow.keras.callbacks as callbacks
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm


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
    with open('Data/FamilyRZC.pickle', 'rb') as f:
        Family_Node_R = pickle.load(f)
        Family_Node_Z = pickle.load(f)
        Family_Node_C = pickle.load(f)
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
    G00_loop = []
    G02_loop = []
    G20_loop = []
    R_loop = []
    Z_loop = []
    C_loop = []
    Rep_loop = []
    Imp_loop = []
    for i, val in enumerate(idx):
        idx_r = val % nr
        idx_z = val // nr
        G00_loop.append(Family_Node_G00[idx_z, idx_r, :, :].flatten())
        G02_loop.append(Family_Node_G02[idx_z, idx_r, :, :].flatten())
        G20_loop.append(Family_Node_G20[idx_z, idx_r, :, :].flatten())
        R_loop.append(Family_Node_R[idx_z, idx_r, :, :].flatten())
        Z_loop.append(Family_Node_Z[idx_z, idx_r, :, :].flatten())
        C_loop.append(Family_Node_C[idx_z, idx_r, :, :].flatten())
        Rep_loop.append(Family_Node_Rep[idx_z, idx_r, :, :].flatten())
        Imp_loop.append(Family_Node_Imp[idx_z, idx_r, :, :].flatten())

    G00_train = np.array(G00_loop)
    G20_train = np.array(G20_loop)
    G02_train = np.array(G02_loop)
    R_train = np.array(R_loop)
    Z_train = np.array(Z_loop)
    C_train = np.array(C_loop)
    Rep_train = np.array(Rep_loop)
    Imp_train = np.array(Imp_loop)
    # k_train = k_star.flatten()[idx, None]
    rho_train = Rho_star.flatten()[idx, None]
    Rep_target = Re_P_star.flatten()[idx, None]
    Imp_target = Im_P_star.flatten()[idx, None]

    return C_train, rho_train, Rep_target, Imp_target, G00_train, G02_train, G20_train, \
           R_train, Z_train, Rep_train, Imp_train, horizont


if __name__ == "__main__":
    # prepare data
    c_train, rho_train, Re_p_target, Im_p_target, G00_train, G02_train, G20_train, r_train, z_train, \
    Re_p_train, Im_p_train, horizont = data_tmp()

    # flag
    # IsTrain = False
    IsTrain = True

    # Variables and Fields
    r = Variable("r", units=horizont)
    z = Variable("z", units=horizont)
    c = Variable("c", units=horizont)
    G00 = Variable("G00", units=horizont)
    G02 = Variable("G02", units=horizont)
    G20 = Variable("G20", units=horizont)
    # p_real = Variable("p_real", units=horizont, dtype='float64')
    # p_imag = Variable("p_imag", units=horizont, dtype='float64')
    omega = 150.0*3.1415926

    layers = 4*[50]
    p_real = Functional(Field("p_real", units=horizont), [r, z, c], layers, 'tanh')
    p_imag = Functional(Field("p_imag", units=horizont), [r, z, c], layers, 'tanh')

    # Define data constrains
    d1 = Data(p_real)
    d2 = Data(p_imag)
    c1 = PDE((omega / c)**2*dot(G00, p_real) + dot(G20, p_real) + dot(G02, p_real))
    c2 = PDE((omega / c)**2*dot(G00, p_imag) + dot(G20, p_imag) + dot(G02, p_imag))

    # data rename
    data_d1 = Re_p_train
    data_d2 = Im_p_train
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    # constraints rename
    input = [r, z, c, G00, G02, G20]
    input_data = [r_train, z_train, c_train, G00_train, G02_train, G20_train]
    cons = [d1, d2, c1, c2]
    cons_data = [data_d1, data_d2, data_c1, data_c2]

    model = SciModel(input, cons, optimizer='scipy-l-bfgs-b')
    # callbacks
    current_file_path = pathlib.Path(__file__).parents[0]
    checkpoint_filepath = str(current_file_path.joinpath('callbacks/PDDO/test.ckpt'))
    save_path = str(current_file_path.joinpath('model/Helmholtz2D.hdf5'))
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
            epochs=1000,
            batch_size=100,
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

    # test
    c_eval, rho_eval, Re_p_target_eval, Im_p_target_eval, G00_eval, G02_eval, G20_eval, r_eval, z_eval, \
    Re_p_eval, Im_p_eval, horizont = data_tmp(random=False)
    # load .mat
    data = loadmat('Data/cylinder_pre_c0_w0.mat')
    R = data['R']  # R x 1
    Z = data['Z']  # Z x 1
    dr = 50
    dz = 5
    r_grid, z_grid = np.meshgrid(R[:, 0:-1], Z[:, 0:-1])
    # shape
    lr = R.shape[1]
    lz = Z.shape[1]
    tl_eval = np.sqrt(Re_p_target_eval ** 2 + Im_p_target_eval ** 2).reshape(lz-1, lr-1)

    eval_input = [r_eval, z_eval, c_eval, G00_eval, G02_eval, G20_eval]
    p_real_pred = np.sum(p_real.eval(model, eval_input)*G00_eval, axis=1)
    p_imag_pred = np.sum(p_imag.eval(model, eval_input)*G00_eval, axis=1)
    tl_pred = np.sqrt(p_real_pred ** 2 + p_imag_pred ** 2).reshape(lz-1, -1)

    fig = plt.figure(2, figsize=(10, 8))
    ax = plt.subplot(2, 2, 1)
    ax.invert_yaxis()
    h0 = ax.pcolormesh(r_grid / 1e3, z_grid, tl_eval, cmap='jet', shading='nearest')
    ax.set_title('exact')
    ax.set_xlabel('$range, km$')
    ax.set_ylabel('$depth, m$')

    ax = plt.subplot(2, 2, 2)
    ax.invert_yaxis()
    # h10 = ax.plot(xx, tl_exsol, 'k', linewidth=1.5, label='exact')
    # h11 = ax.plot(xx, tl_pred, 'r-.', linewidth=1.5, label='predict')
    h1 = ax.pcolormesh(r_grid / 1e3, z_grid, tl_pred, cmap='jet', shading='nearest')
    ax.set_title('predict')
    ax.set_xlabel('$range, km$')
    ax.set_ylabel('$depth, m$')
    # plt.legend()

    ax = plt.subplot(2, 1, 2)
    ax.invert_yaxis()
    h2 = ax.pcolormesh(r_grid / 1e3, z_grid, abs(tl_pred - tl_eval), cmap='jet', shading='nearest')
    ax.set_xlabel('$range, km$')
    ax.set_ylabel('$depth, m$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h0, cax=cax)
    ax.set_title('point-wist')
    plt.tight_layout()
    plt.savefig('./figures/fig5-2.png')

    plt.show()
