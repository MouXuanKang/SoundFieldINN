# conda install tensorflow-gpu==2.1.0 matplotlib
# pip install sciann
# sciann version: 0.6.3.1
import pathlib
import numpy as np
import time
import sciann as sn
from sciann_datagenerator import DataGeneratorXY
from sciann.utils.math import diff
import tensorflow.keras.callbacks as callbacks
from scipy.special import hankel1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def data_tmp(start=0, end=500, num=50000, amp=1.0, sz=0.0, sr=0.0):
    dg = DataGeneratorXY(
        X=[start, end],
        Y=[start, end],
        targets=["domain"],
        num_sample=num,
    )
    # dg.plot_data()
    xx = np.squeeze(dg.input_data[0])
    yy = np.squeeze(dg.input_data[1])
    rr = np.sqrt((xx-sr)**2 + (yy-sz)**2)

    # Find source location id in dg.input_data
    # TOLx = 1e-6
    # TOLz = 1e-6
    # sids, _ = np.where(np.logical_and(np.abs(dg.input_data[0]-sr) < TOLx, np.abs(dg.input_data[1]-sz) < TOLz))

    # Green(x,y) = Hankel_0^1(k * r)
    freq = 150.0
    Pi = 3.141592653

    omega = 2 * Pi * freq
    m0 = 1/1500.0
    m1 = 1/1450.0
    k0 = omega * m0
    k1 = omega * m1
    u0_real = np.real(amp * hankel1(0, k1*rr))
    u0_imag = np.imag(amp * hankel1(0, k1*rr))

    G0_real = np.real(amp * hankel1(0, k0*rr))
    G0_imag = np.imag(amp * hankel1(0, k0*rr))

    # # complex valued part is phase shifted by 90°
    # k_out = k0 * np.ones(u0_real.shape)

    dp_real = u0_real - G0_real
    dp_imag = u0_imag - G0_imag
    return xx, yy, dp_real, dp_imag, omega, G0_real, G0_imag, m0, m1


if __name__ == "__main__":
    # prepare data
    r_data, z_data, p_real_data, p_imag_data, omega, G_real_data, G_imag_data, m0, m1 = data_tmp()
    isTrain = True
    # isTrain = False
    k0_data = omega * m0 * np.ones(r_data.shape)
    k1_data = omega * m1 * np.ones(r_data.shape)

    # Variable and Fields
    r = sn.Variable("r", dtype='float64')
    z = sn.Variable("z", dtype='float64')
    k0 = sn.Variable("k0", dtype='float64')
    k1 = sn.Variable("k1", dtype='float64')
    G_real = sn.Variable("G_real", dtype='float64')
    G_imag = sn.Variable("G_imag", dtype='float64')

    p_real = sn.Functional("p_real", [r, z, k0, k1, G_real], 8*[40], "tanh")
    p_imag = sn.Functional("p_imag", [r, z, k0, k1, G_imag], 8*[40], "tanh")

    # pde ($\Delta p + k^2 \cdot p = 0$) split into real- and complex-valued part
    laplace_Re_dp = diff(p_real, r, order=2) + diff(p_real, z, order=2)
    laplace_Im_dp = diff(p_imag, r, order=2) + diff(p_imag, z, order=2)

    L1 = laplace_Re_dp + k1**2 * p_real + (k1**2 - k0**2)*G_real
    L2 = laplace_Im_dp + k1**2 * p_imag + (k1**2 - k0**2)*G_imag
    c1 = sn.Data(p_real)
    c2 = sn.Data(p_imag)

    # model and input
    input = [r, z, k0, k1, G_real, G_imag]
    input_value = [r_data, z_data, k0_data, k1_data, G_real_data, G_imag_data]
    target = [sn.PDE(L1), sn.PDE(L2), c1, c2]
    target_value = ['zeros', 'zeros', p_real_data, p_imag_data]
    model = sn.SciModel(
        input,
        target,
        load_weights_from='model/2D_dp_[0,10]x[0,10]_zs[0,0]_Gc1.5.hdf5')
        # optimizer='scipy-l-BFGS-B')

    # callbacks
    current_file_path = pathlib.Path(__file__).parents[0]
    checkpoint_filepath = str(current_file_path.joinpath('callbacks/N2Green/N2Green_dp.ckpt'))
    save_path = str(current_file_path.joinpath('model/2D_dp_[0,100]x[0,100]_zs[0,0]_Gc1.5.hdf5'))
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True
    )
    if isTrain:
        t = time.time()

        history = model.train(
            input_value,
            target_value,
            epochs=1500,
            batch_size=1000,
            adaptive_weights={"method": "NTK", "freq": 100},
            # callbacks=[model_checkpoint_callback],
            shuffle=True,
            learning_rate=0.001,
            reduce_lr_after=100,
            stop_loss_value=1e-8
        )
        t = time.time() - t
        #
        model.save_weights(save_path)

        fig0 = plt.figure(10)
        plt.plot(history.history['p_real_loss'], 'r', label='$p_{real}$-loss')
        plt.plot(history.history['p_imag_loss'], 'b', label='$p_{imag}$-loss')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('Mean-Squared-Error')
        plt.title('Loss')
        plt.legend()
        plt.savefig('fig2_loss_history.png')
        # plt.show()
    else:
        model.load_weights(save_path)

    # predictions
    N = 1000
    x_max = 100
    y_max = 100
    xx = np.linspace(0, x_max, N)
    yy = np.linspace(0, y_max, N)
    sx = 0
    sy = 0

    x_test, y_test = np.meshgrid(xx, yy)
    x_eval = x_test.flatten()
    y_eval = y_test.flatten()
    r_eval = np.sqrt(x_eval**2 + y_eval**2)
    k0 = k0_data[0]
    k1 = k1_data[0]
    k0_test = k0 * np.ones(x_eval.shape)
    k1_test = k1 * np.ones(x_eval.shape)
    G_real_test = np.real(hankel1(0, k0*r_eval))
    G_imag_test = np.imag(hankel1(0, k0*r_eval))

    p_real_pred = p_real.eval(model, [x_eval, y_eval, k0_test, k1_test, G_real_test, G_imag_test])
    p_imag_pred = p_imag.eval(model, [x_eval, y_eval, k0_test, k1_test, G_real_test, G_imag_test])

    p_real_pred = (p_real_pred + G_real_test).reshape(x_test.shape)
    p_imag_pred = (p_imag_pred + G_imag_test).reshape(x_test.shape)
    k1_test = k1_test.reshape(x_test.shape)

    # exact solution
    p_real_exsol = np.real(hankel1(0, k1_test * np.sqrt(x_test ** 2 + y_test ** 2)))
    p_imag_exsol = np.imag(hankel1(0, k1_test * np.sqrt(x_test ** 2 + y_test ** 2)))

    # pre2tl
    pr_exsol = (p_real_exsol / np.real(hankel1(0, k1_test))) ** 2
    pi_exsol = (p_imag_exsol / np.imag(hankel1(0, k1_test))) ** 2
    tl_exsol = -20 * np.log10(np.sqrt(pr_exsol + pi_exsol))

    pr_pred = (p_real_pred / np.real(hankel1(0, k1_test))) ** 2
    pi_pred = (p_imag_pred / np.imag(hankel1(0, k1_test))) ** 2
    tl_pred = -20 * np.log10(np.sqrt(pr_pred + pi_pred))

    # pressure
    fig = plt.figure(1, figsize=(8, 3))
    ax = plt.subplot(1, 2, 1)
    ax.invert_yaxis()
    h0 = ax.pcolormesh(xx, yy, p_real_exsol, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h0, cax=cax1)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('exact solution')

    ax = plt.subplot(1, 2, 2)
    ax.invert_yaxis()
    h2 = ax.pcolormesh(xx, yy, p_real_pred, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax3 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h2, cax=cax3)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('predicted solution')
    plt.suptitle('real')
    plt.tight_layout()
    plt.savefig('fig3-1.png')

    fig = plt.figure(2, figsize=(8, 3))
    ax = plt.subplot(1, 2, 1)
    ax.invert_yaxis()
    h1 = ax.pcolormesh(xx, yy, p_imag_exsol, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h1, cax=cax2)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('exact solution')

    ax = plt.subplot(1, 2, 2)
    ax.invert_yaxis()
    h3 = ax.pcolormesh(xx, yy, p_imag_pred, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax4 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h3, cax=cax4)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('predicted solution')
    plt.suptitle('image')
    plt.tight_layout()
    plt.savefig('fig3-2.png')

    fig = plt.figure(3, figsize=(8, 3))
    ax = plt.subplot(1, 2, 1)
    ax.invert_yaxis()
    h1 = ax.pcolormesh(xx, yy, tl_exsol, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h1, cax=cax2)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('exact solution')

    ax = plt.subplot(1, 2, 2)
    ax.invert_yaxis()
    h3 = ax.pcolormesh(xx, yy, tl_pred, cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax4 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h3, cax=cax4)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('predicted solution')
    plt.tight_layout()
    plt.savefig('fig3-3.png')

    fig = plt.figure(4, figsize=(8, 3))
    ax = plt.subplot(1, 2, 1)
    ax.invert_yaxis()
    h4 = ax.pcolormesh(xx, yy, abs(p_real_exsol - p_real_pred), cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax5 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h4, cax=cax5)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('real')

    ax = plt.subplot(1, 2, 2)
    ax.invert_yaxis()
    h5 = ax.pcolormesh(xx, yy, abs(p_imag_exsol - p_imag_pred), cmap='jet_r', shading='auto')
    divider = make_axes_locatable(ax)
    cax6 = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(h5, cax=cax6)
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$y, m$')
    ax.set_title('image')
    plt.suptitle('point-wise error')
    plt.tight_layout()
    plt.savefig('fig3-4.png')

    fig = plt.figure(5, figsize=(8, 3))
    dx = x_max / N
    dy = y_max / N
    zy = 5
    ny = int(zy / dy)
    ax = plt.subplot(1, 2, 1)
    ax.invert_yaxis()
    h6 = ax.plot(xx, p_real_exsol[ny, :], 'k', linewidth=1.5, label='exact')
    h7 = ax.plot(xx, p_real_pred[ny, :], 'r-.', linewidth=1.5, label='predict')
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$pressure$')
    ax.set_title('real')
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    ax.invert_yaxis()
    h8 = ax.plot(xx, p_imag_exsol[ny, :], 'k', linewidth=1.5, label='exact')
    h9 = ax.plot(xx, p_imag_pred[ny, :], 'r-.', linewidth=1.5, label='predict')
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$pressure$')
    ax.set_title('image')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig3-5.png')

    fig1 = plt.figure(6, figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    ax.invert_yaxis()
    h10 = ax.plot(xx, tl_exsol[ny, :], 'k', linewidth=1.5, label='exact')
    h11 = ax.plot(xx, tl_pred[ny, :], 'r-.', linewidth=1.5, label='predict')
    ax.set_xlabel('$x, m$')
    ax.set_ylabel('$TL, dB$')
    plt.legend()

    plt.tight_layout()
    plt.savefig('fig3-6.png')

    plt.show()
