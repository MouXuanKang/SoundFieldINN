import pathlib
import matplotlib.pyplot as plt
import numpy as np
import sciann as sn
import tensorflow.keras.callbacks as callbacks
from sciann.utils.math import diff
from sciann_datagenerator import DataGeneratorXYZ
import time
from numpy import pi, ones, squeeze, sqrt, sin, cos


def data_tmp(start=1, end=2, num=10000, kk=1, amp=1):
    dg = DataGeneratorXYZ(
        X=[start, end],
        Y=[start, end],
        Z=[start, end],
        targets=["domain"],
        num_sample=num,
    )
    dg.plot_data()
    xx = squeeze(dg.input_data[0])
    yy = squeeze(dg.input_data[1])
    zz = squeeze(dg.input_data[2])
    rr = sqrt(xx**2 + yy**2 + zz**2)
    u_real = amp * cos(kk * rr) / (4 * pi * rr)
    u_image = amp * sin(kk * rr) / (4 * pi * rr)
    # complex valued part is phase shifted by 90°
    k_out = kk * ones(u_real.shape)

    return xx, yy, zz, u_real, u_image, k_out


if __name__ == "__main__":
    x_data, y_data, z_data, u_real_data, u_image_data, k_data = data_tmp(kk=10)

    # prepare data
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    p_real_data = scaler.fit_transform(p_real_data.reshape(-1, 1))
    p_imag_data = scaler.transform(p_imag_data.reshape(-1, 1))
    """

    x = sn.Variable("x", dtype='float64')
    y = sn.Variable("y", dtype='float64')
    z = sn.Variable("z", dtype='float64')
    k = sn.Variable("k", dtype='float64')

    p_real = sn.Functional("p_real", [x, y, z, k], 8*[20], "tanh")
    p_image = sn.Functional("p_image", [x, y, z, k], 8*[20], "tanh")

    # pde ($\Delta p + k^2 \cdot p = 0$) split into real- and complex-valued part
    L1 = diff(p_real, x, order=2) + diff(p_real, y, order=2) + diff(p_real, z, order=2) + k**2 * p_real
    L2 = diff(p_image, x, order=2) + diff(p_image, y, order=2) + diff(p_image, z, order=2) + k**2 * p_image
    c1 = sn.Data(p_real)
    c2 = sn.Data(p_image)

    # model and input
    input = [x, y, z, k]
    input_value = [x_data, y_data, z_data, k_data]
    target = [c1, c2, sn.PDE(100 * L1), sn.PDE(100 * L2)]
    target_value = [u_real_data, u_image_data, 'zeros', 'zeros']
    model = sn.SciModel(input, target)

    # callbacks
    current_file_path = pathlib.Path(__file__).parents[0]
    checkpoint_filepath = str(current_file_path.joinpath('callbacks/N3Green/N3Green.ckpt'))
    save_path = str(current_file_path.joinpath('callbacks/N3Green/N3Green.hdf5'))
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        learning_rate=1e-6
    )
    # t = time.time()
    # history = model.train(
    #     input_value,
    #     target_value,
    #     epochs=1000,
    #     batch_size=64,
    #     adaptive_weights={"method": "NTK", "freq": 100},
    #     callbacks=[model_checkpoint_callback],
    # )
    # t = time.time() - t
    # model.save_weights(save_path)
    #
    # fig = plt.figure()
    # plt.plot(history.history['p_real_loss'], 'r', label='$p_{real}$-loss')
    # plt.plot(history.history['p_image_loss'], 'b', label='$p_{image}$-loss')
    # plt.yscale('log')
    # plt.xlabel('epoch')
    # plt.ylabel('Mean-Squared-Error')
    # plt.title('Loss')
    # plt.legend()
    # plt.savefig('N3Green_Loss.png')
    # plt.show()
    model.load_weights(save_path)

    # predictions
    xx = np.linspace(1, 2, 200)
    yy = np.linspace(1, 2, 200)
    zz = np.ones(xx.shape)
    kk = 10 * np.ones(zz.shape)

    x_test, y_test = np.meshgrid(xx, yy)
    x_eval = x_test.flatten()
    y_eval = y_test.flatten()
    k_eval = 10 * np.ones(x_eval.shape)
    z_eval = np.ones(x_eval.shape)
    p_real_pred = p_real.eval(model, [x_eval, y_eval, z_eval, k_eval])
    p_image_pred = p_image.eval(model, [x_eval, y_eval, z_eval, k_eval])
    input = [x_test.flatten, y_test.flatten, z_eval.flatten, k_eval.flatten]

    p_real_pred = p_real_pred.reshape(x_test.shape)
    p_image_pred = p_image_pred.reshape(x_test.shape)
    # exact solution
    p_real_exsol = cos(kk * sqrt(x_test ** 2 + y_test ** 2 + 1)) / (4 * pi * sqrt(x_test ** 2 + y_test ** 2+1))
    p_image_exsol = sin(kk * sqrt(x_test ** 2 + y_test ** 2 + 1)) / (4 * pi * sqrt(x_test ** 2 + y_test ** 2+1))

    # pressure
    # exact solution
    fig2 = plt.figure(figsize=[15, 5])
    plt.suptitle('#3-D Real Pressure(up), Image Pressure(down)')
    plt.subplot(2, 3, 1)
    plt.pcolor(xx, yy, p_real_exsol, cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    plt.title('Exact $G(x,y)$')

    plt.subplot(2, 3, 4)
    plt.pcolor(xx, yy, p_image_exsol, cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    # plt.title('Image Pressure')

    # predict
    plt.subplot(2, 3, 2)
    plt.pcolor(xx, yy, p_real_pred, cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    plt.title('Predicted $G(x,y)$')

    plt.subplot(2, 3, 5)
    plt.pcolor(xx, yy, p_image_pred, cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    # plt.title('Image Pressure')

    # predict
    plt.subplot(2, 3, 3)
    plt.pcolor(xx, yy, np.abs(p_real_pred - p_real_exsol), cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    plt.title('Absolute error')

    plt.subplot(2, 3, 6)
    plt.pcolor(xx, yy, np.abs(p_image_pred - p_image_exsol), cmap='jet', shading='auto')
    plt.xlabel('x,m')
    plt.ylabel('y,m')
    plt.colorbar()
    # plt.title('Image Pressure')

    plt.tight_layout()
    plt.savefig('fig3.png')
    plt.show()
