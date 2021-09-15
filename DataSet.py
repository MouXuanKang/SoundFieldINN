"""
@Title:
    Solving Helmholtz equation with PDDO-PINNs.
    该代码用于准备网络输入所需要的数据，
    I.  PDDO系数和周边算子分布的四维array保存在.pickle，
    II. 声场计算程序ram仿真结果和其他环境参数保存在.mat中，
    III.日志文件保存在为log.txt
@author:
    Xu Liang.
    Department of Underwater Acoustic,
    Harbin Engineering University.
    2013053118@hrbeu.edu.cn
Created on 2021
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
from sciann_datagenerator import DataGeneratorXYT
import pickle
import tqdm
from time import sleep


def axisEqual3D(axx, nn):
    extents = np.array([getattr(axx, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / nn
    for ctr, dim in zip(centers, 'xyz'):
        getattr(axx, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def p_operator_2d(xsi_1, xsi_2, Delta_mag):
    # xsi1 is x, xsi2 is y, delta_mag = sqrt(deltax(i)**2+deltay(i)**2+deltaz(i)**2)
    xsi1p = xsi_1 / Delta_mag
    xsi2p = xsi_2 / Delta_mag
    plist = np.array([[1, xsi1p, xsi2p, xsi1p ** 2, xsi2p ** 2, xsi1p * xsi2p],
                      [xsi1p, xsi1p ** 2, xsi1p * xsi2p, xsi1p ** 3, xsi1p * xsi2p ** 2, xsi1p ** 2 * xsi2p],
                      [xsi2p, xsi1p * xsi2p, xsi2p ** 2, xsi1p ** 2 * xsi2p, xsi2p ** 3, xsi1p * xsi2p ** 2],
                      [xsi1p ** 2, xsi1p ** 3, xsi1p ** 2 * xsi2p, xsi1p ** 4, xsi1p ** 2 * xsi2p ** 2,
                       xsi1p ** 3 * xsi2p],
                      [xsi2p ** 2, xsi1p * xsi2p ** 2, xsi2p ** 3, xsi1p ** 2 * xsi2p ** 2, xsi2p ** 4,
                       xsi1p * xsi2p ** 3],
                      [xsi1p * xsi2p, xsi1p ** 2 * xsi2p, xsi1p * xsi2p ** 2, xsi1p ** 3 * xsi2p, xsi1p * xsi2p ** 3,
                       xsi1p ** 2 * xsi2p ** 2]
                      ])
    return plist


def weights_2d(xsi_1, xsi_2, Delta_mag):
    # the wight of Family-Nodes, xsi1 is x, xsi2 is y, delta_mag = sqrt(deltax(i)**2+deltay(i)**2+deltaz(i)**2)
    xsi_mag = np.sqrt(xsi_1 ** 2 + xsi_2 ** 2)
    wt = np.exp(-4 * (xsi_mag / Delta_mag) ** 2)
    return wt


def b_operator_2d():
    # vector b
    b = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 2, 0, 0],
                  [0, 0, 0, 0, 2, 0],
                  [0, 0, 0, 0, 0, 1]])

    return b


def FormDiffA_mat2D(Family_r, Family_z, d_volume):
    # k is family number, d_volume is dx[k]*dy[k], delta_mag = sqrt(deltax(i)**2+deltay(i)**2+deltaz(i)**2)
    # numfam is number of family, family_node is node number
    fam_num = len(Family_r)
    DiffA_mat2D = np.zeros((6, 6), dtype=float)
    d_r = d_volume[0]
    d_z = d_volume[1]
    Delta_Volume = d_r * d_z
    Delta_mag = np.sqrt(d_r**2 + d_z**2)
    for i_mem in range(fam_num):
        xsi_1 = np.linalg.norm(Family_r[i_mem])
        xsi_2 = np.linalg.norm(Family_z[i_mem])
        p = p_operator_2d(xsi_1, xsi_2, Delta_mag)
        w = weights_2d(xsi_1, xsi_2, Delta_mag)
        coef = w * Delta_Volume
        # print(i_mem+1, ' / ', fam_num, coef)
        DiffA_mat2D = coef * p + DiffA_mat2D
    return DiffA_mat2D


def FormDiffG_cont2D(Family_r, Family_z, d_volume, A_mat, b_vec):
    fam_num = len(Family_r)
    DiffG00_mat2D = []
    DiffG10_mat2D = []
    DiffG01_mat2D = []
    DiffG20_mat2D = []
    DiffG02_mat2D = []
    d_r = d_volume[0]
    d_z = d_volume[1]
    # Delta_Volume = dr * dz
    Delta_mag = np.sqrt(d_r ** 2 + d_z ** 2)
    a_vec = np.linalg.inv(A_mat) * b_vec
    for i_mem in range(fam_num):
        xsi_1 = np.linalg.norm(Family_r[i_mem])
        xsi_2 = np.linalg.norm(Family_z[i_mem])
        p_list = p_operator_2d(xsi_1, xsi_2, Delta_mag)
        w = weights_2d(xsi_1, xsi_2, Delta_mag)
        DiffG00_mat2D = np.append(DiffG00_mat2D, [np.sum(a_vec[0, :] * p_list[:, 0] * w)])
        DiffG10_mat2D = np.append(DiffG10_mat2D, [np.sum(a_vec[1, :] * p_list[:, 0] * w)])
        DiffG01_mat2D = np.append(DiffG01_mat2D, [np.sum(a_vec[2, :] * p_list[:, 0] * w)])
        DiffG20_mat2D = np.append(DiffG20_mat2D, [np.sum(a_vec[3, :] * p_list[:, 0] * w)])
        DiffG02_mat2D = np.append(DiffG02_mat2D, [np.sum(a_vec[4, :] * p_list[:, 0] * w)])
    return DiffG00_mat2D, DiffG10_mat2D, DiffG01_mat2D, DiffG20_mat2D, DiffG02_mat2D


def FormDiffB_vec2D():
    vec_b = b_operator_2d()
    return vec_b


if __name__ == "__main__":
    print('本程序生成PDDO-PINN所需数据集~')
    print('程序开始！一定不要出错呀！（＞﹏＜）')
    current_path = pathlib.Path(__file__).parents[0]
    par_filepath = str(current_path.joinpath('Data/bin/parameters.txt'))
    rep_filepath = str(current_path.joinpath('Data/bin/real.grid'))
    img_filepath = str(current_path.joinpath('Data/bin/imag.grid'))
    ssp_filepath = str(current_path.joinpath('Data/bin/ssp.grid'))
    rho_filepath = str(current_path.joinpath('Data/bin/rho.grid'))

    # parameters
    par = np.loadtxt(par_filepath)
    omega = par[0]
    Range_max = par[1]
    dr = par[2]
    nr = int(Range_max / dr)
    nz = int(par[3]) + 2
    dz = par[4]
    iz = int(par[3])
    # nz = int(par[5])
    eps = 1e-20
    with open(rep_filepath, mode="rb") as f1:
        temp = np.fromfile(f1, dtype=np.float32)
        Re_P_data = np.reshape(temp, [nr, nz])
        f1.close()

    with open(img_filepath, mode="rb") as f2:
        temp = np.fromfile(f2, dtype=np.float32)
        Im_P_data = np.reshape(temp, [nr, nz])
        f2.close()

    with open(ssp_filepath, mode="rb") as f3:
        temp = np.fromfile(f3, dtype=np.float32)
        SSP_data = np.reshape(temp, [nr, nz])
        f3.close()

    with open(rho_filepath, mode="rb") as f4:
        temp = np.fromfile(f4, dtype=np.float32)
        Rho_data = np.reshape(temp, [nr, nz])
        f4.close()

    SSP_star = SSP_data[:, 1:-1]  # R x Z
    omega = omega * np.ones(SSP_star.shape)
    K_star = omega / SSP_star

    R_star = np.arange(dr, Range_max + dr, dr)
    Z_star = np.arange(dz, (iz+1) * dz, dz, dtype=int)

    R = R_star.shape[0]  # R
    Z = Z_star.shape[0]  # Z

    # Rearrange Data
    Re_P_star = Re_P_data[:, 1:nz-1]  # R x Z
    Im_P_star = Im_P_data[:, 1:nz-1]  # R x Z
    TL = np.sqrt(Re_P_star ** 2 + Im_P_star ** 2)
    # plt.figure(1)
    # plt.imshow(TL.T)  # 199x200
    # plt.show()
    ru = 1.1
    rd = 2.0
    zu = 30.0
    zd = 60.0

    box_lb = np.array([ru, zu])
    box_ub = np.array([rd, zd])

    dg = DataGeneratorXYT(
        X=[R_star[0], R_star[-1]],
        Y=[Z_star[0], Z_star[-1]],
        T=[1450.0, 1550.0],
        targets=["domain"],
        num_sample=5000,
    )
    R_train = dg.input_data[0]
    Z_train = dg.input_data[1]
    C_train = dg.input_data[2]
    RR, ZZ = np.meshgrid(R_star, Z_star)
    # Plotting

    # Row 0: Pressure
    fig = plt.figure(1)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.1, bottom=0.15, left=0.12, right=.9, wspace=0)
    ax = plt.subplot(gs0[:, :])
    # ax = plt.subplot(2, 1, 1)
    h = ax.pcolormesh(RR / 1000, ZZ, TL.T, cmap='jet', shading='nearest')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1.5)
    ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1.5)
    ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], 'k', linewidth=1.5)
    ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], 'k', linewidth=1.5)

    ax.set_xlabel('$Range,km$')
    ax.set_ylabel('$Depth,m$')
    ax.set_title('Pressure.abs', fontsize=10)
    plt.savefig('./figures/fig5.png')

    n1 = 1e3  # R
    n2 = 10  # C
    n3 = .5e2  # Z
    n4 = 2.0
    fig2 = plt.figure(2)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=0.8, bottom=0.1, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0], projection='3d')
    ax.axis('off')

    r1 = [R_train.min() / n1, R_train.max() / n1]
    r2 = [C_train.min() / n2, C_train.max() / n2]
    r3 = [Z_train.min() / n3, Z_train.max() / n3]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == r2[1] - r2[0] or \
                np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(R_train / n1, C_train / n2, Z_train / n3, s=0.05)
    ax.contourf(RR / n1, Re_P_star.T, np.flipud(ZZ) / n3, zdir='y', offset=C_train.mean() / n2, cmap='rainbow',
                alpha=0.8)
    ax.text(R_train.mean() / n1, C_train.min() / n2 - 1, Z_train.min() / n3 - 1, '$R$')
    ax.text(R_train.max() / n1 + 1, C_train.mean() / n2, Z_train.min() / n3 - 1, '$C$')
    ax.text(R_train.min() / n1 - 1, C_train.min() / n2 - 0.5, Z_train.mean() / n3, '$Z$')
    ax.text(R_train.min() / n1 - 2, C_train.mean() / n2, Z_train.max() / n3 + 2, '$P(r,z,c)$')
    ax.set_xlim3d(r1)  # R
    ax.set_ylim3d(r2)  # C
    ax.set_zlim3d(r3)  # Z
    axisEqual3D(ax, n4)
    ax.set_title('Real-Pre', fontsize=10)

    ax = plt.subplot(gs1[:, 1], projection='3d')
    ax.axis('off')
    r1 = [R_train.min() / n1, R_train.max() / n1]
    r2 = [C_train.min() / n2, C_train.max() / n2]
    r3 = [Z_train.min() / n3, Z_train.max() / n3]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == r2[1] - r2[0] or \
                np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(R_train / n1, C_train / n2, Z_train / n3, s=0.05)
    ax.contourf(RR / n1, Im_P_star.T, np.flipud(ZZ) / n3, zdir='y', offset=C_train.mean() / n2, cmap='rainbow',
                alpha=0.8)
    ax.text(R_train.mean() / n1, C_train.min() / n2 - 1, Z_train.min() / n3 - 1, '$R$')
    ax.text(R_train.max() / n1 + 1, C_train.mean() / n2, Z_train.min() / n3 - 1, '$C$')
    ax.text(R_train.min() / n1 - 1, C_train.min() / n2 - 0.5, Z_train.mean() / n3, '$Z$')
    ax.text(R_train.min() / n1 - 2, C_train.mean() / n2, Z_train.max() / n3 + 2, '$P(r,z,c)$')
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax, n4)
    ax.set_title('Image-Pre', fontsize=10)
    plt.savefig('./figures/fig6.png')

    # plt.show()
    # Family node


    def GenerateNodeFamilies(coord_x, coord_y, dist, edge, d_volume):
        totnode = len(coord_x)
        d_r = d_volume[0]
        d_z = d_volume[1]
        family_list = np.tile(dist, totnode).reshape(-1, 4)
        # print('Default FamilyList is [+ndr, -ndr, +ndz, -ndz]', dist)
        test = -1
        # print(test + 1, (coord_y[test], coord_x[test]))
        # print('计算族群点分布ing...〒▽〒')
        for i1 in range(totnode):
            x1 = coord_x[i1] + dist[0] * d_r
            x2 = coord_x[i1] - dist[1] * d_r
            y1 = coord_y[i1] + dist[2] * d_z
            y2 = coord_y[i1] - dist[3] * d_z
            if x1 > edge[1]:
                family_list[i1][0] = (edge[1] - coord_x[i1]) // d_r

            if x2 < edge[0]:
                family_list[i1][1] = coord_x[i1] // d_r

            if y1 > edge[3]:
                family_list[i1][2] = (edge[3] - coord_y[i1]) // d_z

            if y2 < edge[2]:
                family_list[i1][3] = coord_y[i1] // d_z

        return family_list


    Family_size = [3, 3, 3, 3]

    # dist:|[x1, x2, y1, y2]|, edge:[dr, r, dz, z],
    # x + x1 < r, x - x2 > 0, y + y1 < z, y - y2 > 0
    r_list = RR.flatten()
    z_list = ZZ.flatten()
    FamilyList = GenerateNodeFamilies(r_list, z_list,
                                      Family_size,
                                      [R_star[0], R_star[-1], Z_star[0],
                                       Z_star[-1]], [dr, dz])
    totnodes = len(r_list)
    delta_mag = np.sqrt(dz ** 2 + dr ** 2)
    Family_Node_R_save = np.empty([len(Z_star),
                                   len(R_star),
                                   Family_size[0] + Family_size[1] + 1,
                                   Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_Z_save = np.empty([len(Z_star),
                                   len(R_star),
                                   Family_size[0] + Family_size[1] + 1,
                                   Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_C_save = np.ones([len(Z_star),
                                  len(R_star),
                                  Family_size[0] + Family_size[1] + 1,
                                  Family_size[2] + Family_size[3] + 1], dtype=float, order='C') * 1500.0
    Family_Node_RP_save = np.empty([len(Z_star),
                                    len(R_star),
                                    Family_size[0] + Family_size[1] + 1,
                                    Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_IP_save = np.empty([len(Z_star),
                                    len(R_star),
                                    Family_size[0] + Family_size[1] + 1,
                                    Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_G00_save = np.empty([len(Z_star),
                                     len(R_star),
                                     Family_size[0] + Family_size[1] + 1,
                                     Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_G10_save = np.empty([len(Z_star),
                                     len(R_star),
                                     Family_size[0] + Family_size[1] + 1,
                                     Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_G01_save = np.empty([len(Z_star),
                                     len(R_star),
                                     Family_size[0] + Family_size[1] + 1,
                                     Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_G20_save = np.empty([len(Z_star),
                                     len(R_star),
                                     Family_size[0] + Family_size[1] + 1,
                                     Family_size[2] + Family_size[3] + 1], dtype=float, order='C')
    Family_Node_G02_save = np.empty([len(Z_star),
                                     len(R_star),
                                     Family_size[0] + Family_size[1] + 1,
                                     Family_size[2] + Family_size[3] + 1], dtype=float, order='C')

    # output2mat
    # Rho_star = Rho_data[int(ru/dr):int(rd/dr)+1, int(zu/dz):int(zd/dz)+1]  # R x Z
    print('.mat注入ing...（￣︶￣）')
    c_star = SSP_data.T[:, :]
    CC = c_star[1:-1, :]
    rho_star = Rho_data.T[:, :]
    omega_star = omega.T[:, :]
    scipy.io.savemat('Data/cylinder_pre_c0_w0.mat', {
        'ReP_star': Re_P_star,
        'ImP_star': Im_P_star,
        'c_star': c_star,
        'rho_star': rho_star,
        'Family_list': FamilyList,
        'Family_size': Family_size,
        'R': R_star,
        'Z': Z_star,
        'omega_star': omega_star[0][0],
        'c0': 1500.0,
        'Versions': 'ramgeo1.5'
    })

    print('循环计算周边动力学系数ing...〒▽〒')

    for i in tqdm.tqdm(range(totnodes)):
        # for i in [0]:
        Node_Number = FamilyList[i]
        # coord of target:(iz, ir), totnodes = len(R_star) * len(Z_star)
        # print(i, (z_list[i - 1], r_list[i - 1]))
        iz = i // len(R_star)  #
        ir = i % len(R_star)  #
        # x_left = Family_size[0] - Node_Number[0] = 0
        x_left = -Node_Number[0] + Family_size[0]
        x_right = Node_Number[1] + Family_size[1]
        y_left = -Node_Number[2] + Family_size[2]
        y_right = Node_Number[3] + Family_size[3]
        x_size = x_right - x_left
        y_size = y_right - y_left
        # Coord of Family Node
        Family_Node_r = RR[iz - Node_Number[3] + 1:iz + Node_Number[2] + 1,
                           ir - Node_Number[1] + 1:ir + Node_Number[0] + 1]
        Family_Node_z = ZZ[iz - Node_Number[3] + 1:iz + Node_Number[2] + 1,
                           ir - Node_Number[1] + 1:ir + Node_Number[0] + 1]
        Family_Node_C = CC[iz - Node_Number[3] + 1:iz + Node_Number[2] + 1,
                           ir - Node_Number[1] + 1:ir + Node_Number[0] + 1]
        family_nr_flatten = Family_Node_r.flatten()
        family_nz_flatten = Family_Node_z.flatten()
        index_target = \
            np.where(np.logical_and(family_nz_flatten == iz * dz + dz, family_nr_flatten == ir * dr + dr))[0]
        # print(i, (z_list[i - 1], r_list[i - 1]))
        # print(family_nz_flatten[index_target[0]], family_nr_flatten[index_target[0]])
        family_nr_sum = np.delete(family_nr_flatten, index_target[0]) - ir * dr - dr
        family_nz_sum = np.delete(family_nz_flatten, index_target[0]) - iz * dz - dz
        Family_Node_R_save[iz, ir, y_left:y_right, x_left:x_right] = Family_Node_r
        Family_Node_Z_save[iz, ir, y_left:y_right, x_left:x_right] = Family_Node_z
        Family_Node_C_save[iz, ir, y_left:y_right, x_left:x_right] = Family_Node_C
        family_node = np.vstack((family_nz_sum, family_nr_sum))
        A_family_node_mat = FormDiffA_mat2D(family_nr_sum, family_nz_sum, [dr, dz])
        b_family_node_vec = FormDiffB_vec2D()
        G00, G10, G01, G20, G02 = FormDiffG_cont2D(family_nr_sum, family_nz_sum,
                                                   [dr, dz],
                                                   A_family_node_mat, b_family_node_vec)
        G_loop = np.vstack((G00, G10, G01, G20, G02))

        G00_shape = np.insert(G00.T, 0, 0).reshape(y_size, x_size)
        G10_shape = np.insert(G10.T, 0, 0).reshape(y_size, x_size)
        G01_shape = np.insert(G01.T, 0, 0).reshape(y_size, x_size)
        G20_shape = np.insert(G20.T, 0, 0).reshape(y_size, x_size)
        G02_shape = np.insert(G02.T, 0, 0).reshape(y_size, x_size)
        Family_Node_G00_save[iz, ir, y_left:y_right, x_left:x_right] = G00_shape
        Family_Node_G10_save[iz, ir, y_left:y_right, x_left:x_right] = G10_shape
        Family_Node_G01_save[iz, ir, y_left:y_right, x_left:x_right] = G01_shape
        Family_Node_G20_save[iz, ir, y_left:y_right, x_left:x_right] = G20_shape
        Family_Node_G02_save[iz, ir, y_left:y_right, x_left:x_right] = G02_shape

        temp_rep = Re_P_data[iz - Node_Number[3] + 1:iz + Node_Number[2] + 1,
                             ir - Node_Number[1] + 1:ir + Node_Number[0] + 1].flatten()
        temp_imp = Im_P_data[iz - Node_Number[3] + 1:iz + Node_Number[2] + 1,
                             ir - Node_Number[1] + 1:ir + Node_Number[0] + 1].flatten()
        temp_rep_target = temp_rep[index_target[0]]
        temp_imp_target = temp_imp[index_target[0]]
        temp_re = np.delete(temp_rep, index_target[0])
        temp_im = np.delete(temp_imp, index_target[0])
        temp_rep = np.hstack((temp_rep_target, temp_re))
        temp_imp = np.hstack((temp_imp_target, temp_im))

        Family_Node_RP_save[iz, ir, y_left:y_right, x_left:x_right] = temp_rep.reshape(y_size, x_size)
        Family_Node_IP_save[iz, ir, y_left:y_right, x_left:x_right] = temp_imp.reshape(y_size, x_size)

        # name = 'G' + str(i)       # 功能，随循环生成变量名保存变量到.pickle
        # locals()[name] = G_loop   # 弃用，原因：主函数中调用不方便，变量名过多
        # pickle.dump(locals()[name], f)
        sleep(0.01)
    print('.pickle注入ing...（￣︶￣）')
    with open('Data/G00.pickle', 'wb') as f1:
        pickle.dump(Family_Node_G00_save, f1)
    f1.close()
    with open('Data/G10.pickle', 'wb') as f2:
        pickle.dump(Family_Node_G10_save, f2)
    f2.close()
    with open('Data/G01.pickle', 'wb') as f3:
        pickle.dump(Family_Node_G01_save, f3)
    f3.close()
    with open('Data/G20.pickle', 'wb') as f4:
        pickle.dump(Family_Node_G20_save, f4)
    f4.close()
    with open('Data/G02.pickle', 'wb') as f5:
        pickle.dump(Family_Node_G02_save, f5)
    f5.close()
    with open('Data/RePImP.pickle', 'wb') as f6:
        pickle.dump(Family_Node_RP_save, f6)
        pickle.dump(Family_Node_IP_save, f6)
    f6.close()
    with open('Data/FamilyRZC.pickle', 'wb') as f7:
        pickle.dump(Family_Node_R_save, f7)
        pickle.dump(Family_Node_Z_save, f7)
        pickle.dump(Family_Node_C_save, f7)
    f7.close()
    sleep(0.5)

    # how2load :
    # f1 = open('G5.pickle', 'rb')
    # G = pickle.load(f1)  # will change in loop

    print('完结撒花！（ ＞﹏＜）')
