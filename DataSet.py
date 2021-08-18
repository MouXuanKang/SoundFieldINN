import pathlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
from sciann_datagenerator import DataGeneratorXYT
from PDDO import p_operator_2d, b_operator_2d, FormDiffA_mat2D, FormDiffB_vec2D


def axisEqual3D(axx, nn):
    extents = np.array([getattr(axx, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / nn
    for ctr, dim in zip(centers, 'xyz'):
        getattr(axx, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":
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

    SSP_star = SSP_data[:, 1:iz]  # R x Z
    omega = omega * np.ones(SSP_star.shape)
    K_star = omega / SSP_star

    R_star = np.arange(dr, Range_max + dr, dr)
    Z_star = np.arange(dz, nz * dz, dz, dtype=int)

    R = R_star.shape[0]  # R
    Z = Z_star.shape[0]  # Z

    # print("data.shape:", RR.shape)
    # Rearrange Data
    Re_P_star = Re_P_data[:, 1:nz]  # R x Z
    Im_P_star = Im_P_data[:, 1:nz]  # R x Z
    TL = np.sqrt(Re_P_star ** 2 + Im_P_star ** 2)

    ru = 0.1
    rd = 9.0
    zu = 1.0
    zd = 380.0

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
    #
    # Family node


    def GenerateNodeFamilies(coord_x, coord_y, dist, edge, d_volume):
        totnode = len(coord_x)
        d_r = d_volume[0]
        d_z = d_volume[1]
        family_list = np.tile(dist, totnode).reshape(-1, 4)
        for i in range(totnode):
            x1 = coord_x[i] + dist[0]
            x2 = coord_x[i] - dist[1]
            y1 = coord_y[i] + dist[2]
            y2 = coord_y[i] - dist[3]
            if x1 > edge[1]:
                family_list[i][1] = (edge[1] - coord_x[i]) // d_r
            if x2 < edge[0]:
                family_list[i][0] = (coord_x[i]) // d_r
            if y1 > edge[3]:
                family_list[i][3] = (edge[3] - coord_y[i]) // d_z
            if y2 < edge[2]:
                family_list[i][2] = (coord_y[i]) // d_z
        return family_list


    # dist:|[x1, x2, y1, y2]|, edge:[0, r, 0, z],
    rr, zz = np.meshgrid(R_star, Z_star)
    r_list = rr.reshape(-1, 1)
    z_list = zz.reshape(-1, 1)
    FamilyList = GenerateNodeFamilies(r_list, z_list,
                                      [2, 2, 20, 20],
                                      [R_star[0], R_star[-1], Z_star[0],
                                       Z_star[-1]], [dr, dz])
    totnodes = len(r_list)
    G = []
    delta_mag = np.sqrt((rr[2, 1] - rr[3, 1])**2 + (zz[2, 1] - zz[2, 2])**2)
    for i in range(totnodes):
        Node_Number = FamilyList[i]
        ir = i // len(R_star)
        iz = i % nr
        # i-Node_Number[1] : i+Node_Number[0]
        # i-Node_Number[3] : i+Node_Number[2]
        Family_Node_r = rr[i-Node_Number[1]:i+Node_Number[0]]
        Family_Node_z = zz[i-Node_Number[3]:i+Node_Number[2]]
        for r_node, xsi1 in enumerate(Family_Node_r):
            for z_node, xsi2 in enumerate(Family_Node_z):
                plist = p_operator_2d(xsi1, xsi2, delta_mag)
                mat_A = FormDiffA_mat2D(xsi1, xsi2)
                vec_b = FormDiffB_vec2D()
                vec_a = np.linalg.inv(mat_A) * vec_b
                # G[i] = sum(mat_A*plist[:, 0])
                G = np.stack(G, vec_a * plist[0, :])


    # output2mat
    # Rho_star = Rho_data[int(ru/dr):int(rd/dr)+1, int(zu/dz):int(zd/dz)+1]  # R x Z
    RePa_star = Re_P_data.T[:, 0: -1]
    ImPa_star = Im_P_data.T[:, 0: -1]
    RePb_star = Re_P_data.T[:, 1:]
    ImPb_star = Im_P_data.T[:, 1:]
    ca_star = SSP_data.T[:, 0: -1]
    cb_star = SSP_data.T[:, 1:]
    rhoa_star = Rho_data.T[:, 0: -1]
    rhob_star = Rho_data.T[:, 1:]
    omega_star = omega.T[:, 1:]
    Z_star = Z_star
    scipy.io.savemat('Data/cylinder_pre_c0_w0.mat', {
        'RePa_star': RePa_star,
        'ImPa_star': ImPa_star,
        'RePb_star': RePb_star,
        'ImPb_star': ImPb_star,
        'ca_star': ca_star,
        'cb_star': cb_star,
        'rhoa_star': rhoa_star,
        'rhob_star': rhob_star,
        'R': R_star,
        'Z': Z_star,
        'Family': FamilyList,
        'omega_star': omega_star,
        'c0': 1500.0,
        'Versions': 'ramgeo1.5'
    })
