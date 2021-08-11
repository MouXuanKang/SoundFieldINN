"""
@Title:
    Solving Helmholtz equation with PDDO-PINNs.
    
@author:
    Xu Liang
    Department of Underwater Acoustic
    Harbin Engineering University
    2013053118@hrbeu.edu.cn

Created on 2021
"""
import numpy as np


def p_operator_2d(xsi1, xsi2, delta_mag):
    xsi1p = xsi1 / delta_mag
    xsi2p = xsi2 / delta_mag
    plist = np.array([[1, xsi1p, xsi2p, xsi1p**2, xsi2p**2, xsi1p * xsi2p],
                      [xsi1p, xsi1p**2, xsi1p * xsi2p, xsi1p**3, xsi1p * xsi2p**2, xsi1p**2 * xsi2p],
                      [xsi2p, xsi1p*xsi2p, xsi2p**2, xsi1p**2 * xsi2p, xsi2p**3, xsi1p * xsi2p**2],
                      [xsi1p**2, xsi1p**3, xsi1p**2 * xsi2p, xsi1p**4, xsi1p**2 * xsi2p**2, xsi1p**3 * xsi2p],
                      [xsi2p**2, xsi1p * xsi2p**2, xsi2p**3, xsi1p**2 * xsi2p**2, xsi2p**4, xsi1p * xsi2p**3],
                      [xsi1p * xsi2p, xsi1p**2 * xsi2p, xsi1p * xsi2p**2, xsi1p**3 * xsi2p, xsi1p * xsi2p**3,
                       xsi1p**2 * xsi2p**2]
                      ])
    return plist


def weights_2d(xsi1, xsi2, delta_mag):
    xsi_mag = np.sqrt(xsi1**2 + xsi2**2)
    wt = np.exp(-4 * (xsi_mag / delta_mag)**2)
    # wt = 1.0
    # wt = (xsi_mag/delta_mag)^2

    # weight = np.array([[wt, wt, wt, wt, wt, wt],
    #                    [wt, wt, wt, wt, wt, wt],
    #                    [wt, wt, wt, wt, wt, wt],
    #                    [wt, wt, wt, wt, wt, wt],
    #                    [wt, wt, wt, wt, wt, wt],
    #                    [wt, wt, wt, wt, wt, wt]
    #                    ])
    weight = wt
    return weight


def b_operator_2d():
    b = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 2, 0, 0],
                  [0, 0, 0, 0, 2, 0],
                  [0, 0, 0, 0, 0, 1]])

    return b


def FormDiffA_mat2D(k, dvolume, delta_mag):
    d_mag = delta_mag
    # morder = str(n1order) + str(n2order)
    # nsize = getSize2D(n1order, n2order)
    # A00 = []
    # A01 = []
    # A10 = []
    # A11 = []
    # A02 = []
    # A20 = []
    # A22 = []
    A = np.zeros((6, 6), dtype=float)
    for imem in range(numfam[k]):
        i = node[imem]
        xsi1 = coord[i, 1] - coord[k, 1]  # x
        xsi2 = coord[i, 2] - coord[k, 2]  # y
        p = p_operator_2d(xsi1, xsi2, d_mag)
        w = weights_2d(xsi1, xsi2, d_mag)
        temp = w*dvolume[0]
        A[0, 0] = temp * p[0, 0] + A[0, 0]
        A[1:2, 0] = temp * p[1:2, 0] + A[1:2, 0]
        A[0, 1:2] = temp * p[0, 1:2] + A[0, 1:2]
        A[1:2, 1:2] = temp * p[1:2, 1:2] + A[1:2, 1:2]
        A[0, 3:5] = temp * p[0, 3:5] + A[0, 3:5]
        A[3:5, 0] = temp * p[3:5, 0] + A[3:5, 0]
        A[3:5, 3:5] = temp * p[3:5, 3:5] + A[3:5, 3:5]

    # A[0] = {'norder': '00', 'Amat': A00}
    # A[1] = {'norder': '01', 'Amat': A01}
    # A[2] = {'norder': '10', 'Amat': A10}
    # A[3] = {'norder': '11', 'Amat': A11}
    # A[4] = {'norder': '02', 'Amat': A02}
    # A[5] = {'norder': '20', 'Amat': A20}
    # A[6] = {'norder': '22', 'Amat': A22}
    # A[1].get('Amat')
    return A


def FormDiffB_vec2D():
    # morder = str(n1order) + str(n2order)
    matric = b_operator_2d()
    b = []
    # b00 = matric[:, 0]
    # b11 = matric[:, 1:2]
    # b22 = matric[:, 3:5]
    # b[0] = {'norder': '00', 'bmat': b00}
    # b[1] = {'norder': '11', 'bmat': b11}
    # b[2] = {'norder': '22', 'bmat': b22}
    b = matric
    return b


if __name__ == "__main__":
    totnodes = 3
    coord = np.array([[1, 2, 2],
                      [2, 3, 4],
                      [3, 4, 5]])
    d_volume = [1e-4]
    numfam = np.array([1])
    node = np.array([1])
    deltax = 0.2
    deltay = 0.2
    d_mag = np.sqrt(deltax**2 + deltay**2)
    P = p_operator_2d(2, 2, d_mag)
    W = weights_2d(2, 2, d_mag)
    A = FormDiffA_mat2D(0, d_volume, d_mag)
