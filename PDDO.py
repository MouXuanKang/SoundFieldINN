import numpy as np

if __name__ == "__main__":
    totnodes = 10020
    deltax = ([4])
    deltay = ([4])
    numfam = []
    node = []
    coord = np.array([[1, 2, 2],
                      [2, 3, 4],
                      [3, 4, 5]])




def p_operator_2d(xsi1, xsi2, delta_mag):
    xsi1p = xsi1 / delta_mag
    xsi2p = xsi2 / delta_mag
    plist = np.array([[1, xsi1p, xsi2p, xsi1p ^ 2, xsi2p ^ 2, xsi1p * xsi2p],
                      [xsi1p, xsi1p ^ 2, xsi1p * xsi2p, xsi1p ^ 3, xsi1p * xsi2p ^ 3, xsi1p ^ 2 * xsi2p],
                      [xsi2p, xsi1p * xsi2p, xsi2p ^ 2, xsi1p ^ 2 * xsi2p, xsi2p ^ 3, xsi1p * xsi2p ^ 2],
                      [xsi1p ^ 2, xsi1p ^ 3, xsi1p ^ 2 * xsi2p, xsi1p ^ 4, xsi1p ^ 2 * xsi2p ^ 2, xsi1p ^ 3 * xsi2p],
                      [xsi2p ^ 2, xsi1p * xsi2p ^ 2, xsi2p ^ 3, xsi1p ^ 2 * xsi2p ^ 2, xsi2p ^ 4, xsi1p * xsi2p ^ 3],
                      [xsi1p * xsi2p, xsi1p ^ 2 * xsi2p, xsi1p * xsi2p ^ 2, xsi1p ^ 3 * xsi2p, xsi1p * xsi2p ^ 3,
                       xsi1p ^ 2 * xsi2p ^ 2]
                      ])
    return plist


def weights_2d(xsi1, xsi2, delta_mag):
    xsi_mag = np.sqrt(xsi1 ^ 2 + xsi2 ^ 2)
    wt = np.exp(-4 * (xsi_mag / delta_mag) ^ 2)
    # wt = 1.0
    # wt = (xsi_mag/delta_mag)^2

    weight = np.array([[wt, wt, wt, wt, wt, wt],
                       [wt, wt, wt, wt, wt, wt],
                       [wt, wt, wt, wt, wt, wt],
                       [wt, wt, wt, wt, wt, wt],
                       [wt, wt, wt, wt, wt, wt],
                       [wt, wt, wt, wt, wt, wt]
                       ])
    return weight


def b_operator_2d():
    b = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 2, 0, 0],
                  [0, 0, 0, 0, 2, 0],
                  [0, 0, 0, 0, 0, 1]])

    return b


def getSize2D(n1order, n2order):
    iterm = 0
    morder = str(n1order) + str(n2order)
    match morder:
        case '00':
            iterm = 1
        case '01':
            iterm = 2
        case '10':
            iterm = 2
        case '11':
            iterm = 3
        case '20':
            iterm = 3
        case '02':
            iterm = 3
        case '21':
            iterm = 4
        case '12':
            iterm = 4
        case '22':
            iterm = 6
    nsize = iterm
    return nsize


def FormDiffAmat2D(n1order, n2order, k, dvolume):
    delta_mag = np.sqrt(deltax[k] ^ 2 + deltay[k] ^ 2)
    Amat = np.array([])
    morder = str(n1order) + str(n2order)
    nsize = getSize2D(n1order, n2order)
    for imem in range(numfam[k]):
        i = node[imem]
        xsi1 = coord[i, 1] - coord[k, 1]  # x
        xsi2 = coord[i, 2] - coord[k, 2]  # y
        p = p_operator_2d(xsi1, xsi2, delta_mag)
        w = weights_2d(xsi1, xsi2, delta_mag)
        match morder:
            case '00':
                Amat[0, 0] += w[0, 0].dot(p[0, 0]) * dvolume[i]
            case '01':
                Amat[1:2, 0] += w[1, 0].dot(p[1:2, 0]) * dvolume[i]
            case '10':
                Amat[0, 1:2] += w[0, 1].dot(p[0, 1:2]) * dvolume[i]
            case '11':
                Amat[1:2, 1:2] += w[1, 1].dot(p[1:2, 1:2]) * dvolume[i]
            case '02':
                Amat[0, 3:5] += w[2, 0].dot(p[0, 3:5]) * dvolume[i]
            case '20':
                Amat[3:5, 0] += w[0, 2].dot(p[3:5, 0]) * dvolume[i]
            case '12':
                Amat[1:2, 3:5] += w[3, 0].dot(p[1:2, 3:5]) * dvolume[i]
            case '21':
                Amat[3:5, 1:2] += w[0, 3].dot(p[3:5, 1:2]) * dvolume[i]
            case '22':
                Amat[3:5, 3:5] += w[3, 3].dot(p[3:5, 3:5]) * dvolume[i]
    return Amat


def FormDiffBvec2D(n1order, n2order, nsize):
    morder = str(n1order) + str(n2order)
    matric = b_operator_2d()
    b = []
    match morder:
        case '00':
            b = matric[:, 1]
            return b
        case '11':
            b = matric[:, 2:3]
            return b
        case '22':
            b = matric[:, 4:6]
            return b


