import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

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
    idx_r = val // nz
    idx_z = val % nz
    G00_loop.append(Family_Node_G00[idx_z, idx_r, :, :].flatten())
    G02_loop.append(Family_Node_G02[idx_z, idx_r, :, :].flatten())
    G20_loop.append(Family_Node_G20[idx_z, idx_r, :, :].flatten())
    R_loop.append(Family_Node_R[idx_z, idx_r, :, :].flatten())
    Z_loop.append(Family_Node_Z[idx_z, idx_r, :, :].flatten())
    C_loop.append(Family_Node_C[idx_z, idx_r, :, :].flatten())
    Rep_loop.append(Family_Node_Rep[idx_z, idx_r, 0, 0].flatten())
    Imp_loop.append(Family_Node_Imp[idx_z, idx_r, 0, 0].flatten())

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

data = loadmat('Data/cylinder_pre_c0_w0.mat')
R = data['R']  # R x 1
Z = data['Z']  # Z x 1
dr = 50
dz = 5
dv = dr*dz
r_grid, z_grid = np.meshgrid(R[:, 0:-1], Z[:, 0:-1])
# shape
lr = R.shape[1]
lz = Z.shape[1]
tl_eval = np.sqrt(Rep_target ** 2 + Imp_target ** 2).reshape(lz-1, -1)

# Rep_PDO = Rep_train[:, 0]
# Imp_PDO = Imp_train[:, 0]
Rep_PDO = np.sum(Rep_train*G00_train, axis=1) * dv
Imp_PDO = np.sum(Imp_train*G00_train, axis=1) * dv
tl_pddo = np.sqrt(Rep_PDO ** 2 + Imp_PDO ** 2).reshape(lz-1, -1)

fig = plt.figure(1, figsize=(9, 4))
ax = plt.subplot(1, 2, 1)
ax.invert_yaxis()
h0 = ax.pcolormesh(r_grid / 1e3, z_grid, tl_eval, cmap='jet', shading='nearest')
ax.set_title('exact')
ax.set_xlabel('$range, km$')
ax.set_ylabel('$depth, m$')

ax1 = plt.subplot(1, 2, 2)
ax1.invert_yaxis()
h1 = ax1.pcolormesh(r_grid / 1e3, z_grid, tl_pddo-tl_eval, cmap='jet', shading='nearest')
ax1.set_title('PDDO')
ax1.set_xlabel('$range, km$')
ax1.set_ylabel('$depth, m$')
plt.show()

