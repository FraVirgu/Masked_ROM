import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from dlroms.roms import POD
from librom.packages import (
    fespace, importdata, PODprojectionerrors,
    PODMINN1, PODMINN2, train_flux, train_solver,
)

numpy_seed = 13435
torch_seed = 10001
np.random.seed(numpy_seed)
torch.manual_seed(torch_seed)

l1, l2, linf, h1, meshV, Vh = fespace(20)
u3d_new, u3d_old, u1d, eta, eta_ext, rgamma, d, u3d_min, u3d_max = importdata(
    './outputs/FOM_DATA_new/'
)

scaling_eta_dtn = 1e15
scaling_eta = 5e13
ratio = scaling_eta / scaling_eta_dtn

eta *= scaling_eta_dtn
eta_ext *= scaling_eta
rgamma *= scaling_eta

ntrain = 1600 + 64 * 27
ntest = len(d) - ntrain
modes_u = 50
modes_eta = 250

V_u, _ = POD(u3d_new[:ntrain], k=modes_u, inner=None)
V_eta, _ = POD(eta[:ntrain], k=modes_eta, inner=None)

print('\nPOD on 3D pressures\n')
PODprojectionerrors(
    u3d_new, u3d_new.mm(V_u.T), V_u,
    l1, l2, linf, h1, ntrain,
)

# ROM flussi: surrogate della mappa DtN
psi1, psi2 = PODMINN1(Vh, modes_eta, torch_seed)

def NN1(u, d):
    return psi1(u) * psi2(d)

def ROM1(u, d):
    return NN1(u, d).mm(V_eta)

def loss_flux(eta, u3d_old, d):
    return (eta.mm(V_eta.T) - NN1(u3d_old, d)).pow(2).mean()

def error_flux(utrue, u3d_old, d, norm):
    return (norm(utrue - ROM1(u3d_old, d)) / norm(utrue)).mean().item()

optimizer_flux = torch.optim.LBFGS(
    psi1.parameters() + psi2.parameters(), lr=1
)
err_flux, time_flux = train_flux(
    eta, u3d_old, d, ntrain, ntest,
    loss_flux, error_flux, optimizer_flux,
    epochs=200,
)

# ROM solver: surrogate del subsolver DD
phi1, phi2, omega, decoder = PODMINN2(Vh, modes_u, torch_seed)

def NN2(u1d, d, eta):
    return decoder(torch.cat([phi1(u1d), phi2(d), omega(eta)], dim=-1))

def ROM2(u1d, d, eta):
    return NN2(u1d, d, eta).mm(V_u)

def loss_u(u3d_new, u1d, d, eta):
    return (u3d_new.mm(V_u.T) - NN2(u1d, d, eta)).pow(2).mean()

def loss_12(u1d, d, eta, rgamma):
    eta_pred = -ratio * ROM1(ROM2(u1d, d, eta), d).detach() + rgamma
    return l2(eta - eta_pred).mean()

def loss_21(u3d_new, u1d, d, rgamma):
    eta_pred = rgamma - ratio * ROM1(u3d_new, d).detach()
    return l2(u3d_new - ROM2(u1d, d, eta_pred)).mean()

def loss_solver(u3d_new, u1d, d, eta, rgamma):
    return (
        0.2 * loss_u(u3d_new, u1d, d, eta)
        + 0.4 * loss_12(u1d, d, eta, rgamma)
        + 0.4 * loss_21(u3d_new, u1d, d, rgamma)
    )

def error_solver(utrue, u1d, d, eta, norm):
    return (norm(utrue - ROM2(u1d, d, eta)) / norm(utrue)).mean().item()

optimizer_solver = torch.optim.LBFGS(
    phi1.parameters() + phi2.parameters() + omega.parameters() + decoder.parameters(),
    lr=1,
)
err_solver, time_solver = train_solver(
    u3d_new, u1d, d, eta_ext, rgamma, ntrain, ntest,
    loss_solver, error_solver, optimizer_solver,
    epochs=50,
)
