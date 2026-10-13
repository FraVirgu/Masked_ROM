import os
import warnings

import dolfin
import numpy as np
import torch
from dolfin import BoxMesh, Point
from IPython.display import clear_output
from torch import tanh

from dlroms import fespaces as fe
from dlroms.cores import GPU
from dlroms.dnns import Clock, Dense
from dlroms.minns import H1, L1, L2, Linf, Local
from dlroms.roms import num2p


def to01_u(T):
    vmax = T.max(axis=0)[0]
    vmin = T.min(axis=0)[0]
    return (T - vmin) / (vmax - vmin), vmin, vmax


def fespace(n=10):
    mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), n, n, n)
    Vh = fe.space(mesh, 'CG', 1)

    l1 = L1(Vh)
    l2 = L2(Vh)
    h1 = H1(Vh)
    linf = Linf()
    for norm in (l1, l2, h1, linf):
        norm.cuda()

    return l1, l2, linf, h1, mesh, Vh


def importdata(folder, seed=0):
    np.random.seed(seed)
    warnings.filterwarnings('ignore')

    u3d_new, u3d_min, u3d_max = to01_u(GPU.load(os.path.join(folder, 'u3d_new.npy')))
    u3d_old = (GPU.load(os.path.join(folder, 'u3d_old.npy')) - u3d_min) / (u3d_max - u3d_min)
    u1d = (GPU.load(os.path.join(folder, 'u1d.npy')) - 0.05) / 0.02

    eta = GPU.load(os.path.join(folder, 'eta.npy'))
    eta_ext = GPU.load(os.path.join(folder, 'eta_ext.npy'))
    rgamma = GPU.load(os.path.join(folder, 'rgamma_u.npy'))
    d = GPU.load(os.path.join(folder, 'dist.npy'))

    return u3d_new, u3d_old, u1d, eta, eta_ext, rgamma, d, u3d_min, u3d_max


def PODprojectionerrors(u, ub, V, l1, l2, linf, h1, ntrain):
    norms = [('L1', l1), ('L2', l2), ('L_inf', linf), ('H1', h1)]

    for name, norm in norms:
        err_train = (norm(u[:ntrain] - ub[:ntrain].mm(V)) / norm(u[:ntrain])).mean().item()
        err_test = (norm(u[ntrain:] - ub[ntrain:].mm(V)) / norm(u[ntrain:])).mean().item()
        print(f'{name} projection error (training-test): {num2p(err_train)} - {num2p(err_test)}')


def stretch(x, alpha=3.0):
    return 0.5 * (
        1.0 + torch.tanh(alpha * (x - 0.5)) / torch.tanh(torch.tensor(alpha * 0.5))
    )


def PODMINN1(Vh, modes, seed):
    smallmesh = dolfin.cpp.generation.UnitCubeMesh(8, 8, 8)
    coords = torch.tensor(smallmesh.coordinates(), dtype=torch.float64)
    for i in range(3):
        coords[:, i] = stretch(coords[:, i])
    smallmesh.coordinates()[:] = coords.numpy()

    smallVh = fe.space(smallmesh, 'CG', 1)
    hdof = fe.coordinates(Vh)
    cdof = fe.coordinates(smallVh)

    psi1 = Local(hdof, cdof, 0.2, activation=tanh) + Dense(
        smallVh.dim(), modes, activation=None
    )
    psi2 = Local(hdof, cdof, 0.2, activation=tanh) + Dense(
        smallVh.dim(), modes, activation=None
    )

    for net in (psi1, psi2):
        net.Xavier()
        net.cuda()

    return psi1, psi2


def PODMINN2(Vh, modes, seed):
    smallmesh = dolfin.cpp.generation.UnitCubeMesh(10, 10, 10)
    smallVh = fe.space(smallmesh, 'CG', 1)
    hdof = fe.coordinates(Vh)
    cdof = fe.coordinates(smallVh)
    activation = torch.nn.Tanh()

    phi1 = (
        Local(hdof, cdof, 0.3, activation=activation)
        + Local(cdof, cdof, 0.3, activation=activation)
        + Dense(smallVh.dim(), modes, activation=None)
    )
    phi2 = (
        Local(hdof, cdof, 0.3, activation=activation)
        + Local(cdof, cdof, 0.3, activation=activation)
        + Dense(smallVh.dim(), modes, activation=None)
    )
    omega = (
        Local(hdof, cdof, 0.3, activation=activation)
        + Local(cdof, cdof, 0.3, activation=activation)
        + Dense(smallVh.dim(), modes, activation=None)
    )
    decoder = (
        Dense(3 * modes, 2 * modes, activation=torch.nn.LeakyReLU(0.1))
        + Dense(2 * modes, 2 * modes, activation=torch.nn.LeakyReLU(0.1))
        + Dense(2 * modes, modes, activation=None)
    )

    for net in (phi1, phi2, omega, decoder):
        net.Xavier()
        net.cuda()

    return phi1, phi2, omega, decoder


def train_flux(eta, u3d_old, d, ntrain, ntest, lossf, errorf, optimizer, epochs=200, verbose=True):
    _, l2, linf, _, _, _ = fespace(20)
    etrain, etest = eta[:ntrain], eta[-ntest:]
    utrain, utest = u3d_old[:ntrain], u3d_old[-ntest:]
    dtrain, dtest = d[:ntrain], d[-ntest:]

    errors = []
    clock = Clock()
    clock.start()

    for e in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = lossf(etrain, utrain, dtrain)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            errors.append([
                errorf(etrain, utrain, dtrain, l2), errorf(etest, utest, dtest, l2),
                errorf(etrain, utrain, dtrain, linf), errorf(etest, utest, dtest, linf),
            ])

        if verbose:
            clear_output(wait=True)
            err = errors[-1]
            print(
                f'Flux ROM - epoch {e + 1}: '
                f'{num2p(err[0])} / {num2p(err[1])} L2, '
                f'{num2p(err[2])} / {num2p(err[3])} L_inf (train/test)'
            )

    clock.stop()
    print('Training complete. Elapsed time: ' + clock.elapsedTime())
    return errors, clock.elapsed()


def train_solver(u3d_new, u1d, d, eta_ext, rgamma, ntrain, ntest, lossf, errorf, optimizer, epochs=50, verbose=True):
    _, l2, linf, _, _, _ = fespace(20)
    utrain, utest = u3d_new[:ntrain], u3d_new[-ntest:]
    u1train, u1test = u1d[:ntrain], u1d[-ntest:]
    dtrain, dtest = d[:ntrain], d[-ntest:]
    etrain, etest = eta_ext[:ntrain], eta_ext[-ntest:]
    rtrain = rgamma[:ntrain]

    errors = []
    clock = Clock()
    clock.start()

    for e in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = lossf(utrain, u1train, dtrain, etrain, rtrain)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            errors.append([
                errorf(utrain, u1train, dtrain, etrain, l2), errorf(utest, u1test, dtest, etest, l2),
                errorf(utrain, u1train, dtrain, etrain, linf), errorf(utest, u1test, dtest, etest, linf),
            ])

        if verbose:
            clear_output(wait=True)
            err = errors[-1]
            print(
                f'Solver ROM - epoch {e + 1}: '
                f'{num2p(err[0])} / {num2p(err[1])} L2, '
                f'{num2p(err[2])} / {num2p(err[3])} L_inf (train/test)'
            )

    clock.stop()
    print('Training complete. Elapsed time: ' + clock.elapsedTime())
    return errors, clock.elapsed()
