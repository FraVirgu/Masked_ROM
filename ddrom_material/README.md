# DD-ROM training example

This directory contains a reference implementation of the DD-ROM training workflow based on [`dlroms`](https://github.com/NicolaRFranco/dlroms).

The full-order datasets used in the application are **not included**. The purpose of the example is to provide the software environment, the DD-ROM architectures, and the training workflow used in the application.

## Contents

```text
ddrom_material/
├── install.sh
├── train_ddrom.py
└── librom/
    ├── __init__.py
    └── packages.py
```

`librom/packages.py` contains the application-specific finite-element utilities, DD-ROM architectures, training routines, and model-saving functions. `train_ddrom.py` assembles the surrogate maps, losses, optimizers, and training stages.

## 1. Installation

The scripts use the advanced FEniCS-based functionality of `dlroms`, together with additional packages required by the DD application.

Run:

```bash
bash install.sh
```

The script creates a Conda environment named `ddrom`, installs FEniCS and the Python dependencies, clones and installs `dlroms`, and installs the additional FEniCS utilities used by the DD code (`fenics_ii`, `cbc.block`, and `ulfy`).

For subsequent sessions activate the environment with:

```bash
conda activate ddrom
```

The training scripts assume a CUDA-capable compute node. The installation check reports whether CUDA is visible in the current shell; it is normal for CUDA not to be visible on some login nodes.

For the general `dlroms` API and additional examples, refer to the upstream repository:

https://github.com/NicolaRFranco/dlroms

## 2. Expected data layout

The full-order data are application-specific and are not distributed with this example. The training code expects the following layout:

```text
outputs/
├── FOM_DATA/
│   ├── u3d_new.npy
│   ├── u3d_old.npy
│   ├── u3d_coarse.npy
│   ├── eta.npy
│   ├── eta_ext.npy
│   ├── rgamma_eta.npy
│   ├── rgamma_u.npy
│   ├── dist.npy
│   ├── u1d.npy
│   ├── r.npy
│   └── r0.npy
└── ROM_DATA/
    ├── u3d_min.npy
    ├── u3d_max.npy
    └── ...
```

Depending on the loading flags in `train_ddrom.py`, `ROM_DATA` may also contain previously computed POD bases and trained network parameters (for example `V_u0.npy`, `psi1.npz`, `phi1_0.npz`, `phi2_0.npz`, `omega_0.npz`, and `decoder_0.npz`).

## 3. Running the reference training script

After activating the environment and providing data with the expected layout, run:

```bash
python train_ddrom.py
```

The flags near the beginning of `train_ddrom.py` control whether the different surrogate components are loaded or trained. They follow the workflow of the supplied DD-ROM application and are intentionally left application-specific.

The script contains the main stages used in the example:

- POD reduction of the 3D field;
- DtN surrogate architecture;
- subsolver surrogate and closure;
- coarse-solver surrogate;
- LBFGS-based training and model saving.

The file is meant as a reference training implementation rather than a self-contained benchmark: generation and preprocessing of the full-order snapshots remain part of the application workflow.
