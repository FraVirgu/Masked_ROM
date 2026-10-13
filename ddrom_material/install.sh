#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ddrom"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 1
fi

# Enable `conda activate` also when this file is executed as a script.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Base environment used by the DD-ROM application. Reuse it if it already
# exists, so the script can also be run again to repair/update dependencies.
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
    conda create --name "${ENV_NAME}" -c conda-forge fenics python=3.10 pip git -y
fi
conda activate "${ENV_NAME}"

# Python dependencies used by the DD-ROM scripts.
python -m pip install \
    matplotlib \
    scipy \
    networkx \
    tqdm \
    gmsh \
    meshio \
    ipython

mkdir -p "${EXTERNAL_DIR}"

clone_if_missing() {
    local repo_url="$1"
    local target_dir="$2"

    if [ -d "${target_dir}/.git" ]; then
        echo "Repository already present: ${target_dir}"
    else
        git clone "${repo_url}" "${target_dir}"
    fi
}

# DLROMs library. The upstream README also supports direct installation from
# the Git repository; keeping a local clone is useful here for inspection and
# development during the thesis work.
clone_if_missing \
    "https://github.com/NicolaRFranco/dlroms.git" \
    "${EXTERNAL_DIR}/dlroms"
python -m pip install -e "${EXTERNAL_DIR}/dlroms"

# Additional FEniCS utilities required by the DD application. cbc.block is a
# dependency of fenics_ii; ulfy is used by several fenics_ii workflows.
clone_if_missing \
    "https://bitbucket.org/fenics-apps/cbc.block.git" \
    "${EXTERNAL_DIR}/cbc.block"
python -m pip install "${EXTERNAL_DIR}/cbc.block"

clone_if_missing \
    "https://github.com/MiroK/ulfy.git" \
    "${EXTERNAL_DIR}/ulfy"
python -m pip install "${EXTERNAL_DIR}/ulfy"

clone_if_missing \
    "https://github.com/MiroK/fenics_ii.git" \
    "${EXTERNAL_DIR}/fenics_ii"
python -m pip install "${EXTERNAL_DIR}/fenics_ii"

# Basic installation check. CUDA availability may be false on a login node;
# the DD-ROM scripts themselves assume a CUDA-capable compute node.
python - <<'PY'
import dolfin
import dlroms
import torch

print("FEniCS/dolfin import: OK")
print("dlroms import:       OK")
print("PyTorch import:      OK")
print("CUDA available:     ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:         ", torch.cuda.get_device_name(0))
else:
    print("NOTE: CUDA is not visible in this shell. Run training on a GPU compute node.")
PY

echo
echo "Installation completed."
echo "For subsequent sessions run: conda activate ${ENV_NAME}"
