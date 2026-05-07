Running :
python3 Solver.py -rad 0.05 -mesh 1 -nnn 10 -which SubDomain

python3 Domain.py -name net01 -test SubDomain -inlet 5 -outlet 3

Codes for 1D graph generation and solution of 3D-1D coupled problem. The purpose is the generation of a "masked" FOM for the training of 3D-1D "masked" ROM.

rm -rf /home/francesco-virgulti/miniconda3/envs/fenics-env

conda create -n fenics-env -c conda-forge --strict-channel-priority --yes
python=3.10 fenics "ucx<1.20" binutils_linux-64 gcc_linux-64

conda activate fenics-env
python3 -c "import dolfin; print('Success! DOLFIN version:', dolfin.**version**)"

conda install -c conda-forge scipy matplotlib sympy

Here's a summary of all the fixes applied:

**1. Patched `ufl_legacy/__init__.py`** — removed broken `pkg_resources` imports:

```bash
sed -i '/import pkg_resources/d' .../ufl_legacy/__init__.py
sed -i '/__version__ = pkg_resources/d' .../ufl_legacy/__init__.py
```

**2. Patched `xii/assembler/ufl_utils.py`** — replaced `ufl_legacy` Terminal with plain `ufl`:

```bash
sed -i 's/from ufl_legacy.core.terminal import Terminal/from ufl.core.terminal import Terminal/' .../ufl_utils.py
```

**3. Patched `xii/assembler/average_form.py`** — replaced `ufl_legacy` imports:

```bash
sed -i 's/from ufl_legacy.corealg.traversal import traverse_unique_terminals/from ufl.corealg.traversal import traverse_unique_terminals/' .../average_form.py
sed -i 's/import ufl_legacy as ufl/import ufl/' .../average_form.py
```

**4. Patched all remaining xii files** — bulk replaced all `ufl_legacy` references:

```bash
grep -rl "import ufl_legacy as ufl" .../xii/ | xargs sed -i 's/import ufl_legacy as ufl/import ufl/'
grep -rl "ufl_legacy" .../xii/ | xargs sed -i 's/ufl_legacy/ufl/g'
```

**Root cause:** `xii` (fenics-ii) was installed in a version targeting **FEniCSx** (which uses `ufl_legacy` as a compatibility shim), but your environment runs **legacy FEniCS 2019.1.0** which uses plain `ufl` directly. All fixes were simply replacing `ufl_legacy` references with `ufl`.
