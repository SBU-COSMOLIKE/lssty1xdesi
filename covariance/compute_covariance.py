"""Compute and save the lssty1xdesi covariance through the production interface.

The covariance matrix of the data vector (9964 Fourier entries: 15 EE
spectra, 76 clustering autos and 97 clustering cross pairs, 53
bandpowers each) is the sum of three terms: Gaussian (G), super-sample
covariance (SSC) and connected non-Gaussian (cNG). This script evaluates
them at the one cosmology written in a Cobaya-style yaml file and saves G,
SSC, cNG, their sum, the coordinates of every row and the resolved settings
in one .npz archive (the yaml's output key, or --output; an existing
archive is replaced only with --overwrite). The production interface is
the covariance part of the compiled module cosmolike_lssty1xdesi_interface,
called directly with numpy arrays; the notebook wrappers reach the same C
kernels with more copying.

From an activated Cocoa installation, in the folder cocoa/Cocoa:
    python projects/lssty1xdesi/covariance/compute_covariance.py \
        projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE.yaml

See --help and covariance/README.md for space and accuracy options. The
survey settings (lssty1xdesi_covariance.py) and the numerical model are shared
with the notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb. A library compiled
without covariance support stops the script with a message that points to
the project README.
"""

import os
from pathlib import Path
import sys

# OpenBLAS, Intel MKL and Apple's Accelerate (vecLib) read these variables
# once, when numpy first loads them, so they are set before any numerical
# import: one BLAS thread per process. Cosmolike's own OpenMP threads
# follow OMP_NUM_THREADS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/lssty1xdesi (two levels above this file); core = the
# cosmolike_core checkout, which holds the cosmolike_notebook_utils package.
# sys.path is the list of directories Python searches on import; the
# compiled module (interface/) and core go first. lssty1xdesi_covariance is
# found in this script's own directory, which Python adds to sys.path when
# it runs a script.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_lssty1xdesi_interface as ci
import lssty1xdesi_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# __name__ is "__main__" only when this file runs as a script. run_covariance
# parses the command line, checks the output path and the yaml before any
# numerical work, runs CAMB and the covariance terms, and writes the archive.
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="fourier", joint=False,
    )
