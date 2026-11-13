"""Write plots/velocity_schematic/data.json: the numbers behind the
didactic schematic of the velocity decomposition and the bin-edge blur.

The schematic itself is hand drawn by plot_velocity_schematic.py; this
generator only computes its quantitative elements, so the drawn scales
are honest: the Fingers-of-God blur sigma_chi = (1 + z) sigma_v / H(z)
at z = 0.5 with the LRG1 fiducial sigma_v = 357 km/s and H(z) of flat
LambdaCDM at the note's fiducial (H_0 = 67.32 km/s/Mpc, Omega_m = 0.3;
the 0.3% neutrino correction to E(z) does not matter for a sketch),
against the 62 Mpc bin width, plus one example galaxy velocity and the
comoving shift it produces.  No CAMB call is needed.
"""
import json
import math
import os

H0, OMEGA_M = 67.32, 0.3
Z = 0.5
SIGMA_V = 357.0          # km/s, the LRG1 fiducial dispersion
BIN_WIDTH = 62.0         # Mpc, the equal-comoving-width layout
EXAMPLE_V = 587.0        # km/s, one galaxy's velocity (about 1.6 sigma)

E = math.sqrt(OMEGA_M*(1.0 + Z)**3 + (1.0 - OMEGA_M))
H = H0*E
sigma_chi = (1.0 + Z)*SIGMA_V/H
shift = (1.0 + Z)*EXAMPLE_V/H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "velocity_schematic")
os.makedirs(OUT, exist_ok=True)
data = {
    "about": ("make_velocity_schematic_data.py: quantitative elements of "
              "the hand-drawn schematic (the two parts of a galaxy's "
              "velocity, and the Fingers-of-God blur across a thin bin "
              "edge).  sigma_chi = (1 + z) sigma_v / H(z) at z = 0.5 with "
              "the LRG1 fiducial sigma_v and flat LambdaCDM "
              "(H_0 = 67.32 km/s/Mpc, Omega_m = 0.3); bin width of the "
              "equal-comoving-width layout; one example galaxy velocity "
              "and its comoving shift (1 + z) v / H(z)."),
    "z": Z,
    "sigma_v_kms": SIGMA_V,
    "H_kms_per_Mpc": round(H, 2),
    "sigma_chi_Mpc": round(sigma_chi, 2),
    "bin_width_Mpc": BIN_WIDTH,
    "example_v_kms": EXAMPLE_V,
    "example_shift_Mpc": round(shift, 2),
}
with open(os.path.join(OUT, "data.json"), "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print("wrote", os.path.join(OUT, "data.json"))
print("H(0.5) = %.2f km/s/Mpc, sigma_chi = %.2f Mpc, shift = %.2f Mpc"
      % (H, sigma_chi, shift))
