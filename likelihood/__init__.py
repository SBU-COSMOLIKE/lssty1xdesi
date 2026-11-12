# Marks this directory as a Python package. Cocoa links
# projects/lssty1xdesi/likelihood into Cobaya's likelihoods directory under the
# name lssty1xdesi, so Python imports it as cobaya.likelihoods.lssty1xdesi. The
# package holds the LSST likelihood modules (cosmic_shear, combo_xi_ggl,
# combo_xi_gg, combo_2x2pt, combo_3x2pt, and lsst_y1_3x2pt for the LSST Y1
# 3x2pt case), the DESI clustering likelihood desi_cl with one class per
# fixed-width layout (desi_cl_dr1_dz0p02, desi_cl_dr1_dz0p05,
# desi_cl_dr1_dz0p20), and the classes of the paper's bias model, model B +
# tilts on the pessimistic linear-bias prior (desi_cl_model_b and
# desi_cl_dr1_dz0p*_model_b). Each module has a yaml file of option
# defaults. The nuisance-parameter priors are params_source.yaml,
# params_lens.yaml and the DESI fragments params_desi*.yaml (written by
# scripts/make_desi_params.py); _cosmolike_prototype_base.py is the shared
# code. The subpackage exploration holds the parameter fragment the paper
# does not use (the DR3 forecast fiducials).
