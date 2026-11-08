# Marks this directory as a Python subpackage of the lssty1xdesi
# likelihoods. Cobaya imports it as cobaya.likelihoods.lssty1xdesi.exploration,
# so a yaml names a class here as lssty1xdesi.exploration.<class>, for
# example lssty1xdesi.exploration.desi_cl_model_a. It holds the likelihood
# classes and parameter fragments that the paper model (model B + tilts on
# the pessimistic linear-bias prior, the desi_cl*_model_b classes of the
# parent folder) does not use: model A and model A + tilts, model B without
# tilts, model B on the growth-tied linear bias, and the DR3 forecast
# fiducials. A class yaml here names the linear fragments of the parent
# folder as ../params_desi*.yaml; cobaya resolves a !defaults entry
# relative to the folder of the yaml that names it.
# projects/lssty1xdesi/exploration/README.md lists every file and the
# examples that run them.
