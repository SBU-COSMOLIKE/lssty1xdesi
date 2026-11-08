"""Write a DESI layout's fiducial data vector from one EXAMPLE yaml.

The data vector of a DESI dataset is a model at a fiducial point, not a
measurement. A .dataset file is the small text file listing the files
the likelihood reads, one "key = value" line each; its data_file line
names the data vector. An example evaluates to chi2 ~ 0 exactly when
that vector is the example's own model at the example's own point,
computed with the same accuracy settings.

This script builds the cobaya model of one EXAMPLE yaml, evaluates it at
the point of the yaml's evaluate sampler (its override block, the
fiducial), with the likelihood options the yaml sets, and writes the
theory vector into the data file of the example's dataset: two columns,
entry index and value, the format of the likelihood's print_datavector.
Masked entries are 0, as the likelihood computes them.

Each DESI layout ships two datasets (<stem> = lssty1xdesi_desi_<case>,
<case> = dr3 for Delta d_com = 62 Mpc, dr1_dz0p02, dr1_dz0p05,
dr1_dz0p20):

  <stem>.dataset        -> <stem>_fid.datavector, the paper's model
                           (model B + tilts on the pessimistic linear-bias
                           prior) at its fiducial, written from
                           EXAMPLE_EVALUATE<n>.yaml;
  <stem>_linear.dataset -> <stem>_linear_fid.datavector, the growth-tied
                           linear bias at the same cosmology, written from
                           exploration/EXAMPLE_EVALUATE_LINEAR<n>.yaml and
                           read by every example in exploration/.

Usage (from the Cocoa/ folder, cocoapy311 active, start_cocoa.sh
sourced):

    python ./projects/lssty1xdesi/scripts/make_desi_fiducial.py \
        ./projects/lssty1xdesi/EXAMPLE_EVALUATE1.yaml
    python ./projects/lssty1xdesi/scripts/make_desi_fiducial.py \
        ./projects/lssty1xdesi/exploration/EXAMPLE_EVALUATE_LINEAR1.yaml

With --output <file> the vector is written there instead, which compares
a regenerated vector with the shipped one without replacing it.
scripts/make_desi_dz_case.py --stage fiducial calls write_fiducial for
the two examples of a Delta z case.
"""
import argparse
import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")

# top-level keys of an EXAMPLE yaml that configure a cobaya run, not the
# model: get_model refuses some of them, and none changes the vector
RUN_KEYS = ("sampler", "output", "force", "resume", "debug", "timing",
            "stop_at_error")


def dataset_entry(dataset, key):
    """The value of one "key = value" line of a .dataset file.

    Arguments:
      dataset = path of the .dataset file.
      key     = the entry name, e.g. "data_file".

    Returns:
      the value as a string, stripped.

    Raises:
      KeyError when the file has no such line.
    """
    with open(dataset) as handle:
        for row in handle:
            # a line commented out with "#" is not an entry
            if row.lstrip().startswith("#") or "=" not in row:
                continue
            name, value = row.split("=", 1)
            if name.strip() == key:
                return value.strip()
    raise KeyError(f"{key} is not in {dataset}")


def write_fiducial(example, output=None):
    """Evaluate one EXAMPLE yaml at its fiducial point and write the vector.

    The vector goes first into a temporary file beside the target and
    replaces the target only after the evaluation succeeded and its
    layout matched the dataset's mask, so a failed run never leaves a
    truncated data file behind.

    Arguments:
      example = path of the EXAMPLE yaml (one likelihood block, an
                evaluate sampler with an override block).
      output  = file to write instead of the dataset's data_file, or
                None to replace the shipped data vector.

    Returns:
      (target path, number of rows, number of nonzero rows, chi2 of the
      evaluation against the data vector the dataset held before).

    Raises:
      ValueError when the yaml has no single likelihood block, no
      data_file or no evaluate override, or when the written vector does
      not have the
      mask's length or is nonzero on a masked entry.

    Side effects: writes the target file; the likelihood's
    print_datavector writes the temporary file.
    """
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(example)
    blocks = list(info.get("likelihood", {}))
    if len(blocks) != 1:
        raise ValueError(f"{example}: needs exactly one likelihood block, "
                         f"found {blocks}; pass an example of one DESI layout")
    name = blocks[0]
    evaluate = info.get("sampler", {}).get("evaluate", {}) or {}
    if "override" not in evaluate:
        raise ValueError(f"{example}: no sampler: evaluate: override block, "
                         "so it names no fiducial point")
    point = dict(evaluate["override"])
    like = info["likelihood"][name]
    if "data_file" not in like:
        raise ValueError(f"{example}: the {name} block sets no data_file; "
                         "name the .dataset file in the example")
    dataset = os.path.join(DATA, like["data_file"])
    mask_file = os.path.join(DATA, dataset_entry(dataset, "mask_file"))
    if output is None:
        target = os.path.join(DATA, dataset_entry(dataset, "data_file"))
    else:
        target = os.path.abspath(output)
    temporary = target + ".tmp"

    for key in RUN_KEYS:
        info.pop(key, None)
    like["print_datavector"] = True
    like["print_datavector_file"] = temporary
    model = get_model(info)
    # the override may list derived or fixed names; logposterior takes
    # the sampled parameters only
    sampled = model.parameterization.sampled_params()
    point = {k: v for k, v in point.items() if k in sampled}
    post = model.logposterior(point, cached=False)
    chi2_old = -2.0*float(post.loglikes[0])

    vector = np.loadtxt(temporary)
    mask = np.loadtxt(mask_file)
    if vector.shape[0] != mask.shape[0]:
        raise ValueError(f"{example}: the theory vector has {vector.shape[0]} "
                         f"rows, the mask {mask_file} {mask.shape[0]}")
    live = vector[:, 1] != 0.0
    stray = int(np.sum(live & (mask[:, 1] == 0)))
    if stray:
        raise ValueError(f"{example}: {stray} nonzero entries on masked rows; "
                         "the vector and the mask disagree on the layout")
    os.replace(temporary, target)
    return target, int(vector.shape[0]), int(np.sum(live)), chi2_old


def main():
    parser = argparse.ArgumentParser(
        description="Write the fiducial data vector of one DESI EXAMPLE yaml.")
    parser.add_argument("example", help="the EXAMPLE yaml to evaluate")
    parser.add_argument("--output", default=None,
                        help="write here instead of the dataset's data_file")
    args = parser.parse_args()
    target, rows, live, chi2_old = write_fiducial(args.example, args.output)
    print(f"wrote {target}: {rows} rows, {live} nonzero; chi2 of the "
          f"example against the vector it replaced = {chi2_old:.6g}")
    return 0


if __name__ == "__main__":
    main()
