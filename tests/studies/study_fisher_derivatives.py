"""Opt-in study of the accuracy of the Fisher derivatives.

Run this file explicitly with pytest. Its study_ prefix deliberately does
not match pytest's default test_*.py or *_test.py discovery patterns, so an
ordinary regression run does not repeat this expensive study (about a day
of evaluations on a laptop: the ladder of run_fisher_derivatives.py on the
Delta z = 0.05 layout and its spot check on Delta d_com = 62 Mpc).

The child processes print, for every configuration, the largest change
of the fiducial vector after all the steps (zero when every parameter is
restored) and the changes of the paper model's errors, figures of merit
and break-test shifts against the base configuration, flagged against
the tolerance; the test imposes no pass limit (the README states the
tolerance and the measured values). Only a failing stage fails it.
"""
from pathlib import Path
import subprocess
import sys

# the ladder of the README, in the order it runs: the cut, the reference
# and the MCMC settings (with their step ladders), the chi grid, the other
# controls, the candidate and DerivKit
LADDER = ["reference", "mcmc", "chi_grid_mcmc", "chi_grid_half", "chi_grid_double",
          "chi_grid_quad", "accuracyboost_2", "accuracyboost_3", "integration_accuracy_0",
          "camb_accuracy_1p05", "camb_accuracy_1p5", "camb_accuracy_3", "k_per_logint_15",
          "camb_kmax_40", "rank_5", "lowrank_lmax_1000", "candidate"]
DERIVKIT = ["candidate", "mcmc"]
LAYOUT = "dz0p05"
# the spot check of the thinnest layout: the candidate's stencil block at
# the step ladder, the per-bin columns from the model A and model B
# notebooks' caches in chains/ (candidate_chi_double_ladder, at twice the chi grid, is the
# fallback when the FoM moves with the step)
SPOT_LAYOUT = "dcom62"
SPOT_CHECK = ["candidate_ladder"]


def test_fisher_derivative_study(tmp_path):
    """Run every stage of the study on one layout into a fresh folder.

    Arguments:
      tmp_path = pytest-provided empty directory belonging to this test.

    Returns:
      None. The study writes its arrays below tmp_path/work and its JSON
      summaries below tmp_path/results; nothing in the repository changes.
      The spot check reads the model A and model B caches of chains/
      (the per-bin columns and the alpha_2 and SN2 templates), so the
      notebooks EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb and
      EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb must have run first.

    Each stage runs in its own Python process, because the compiled
    interface holds one model at a time and its accuracy baselines are
    fixed at the first initialization.
    """
    script = Path(__file__).with_name("run_fisher_derivatives.py")
    work = tmp_path / "work"
    results = tmp_path / "results"
    commands = [["cuts"]]
    for config in LADDER:
        commands.append(["derive", "--config", config])
    for config in DERIVKIT:
        commands.append(["dkit", "--config", config])
    commands.append(["compare"])
    for stage in commands:
        command = [sys.executable, str(script), stage[0], "--layout", LAYOUT,
                   "--work", str(work), "--results", str(results), *stage[1:]]
        subprocess.run(args=command, check=True)
    chains = Path(__file__).resolve().parents[2] / "chains"
    commands = [["cuts"]]
    for config in SPOT_CHECK:
        commands.append(["derive", "--config", config])
    commands.append(["compare", "--base", SPOT_CHECK[0], "--perbin-from", str(chains)])
    for stage in commands:
        command = [sys.executable, str(script), stage[0], "--layout", SPOT_LAYOUT,
                   "--work", str(work), "--results", str(results), *stage[1:]]
        subprocess.run(args=command, check=True)
    print(f"Saved the Fisher-derivative study: {results}", flush=True)
