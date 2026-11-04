# Data-vector and likelihood tests

The frozen snapshot `frozen/` and its manifest `manifest_sha256.json`
live in the parent `tests/` directory; the paths below are relative to
it.

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads). These
tests also measure the accuracy of the numerical settings, the hybrid
emulated pipelines and the baryonic-feedback models, and report whether
they are accurate enough for data analysis (advisory: no pass/fail).

The mechanism of every shared check family - the frozen-reference and
race contract, the zero-baseline construction, the sweeps and their
`--high`/`--mask` options, the accuracy settings and their meanings -
is documented once in
[the shared data-vector checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_checks)
of the CosmoLike core. This file records what this project runs and
what it measured.

The project's one measured configuration is the Fourier band-power
vector of `lssty1xdesi.desi_cl`: 15 LSST Y1 cosmic-shear pairs, 76
DESI DR3 thin-bin auto spectra and 97 adjacent-bin cross spectra, 53
band powers each (9,964 points). The project declares no source
nuisance parameters yet (intrinsic alignment amplitudes, shear
calibration, source photo-z shifts), so the TATT variants and the
CFASTPT-vs-FASTPT sweep of the donor suite have nothing to exercise;
declaring those parameters and porting those tests is a follow-up of
its own. The sampled nuisance block is DESI's: 76 linear biases
(`DESI_B1_*`) and 76 finger-of-god velocity dispersions
(`DESI_SIGV_*`).

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The Halofit vs EE2 checks](#halofit_ee2)
    2. [The EE2 modification tests](#ee2_tests)
    3. [Advisory checks](#advisory_checks)
    4. [Accuracy checks](#accuracy_checks)
    5. [The N-random-models check](#nmodels_check)
    6. [Baryonic feedback checks](#baryon_checks)
    7. [The photo-z convention checks](#photoz_conventions)
    8. [The non-Limber checks](#nonlimber_checks)
    9. [The low-rank engine check](#lowrank_check)
    10. [The n(z) reader and cache checks](#nz_readers)
    11. [The sector-ladder cache check](#cache_ladder)
3. [Appendix](#appendix)
    1. [FAQ: What does this project's snapshot hold?](#frozen_copy)
    2. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`. Every command
below assumes `start_cocoa.sh` has been sourced once:

    source start_cocoa.sh

Run the full suite with

    python -m pytest ./projects/lssty1xdesi/tests/data_vector

or, without pytest,

    python -m unittest discover -s ./projects/lssty1xdesi/tests/data_vector -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the computed
$\chi^2$, the stored reference, the difference, and the pass limit.
The test files force `OMP_NUM_THREADS=4` internally. The accuracy,
nonlinear and baryon comparisons dominate the runtime; the
frozen-reference, photo-z, non-Limber and cache checks together take
tens of minutes on a laptop.

## The tests <a name="the_tests"></a>

The standard configuration gets two tests - the frozen-reference
drift check and the race check, in NLA
([mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen)).
The suite follows the shared
[test-file naming convention](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_naming): a file name
means the same thing in every project. This project's
configurations and test numbers:

| file | configuration | tests |
|---|---|---|
| `test_example1.py` | EE + gg + gg-cross band powers (`lssty1xdesi.desi_cl`) | 1-2 |
| `test_nonlinear.py` | Halofit vs EE2 on the full vector | NL1 |
| `test_ee2.py` | modification pin; race | E-pin |
| `test_emul2.py` | hybrid emulators (advisory) | E1-E2 |
| `test_accuracy.py` | the knob scan and the all-knobs check | A0, A1 |
| `test_baryons.py` | feedback drift pins | BD1-BD7 |
| `test_accuracy_baryons.py` | feedback accuracy reports | BF0-BF7 |
| `test_photoz_conventions.py` | the runtime n(z) conventions | - |
| `test_nonlimber_gg.py` | non-Limber vs Limber clustering | - |
| `test_nonlimber_lowrank_gg.py` | low-rank vs FKEM engine | - |
| `test_nz_histogram_reader.py`, `test_nz_cache_roundtrip.py` | n(z) file reading and cache rebuilds | - |
| `test_cache_consistency.py` | the sector-ladder cache check | - |

Measured at the 2026-11-04 freeze: `example1` reproduces its frozen
reference at $\chi^2 = 1.2\times10^{-11}$.

### The Halofit vs EE2 checks (`test_nonlinear.py`, NL1) <a name="halofit_ee2"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_halofit_ee2):
NL1 evaluates the full band-power vector with both nonlinear-power
sources at the ten shared cosmologies. Measured on 2026-11-04:

- median $\Delta\chi^2 = 447$, maximum $4076$ (advisory; no pass
  limit).
- These are far larger than LSST Y1's shear-only numbers because the
  9,964-point vector weighs 173 thin-bin clustering spectra with
  spectroscopic precision: a percent-level difference between the two
  nonlinear $P(k)$ prescriptions is resolved at high significance.
  The two sources are not interchangeable for this project; an
  analysis must pick one and state it.

### The EE2 modification tests (`test_ee2.py`) <a name="ee2_tests"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_ee2):
the test pins the physics of Cocoa's EuclidEmulator2 modifications
against the pre-modification build, plus the race check with EE2 on.
Measured on 2026-11-04: max $\Delta\chi^2 = 5.2\times10^{-7}$ over
the ten cosmologies (limit 0.2) - the modifications change no emulated
physics on this project's vector.

### Advisory checks (`test_emul2.py`, E1-E2) <a name="advisory_checks"></a>

These checks test the hybrid emulators: trained machine-learning
networks replace the Boltzmann code. There is no pass/fail; each
check prints the emulated $\chi^2$, its drift against the stored
emulator reference, its error against the exact-physics $\chi^2$ at
the same point, and a recommendation (RECOMMENDED below 0.2).

Measured at the 2026-11-04 freeze: the emulated fiducial scores
$\chi^2 = 8.76$ against the exact-physics frozen vector. The
spectroscopic thin bins resolve transfer-function differences that
LSST Y1's photometric vector blurs, so the shipped networks are NOT
recommended for data analysis on this project until retrained at this
precision.

> [!NOTE]
> The trained-network files are read from
> `external_modules/data/emultrf`, not from the `frozen/` snapshot; the
> network device is pinned to `cpu` so the numbers do not depend on
> GPU availability.

### Accuracy checks (`test_accuracy.py`, A0-A1) <a name="accuracy_checks"></a>

[Mechanism, setting meanings and cost order](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_accuracy):
check A0 scans one setting at a time on the NLA configuration
(including `accuracyboost: 5` as a stress test); A1 re-evaluates the
vector with every setting pushed at once. Beyond the shared knobs,
the scan covers this project's own non-Limber controls: the FFTLog
$\chi$-grid density (`nonlimber_accuracyboost`, default 16 - the
0.05-wide comoving bins alias on a coarse grid, so this is the
accuracy-critical control), the kept SVD ranks
(`nonlimber_lowrank_rank`, default 5) and the engine's multipole cap
(`nonlimber_lowrank_lmax`, default 1000).

Measured on 2026-11-04 (A0, one knob at a time, advisory):

| setting | moved | $\Delta\chi^2$ |
|---|---|---:|
| `accuracyboost` (cosmolike) | 1 → 3 | +2.400 |
| `accuracyboost` (stress) | 1 → 5 | +2.298 |
| `internal_accuracyboost` | 1 → 2 | 0.000 |
| `integration_accuracy` | 0 → 10 | 0.000 |
| `lmax` | 5000 → 20000 | 0.000 |
| `kmax_boltzmann` + camb `kmax` | → 40, 50 | +0.00001 |
| camb `AccuracyBoost` | 1.05 → 2 | +0.086 |
| camb `k_per_logint` | 10 → 50 | +0.000002 |
| `nonlimber_accuracyboost` | 16 → 32 | +1.667 |
| `nonlimber_lowrank_rank` | 5 → 8 | 0.000 |
| `nonlimber_lowrank_lmax` | 1000 → 2000 | +0.0024 |

A1 (every knob at once): $\Delta\chi^2 = +2.454$.

Two knobs dominate and deserve attention before a production analysis:
the cosmolike `accuracyboost` (which refines the Limber projection
grids under the shear and cross blocks) and the non-Limber $\chi$-grid
boost, whose 16 → 32 step still moves $\Delta\chi^2$ by 1.7 on a
9,964-point vector. Both sit near the survey's statistical precision
rather than comfortably below it; the per-knob scan localizes any
future drift to its control.

Run them on their own, or skip them (high-accuracy evaluations take
minutes each):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py
    python -m pytest ./projects/lssty1xdesi/tests/data_vector --ignore ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py

### The N-random-models check (opt-in) <a name="nmodels_check"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_nmodels):
seeded random points across the NLA prior, each scored high-accuracy
against its own default-settings vector. Enable and run it with

    COCOA_ACCURACY_NMODELS=10 python -m pytest \
        ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py -k nmodels

Running the file as a script accepts `--nmodels N` in place of the
environment variable.

### Baryonic feedback checks (`test_accuracy_baryons.py` BF0-BF7, `test_baryons.py` BD1-BD7) <a name="baryon_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_baryons):
the accuracy checks (BF) report the default-versus-pushed
$\Delta\chi^2$ with each feedback method on; the drift tests (BD) pin
each method's frozen prediction. The seven methods are the three SP(k)
feedback relations, BCEmu, the Flamingo response emulator, BACCOemu
and BCemu2025. Measured on 2026-11-04: every BD pin reproduces its
frozen vector, and each BF high-versus-default sits at
$\Delta\chi^2 \approx 2.45$ (2.10 for one method) - the same two
knobs as the no-feedback A1, confirming feedback adds no numerical
sensitivity of its own. The commented `bfmt` theory block in
`EXAMPLE_EVALUATE1.yaml` shows how to turn each method on.

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_accuracy_baryons.py
    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_baryons.py

### The photo-z convention checks (`test_photoz_conventions.py`) <a name="photoz_conventions"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_photoz):
the frozen fiducial under the default, each alternative convention,
and the default again. This project's n(z) files tabulate bin centers,
so its default z-column reading is `Z_MID`
(`photoz_zmid_convention: 1` in `likelihood/desi_cl.yaml`), and the
sweep's alternatives are the two other interpolants and the `Z_LOW`
reading. Measured on 2026-11-04:

- default (cubic spline / Z_MID): $\chi^2 = $ 0.000000 (the
  frozen reference).
- linear: $\Delta\chi^2 = $ 0.479; Steffen:
  $\Delta\chi^2 = $ 0.087.
- Z_LOW: $\Delta\chi^2 = $ 3.47 - the half-bin z-column
  reading, which rigidly shifts every thin spectroscopic bin.

### The non-Limber checks (`test_nonlimber_gg.py`) <a name="nonlimber_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_nonlimber):
`adopt_limber_gg: 0` in this project, with no Limber hand-off below
the engine's cap: the 0.05-wide spectroscopic bins keep real
non-Limber power far past the $\ell = 150$ hand-off that photometric
surveys use, so the full density-density term is computed by the
low-rank engine up to its multipole cap.

Galaxy clustering, measured on 2026-11-04 (the test stores
`measured = ` 21157.6 and re-asserts it to 10% relative - a
drift pin on both engines, not an accuracy limit):

- $\Delta\chi^2 = $ 21157.6 between the non-Limber and
  pure-Limber predictions of the 9,964-point vector - the thin bins
  make the Limber approximation unusable for this project, not merely
  inaccurate.

### The low-rank engine check (`test_nonlimber_lowrank_gg.py`) <a name="lowrank_check"></a>

The project computes its non-Limber clustering with the low-rank
FFTLog engine (`nonlimber_lowrank_gg: 1`): a per-bin SVD compresses
$\sqrt{P_{gg}}$ to `nonlimber_lowrank_rank` rank-one profiles, whose
FFTLog projections rebuild every auto and cross spectrum up to
`nonlimber_lowrank_lmax`, with no Limber hand-off inside that range.
With the engine off (`nonlimber_lowrank_gg: 0`), clustering uses
FKEM: the FFTLog integral of the separable linear spectrum plus, in
Limber, what linear theory misses, below the cap `lmax_nonlimber`
(default 150), and pure Limber above. The cross pairs get the same
split, through the FKEM twin of the cosmolike core.

The test pins the difference between the two engines at the default
settings (`DCHI2_MEASURED = 4243.4`, re-asserted to 10% relative), a
drift check on both. The frozen data vector is the engine's own
prediction, so the pin equals the χ² the FKEM run scores against it.

To compare the methods fairly, FKEM must run to the same multipole
as the engine and cover the cross pairs. Measured on 2026-11-04, on
the same chi grid (`nonlimber_accuracyboost: 64`):

| FKEM configuration | $\Delta\chi^2$ vs the engine |
|---|---:|
| cap 150, cross pairs in Limber (the previous FKEM path) | 17,388 |
| cap 150, cross pairs with the FKEM split (the default) | 4,243 |
| cap 550, cross pairs with the FKEM split (every band) | **74.0** |

Each row isolates one approximation: the Limber cross pairs cost about
13,100, and the cap at 150 about 4,200. The remaining 74.0 is the
methods' genuine difference: FKEM keeps the nonlinear part of the
density term in Limber, while the engine projects the full nonlinear
$P_{gg}$ without approximation. It is spread over many spectra - the
largest single contributions are mid-redshift cross pairs of about
4.4 each - with 70.6 from the auto block and 59.5 from the cross
block alone (the two overlap through the correlations of the
covariance, so they do not add). At the equal cap, FKEM evaluates in
about half the engine's time (4.7 s against 10.2 s, same run,
8 threads).

The rank-5 compression
itself costs nothing at this precision (the A0 scan's rank 5 → 8 row
moves 0.000).

### The n(z) reader and cache checks (`test_nz_histogram_reader.py`, `test_nz_cache_roundtrip.py`) <a name="nz_readers"></a>

The histogram reader test checks the file-to-spline path of the n(z)
machinery against an independent Python reading; the roundtrip test
flips the runtime n(z) flags away from and back to the defaults in
one process and asserts the rebuilt cache reproduces the original
vector bit for bit. Both matter more here than in photometric
projects: the FoG convolution (`external_nz_modeling: 1`) rebuilds
the 76 thin lens distributions at every evaluation.

### The sector-ladder cache check (`test_cache_consistency.py`) <a name="cache_ladder"></a>

The deterministic mixed-update ladder and its bitwise assertions are
described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_cache_ladder).
This project samples cosmology, galaxy bias and the FoG velocity
dispersions only, so the source-side rungs (IA, source photo-z, shear
calibration) drop out of the ladder, the TATT variant skips, and bias
still moves at the scramble point.

## Appendix <a name="appendix"></a>

## :interrobang: FAQ: What does this project's snapshot hold? <a name="frozen_copy"></a>

The snapshot-protection mechanism is described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen).
This project's `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_example1.py` | the band-power configuration as a yaml string, plus the exact evaluation point |
| `frozen_config_emul2_example1.py` | the same configuration with the hybrid emulators, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vector, covariance, n(z) histograms, window functions and mask, plus the baryonic-feedback vectors the generator writes |
| `reference_chi2.json` | the reference $\chi^2$ values, one per configuration and variant |
| `EXAMPLE_EVALUATE1.yaml` | snapshot of the live example, kept only so a human can diff how it drifted since the freeze |
| `EXAMPLE_EMUL2_EVALUATE1.yaml` | snapshot of the hybrid example, kept only so a human can diff how it drifted since the freeze |

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vector, n(z), covariance, examples,
or likelihood defaults requires a re-freeze. The `_meta` block of
`frozen/reference_chi2.json` records when the current references were
generated.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/lssty1xdesi/tests/generate_frozen_reference.py --overwrite

It deletes and rebuilds `frozen/` from the current project, prints one
reference $\chi^2$ per configuration, and rewrites the manifest. This
step does not write the baryonic-feedback vectors of
`test_baryons.py`.

**Step :three:**: write the baryonic-feedback vectors

    python ./projects/lssty1xdesi/tests/generate_frozen_reference.py --baryons

It evaluates each feedback method once, writes its vector and dataset
file under `frozen/data/`, and re-pins the manifest. Review the
printed $\chi^2$ values of both steps before committing: they become
the new references the tests compare against.

**Step :four:**: re-measure and re-pin the two non-Limber deltas. The
Limber-vs-non-Limber and engine-agreement values stored in
`test_nonlimber_gg.py` (`measured=`) and
`test_nonlimber_lowrank_gg.py` (`DCHI2_MEASURED`) belong to the frozen
vector; after a re-freeze, run each file once with `-s`, read the
printed `delta^T C^-1 delta`, and update the stored value.
