"""Write pair_graph_bandwidth2/data.json for the Fisher machinery note: the
pair graph of the four DESI layouts when every pair of bins two apart in
one galaxy sample, C_{i,i+2}, joins the cross pairs the data files list.

Words used below. The DESI data vector of one bin layout holds the auto
spectrum C_ii of every thin redshift bin i and the cross spectrum C_ij of
every pair (i, j) listed in data/<dataset>_pairs.txt; these listed pairs
are "today's pairs". Two bins are partners when a measured cross spectrum
contains both, and the degree of a bin is its number of partners. The
pair graph draws each bin as a dot and each pair as a line between two
dots.

The notebooks compute the per-bin bias derivatives in batches: a batch is
a set of bins of which no two are partners, so every spectrum holds at
most one stepped bin and one five-point stencil gives the derivative
column of every bin of the batch (plots/batching/ draws this for today's
pairs). Splitting the bins into batches is a coloring of the pair graph:
each batch is a color, and two partners never share a color.

For each layout the script computes, on the bandwidth-2 list:

  - the list itself: today's pairs plus every pair (i, i+2) whose two bins
    belong to the same sample, the column 'sample' of _bins.txt (bgs, lrg
    or elg). The sample decides, not the group, so the pairs across a
    group boundary of one sample (LRG1-LRG2, LRG2-LRG3, ELG1-ELG2) are
    added and the BGS-LRG1 pairs are not;
  - the degree of every bin and its 'earlier' count, the number of its
    partners with a lower bin index;
  - the greedy coloring in bin (data-vector) order, the one pair_coloring
    of scripts/compute_fisher_base_mpi.py runs: each bin, in index order,
    takes the lowest batch that none of its earlier partners holds. A
    bin's batch is therefore at most 1 + its earlier count, and the greedy
    pass never needs more than 1 + max(earlier) batches;
  - the fewest batches any assignment could use (the chromatic number of
    the pair graph). A clique is a set of bins that are all partners of
    each other; it needs one batch per bin, so the largest clique bounds
    the fewest batches from below. A DSATUR coloring (degree of
    saturation, Brelaz 1979) that uses exactly that many batches shows the
    bound is reached. DSATUR colors one bin at a time: always the
    uncolored bin whose colored partners already hold the most distinct
    batches, which takes the lowest batch none of them holds. The script
    stops with an error when DSATUR needs more batches than the clique
    has bins;
  - for comparison, the greedy coloring in mean-redshift (zbar) order.

Inputs, all inside the repository and read only:

  - notes/cross_correlations/plots/nz_overlap_layouts/data.json (written
    by make_cross_correlations_data.py in that plots/ folder): every bin's
    group, sample, rounded zbar and raw n(z), today's pairs, the color,
    marker and label of each layout and the redshift range LRG3 and ELG1
    share. The figure redraws these n(z);
  - data/lssty1xdesi_desi_<layout>_bins.txt and _pairs.txt, for the
    layouts dr1_dz0p20, dr1_dz0p05, dr1_dz0p02 (Delta z = 0.2, 0.05,
    0.02) and dr3 (Delta d_com = 62 Mpc): the sample, group and mean
    redshift of every bin and today's pairs. They are checked against the
    data.json above, so the drawn n(z) and the batches describe the same
    bins.

Run from any folder, inside the repository:
  python make_pair_graph_bandwidth2_data.py
or with the project's data folder named explicitly:
  python make_pair_graph_bandwidth2_data.py /path/to/projects/lssty1xdesi/data
then redraw with pair_graph_bandwidth2/plot_pair_graph_bandwidth2.py.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in projects/lssty1xdesi/notes/fisher_machinery/plots/,
# three folders below the project, whose data/ holds the bin tables and
# pair lists
DEFAULT_DATA = os.path.join(HERE, '..', '..', '..', 'data')
# the bins, n(z) and today's pairs of the four layouts, as the
# cross-correlation note draws them
SOURCE = os.path.join(HERE, '..', '..', 'cross_correlations', 'plots',
                      'nz_overlap_layouts', 'data.json')
# the plot folder whose data.json this script writes
OUTPUT_DIR = os.path.join(HERE, 'pair_graph_bandwidth2')

# The dataset stem of each layout, by the 'key' of the layout in SOURCE.
# The stem names the layout's files in data/ (<stem>_bins.txt and
# <stem>_pairs.txt); dr3 is the Delta d_com = 62 Mpc layout.
DATASETS = {
    'dr1_dz0p20': 'lssty1xdesi_desi_dr1_dz0p20',
    'dr1_dz0p05': 'lssty1xdesi_desi_dr1_dz0p05',
    'dr1_dz0p02': 'lssty1xdesi_desi_dr1_dz0p02',
    'dr3': 'lssty1xdesi_desi_dr3',
}
# SOURCE stores zbar rounded to 5 decimals, so it differs from the
# 8-decimal value of _bins.txt by at most 5e-6; a larger difference means
# the two files describe different bins
ZBAR_TOLERANCE = 1.0e-5
# largest_clique tries every subset of a bin's higher-index partners,
# 2^d subsets for d partners. The bandwidth-2 lists have d <= 7 (128
# subsets); at d = 20 a bin alone needs about a million, so the search
# refuses larger counts instead of running for minutes
MAX_LATER_PARTNERS = 20


def read_rows(path):
    '''Return the data rows of a text file, each split into words.

    Lines that are empty or start with # are skipped, as in read_rows of
    the notebooks and of plots/batching/make_batching_data.py.

    Arguments:
      path = the file to read.

    Returns:
      list of lists of strings, one per data line.
    '''
    rows = []
    # the with-block closes the file when the loop ends
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                rows.append(line.split())
    return rows


def read_layout_files(data_dir, dataset):
    '''Read one layout's bin table and pair list from the data folder.

    The bin table has one row per bin: bin index, sample (bgs, lrg or
    elg), chunk_index, chunk_name (the group: BGS, LRG1, LRG2, LRG3, ELG1
    or ELG2) and the mean redshift zbar. The pair list has one row "i j"
    (0-based, i < j) per measured cross spectrum.

    Arguments:
      data_dir = the project's data folder.
      dataset  = the dataset stem, e.g. 'lssty1xdesi_desi_dr3'.

    Returns:
      dict: 'samples', 'groups' (lists of str), 'zbar' (list of float),
      one entry per bin in bin order, and 'pairs' (sorted list of (i, j)).

    Raises:
      ValueError when the bin column is not 0, 1, 2, ... or a pair is not
      two distinct bins of the table with i < j.
    '''
    samples = []
    groups = []
    zbar = []
    for i, row in enumerate(read_rows(os.path.join(data_dir, dataset + '_bins.txt'))):
        if int(row[0]) != i:
            raise ValueError(f'{dataset}_bins.txt row {i} has bin index {row[0]}; the '
                             'bins must be listed in order 0, 1, 2, ...; regenerate the '
                             'bin table')
        samples.append(row[1])
        groups.append(row[3])
        zbar.append(float(row[4]))
    nbins = len(samples)
    pairs = []
    for row in read_rows(os.path.join(data_dir, dataset + '_pairs.txt')):
        a = int(row[0])
        b = int(row[1])
        if not (0 <= a < b < nbins):
            raise ValueError(f'{dataset}_pairs.txt names the pair ({a}, {b}); a pair '
                             f'must be two bins i < j of the table (0 to {nbins - 1}); '
                             'regenerate the pair file')
        pairs.append((a, b))
    pairs.sort()
    return {'samples': samples, 'groups': groups, 'zbar': zbar, 'pairs': pairs}


def check_against_source(layout, tables):
    '''Check that SOURCE and the data files describe the same bins and pairs.

    The figure draws the n(z) of SOURCE and the batches computed from the
    data files; both must hold the same bins (sample, group and zbar,
    within the rounding of SOURCE) and the same list of today's pairs.

    Arguments:
      layout = one entry of SOURCE's 'layouts' list.
      tables = what read_layout_files returned for that layout.

    Returns:
      nothing when everything agrees.

    Raises:
      ValueError naming the first quantity that differs.
    '''
    key = layout['key']
    fix = ('; rerun notes/cross_correlations/plots/make_cross_correlations_data.py '
           'so its data.json follows the data files')
    if len(layout['bins']) != len(tables['samples']):
        raise ValueError(f'{key}: SOURCE has {len(layout["bins"])} bins, the bin table '
                         f'{len(tables["samples"])}' + fix)
    for i, b in enumerate(layout['bins']):
        if b['sample'] != tables['samples'][i] or b['group'] != tables['groups'][i]:
            raise ValueError(f'{key}: bin {i} is {b["sample"]}/{b["group"]} in SOURCE '
                             f'and {tables["samples"][i]}/{tables["groups"][i]} in the '
                             'bin table' + fix)
        if abs(b['zbar'] - tables['zbar'][i]) > ZBAR_TOLERANCE:
            raise ValueError(f'{key}: bin {i} has zbar {b["zbar"]} in SOURCE and '
                             f'{tables["zbar"][i]} in the bin table, more than '
                             f'{ZBAR_TOLERANCE} apart' + fix)
    source_pairs = []
    for p in layout['pairs']:
        source_pairs.append((p[0], p[1]))
    source_pairs.sort()
    if source_pairs != tables['pairs']:
        raise ValueError(f'{key}: SOURCE lists {len(source_pairs)} pairs and the pair '
                         f'file {len(tables["pairs"])}, not the same list' + fix)


def bandwidth2_added(samples, today):
    '''Return the pairs (i, i+2) of one sample that today's list lacks.

    Arguments:
      samples = the sample of every bin, in bin order.
      today   = today's pairs, a list of (i, j).

    Returns:
      sorted list of (i, i + 2) with samples[i] == samples[i + 2] and
      (i, i + 2) not in today.
    '''
    listed = set(today)
    added = []
    for i in range(len(samples) - 2):
        if samples[i] == samples[i + 2] and (i, i + 2) not in listed:
            added.append((i, i + 2))
    return added


def partner_sets(nbins, pairs):
    '''Return partners[i] = the set of bins that share a pair with bin i.

    Arguments:
      nbins = the number of bins.
      pairs = list of (a, b) 0-based bin pairs.

    Returns:
      list of nbins sets of int.
    '''
    partners = []
    for _ in range(nbins):
        partners.append(set())
    for a, b in pairs:
        partners[a].add(b)
        partners[b].add(a)
    return partners


def pair_coloring(nbins, pairs):
    """Color the cross-pair graph greedily: no pair joins two bins of one color.

    Bin i takes the smallest color that none of its lower-numbered
    neighbors (the bins it shares a measured cross pair with) has. This
    is the notebook's coloring, so the batches are its own.

    The code is copied unchanged from scripts/compute_fisher_base_mpi.py,
    which runs the notebook's batches with MPI. That script imports
    mpi4py when it loads, which starts MPI, so this script copies the
    function instead of importing it and needs only the Python standard
    library.

    Arguments:
      nbins = the number of lens bins.
      pairs = list of (a, b) 0-based bin pairs.

    Returns:
      list of int, the color of each bin (0, 1, ...).
    """
    neighbors = []
    for _ in range(nbins):
        neighbors.append(set())
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    color = []
    for i in range(nbins):
        used = set()
        for j in neighbors[i]:
            if j < i:
                used.add(color[j])
        c = 0
        while c in used:
            c += 1
        color.append(c)
    return color


def greedy_in_order(nbins, pairs, order):
    '''Color the pair graph greedily, visiting the bins in a given order.

    Each bin, when its turn comes, takes the lowest color that none of
    its already colored partners holds. With order = 0, 1, 2, ... this is
    pair_coloring; the script also runs it in zbar order, to show how
    much the count depends on the visiting order.

    Arguments:
      nbins = the number of bins.
      pairs = list of (a, b) 0-based bin pairs.
      order = every bin index once, in the order to color them.

    Returns:
      list of int, the color of each bin (0, 1, ...).
    '''
    partners = partner_sets(nbins, pairs)
    # -1 marks a bin not colored yet
    color = [-1]*nbins
    for v in order:
        used = set()
        for u in partners[v]:
            if color[u] >= 0:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def dsatur(nbins, pairs):
    '''Color the pair graph with DSATUR (degree of saturation, Brelaz 1979).

    The saturation of an uncolored bin is the number of distinct colors
    its colored partners hold. At every step DSATUR colors the uncolored
    bin of largest saturation (ties: most partners, then lowest index)
    with the lowest color none of its partners holds. Coloring the most
    constrained bin first often reaches the fewest colors where the
    greedy pass in index order does not.

    Arguments:
      nbins = the number of bins.
      pairs = list of (a, b) 0-based bin pairs.

    Returns:
      list of int, the color of each bin (0, 1, ...).
    '''
    partners = partner_sets(nbins, pairs)
    # -1 marks a bin not colored yet
    color = [-1]*nbins
    for _ in range(nbins):
        best = -1
        best_rank = None
        for v in range(nbins):
            if color[v] >= 0:
                continue
            held = set()
            for u in partners[v]:
                if color[u] >= 0:
                    held.add(color[u])
            # Python compares tuples entry by entry: saturation first, then
            # the number of partners. Only a strictly larger rank replaces
            # the best, so among equal ranks the lowest index wins.
            rank = (len(held), len(partners[v]))
            if best < 0 or rank > best_rank:
                best = v
                best_rank = rank
        used = set()
        for u in partners[best]:
            if color[u] >= 0:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[best] = c
    return color


def largest_clique(nbins, partners):
    '''Return the size of the largest set of bins all paired with each other.

    Such a set (a clique of the pair graph) needs one batch per bin, so
    its size is a lower bound on the number of batches. The search is
    exact: from every bin i it tries every subset of the partners above
    i, so each clique is found from its lowest bin.

    Arguments:
      nbins    = the number of bins.
      partners = partners[i] = the set of bins paired with bin i.

    Returns:
      int, the largest clique size (1 when there are no pairs).

    Raises:
      ValueError when a bin has more than MAX_LATER_PARTNERS partners
      above it (the subset count 2^d would make the search too long).
    '''
    best = 1
    for i in range(nbins):
        above = []
        for j in partners[i]:
            if j > i:
                above.append(j)
        above.sort()
        if len(above) > MAX_LATER_PARTNERS:
            raise ValueError(f'bin {i} has {len(above)} partners above it; the clique '
                             f'search tries 2^d subsets and accepts d <= '
                             f'{MAX_LATER_PARTNERS}; use a branch-and-bound search')
        # bit t of mask says whether above[t] is in the subset
        for mask in range(1, 2**len(above)):
            chosen = []
            for t in range(len(above)):
                if mask >> t & 1:
                    chosen.append(above[t])
            joined = True
            for s in range(len(chosen)):
                for t in range(s + 1, len(chosen)):
                    if chosen[t] not in partners[chosen[s]]:
                        joined = False
            if joined:
                best = max(best, len(chosen) + 1)
    return best


def check_coloring(key, name, color, pairs):
    '''Check that no pair joins two bins of one color.

    Arguments:
      key   = the layout key, for the message.
      name  = the coloring's name, for the message.
      color = the color of every bin.
      pairs = list of (a, b) 0-based bin pairs.

    Returns:
      int, the number of colors used.

    Raises:
      RuntimeError when a pair joins two bins of one color (the coloring
      function would then be wrong).
    '''
    for a, b in pairs:
        if color[a] == color[b]:
            raise RuntimeError(f'{key}: the {name} coloring gives bins {a} and {b} the '
                               f'same batch {color[a] + 1}, but they share a cross '
                               'spectrum')
    return max(color) + 1


def degree_histogram(degree):
    '''Return {degree as str: number of bins with that degree}, by degree.'''
    # Counter counts how often each value occurs; the keys become strings
    # because json object keys are strings
    counts = collections.Counter(degree)
    histogram = {}
    for d in sorted(counts):
        histogram[str(d)] = counts[d]
    return histogram


def layout_entry(layout, tables):
    '''Return the data.json entry of one layout and print its summary.

    Arguments:
      layout = one entry of SOURCE's 'layouts' list.
      tables = what read_layout_files returned for that layout.

    Returns:
      dict: the layout's label, color, marker, dz_node and LRG3/ELG1
      range (copied from SOURCE), its bins (SOURCE's n(z) plus degree,
      earlier count and 1-based greedy batch), today's and the added
      pairs, and a summary of the counts.

    Raises:
      RuntimeError when a coloring is invalid, a greedy batch exceeds
      1 + earlier, or DSATUR does not reach the clique bound.
    '''
    key = layout['key']
    samples = tables['samples']
    groups = tables['groups']
    nbins = len(samples)
    today = tables['pairs']
    added = bandwidth2_added(samples, today)
    allpairs = sorted(set(today) | set(added))

    degree = [0]*nbins
    earlier = [0]*nbins
    degree_today = [0]*nbins
    for a, b in allpairs:
        degree[a] += 1
        degree[b] += 1
        # a < b, so a is an earlier partner of b
        earlier[b] += 1
    for a, b in today:
        degree_today[a] += 1
        degree_today[b] += 1
    # the same rule within one group (LRG1, LRG2 and LRG3 kept apart), for
    # the summary only: how much the group-boundary pairs add to the degree
    degree_group = [0]*nbins
    within_group = set(today)
    for i in range(nbins - 2):
        if groups[i] == groups[i + 2]:
            within_group.add((i, i + 2))
    for a, b in within_group:
        degree_group[a] += 1
        degree_group[b] += 1

    color = pair_coloring(nbins, allpairs)
    nbatch = check_coloring(key, 'greedy (bin order)', color, allpairs)
    for i in range(nbins):
        if color[i] > earlier[i]:
            raise RuntimeError(f'{key}: bin {i} got batch {color[i] + 1} with only '
                               f'{earlier[i]} earlier partners; the greedy pass cannot '
                               'do that')
    # zbar order: sorting (zbar, index) pairs breaks equal zbar by index
    by_zbar = []
    for i in range(nbins):
        by_zbar.append((tables['zbar'][i], i))
    by_zbar.sort()
    order = []
    for _, i in by_zbar:
        order.append(i)
    greedy_zbar = check_coloring(key, 'greedy (zbar order)',
                                 greedy_in_order(nbins, allpairs, order), allpairs)
    n_dsatur = check_coloring(key, 'DSATUR', dsatur(nbins, allpairs), allpairs)
    partners = partner_sets(nbins, allpairs)
    clique = largest_clique(nbins, partners)
    if n_dsatur != clique:
        raise RuntimeError(f'{key}: DSATUR uses {n_dsatur} batches and the largest '
                           f'clique has {clique} bins, so the fewest batches is not '
                           'settled; run an exact search (fewest_batches of '
                           'plots/batching/make_batching_data.py)')

    dmax = max(degree)
    max_degree_bins = []
    for i in range(nbins):
        if degree[i] != dmax:
            continue
        max_degree_bins.append({
            'bin': i,
            'bin_1based': i + 1,
            'group': groups[i],
            'zbar': layout['bins'][i]['zbar'],
            'degree': degree[i],
            'earlier': earlier[i],
            'batch': color[i] + 1,
            'partners': sorted(partners[i])})
    across = []
    for a, b in added:
        if groups[a] != groups[b]:
            across.append([a, b, groups[a], groups[b]])
    sizes = []
    for c in range(nbatch):
        sizes.append(color.count(c))
    summary = {
        'nbins': nbins,
        'npairs_today': len(today),
        'npairs_added': len(added),
        'added_across_group_boundary': across,
        'degree_histogram': degree_histogram(degree),
        'degree_histogram_today': degree_histogram(degree_today),
        'max_degree': dmax,
        'max_degree_today': max(degree_today),
        'max_degree_if_within_group': max(degree_group),
        'max_earlier': max(earlier),
        'greedy_batches': nbatch,
        'greedy_bound_from_degree': dmax + 1,
        'greedy_bound_from_earlier': max(earlier) + 1,
        'largest_clique': clique,
        # DSATUR reaches the clique bound (checked above), so it is the minimum
        'fewest_batches': clique,
        'greedy_in_zbar_order': greedy_zbar,
        'dsatur': n_dsatur,
        'batch_sizes': sizes,
        'max_degree_bins': max_degree_bins}
    bins = []
    for i, b in enumerate(layout['bins']):
        bins.append({
            'group': b['group'],
            'sample': b['sample'],
            'zbar': b['zbar'],
            'z_first': b['z_first'],
            'n': b['n'],
            'degree': degree[i],
            'earlier': earlier[i],
            'batch': color[i] + 1})
    pairs_today = []
    for p in today:
        pairs_today.append(list(p))
    pairs_added = []
    for p in added:
        pairs_added.append(list(p))

    print(f"{layout['label']:22s} {nbins:3d} bins, {len(today)} + {len(added)} pairs; "
          f"degree hist {summary['degree_histogram']} (today "
          f"{summary['degree_histogram_today']}); max degree {dmax} (today "
          f"{max(degree_today)}, within-group rule {max(degree_group)}); max earlier "
          f"{max(earlier)}; greedy {nbatch} {sizes}; clique {clique}; fewest {clique}; "
          f"zbar-order greedy {greedy_zbar}; DSATUR {n_dsatur}")
    for e in max_degree_bins:
        one_based = []
        for p in e['partners']:
            one_based.append(p + 1)
        print(f"    bin {e['bin_1based']:3d} {e['group']} zbar {e['zbar']:.3f}: degree "
              f"{e['degree']}, earlier {e['earlier']}, batch {e['batch']}; partners "
              f"(1-based) {one_based}")
    return {'key': key,
            'label': layout['label'],
            'color': layout['color'],
            'marker': layout['marker'],
            'dz_node': layout['dz_node'],
            'lrg3_elg1_range': layout['lrg3_elg1_range'],
            'bins': bins,
            'pairs_today': pairs_today,
            'pairs_added': pairs_added,
            'summary': summary}


ABOUT = (
    'Written by notes/fisher_machinery/plots/make_pair_graph_bandwidth2_data.py '
    'from notes/cross_correlations/plots/nz_overlap_layouts/data.json (bins, raw '
    'n(z) at unit peak on nodes of spacing dz_node from z_first, today\'s pairs, '
    'the LRG3/ELG1 shared range) and the project\'s data/lssty1xdesi_desi_<layout>'
    '_bins.txt and _pairs.txt (layouts dr1_dz0p20, dr1_dz0p05, dr1_dz0p02 and dr3, '
    'the last one Delta d_com = 62 Mpc; checked equal to the data.json above). '
    'Bins and pairs are 0-based in data-vector order. pairs_added: every (i, i+2) '
    'with bins i and i+2 in the same sample (bgs, lrg or elg, so LRG1-LRG2, '
    'LRG2-LRG3 and ELG1-ELG2 boundary pairs count, BGS-LRG1 do not) that today\'s '
    'list lacks. Per bin: degree = number of partners in today\'s pairs plus the '
    'added ones; earlier = partners with a lower bin index; batch = 1-based batch '
    'of the greedy coloring in bin order (pair_coloring of '
    'scripts/compute_fisher_base_mpi.py: each bin takes the lowest batch none of '
    'its earlier partners holds, so batch <= earlier + 1). largest_clique: exact '
    'search; fewest_batches: the clique size, reached by the DSATUR coloring '
    '(dsatur = its batch count); greedy_in_zbar_order: the greedy pass in '
    'mean-redshift order.')


def main():
    '''Read the four layouts, batch the bandwidth-2 lists and write data.json.

    Side effects: overwrites pair_graph_bandwidth2/data.json next to this
    script; nothing else is written.

    Raises:
      FileNotFoundError when the data folder, SOURCE or the plot folder
      is missing.
    '''
    data_dir = DEFAULT_DATA
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'the data folder {data_dir} does not exist; pass the '
                                'project data folder as the first argument')
    if not os.path.isfile(SOURCE):
        raise FileNotFoundError(f'{SOURCE} does not exist; run make_cross_correlations'
                                '_data.py in notes/cross_correlations/plots/ first')
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(f'the plot folder {OUTPUT_DIR} does not exist; it holds '
                                'plot_pair_graph_bandwidth2.py and receives data.json')
    with open(SOURCE) as f:
        source = json.load(f)
    # read and check every layout before writing anything
    layouts = []
    for layout in source['layouts']:
        if layout['key'] not in DATASETS:
            raise ValueError(f'SOURCE has the layout {layout["key"]}, which DATASETS '
                             f'does not name ({sorted(DATASETS)}); add its dataset stem')
        tables = read_layout_files(data_dir, DATASETS[layout['key']])
        check_against_source(layout, tables)
        layouts.append(layout_entry(layout, tables))
    with open(os.path.join(OUTPUT_DIR, 'data.json'), 'w') as f:
        json.dump({'about': ABOUT, 'layouts': layouts}, f)


if __name__ == '__main__':
    main()
