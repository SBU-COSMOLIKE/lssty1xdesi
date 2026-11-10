"""Write data.json for the batching figure of the Fisher machinery note:
how the notebooks compute the per-bin bias derivatives in batches.

The DESI data vector of one bin layout holds, for every thin redshift bin
i, its auto spectrum C_ii in N_ell = 53 multipole bands, and the cross
spectrum C_ij of every pair of bins listed in the dataset's pair file
(bins whose redshift distributions overlap or touch). In front of these
spectra sit n_ee entries that the data format reserves for cosmic-shear
blocks; they stay zero for DESI alone.

A per-bin derivative such as dm/db_{2,i} (m the data vector, b_{2,i} the
quadratic bias of bin i) is nonzero only in the spectra that contain bin
i. The notebooks therefore step b_2 in a whole set of bins at once, a
"batch", in which no two bins share a measured cross spectrum: every
spectrum then holds at most one stepped bin, and one five-point stencil
(four data-vector evaluations, at +-h and +-2h) gives the column of every
bin of the batch. The batches are the "colors" of pair_coloring in the
builder of EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (and of
scripts/compute_fisher_base_mpi.py); this script copies that function and
spectrum_rows unchanged, so the figure shows what the notebook runs.

The script reads the bin table and the pair list of each layout the way
layout_covariance of the notebooks does (data/<dataset>_bins.txt and
data/<dataset>_pairs.txt), runs the same greedy batching, and stores:

  - for Delta d_com = 62 Mpc (76 bins): every bin (its group, mean
    redshift zbar_i, batch and cross partners), the pair list, the six
    groups, the batches and the row map (which data-vector rows each
    bin's column fills);
  - for all four layouts: the number of bins, pairs and batches, the
    batch sizes, the number of cross partners per bin, and the fewest
    batches any assignment could use (an exhaustive search, a diagnostic
    that the figure does not draw).

Run from any folder, inside the repository:
  python make_batching_data.py
or with the project's data folder named explicitly:
  python make_batching_data.py /path/to/projects/lssty1xdesi/data
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in projects/lssty1xdesi/notes/fisher_machinery/plots/
# batching/, four folders below the project, whose data/ holds the files
DEFAULT_DATA = os.path.join(HERE, '..', '..', '..', '..', 'data')

# The four layouts of the notebooks (the LAYOUTS dictionary of the builder
# of EXAMPLE_EVALUATE_FISHER_MODEL.ipynb): the name the notebooks use, the
# label the figures print, and the dataset stem that names its data files.
# The order is the order of the notebooks.
LAYOUTS = [
    {'name': 'Delta d_com = 62 Mpc',
     'label': r'$\Delta d_\mathrm{com} = 62$ Mpc',
     'dataset': 'lssty1xdesi_desi_dr3'},
    {'name': 'Delta z = 0.02',
     'label': r'$\Delta z = 0.02$',
     'dataset': 'lssty1xdesi_desi_dr1_dz0p02'},
    {'name': 'Delta z = 0.05',
     'label': r'$\Delta z = 0.05$',
     'dataset': 'lssty1xdesi_desi_dr1_dz0p05'},
    {'name': 'Delta z = 0.2',
     'label': r'$\Delta z = 0.2$',
     'dataset': 'lssty1xdesi_desi_dr1_dz0p20'},
]
# the layout whose bins the figure draws one by one
FOCUS = 'Delta d_com = 62 Mpc'
# The stretch of 62 Mpc bins that panels (b) and (c) of the figure draw,
# as 0-based indices (the figure and the note add 1): the last two LRG2
# bins and the first three LRG3 bins (36 to 40), and the first three ELG1
# bins (49 to 51), which overlap LRG3 in redshift. All
# four batches meet here, and index 50 has all four of its partners in the
# stretch. Eight bins keep panel (c) at 20 rows, one name per row.
STRETCH = [36, 37, 38, 39, 40, 49, 50, 51]
# the bias operators that get per-bin columns (OPERATOR_STEPS of the
# notebooks): b_1, b_2, b_{s^2} and the k^2 counterterm alpha_0
FAMILIES = ['b1', 'b2', 'bs2', 'a0']
# the five-point stencil evaluates the vector at p + offset*h for these
# offsets (FIVE_POINT_OFFSETS of interface/cosmolike_lssty1xdesi_fisher.py);
# the center point has weight zero and is never evaluated
FIVE_POINT_OFFSETS = [2.0, 1.0, -1.0, -2.0]


def read_rows(path):
    '''Return the data rows of a text file, each split into words.

    Lines that are empty or start with # are skipped, as in read_rows of
    the notebooks.

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


def dataset_entry(data_dir, stem, key):
    '''Return the value of one "key = value" line of a .dataset file.

    The .dataset file is the small text file that lists which data files
    the likelihood reads, one "key = filename" line each.

    Arguments:
      data_dir = the project's data folder.
      stem     = the dataset name, without the .dataset extension.
      key      = the key to look up, e.g. 'ell_file'.

    Returns:
      str, the value after the first '='.

    Raises:
      KeyError when no line of the file has that key.
    '''
    with open(os.path.join(data_dir, stem + '.dataset')) as f:
        lines = f.readlines()
    for row in lines:
        if row.split('=')[0].strip() == key:
            return row.split('=', 1)[1].strip()
    raise KeyError(f'{key} is not in {stem}.dataset; the batching data needs it '
                   'to find the multipole bands and the data vector')


def pair_coloring(nbins, pairs):
    '''A greedy coloring of the pair graph: no pair joins two bins of one color.'''
    # Copied unchanged from the notebook builders. Each bin, in index
    # order, takes the lowest color (batch) that none of its lower-index
    # partners holds; color 0 is batch 1 of the figure.
    neighbors = [set() for _ in range(nbins)]
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    color = []
    for i in range(nbins):
        used = {color[j] for j in neighbors[i] if j < i}
        c = 0
        while c in used:
            c += 1
        color.append(c)
    return color


def spectrum_rows(nbins, pairs, nell, n_ee):
    '''rows[i] = the data-vector entries of every spectrum that involves bin i.'''
    # Copied unchanged from the notebook builders (numpy replaced by plain
    # lists, so this script needs no numpy). The spectra follow the
    # n_ee shear entries: the nbins auto spectra first, then the cross
    # spectra in the order of the pair file, nell entries each.
    rows = []
    for i in range(nbins):
        r = list(range(n_ee + i*nell, n_ee + (i + 1)*nell))
        for p, (a, b) in enumerate(pairs):
            if i in (a, b):
                start = n_ee + (nbins + p)*nell
                r += list(range(start, start + nell))
        rows.append(r)
    return rows


def as_ranges(rows):
    '''Return a sorted list of row indices as [start, stop) ranges.

    The rows of one bin are whole spectra, nell consecutive entries each,
    so a few ranges replace hundreds of indices in data.json.

    Arguments:
      rows = increasing integer row indices.

    Returns:
      list of [start, stop) pairs, stop excluded, merged where they touch.
    '''
    ranges = []
    for r in rows:
        if ranges and ranges[-1][1] == r:
            ranges[-1][1] = r + 1
        else:
            ranges.append([r, r + 1])
    return ranges


def largest_clique(nbins, partners):
    '''Return the size of the largest set of bins all paired with each other.

    Such a set (a clique of the pair graph) needs one batch per bin, so
    its size is a lower bound on the number of batches. Each bin has at
    most five partners, so every subset of a bin's partners is tried.

    Arguments:
      nbins    = the number of bins.
      partners = partners[i] = the set of bins paired with bin i.

    Returns:
      int, the largest clique size (1 when there are no pairs).
    '''
    best = 1
    for i in range(nbins):
        # only partners above i, so each clique is counted from its lowest bin
        above = sorted(j for j in partners[i] if j > i)
        for mask in range(1, 2**len(above)):
            chosen = [above[t] for t in range(len(above)) if mask >> t & 1]
            joined = True
            for s in range(len(chosen)):
                for t in range(s + 1, len(chosen)):
                    if chosen[t] not in partners[chosen[s]]:
                        joined = False
            if joined:
                best = max(best, len(chosen) + 1)
    return best


def fewest_batches(nbins, pairs, greedy):
    '''Return the fewest batches any assignment of the bins could use.

    This is the chromatic number of the pair graph (the bins are the
    vertices, the measured cross pairs the edges). The largest clique
    (largest_clique) bounds it from below and the greedy count of
    pair_coloring from above; each k in between is tried with a
    backtracking search over the bins, the most connected bins first.
    On the four layouts the first k tried already succeeds, so the
    search never has to prove that an assignment is impossible. The
    figure does not draw this number: it is a diagnostic of how far the
    greedy pass of pair_coloring is from the minimum.

    Arguments:
      nbins  = the number of bins.
      pairs  = the (i, j) cross pairs.
      greedy = the number of batches pair_coloring used.

    Returns:
      int, the smallest k for which an assignment exists.
    '''
    partners = [set() for _ in range(nbins)]
    for a, b in pairs:
        partners[a].add(b)
        partners[b].add(a)
    # most connected bins first: they constrain the search the most
    order = sorted(range(nbins), key=lambda v: -len(partners[v]))
    k = largest_clique(nbins, partners)
    while k < greedy:
        batch = [-1]*nbins
        # an explicit stack of (position in order, next batch to try)
        # replaces recursion, so 90 bins never reach Python's depth limit
        stack = [[0, 0]]
        found = False
        while stack:
            position, trial = stack[-1]
            if position == nbins:
                found = True
                break
            v = order[position]
            placed = False
            while trial < k:
                clash = False
                for u in partners[v]:
                    if batch[u] == trial:
                        clash = True
                        break
                if not clash:
                    batch[v] = trial
                    stack[-1][1] = trial + 1
                    stack.append([position + 1, 0])
                    placed = True
                    break
                trial += 1
            if not placed:
                batch[v] = -1
                stack.pop()
                if stack:
                    # undo the bin of the previous level before its next trial
                    batch[order[stack[-1][0]]] = -1
        if found:
            return k
        k += 1
    # no k below the greedy count works: greedy is optimal
    return greedy


def layout_tables(data_dir, layout):
    '''Read one layout's bins, pairs and band count, and batch its bins.

    Arguments:
      data_dir = the project's data folder.
      layout   = one LAYOUTS entry.

    Returns:
      dict: 'bins' (rows of the bin table), 'pairs' (list of (i, j)),
      'nell', 'n' (the data-vector length), 'n_ee' (the shear entries in
      front of the spectra), 'batch' (the 0-based color of each bin) and
      'partners' (the sorted cross partners of each bin).

    Raises:
      ValueError when a pair names a bin outside the bin table, or when
      the data vector is shorter than the spectra it must hold.
    '''
    stem = layout['dataset']
    bins = read_rows(os.path.join(data_dir, stem + '_bins.txt'))
    pairs = []
    for row in read_rows(os.path.join(data_dir, stem + '_pairs.txt')):
        pairs.append((int(row[0]), int(row[1])))
    nbins = len(bins)
    for a, b in pairs:
        if not (0 <= a < nbins and 0 <= b < nbins):
            raise ValueError(f'{stem}_pairs.txt names the pair ({a}, {b}) but the bin '
                             f'table has bins 0 to {nbins - 1}; regenerate the pair file')
    ell_file = dataset_entry(data_dir, stem, 'ell_file')
    nell = len(read_rows(os.path.join(data_dir, ell_file)))
    # the data vector's length: one line per entry of the shipped vector
    data_file = dataset_entry(data_dir, stem, 'data_file')
    n = len(read_rows(os.path.join(data_dir, data_file)))
    n_ee = n - (nbins + len(pairs))*nell
    if n_ee < 0:
        raise ValueError(f'{data_file} has {n} entries, fewer than the '
                         f'{(nbins + len(pairs))*nell} of its {nbins} auto and '
                         f'{len(pairs)} cross spectra; the files do not match')
    partners = [[] for _ in range(nbins)]
    for a, b in pairs:
        partners[a].append(b)
        partners[b].append(a)
    for i in range(nbins):
        partners[i].sort()
    return {'bins': bins, 'pairs': pairs, 'nell': nell, 'n': n, 'n_ee': n_ee,
            'batch': pair_coloring(nbins, pairs), 'partners': partners}


def layout_summary(layout, tables):
    '''Return the data.json entry of one layout: its sizes and batches.

    Arguments:
      layout = one LAYOUTS entry.
      tables = what layout_tables returned for it.

    Returns:
      dict with the counts the figure draws and the report quotes.

    Raises:
      RuntimeError when the greedy batches let a cross pair join two bins
      of one batch (pair_coloring would then be wrong).
    '''
    nbins = len(tables['bins'])
    batch = tables['batch']
    for a, b in tables['pairs']:
        if batch[a] == batch[b]:
            raise RuntimeError(f'{layout["name"]}: bins {a} and {b} share batch '
                               f'{batch[a] + 1} and a cross spectrum')
    nbatches = max(batch) + 1
    sizes = [batch.count(c) for c in range(nbatches)]
    degree = [len(p) for p in tables['partners']]
    partner_sets = [set(p) for p in tables['partners']]
    histogram = collections.Counter(degree)
    # the typical partner count: the most frequent one
    typical = max(histogram, key=lambda d: histogram[d])
    evals_per_batch = len(FIVE_POINT_OFFSETS)
    return {'name': layout['name'],
            'label': layout['label'],
            'dataset': layout['dataset'],
            'nbins': nbins,
            'npairs': len(tables['pairs']),
            'nell': tables['nell'],
            'n': tables['n'],
            'n_ee': tables['n_ee'],
            'batches': nbatches,
            'batch_sizes': sizes,
            'largest_clique': largest_clique(nbins, partner_sets),
            'fewest_batches': fewest_batches(nbins, tables['pairs'], nbatches),
            'partners_min': min(degree),
            'partners_max': max(degree),
            'partners_typical': typical,
            # {number of partners: number of bins with that many}
            'partners_histogram': {str(d): histogram[d] for d in sorted(histogram)},
            'evaluations_per_family_batched': evals_per_batch*nbatches,
            'evaluations_per_family_per_bin': evals_per_batch*nbins}


def focus_entry(tables):
    '''Return the bin-by-bin content of the layout the figure draws.

    Arguments:
      tables = what layout_tables returned for FOCUS.

    Returns:
      dict: 'bins', 'pairs', 'groups', 'batches', 'row_map', 'block' and
      'stretch' (see the comments of each field below).
    '''
    bins = tables['bins']
    nbins = len(bins)
    pairs = tables['pairs']
    nell = tables['nell']
    n_ee = tables['n_ee']
    rows = spectrum_rows(nbins, pairs, nell, n_ee)
    bin_list = []
    for i, row in enumerate(bins):
        # bin-table columns: bin, sample, chunk_index, group name, zbar
        bin_list.append({'bin': i,
                         'group': row[3],
                         'zbar': float(row[4]),
                         'batch': tables['batch'][i] + 1,
                         'partners': tables['partners'][i]})
    # the six groups, in the order of the bin table
    groups = []
    for entry in bin_list:
        if not groups or groups[-1]['name'] != entry['group']:
            groups.append({'name': entry['group'], 'bins': []})
        groups[-1]['bins'].append(entry['bin'])
    for group in groups:
        z = [bin_list[i]['zbar'] for i in group['bins']]
        group['zbar_range'] = [min(z), max(z)]
    batches = []
    for c in range(max(tables['batch']) + 1):
        members = [i for i in range(nbins) if tables['batch'][i] == c]
        batches.append({'batch': c + 1, 'bins': members})
    # the spectra in data-vector order: autos, then crosses in pair order
    spectra = []
    for i in range(nbins):
        spectra.append({'spectrum': i, 'bins': [i, i],
                        'rows': [n_ee + i*nell, n_ee + (i + 1)*nell]})
    for p, (a, b) in enumerate(pairs):
        start = n_ee + (nbins + p)*nell
        spectra.append({'spectrum': nbins + p, 'bins': [a, b],
                        'rows': [start, start + nell]})
    rows_of_bin = []
    for i in range(nbins):
        own = [s['spectrum'] for s in spectra if i in s['bins']]
        rows_of_bin.append({'bin': i, 'spectra': own, 'rows': as_ranges(rows[i])})
    filled = sum(len(r) for r in rows)
    return {'bins': bin_list,
            'pairs': [list(p) for p in pairs],
            'groups': groups,
            'batches': batches,
            # which rows of the data vector each bin's column fills; every
            # other row of that column is exactly zero
            'row_map': {'n': tables['n'], 'n_ee': n_ee, 'nell': nell,
                        'spectra': spectra, 'rows_of_bin': rows_of_bin},
            # the size of one family's derivative block (rows = data vector,
            # columns = bins) and how much of it can be nonzero: a spectrum
            # cell is one spectrum (nell rows) of one column. Inside a
            # filled cell, the bands where the likelihood leaves the vector
            # at zero (the highest multipoles of each spectrum) stay zero,
            # so filled_entries bounds the nonzero entries from above.
            'block': {'rows': tables['n'], 'columns': nbins,
                      'spectra': len(spectra),
                      'filled_spectrum_cells': nbins + 2*len(pairs),
                      'spectrum_cells': len(spectra)*nbins,
                      'filled_entries': filled,
                      'entries': tables['n']*nbins},
            'stretch': STRETCH}


def main():
    '''Read the four layouts, batch their bins and write data.json.'''
    data_dir = DEFAULT_DATA
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'the data folder {data_dir} does not exist; pass the '
                                'project data folder as the first argument')
    summaries = []
    focus = None
    for layout in LAYOUTS:
        tables = layout_tables(data_dir, layout)
        summaries.append(layout_summary(layout, tables))
        if layout['name'] == FOCUS:
            focus = focus_entry(tables)
    out = {'about': ('Written by make_batching_data.py (this folder) from the bin '
                     'tables and pair lists of data/ (lssty1xdesi_desi_*_bins.txt, '
                     '*_pairs.txt), batched with pair_coloring of the builder of '
                     'EXAMPLE_EVALUATE_FISHER_MODEL.ipynb. Batches are numbered from 1 '
                     '(color 0 of the code is batch 1). Bin indices in this file (bin, partners, '
                     'pairs, groups, batches, row_map, stretch) are 0-based, as in the '
                     'data files and the code; the figure and the note number bins '
                     'from 1, so index i is drawn as bin i + 1. largest_clique and '
                     'fewest_batches are diagnostics of the pair graph, not drawn.'),
           'focus_layout': FOCUS,
           'families': FAMILIES,
           'five_point_offsets': FIVE_POINT_OFFSETS,
           'evaluations_per_batch': len(FIVE_POINT_OFFSETS),
           'layouts': summaries,
           'focus': focus}
    with open(os.path.join(HERE, 'data.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for s in summaries:
        print(f"{s['name']}: {s['nbins']} bins, {s['npairs']} pairs, {s['batches']} "
              f"batches {s['batch_sizes']}, fewest {s['fewest_batches']}, partners "
              f"{s['partners_min']}-{s['partners_max']} (typical "
              f"{s['partners_typical']}) {s['partners_histogram']}")


if __name__ == '__main__':
    main()
