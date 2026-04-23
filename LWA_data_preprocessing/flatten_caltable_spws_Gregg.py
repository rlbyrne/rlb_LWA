#!/usr/bin/env python
"""
flatten_caltable_spw.py

Collapses a multi-SPW CASA calibration table (B-type) into a single-SPW table
with all channels concatenated in frequency order.

This eliminates the need for spwmap gymnastics during applycal  the output
table has one row per (antenna, timestamp) with SPECTRAL_WINDOW_ID=0 and a
single SPECTRAL_WINDOW entry spanning the full band.  Multi-integration/scan
tables retain their full time structure (TIME, SCAN_NUMBER, FIELD_ID, INTERVAL
are all preserved per row).

Requires: python-casacore (pyrap.tables)

Usage:
    python flatten_caltable_spw.py input.B output.B [--dry-run]

Author: OVRO-LWA Pipeline
"""

import os
import sys
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('FlattenSPW')

try:
    import casacore.tables as pt
except ImportError:
    try:
        import pyrap.tables as pt
    except ImportError:
        logger.error("Neither casacore.tables nor pyrap.tables found. "
                     "Install python-casacore.")
        sys.exit(1)


def read_spw_info(caltable_path):
    """Read the SPECTRAL_WINDOW sub-table and return sorted SPW metadata.

    Returns
    -------
    spw_order : list of int
        SPW IDs sorted by ascending center frequency.
    spw_freqs : dict
        {spw_id: 1-D array of channel frequencies in Hz}
    spw_nchan : dict
        {spw_id: number of channels}
    """
    spw_table_path = os.path.join(caltable_path, 'SPECTRAL_WINDOW')
    with pt.table(spw_table_path, ack=False) as t_spw:
        n_spw = t_spw.nrows()
        all_chan_freqs = t_spw.getcol('CHAN_FREQ')   # (n_spw, max_nchan) or ragged
        all_num_chan = t_spw.getcol('NUM_CHAN')       # (n_spw,)

        # Also grab columns we'll need to reconstruct
        col_names = t_spw.colnames()

    # Handle potential transposition (same logic as pipeline_utils)
    if all_chan_freqs.ndim == 2 and all_chan_freqs.shape[0] != n_spw:
        if all_chan_freqs.shape[1] == n_spw:
            logger.warning("CHAN_FREQ array transposed  fixing.")
            all_chan_freqs = all_chan_freqs.T

    spw_freqs = {}
    spw_nchan = {}
    centers = {}
    for spw_id in range(n_spw):
        nc = all_num_chan[spw_id]
        freqs = all_chan_freqs[spw_id, :nc] if all_chan_freqs.ndim == 2 else all_chan_freqs
        spw_freqs[spw_id] = freqs.copy()
        spw_nchan[spw_id] = nc
        centers[spw_id] = np.mean(freqs)

    # Sort SPWs by center frequency
    spw_order = sorted(centers.keys(), key=lambda s: centers[s])
    logger.info(f"Found {n_spw} SPWs, total {sum(spw_nchan.values())} channels.")
    for s in spw_order:
        logger.info(f"  SPW {s}: {spw_nchan[s]} ch, "
                     f"{spw_freqs[s].min()/1e6:.2f}{spw_freqs[s].max()/1e6:.2f} MHz")

    return spw_order, spw_freqs, spw_nchan


def _quantize_times(times, tolerance_s=1.0):
    """Bin timestamps that differ by less than tolerance_s into groups.

    SPWs within the same integration have slightly different TIME centroids
    (sub-second offsets from centroid calculation).  This clusters them so
    grouping by (antenna, quantized_time) correctly stitches across SPWs.

    Returns an array of quantized times (same length as input) where all
    values within a cluster are replaced by the cluster median.
    """
    sorted_unique = np.unique(times)
    # Walk through sorted unique times and group those within tolerance
    clusters = []
    current_cluster = [sorted_unique[0]]
    for t in sorted_unique[1:]:
        if t - current_cluster[0] <= tolerance_s:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)

    # Map each original time to its cluster representative (median)
    time_map = {}
    for cluster in clusters:
        representative = np.median(cluster)
        for t in cluster:
            time_map[t] = representative

    return np.array([time_map[t] for t in times]), len(clusters)


def read_main_table(caltable_path, spw_order, spw_nchan):
    """Read the main cal table and assemble per-(antenna, time) stitched data.

    Rows sharing the same (ANTENNA1, TIME) are stitched across SPWs into a
    single broadband row.  This correctly handles multi-integration/scan
    tables where each timestamp has a full set of per-SPW rows.

    Handles two common casacore issues:
      - CPARAM stored as (n_chan, n_pol) instead of (n_pol, n_chan)
      - Per-SPW TIME centroids that differ by sub-second offsets within
        the same integration

    Returns
    -------
    output_rows : list of dict
        Each dict contains:
            'antenna': int, 'time': float, 'scan': int, 'field': int,
            'interval': float, 'cparam': ndarray (n_pol, total_chan),
            'flag': ndarray (n_pol, total_chan)
        Sorted by (time, antenna).
    n_pol : int
    """
    total_chan = sum(spw_nchan[s] for s in spw_order)

    # Build channel offset map (cumulative channel count in stitched order)
    chan_offset = {}
    offset = 0
    for s in spw_order:
        chan_offset[s] = offset
        offset += spw_nchan[s]

    with pt.table(caltable_path, ack=False) as t:
        n_rows = t.nrows()
        all_ant = t.getcol('ANTENNA1')
        all_spw = t.getcol('SPECTRAL_WINDOW_ID')
        all_time_raw = t.getcol('TIME')

        # Optional metadata columns  read if present
        col_names = t.colnames()
        all_scan = t.getcol('SCAN_NUMBER') if 'SCAN_NUMBER' in col_names else np.ones(n_rows, dtype=int)
        all_field = t.getcol('FIELD_ID') if 'FIELD_ID' in col_names else np.zeros(n_rows, dtype=int)
        all_interval = t.getcol('INTERVAL') if 'INTERVAL' in col_names else np.zeros(n_rows, dtype=float)

        # --- FIX 2: Quantize TIME to merge sub-second SPW centroid offsets ---
        all_time, n_time_clusters = _quantize_times(all_time_raw, tolerance_s=1.0)
        n_raw_times = len(np.unique(all_time_raw))
        if n_time_clusters != n_raw_times:
            logger.info(f"TIME quantization: {n_raw_times} raw timestamps ? "
                         f"{n_time_clusters} integration(s) (tolerance 1.0 s)")

        # --- FIX 1: Detect CPARAM transposition ---
        # Read first row to check axis order.  B-type tables have (n_pol, n_chan)
        # where n_pol is 1 or 2.  If shape[0] matches a known n_chan and
        # shape[1] is small, the array is transposed.
        cparam0 = t.getcell('CPARAM', 0)
        first_spw = int(all_spw[0])
        expected_nchan = spw_nchan.get(first_spw, cparam0.shape[1])

        cparam_transposed = False
        if cparam0.shape[0] == expected_nchan and cparam0.shape[1] <= 2:
            cparam_transposed = True
            n_pol = cparam0.shape[1]
            logger.warning(f"CPARAM is transposed: shape per row is "
                           f"(n_chan={cparam0.shape[0]}, n_pol={cparam0.shape[1]}). "
                           f"Will transpose each row to (n_pol, n_chan).")
        else:
            n_pol = cparam0.shape[0]

        # Identify unique groups after quantization
        unique_times = sorted(set(all_time))
        unique_ants = sorted(set(all_ant))
        n_times = len(unique_times)
        n_ants = len(unique_ants)

        logger.info(f"Polarizations: {n_pol}, Antennas: {n_ants}, "
                     f"Integrations: {n_times}, Main table rows: {n_rows}")
        if n_times > 1:
            dt = np.diff(unique_times)
            logger.info(f"  Time range: {unique_times[0]:.1f}  {unique_times[-1]:.1f} "
                         f"(median ?t = {np.median(dt):.1f} s)")

        # Pre-allocate storage keyed by (ant, quantized_time)
        groups = {}

        for row in range(n_rows):
            ant = int(all_ant[row])
            time_val = all_time[row]
            spw = int(all_spw[row])

            if spw not in chan_offset:
                logger.warning(f"Row {row}: SPW {spw} not in SPW table  skipping.")
                continue

            key = (ant, time_val)
            if key not in groups:
                groups[key] = {
                    'antenna': ant,
                    'time': time_val,
                    'scan': int(all_scan[row]),
                    'field': int(all_field[row]),
                    'interval': float(all_interval[row]),
                    'cparam': np.ones((n_pol, total_chan), dtype=np.complex128),
                    'flag': np.ones((n_pol, total_chan), dtype=bool),
                }

            cparam = t.getcell('CPARAM', row)
            flag = t.getcell('FLAG', row)

            # Transpose if needed: (n_chan, n_pol) ? (n_pol, n_chan)
            if cparam_transposed:
                cparam = cparam.T
                flag = flag.T

            nc = spw_nchan[spw]

            if cparam.shape[1] != nc:
                logger.warning(f"Row {row} (ant={ant}, spw={spw}): CPARAM has "
                               f"{cparam.shape[1]} ch, expected {nc}. Skipping.")
                continue

            start = chan_offset[spw]
            end = start + nc
            groups[key]['cparam'][:, start:end] = cparam[:, :nc]
            groups[key]['flag'][:, start:end] = flag[:, :nc]

    # Sort by (time, antenna) for deterministic output row order
    output_rows = sorted(groups.values(), key=lambda r: (r['time'], r['antenna']))

    # Sanity checks
    n_allflagged = sum(1 for r in output_rows if r['flag'].all())
    if n_allflagged > 0:
        logger.warning(f"{n_allflagged}/{len(output_rows)} output rows are fully flagged.")

    expected_rows = n_ants * n_times
    if len(output_rows) != expected_rows:
        logger.warning(f"Expected {n_ants}×{n_times} = {expected_rows} output rows, "
                       f"got {len(output_rows)}. Some (antenna, time) pairs may be missing.")

    return output_rows, n_pol, cparam_transposed


def write_flattened_table(input_path, output_path, spw_order, spw_freqs,
                          spw_nchan, output_rows, n_pol, cparam_transposed=False):
    """Write a new cal table with a single SPW containing all channels.

    Each entry in output_rows becomes one row in the output, preserving
    the original TIME, SCAN_NUMBER, FIELD_ID, and INTERVAL values so that
    multi-integration tables retain their time structure.

    If cparam_transposed is True, arrays are transposed back to the
    (n_chan, n_pol) storage order that the original table used.

    Strategy: copy the input table, then rewrite the main table rows and
    the SPECTRAL_WINDOW sub-table in place.
    """
    total_chan = sum(spw_nchan[s] for s in spw_order)

    # Build the concatenated frequency axis
    full_freqs = np.concatenate([spw_freqs[s] for s in spw_order])
    assert full_freqs.shape[0] == total_chan

    # --- 1. Deep-copy the input table ---
    logger.info(f"Copying {input_path} ? {output_path}")
    pt.tablecopy(input_path, output_path)

    # --- 2. Rewrite SPECTRAL_WINDOW sub-table ---
    spw_table_path = os.path.join(output_path, 'SPECTRAL_WINDOW')
    with pt.table(spw_table_path, readonly=False, ack=False) as t_spw:
        n_spw_orig = t_spw.nrows()

        # Remove all rows, then add one
        t_spw.removerows(range(n_spw_orig))
        t_spw.addrows(1)

        # Write the single-SPW metadata
        t_spw.putcell('NUM_CHAN', 0, total_chan)
        t_spw.putcell('CHAN_FREQ', 0, full_freqs)

        # Channel widths: derive from frequency spacing
        if total_chan > 1:
            chan_widths = np.diff(full_freqs)
            chan_widths = np.append(chan_widths, chan_widths[-1])
        else:
            chan_widths = np.array([full_freqs[0]])
        t_spw.putcell('CHAN_WIDTH', 0, chan_widths)
        t_spw.putcell('EFFECTIVE_BW', 0, np.abs(chan_widths))
        t_spw.putcell('RESOLUTION', 0, np.abs(chan_widths))

        if 'TOTAL_BANDWIDTH' in t_spw.colnames():
            t_spw.putcell('TOTAL_BANDWIDTH', 0, full_freqs[-1] - full_freqs[0])
        if 'REF_FREQUENCY' in t_spw.colnames():
            t_spw.putcell('REF_FREQUENCY', 0, np.mean(full_freqs))
        if 'MEAS_FREQ_REF' in t_spw.colnames():
            t_spw.putcell('MEAS_FREQ_REF', 0, 5)
        if 'NAME' in t_spw.colnames():
            t_spw.putcell('NAME', 0, 'FullBand_Stitched')

        t_spw.flush()

    logger.info(f"SPECTRAL_WINDOW rewritten: 1 SPW, {total_chan} channels, "
                f"{full_freqs[0]/1e6:.2f}{full_freqs[-1]/1e6:.2f} MHz")

    # --- 3. Rewrite main table ---
    with pt.table(output_path, readonly=False, ack=False) as t:
        n_rows_orig = t.nrows()
        col_names = t.colnames()

        # Remove all original rows, add one per (antenna, time) group
        t.removerows(range(n_rows_orig))
        t.addrows(len(output_rows))

        for i, row_data in enumerate(output_rows):
            t.putcell('ANTENNA1', i, row_data['antenna'])
            t.putcell('ANTENNA2', i, row_data['antenna'])
            t.putcell('SPECTRAL_WINDOW_ID', i, 0)
            t.putcell('TIME', i, row_data['time'])

            # Transpose back to original storage order if needed
            cp = row_data['cparam'].T if cparam_transposed else row_data['cparam']
            fl = row_data['flag'].T if cparam_transposed else row_data['flag']
            t.putcell('CPARAM', i, cp)
            t.putcell('FLAG', i, fl)

            if 'FIELD_ID' in col_names:
                t.putcell('FIELD_ID', i, row_data['field'])
            if 'SCAN_NUMBER' in col_names:
                t.putcell('SCAN_NUMBER', i, row_data['scan'])
            if 'INTERVAL' in col_names:
                t.putcell('INTERVAL', i, row_data['interval'])
            if 'SNR' in col_names:
                snr = np.ones((n_pol, total_chan), dtype=np.float64)
                t.putcell('SNR', i, snr.T if cparam_transposed else snr)
            if 'WEIGHT' in col_names:
                t.putcell('WEIGHT', i, np.ones(n_pol, dtype=np.float64))
            if 'PARAMERR' in col_names:
                pe = np.zeros((n_pol, total_chan), dtype=np.float64)
                t.putcell('PARAMERR', i, pe.T if cparam_transposed else pe)

        t.flush()

    unique_times = sorted(set(r['time'] for r in output_rows))
    unique_ants = sorted(set(r['antenna'] for r in output_rows))
    logger.info(f"Main table rewritten: {len(output_rows)} rows "
                f"({len(unique_ants)} antennas × {len(unique_times)} timestamps).")


def flatten_caltable(input_path, output_path, dry_run=False):
    """Top-level driver: read, stitch, write."""

    if not os.path.isdir(input_path):
        logger.error(f"Input table not found: {input_path}")
        return False

    if os.path.exists(output_path):
        logger.error(f"Output path already exists: {output_path}. "
                     "Remove it first or choose a different name.")
        return False

    # Read SPW structure
    spw_order, spw_freqs, spw_nchan = read_spw_info(input_path)

    if len(spw_order) == 1:
        logger.info("Table already has a single SPW  nothing to do.")
        if not dry_run:
            logger.info(f"Copying as-is to {output_path}")
            pt.tablecopy(input_path, output_path)
        return True

    # Read and stitch main table data
    output_rows, n_pol, cparam_transposed = read_main_table(
        input_path, spw_order, spw_nchan
    )

    if dry_run:
        total_chan = sum(spw_nchan[s] for s in spw_order)
        unique_times = sorted(set(r['time'] for r in output_rows))
        unique_ants = sorted(set(r['antenna'] for r in output_rows))
        logger.info(f"[DRY RUN] Would write: {len(unique_ants)} antennas × "
                     f"{len(unique_times)} timestamps × {total_chan} channels × "
                     f"{n_pol} pols = {len(output_rows)} rows ? {output_path}")
        return True

    # Write flattened table
    write_flattened_table(
        input_path, output_path, spw_order, spw_freqs, spw_nchan,
        output_rows, n_pol, cparam_transposed
    )

    # --- Verification ---
    logger.info("Verifying output table...")
    with pt.table(output_path, ack=False) as t:
        n_rows = t.nrows()
        cparam0 = t.getcell('CPARAM', 0)
        times = t.getcol('TIME')
        ants = t.getcol('ANTENNA1')
        unique_t = sorted(set(times))
        unique_a = sorted(set(ants))
        logger.info(f"  Rows: {n_rows} ({len(unique_a)} ants × {len(unique_t)} times)")
        logger.info(f"  CPARAM shape per row: {cparam0.shape}")

    with pt.table(os.path.join(output_path, 'SPECTRAL_WINDOW'), ack=False) as t:
        logger.info(f"  SPW rows: {t.nrows()}, "
                     f"NUM_CHAN: {t.getcell('NUM_CHAN', 0)}")

    logger.info("Done.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Flatten a multi-SPW CASA calibration table into a "
                    "single-SPW table with all channels concatenated.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flatten_caltable_spw.py calibration_20241222_10h.B  flat_20241222_10h.B
  python flatten_caltable_spw.py input.B output.B --dry-run
        """
    )
    parser.add_argument('input', help='Path to input CASA calibration table')
    parser.add_argument('output', help='Path for output flattened table')
    parser.add_argument('--dry-run', action='store_true',
                        help='Read and report structure without writing')
    args = parser.parse_args()

    success = flatten_caltable(args.input, args.output, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
