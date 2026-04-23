import numpy as np
import pyuvdata

# Adapted from Danny Jacobs


def flatten_caltable_spws(input_path, output_path, use_freqs=None):

    cal = pyuvdata.UVCal()
    cal.read(input_path)

    gain = np.ma.masked_where(cal.flag_array, cal.gain_array)
    gain = np.ma.sum(gain, axis=2)
    if use_freqs is None:
        freq_inds = np.arange(cal.Nfreqs)
        use_freqs = cal.freq_array
    else:
        freq_inds = np.digitize(use_freqs, cal.freq_array, right=True)
    gain = gain[:, freq_inds, :]
    gain.shape = (cal.Nants_data, len(freq_inds), 1, cal.Njones)

    # problem. we cant combine flags because the time staggering means that every time is flagged at least once
    flags = np.zeros(gain.shape, dtype=bool)
    flags[np.abs(gain) > 0.999] = 1  # in my file, gain=1 was flagged.
    flags[np.abs(gain) < 1e-9] = 1  # no gain no pain

    cal.select(times=cal.time_array[-1], frequencies=use_freqs)
    cal.flex_spw_id_array = np.zeros_like(use_freqs)
    cal.spw_array = np.array([0])
    cal.Nspws = 1
    cal.gain_array = gain
    cal.flag_array = flags

    cal.write_ms_cal(output_path, clobber=True)


if __name__ == "__main__":
    uv = pyuvdata.UVData()
    uv.read("/fast/rbyrne/20260407_123010-123201_83MHz.ms")
    input_path = "/lustre/pipeline/calibration/results/2026-04-07/10h/successful/20260407_191722/tables/calibration_2026-04-07_10h.B.flagged"
    output_path = "/fast/rbyrne/calibration_2026-04-07_10h_spwcorrected_83MHz.B.flagged"
    flatten_caltable_spws(input_path, output_path, use_freqs=uv.freq_array)
    output_path = "/fast/rbyrne/calibration_2026-04-07_10h_spwcorrected.B.flagged"
    flatten_caltable_spws(input_path, output_path)
