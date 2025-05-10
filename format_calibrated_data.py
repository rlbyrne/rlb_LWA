import pyuvdata
import numpy as np
from calico import calibration_qa
import LWA_preprocessing

def plot_gains():
    use_freq_bands = [
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "69",
        "73",
        "78",
        "82",
    ]
    data_dir = "/lustre/rbyrne/2025-05-05"

    # Plot gains
    calibration_qa.plot_gains(
        [f"{data_dir}/20250505_123014-123204_{use_freq}MHz_compact.calfits" for use_freq in use_freq_bands],
        plot_output_dir=f"/lustre/rbyrne/2025-05-05/gains_plots",
        cal_name=[f"compact_calibrated"],
        plot_reciprocal=False,
        ymin=0,
        ymax=None,
        zero_mean_phase=True,
    )

def format_data_for_fhd():
    
    use_freq_bands = [
        "41",
        "46",
        "50",
        "55",
        "59",
        "64",
        "69",
        "73",
        "78",
        "82",
    ]
    for freq_ind, freq in enumerate(use_freq_bands):
        filename = ""
        uv_new = pyuvdata.UVData()
        uv_new.read(filename)
        LWA_preprocessing.flag_outriggers(
            uv_new,
            remove_outriggers=True,
            inplace=True,
        )
        if freq_ind == 0:
            uv = uv_new
        else:
            uv.fast_concat(uv_new, "freq", inplace=True)
        

if __name__=="__main__":
    plot_gains()