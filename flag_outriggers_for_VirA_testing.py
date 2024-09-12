import numpy as np
import sys
sys.path.append("LWA_data_preprocessing")
import LWA_preprocessing
import pyuvdata
import os

use_files = [
    "20240302_095007_18MHz_calibrated.ms",
    "20240302_095007_23MHz_calibrated.ms",
    "20240302_095007_27MHz_calibrated.ms",
    "20240302_095007_36MHz_calibrated.ms",
    "20240302_095007_41MHz_calibrated.ms",
    "20240302_095007_46MHz_calibrated.ms",
    "20240302_095007_50MHz_calibrated.ms",
    "20240302_095007_55MHz_calibrated.ms",
    "20240302_095007_59MHz_calibrated.ms",
    "20240302_095007_64MHz_calibrated.ms",
    "20240302_095007_73MHz_calibrated.ms",
    "20240302_095007_78MHz_calibrated.ms",
    "20240302_095007_82MHz_calibrated.ms",
]
data_path = "/lustre/rbyrne/2024-03-02/flux_testing"

for datafile in use_files:
    uv = pyuvdata.UVData()
    uv.read(f"{data_path}/{datafile}")
    LWA_preprocessing.flag_outriggers(uv, inplace=True)
    uv.reorder_pols(order="CASA")
    uv.write_ms(f"{data_path}/{datafile[:-3]}_core.ms", clobber=True)
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -taper-inner-tukey 30 -weight briggs 1 -no-update-model-required -mem 10 -no-reorder -name {data_path}/{datafile}_filtered_core {data_path}/{datafile[:-3]}_core.ms"
    )