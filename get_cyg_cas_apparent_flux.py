import sys
import pyuvdata
sys.path.append("/home/rbyrne/rlb_LWA/LWA_data_preprocessing")
import LWA_preprocessing
import os

#file_directory = "/lustre/rbyrne/2024-03-02/ruby/calibration_outputs"
file_directory = "/lustre/rbyrne/2024-03-02/ruby/calibration_models"

for freq_band in [
    "18",
    "23",
    "27",
    "36",
    "41",
    "46",
    "50",
    "55",
    "59",
    "64",
    "73",
    "78",
    "82",
]:
    #filename = f"{file_directory}/{freq_band}_extended_sources_calibrated.ms"
    filename = f"{file_directory}/{freq_band}_deGasperin_sources.ms"
    uv = pyuvdata.UVData()
    uv.read(filename)
    LWA_preprocessing.flag_outriggers(
        uv,
        remove_outriggers=True,
        inplace=True,
    )
    uv.reorder_pols(order="CASA")
    uv.write_ms(f"{file_directory}/{freq_band}_deGasperin_sources_core.ms")
    os.system(f"/opt/bin/wsclean -multiscale -multiscale-scale-bias 0.8 -pol I -size 4096 4096 -scale 0.03125 -niter 0 -taper-inner-tukey 30 -mgain 0.85 -weight briggs 1 -no-update-model-required -mem 10 -no-reorder -name {file_directory}/{freq_band}_deGasperin_sources_core_filtered {file_directory}/{freq_band}_deGasperin_sources_core.ms")