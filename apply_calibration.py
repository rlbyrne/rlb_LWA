import pyuvdata
import os

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
    data_path = "/lustre/rbyrne/2024-03-02/flux_testing"
    datafile = [
        filename
        for filename in os.listdir(data_path)
        if filename.endswith(f"_{freq_band}MHz.ms")
    ][0][:-3]
    cal = pyuvdata.UVCal()
    cal.read(
        f"/lustre/rbyrne/2024-03-02/ruby/calibration_outputs/{freq_band}_extended_sources.calfits"
    )
    uv = pyuvdata.UVData()
    uv.read_ms(f"{data_path}/{datafile}.ms", data_column="DATA")
    pyuvdata.utils.uvcalibrate(uv, cal, inplace=True, time_check=False)
    uv.write_ms(f"{data_path}/{datafile}_calibrated.ms")
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -weight briggs 0 -no-update-model-required -mem 10 -no-reorder -name {data_path}/{datafile}_nofilter {data_path}/{datafile}_calibrated.ms"
    )
    os.system(
        f"/opt/bin/wsclean -pol I -multiscale -multiscale-scale-bias 0.8 -size 4096 4096 -scale 0.03125 -niter 0 -mgain 0.85 -taper-inner-tukey 30 -weight briggs 1 -no-update-model-required -mem 10 -no-reorder -name {data_path}/{datafile}_filtered {data_path}/{datafile}_calibrated.ms"
    )
